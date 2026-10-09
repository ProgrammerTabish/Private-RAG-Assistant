"""Qdrant vector store: one collection, a named dense vector + a BM25 sparse vector.

* point ids are uuid5(chunk_id) -> re-indexing is idempotent (upsert overwrites)
* upserts are batched and retried; the final point count is verified
* the embedder fingerprint is stored in ``index_meta.<mode>.<collection>.json``; querying with a
  different embedder than the one used for indexing is refused (it would
  silently return garbage otherwise)
* modes: ``memory`` (tests), ``path`` (local file), ``server`` (Qdrant on Azure)
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings
from ..errors import IndexError_
from ..logging_setup import get_logger
from ..models import Chunk

log = get_logger("index.store")
NS = uuid.UUID("6f1c3c1e-8a52-4d0e-9c1b-2b8f6a1d9e77")
PAYLOAD_KEYS = ("doc_id", "regulator", "doc_type", "language")

_clients: dict[str, Any] = {}


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NS, chunk_id))


def _client(s: Settings):
    from qdrant_client import QdrantClient

    key = f"{s.qdrant_mode}:{s.qdrant_url or s.qdrant_path}"
    if key in _clients:
        return _clients[key]
    try:
        if s.qdrant_mode == "memory":
            c = QdrantClient(":memory:")
        elif s.qdrant_mode == "path":
            s.qdrant_path.mkdir(parents=True, exist_ok=True)
            c = QdrantClient(path=str(s.qdrant_path))
        else:
            c = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)
            c.get_collections()  # fail fast if the server is unreachable
    except Exception as exc:
        hint = " (another process holds the local Qdrant folder lock - stop the API server or use qdrant_mode=server)" \
            if "already accessed" in str(exc) or "lock" in str(exc).lower() else ""
        raise IndexError_(f"cannot open Qdrant ({s.qdrant_mode}): {exc}{hint}", code="INDEX_UNAVAILABLE") from exc
    _clients[key] = c
    return c


def close_clients() -> None:
    for c in _clients.values():
        try:
            c.close()
        except Exception:
            pass
    _clients.clear()


import atexit  # noqa: E402

atexit.register(close_clients)  # release the local Qdrant lock cleanly on exit


class VectorStore:
    def __init__(self, s: Settings):
        self.s = s
        self.client = _client(s)
        self.collection = s.collection
        # one metadata record per Qdrant target, so a server collection and a local one never overwrite each other
        self.meta_path: Path = s.data_dir / f"index_meta.{s.qdrant_mode}.{s.collection}.json"

    # ---------------------------------------------------------------- schema
    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def dense_dim(self) -> int | None:
        if not self.exists():
            return None
        return self.client.get_collection(self.collection).config.params.vectors["dense"].size

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client import models as m

        if self.exists():
            if recreate:
                self.client.delete_collection(self.collection)
                log.info("collection dropped for rebuild", extra={"collection": self.collection})
            else:
                have = self.dense_dim()
                if have != dim:
                    raise IndexError_(f"collection has dim {have}, embedder produces {dim} - rebuild with --recreate",
                                      code="INDEX_DIM_MISMATCH", have=have, want=dim)
                return
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": m.VectorParams(size=dim, distance=m.Distance.COSINE)},
            sparse_vectors_config={"sparse": m.SparseVectorParams(modifier=m.Modifier.IDF)},
        )
        if self.s.qdrant_mode == "server":
            for k in PAYLOAD_KEYS:
                self.client.create_payload_index(self.collection, k, m.PayloadSchemaType.KEYWORD)
        log.info("collection created", extra={"collection": self.collection, "dim": dim})

    # ---------------------------------------------------------------- write
    def upsert(self, chunks: list[Chunk], dense: np.ndarray | None, sparse: list[tuple[list[int], list[float]]]) -> int:
        """Upsert points. ``dense=None`` writes keyword-only points (dense vector added by a later run)."""
        from qdrant_client import models as m

        if dense is not None and len(dense) != len(chunks):
            raise IndexError_("chunks/vectors length mismatch", code="INDEX_LENGTH_MISMATCH",
                              chunks=len(chunks), dense=len(dense), sparse=len(sparse))
        if not len(chunks) == len(sparse):
            raise IndexError_("chunks/vectors length mismatch", code="INDEX_LENGTH_MISMATCH",
                              chunks=len(chunks), sparse=len(sparse))

        @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=0.5, max=10), reraise=True)
        def _put(points):
            self.client.upsert(self.collection, points=points, wait=True)

        bs = self.s.upsert_batch_size
        written = 0
        for i in range(0, len(chunks), bs):
            points = []
            dpart = dense[i:i + bs] if dense is not None else [None] * len(chunks[i:i + bs])
            for c, d, (si, sv) in zip(chunks[i:i + bs], dpart, sparse[i:i + bs]):
                vec: dict[str, Any] = {} if d is None else {"dense": d.tolist()}
                if si:
                    vec["sparse"] = m.SparseVector(indices=si, values=sv)
                payload = c.model_dump(mode="json") | {"has_dense": d is not None}
                points.append(m.PointStruct(id=point_id(c.chunk_id), vector=vec, payload=payload))
            try:
                _put(points)
            except Exception as exc:
                raise IndexError_(f"upsert failed at batch starting {i}: {exc}", code="INDEX_UPSERT_FAILED",
                                  batch_start=i) from exc
            written += len(points)
        return written

    def delete_missing(self, keep_chunk_ids: set[str]) -> int:
        """Remove points whose chunk no longer exists (document removed or re-chunked)."""
        from qdrant_client import models as m

        stale: list[str] = []
        offset = None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=1000, offset=offset,
                                             with_payload=["chunk_id"], with_vectors=False)
            stale += [p.id for p in pts if (p.payload or {}).get("chunk_id") not in keep_chunk_ids]
            if offset is None:
                break
        if stale:
            self.client.delete(self.collection, points_selector=m.PointIdsList(points=stale), wait=True)
        return len(stale)

    def existing_points(self) -> dict[str, bool]:
        """chunk_id -> has a dense vector. Used to resume a build, to add only new chunks and
        to continue dense coverage of a time-budgeted (keyword-only) index."""
        if not self.exists():
            return {}
        out: dict[str, bool] = {}
        offset = None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=1000, offset=offset,
                                             with_payload=["chunk_id", "has_dense"], with_vectors=False)
            for p in pts:
                pl = p.payload or {}
                if pl.get("chunk_id"):
                    out[pl["chunk_id"]] = bool(pl.get("has_dense", True))   # older points always had dense
            if offset is None:
                break
        return out

    def existing_chunk_ids(self) -> set[str]:
        return set(self.existing_points())

    @property
    def progress_path(self) -> Path:
        return self.meta_path.with_name(self.meta_path.name.replace("index_meta", "index_progress", 1))

    def read_progress(self) -> dict:
        try:
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def write_progress(self, data: dict) -> None:
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.progress_path)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.exists() else 0

    # ---------------------------------------------------------------- read
    def _filter(self, doc_ids: list[str] | None, regulators: list[str] | None):
        from qdrant_client import models as m

        must = []
        if doc_ids:
            must.append(m.FieldCondition(key="doc_id", match=m.MatchAny(any=doc_ids)))
        if regulators:
            must.append(m.FieldCondition(key="regulator", match=m.MatchAny(any=regulators)))
        return m.Filter(must=must) if must else None

    def search_dense(self, vec: np.ndarray, limit: int, doc_ids=None, regulators=None) -> list[tuple[Chunk, float]]:
        res = self.client.query_points(self.collection, query=vec.tolist(), using="dense", limit=limit,
                                       with_payload=True, query_filter=self._filter(doc_ids, regulators))
        return [(Chunk.model_validate(p.payload), float(p.score)) for p in res.points]

    def search_sparse(self, ids: list[int], vals: list[float], limit: int, doc_ids=None, regulators=None) -> list[tuple[Chunk, float]]:
        from qdrant_client import models as m

        if not ids:
            return []
        res = self.client.query_points(self.collection, query=m.SparseVector(indices=ids, values=vals), using="sparse",
                                       limit=limit, with_payload=True, query_filter=self._filter(doc_ids, regulators))
        return [(Chunk.model_validate(p.payload), float(p.score)) for p in res.points]

    # ---------------------------------------------------------------- metadata
    def write_meta(self, meta: dict) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def read_meta(self) -> dict:
        if not self.meta_path.exists():
            raise IndexError_(f"{self.meta_path.name} missing - run 'privrag index' first", code="INDEX_NOT_BUILT")
        return json.loads(self.meta_path.read_text(encoding="utf-8"))
