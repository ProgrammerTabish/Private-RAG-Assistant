"""Stage 2 orchestration: chunks.jsonl -> dense + sparse vectors -> Qdrant.

Skips work when the chunks file and the embedder are unchanged; otherwise
re-embeds, upserts (idempotent ids), removes stale points and verifies that the
collection holds exactly one point per chunk.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

from ..config import Settings
from ..embed.base import chunk_embed_text, embed_in_batches, get_embedder
from ..errors import IndexError_
from ..ingest.pipeline import load_chunks
from ..logging_setup import current_run_id, get_logger, stage, step
from . import sparse as sp
from .store import VectorStore

log = get_logger("index")


def _file_sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_index(settings: Settings, recreate: bool = False) -> dict:
    settings.ensure_dirs()
    t_run = time.perf_counter()
    with stage("index"):
        chunks = load_chunks(settings.chunks_path)
        if not chunks:
            raise IndexError_("chunks.jsonl is empty", code="INDEX_NO_CHUNKS")
        chunks_sha = _file_sha(settings.chunks_path)
        store = VectorStore(settings)

        # ---- skip if nothing changed
        if not recreate and store.exists():
            try:
                meta = store.read_meta()
                if (meta.get("chunks_sha256") == chunks_sha and meta.get("embed_backend") == settings.embed_backend
                        and meta.get("points") == store.count() == len(chunks)):
                    log.info("index up to date - nothing to do", extra={"count": len(chunks)})
                    return {"run_id": current_run_id(), "stage": "index", "status": "up_to_date", "points": len(chunks)}
            except IndexError_:
                pass

        texts = [chunk_embed_text(c) for c in chunks]
        embedder = get_embedder(settings)
        with stage("index.embed"), step("fit embedder", log) as r:
            embedder.fit(texts)   # no-op for pre-trained models
            r.update(backend=embedder.name, dim=embedder.dim)
        with stage("index.embed"), step("embed chunks", log) as r:
            dense = embed_in_batches(embedder, texts, settings.embed_batch_size)
            r.update(count=len(dense), dim=dense.shape[1])
        with stage("index.sparse"), step("bm25 sparse vectors", log) as r:
            avg = sp.average_length(texts)
            sparse = [sp.doc_vector(t, avg) for t in texts]
            empty = sum(1 for i, _ in sparse if not i)
            r.update(count=len(sparse), avg_len=round(avg, 1), empty=empty)

        with stage("index.upsert"):
            # an embedder change means a different vector space -> always rebuild
            try:
                old = store.read_meta()
            except IndexError_:
                old = {}
            must_recreate = recreate or (old and old.get("embed_fingerprint") != embedder.fingerprint())
            if must_recreate and not recreate:
                log.warning("embedder changed since last index - rebuilding collection",
                            extra={"old": old.get("embed_fingerprint"), "new": embedder.fingerprint()})
            store.ensure_collection(dense.shape[1], recreate=bool(must_recreate))
            with step("upsert points", log) as r:
                written = store.upsert(chunks, dense, sparse)
                r.update(count=written)
            with step("remove stale points", log) as r:
                r.update(count=store.delete_missing({c.chunk_id for c in chunks}))
            n = store.count()
            if n != len(chunks):
                raise IndexError_(f"verification failed: collection has {n} points, expected {len(chunks)}",
                                  code="INDEX_VERIFY_FAILED", have=n, want=len(chunks))

        meta = {
            "built_at": dt.datetime.now().isoformat(timespec="seconds"),
            "run_id": current_run_id(),
            "collection": settings.collection,
            "points": n,
            "chunks_sha256": chunks_sha,
            "embed_backend": settings.embed_backend,
            "embed_fingerprint": embedder.fingerprint(),
            "dim": int(dense.shape[1]),
            "bm25_avg_len": avg,
        }
        store.write_meta(meta)
        summary = {"run_id": current_run_id(), "stage": "index", "status": "built", **meta,
                   "duration_s": round(time.perf_counter() - t_run, 2)}
        (settings.reports_dir / f"index_{current_run_id()}.json").write_text(json.dumps(summary, indent=1))
        log.info("index finished", extra={"count": n, "duration_s": summary["duration_s"]})
        return summary
