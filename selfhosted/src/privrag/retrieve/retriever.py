"""Hybrid retrieval: dense (semantic) + BM25 (keyword) -> RRF fusion -> optional
rerank -> per-document cap.

Why both: dense vectors (BGE-M3) bridge English questions and German text;
BM25 nails exact legal references ("§ 25h", "Artikel 19") and German terms.
English questions are expanded with German terms (LLM rewrite + glossary)
before the keyword search.

Degradation: if the dense side fails (embedding endpoint down) the request
continues on BM25 alone with a warning, and vice versa. Only if both fail
does retrieval raise.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path

from ..config import Settings
from ..embed.base import Embedder, chunk_embed_text, get_embedder
from ..errors import IndexError_, PrivRagError, RetrievalError
from ..index import sparse as sp
from ..index.store import VectorStore
from ..logging_setup import get_logger, step
from ..models import Chunk, RetrievedChunk
from .rerank import Reranker, get_reranker

log = get_logger("retrieve")
RRF_K = 60
_GLOSSARY_PATH = Path(__file__).with_name("glossary_en_de.json")


@lru_cache(maxsize=1)
def _glossary() -> list[tuple[re.Pattern, list[str]]]:
    raw = json.loads(_GLOSSARY_PATH.read_text(encoding="utf-8"))
    items = [(k, v) for k, v in raw.items() if not k.startswith("_")]
    items.sort(key=lambda kv: -len(kv[0]))  # longest phrase first
    return [(re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v) for k, v in items]


def glossary_terms(question: str) -> list[str]:
    terms: list[str] = []
    for pat, de in _glossary():
        if pat.search(question):
            terms += [t for t in de if t not in terms]
    return terms


def rrf(rank_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for r, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r + 1)
    return scores


class Retriever:
    def __init__(self, s: Settings, store: VectorStore | None = None, embedder: Embedder | None = None,
                 reranker: Reranker | None = None, use_glossary: bool = True):
        self.s = s
        self.store = store or VectorStore(s)
        try:
            meta = self.store.read_meta()
        except IndexError_ as exc:
            raise RetrievalError(str(exc), code="INDEX_NOT_BUILT") from exc
        if meta.get("embed_backend") != s.embed_backend:
            raise RetrievalError(
                f"index was built with embed_backend={meta.get('embed_backend')} but settings say {s.embed_backend} "
                f"- rebuild the index or fix the config", code="INDEX_EMBEDDER_MISMATCH")
        self.embedder = embedder or get_embedder(s)
        if embedder is None and meta.get("embed_fingerprint"):
            try:
                fp = self.embedder.fingerprint()
            except Exception:
                fp = None
            if fp and fp != meta["embed_fingerprint"]:
                raise RetrievalError(
                    f"index was built with embedding model '{meta['embed_fingerprint']}' but the configured one is "
                    f"'{fp}' - query vectors would not match; rebuild the index or fix the config",
                    code="INDEX_EMBEDDER_MISMATCH")
        self.reranker = reranker if reranker is not None else get_reranker(s)
        self.use_glossary = use_glossary

    def retrieve(self, question: str, german_query: str | None = None, top_k: int | None = None,
                 doc_ids: list[str] | None = None, regulators: list[str] | None = None) -> tuple[list[RetrievedChunk], dict]:
        """Multi-query hybrid retrieval.

        Queries: the original question, plus a German version (LLM rewrite and/or
        glossary terms). Each query runs through dense and BM25 search; all result
        lists are fused with RRF. Measured on the UC13 question set this is the
        single biggest retrieval gain for English questions on German documents.
        """
        top_k = top_k or self.s.top_k
        n = self.s.candidate_k
        debug: dict = {"warnings": [], "timings_ms": {}}
        de_parts = [german_query] if german_query else []
        if self.use_glossary:
            de_parts += glossary_terms(question)
        de_query = " ".join(dict.fromkeys(p for p in de_parts if p and p.strip())) or None
        debug["german_query"] = de_query
        queries = [question] + ([de_query] if de_query else [])

        by_id: dict[str, Chunk] = {}
        lists: dict[str, list[str]] = {}
        failed: dict[str, Exception] = {}

        for kind in ("dense", "sparse"):
            t0 = time.perf_counter()
            for qi, q in enumerate(queries):
                name = f"{kind}_{'q' if qi == 0 else 'de'}"
                if kind in failed:
                    break
                try:
                    with step(f"{name} search", log, level=10) as r:
                        if kind == "dense":
                            hits = self.store.search_dense(self.embedder.embed_query(q), n, doc_ids, regulators)
                        else:
                            ids, vals = sp.query_vector(q)
                            hits = self.store.search_sparse(ids, vals, n, doc_ids, regulators)
                        lists[name] = [c.chunk_id for c, _ in hits]
                        by_id.update({c.chunk_id: c for c, _ in hits})
                        r.update(count=len(hits))
                except Exception as exc:
                    failed[kind] = exc
                    other = "keyword" if kind == "dense" else "dense"
                    debug["warnings"].append(f"{kind} search failed ({getattr(exc, 'code', type(exc).__name__)}) - {other} only")
                    log.warning(f"{kind} search failed - continuing with {other} search", extra={"error": str(exc)})
            debug["timings_ms"][kind] = round((time.perf_counter() - t0) * 1000, 1)

        if len(failed) == 2:
            raise RetrievalError(f"both retrievers failed: {failed['dense']}; {failed['sparse']}", code="RETRIEVE_ALL_FAILED")

        fused = rrf(list(lists.values()))
        dense_ids = lists.get("dense_q", []) or lists.get("dense_de", [])
        sparse_ids = lists.get("sparse_q", []) or lists.get("sparse_de", [])
        dense_rank = {cid: i + 1 for i, cid in enumerate(dense_ids)}
        sparse_rank = {cid: i + 1 for i, cid in enumerate(sparse_ids)}
        ranked = sorted(fused, key=lambda cid: -fused[cid])
        results = [RetrievedChunk(chunk=by_id[cid], score=fused[cid], dense_rank=dense_rank.get(cid),
                                  sparse_rank=sparse_rank.get(cid)) for cid in ranked]

        if self.reranker and results:
            t0 = time.perf_counter()
            try:
                cand = results[: n]
                scores = self.reranker.score(question, [chunk_embed_text(r.chunk) for r in cand])
                for r, sc in zip(cand, scores):
                    r.rerank_score = sc
                results = sorted(cand, key=lambda r: -(r.rerank_score or 0.0))
            except Exception as exc:
                debug["warnings"].append(f"reranker failed ({getattr(exc, 'code', type(exc).__name__)}) - fused order kept")
                log.warning("reranker failed - keeping fused order", extra={"error": str(exc)})
            debug["timings_ms"]["rerank"] = round((time.perf_counter() - t0) * 1000, 1)

        # diversity: avoid 8 chunks from one long document crowding out the second source
        out: list[RetrievedChunk] = []
        per_doc: dict[str, int] = {}
        for r in results:
            if per_doc.get(r.chunk.doc_id, 0) >= self.s.max_chunks_per_doc:
                continue
            per_doc[r.chunk.doc_id] = per_doc.get(r.chunk.doc_id, 0) + 1
            out.append(r)
            if len(out) >= top_k:
                break
        debug["candidates"] = {k: len(v) for k, v in lists.items()} | {"fused": len(fused)}
        return out, debug
