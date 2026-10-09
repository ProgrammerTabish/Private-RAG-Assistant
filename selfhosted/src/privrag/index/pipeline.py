"""Stage 2 orchestration: chunks.jsonl -> dense + sparse vectors -> Qdrant.

Built for slow CPU embedding endpoints (a laptop needs hours for 12k chunks):

* chunks are embedded and upserted in small groups, so work is saved continuously
* an interrupted build resumes: chunks already in the collection (same embedder
  fingerprint) are skipped - Ctrl+C, a crash or a reboot loses at most one group
* when only some documents changed, only their new chunks are embedded
* a different embedder (another vector space) always triggers a full rebuild
* optional time budget (``index_time_budget_s``): every chunk first gets a keyword
  (BM25) point, then dense vectors are added for as many chunks as fit in the budget,
  spread over all documents. Later runs continue the dense coverage.
* progress is logged at least every 15 s with rate and ETA
* at the end stale points are removed and the point count is verified
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from collections import defaultdict

from ..config import Settings
from ..embed.base import Progress, chunk_embed_text, embed_in_batches, get_embedder
from ..errors import IndexError_
from ..ingest.pipeline import load_chunks
from ..logging_setup import current_run_id, get_logger, stage, step
from ..models import Chunk
from . import sparse as sp
from .store import VectorStore

log = get_logger("index")


def _file_sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vdc(k: int) -> float:
    """van der Corput sequence: 0, .5, .25, .75, .125 ... (evenly spread fractions)."""
    x, denom = 0.0, 1.0
    while k:
        denom *= 2
        k, r = divmod(k, 2)
        x += r / denom
    return x


def spread_order(chunks: list[Chunk], idx: list[int]) -> list[int]:
    """Order chunk indices so that a budget-limited dense pass covers every document:
    round-robin over documents, and inside a document positions spread over the whole
    text (start, middle, quarters, ...) instead of only the first pages."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        by_doc[chunks[i].doc_id].append(i)
    keyed: list[tuple[int, int, int]] = []
    for d_rank, (doc, items) in enumerate(sorted(by_doc.items())):
        n = len(items)
        picked: list[int] = []
        seen: set[int] = set()
        k = 0
        while len(picked) < n and k < 4 * n + 8:
            pos = min(int(_vdc(k) * n), n - 1)
            if pos not in seen:
                seen.add(pos)
                picked.append(pos)
            k += 1
        picked += [p for p in range(n) if p not in seen]
        keyed += [(rank, d_rank, items[pos]) for rank, pos in enumerate(picked)]
    return [i for _, _, i in sorted(keyed)]


def run_index(settings: Settings, recreate: bool = False) -> dict:
    settings.ensure_dirs()
    t_run = time.perf_counter()
    budget = settings.index_time_budget_s
    with stage("index"):
        chunks = load_chunks(settings.chunks_path)
        if not chunks:
            raise IndexError_("chunks.jsonl is empty", code="INDEX_NO_CHUNKS")
        chunks_sha = _file_sha(settings.chunks_path)
        store = VectorStore(settings)

        embedder = get_embedder(settings)
        try:
            fp_now = embedder.fingerprint()      # cheap: model name, or hash of the fitted LSA file
        except Exception:
            fp_now = None                        # e.g. LSA not fitted yet

        # ---- skip if nothing changed (same chunks, same embedding model, complete collection)
        if not recreate and store.exists() and fp_now:
            try:
                meta = store.read_meta()
                dense_done = meta.get("dense_points", meta.get("points")) == meta.get("points")
                if (meta.get("chunks_sha256") == chunks_sha and meta.get("embed_fingerprint") == fp_now
                        and meta.get("points") == store.count() == len(chunks)
                        and (dense_done or budget == 0)):
                    log.info("index up to date - nothing to do",
                             extra={"count": len(chunks), "dense_points": meta.get("dense_points", len(chunks))})
                    return {"run_id": current_run_id(), "stage": "index", "status": "up_to_date",
                            "points": len(chunks), "dense_points": meta.get("dense_points", len(chunks))}
            except IndexError_:
                pass

        texts = [chunk_embed_text(c) for c in chunks]
        with stage("index.embed"), step("fit embedder", log) as r:
            embedder.fit(texts)   # no-op for pre-trained models
            r.update(backend=embedder.name)
        fp = embedder.fingerprint()

        # ---- decide: resume / incremental update / full rebuild
        try:
            old_meta = store.read_meta()
        except IndexError_:
            old_meta = {}
        prog = store.read_progress()
        same_space = store.exists() and fp in (prog.get("embed_fingerprint"), old_meta.get("embed_fingerprint"))
        if recreate or not same_space:
            if store.exists() and not recreate:
                log.warning("embedder changed or unknown since last index - rebuilding collection",
                            extra={"old": old_meta.get("embed_fingerprint") or prog.get("embed_fingerprint"), "new": fp})
            mode, existing = "full", {}
            avg = sp.average_length(texts)
            if store.exists():
                store.client.delete_collection(store.collection)
        else:
            existing = store.existing_points()
            mode = "resume" if prog.get("status") == "building" else "incremental"
            # keep the BM25 length normalisation of the points already written
            avg = prog.get("bm25_avg_len") or old_meta.get("bm25_avg_len") or sp.average_length(texts)
        wanted = {c.chunk_id for c in chunks}
        missing = [i for i, c in enumerate(chunks) if c.chunk_id not in existing]
        need_dense = [i for i, c in enumerate(chunks) if not existing.get(c.chunk_id, False)]
        already = sum(1 for c in chunks if existing.get(c.chunk_id, False))
        budget_txt = "no time limit" if budget is None else f"time budget {budget / 60:.1f} min"
        log.info(f"index plan: {mode} - {already} chunks already have dense vectors, {len(need_dense)} to embed "
                 f"({budget_txt})", extra={"mode": mode, "count": len(need_dense), "already": already,
                                            "missing_points": len(missing), "embed_fingerprint": fp,
                                            "budget_s": budget})
        store.write_progress({"status": "building", "embed_fingerprint": fp, "bm25_avg_len": avg,
                              "started": dt.datetime.now().isoformat(timespec="seconds"), "run_id": current_run_id()})

        dim = old_meta.get("dim") if mode != "full" else None
        written = 0

        # ---- budget mode, phase 1: every chunk gets a keyword point right away (seconds)
        if budget is not None and missing:
            with stage("index.sparse"), step("keyword (BM25) points for all chunks", log) as r:
                if dim is None or not store.exists():
                    probe = embedder.embed_documents([texts[missing[0]]])      # learn the vector size
                    dim = int(probe.shape[1])
                    store.ensure_collection(dim)
                written_sparse = store.upsert([chunks[i] for i in missing], None,
                                              [sp.doc_vector(texts[i], avg) for i in missing])
                r.update(count=written_sparse)

        # ---- embed + upsert group by group (saved continuously), within the time budget
        order = spread_order(chunks, need_dense) if budget is not None else need_dense
        # smaller save points in budget mode, so the run stops close to the budget
        group = max(settings.embed_batch_size * 2, 16) if budget is not None else max(settings.embed_batch_size * 8, 64)
        progress = Progress(total=len(chunks), already_done=already,
                            label="BGE-M3 vectors" if budget is not None else "embedding progress",
                            deadline=(t_run + budget) if budget is not None else None,
                            every_n=settings.progress_every)
        stopped_by_budget = False
        with stage("index.embed"), step("embed + upsert", log) as r:
            last_group_s = 0.0
            for g in range(0, len(order), group):
                if budget is not None and (time.perf_counter() - t_run) + last_group_s > budget:
                    stopped_by_budget = True
                    break
                tg = time.perf_counter()
                idx = order[g:g + group]
                dense = embed_in_batches(embedder, [texts[i] for i in idx], settings.embed_batch_size, progress)
                sparse = [sp.doc_vector(texts[i], avg) for i in idx]
                if dim is None or not store.exists():
                    dim = int(dense.shape[1])
                    store.ensure_collection(dim)
                written += store.upsert([chunks[i] for i in idx], dense, sparse)
                last_group_s = time.perf_counter() - tg
            r.update(count=written, mode=mode, stopped_by_budget=stopped_by_budget)
        dense_points = already + written
        if stopped_by_budget:
            log.warning(f"time budget reached: dense vectors for {dense_points}/{len(chunks)} chunks "
                        f"({100 * dense_points / len(chunks):.1f}%); all chunks are keyword-searchable. "
                        f"Run the index again to add more.",
                        extra={"dense_points": dense_points, "count": len(chunks)})

        with stage("index.upsert"):
            if dim is None:          # nothing to embed and no previous meta
                dim = store.dense_dim()
            with step("remove stale points", log) as r:
                r.update(count=store.delete_missing(wanted))
            n = store.count()
            if n != len(chunks):
                raise IndexError_(f"verification failed: collection has {n} points, expected {len(chunks)}",
                                  code="INDEX_VERIFY_FAILED", have=n, want=len(chunks))

        meta = {
            "built_at": dt.datetime.now().isoformat(timespec="seconds"),
            "run_id": current_run_id(),
            "collection": settings.collection,
            "points": n,
            "dense_points": dense_points,
            "dense_coverage": round(dense_points / n, 4),
            "chunks_sha256": chunks_sha,
            "embed_backend": settings.embed_backend,
            "embed_fingerprint": fp,
            "dim": int(dim),
            "bm25_avg_len": avg,
            "mode": mode,
            "embedded_this_run": written,
        }
        store.write_meta(meta)
        store.write_progress({"status": "complete" if dense_points == n else "partial", "embed_fingerprint": fp,
                              "bm25_avg_len": avg, "finished": meta["built_at"]})
        summary = {"run_id": current_run_id(), "stage": "index", "status": "built", **meta,
                   "duration_s": round(time.perf_counter() - t_run, 2)}
        (settings.reports_dir / f"index_{current_run_id()}.json").write_text(json.dumps(summary, indent=1))
        log.info("index finished", extra={"count": n, "dense_points": dense_points, "embedded": written,
                                          "duration_s": summary["duration_s"]})
        return summary
