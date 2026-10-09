"""Stage 2 orchestration: chunks.jsonl -> dense + sparse vectors -> Qdrant.

Built for slow CPU embedding endpoints (a laptop needs hours for 12k chunks):

* chunks are embedded and upserted in small groups, so work is saved continuously
* an interrupted build resumes: chunks already in the collection (same embedder
  fingerprint) are skipped - Ctrl+C, a crash or a reboot loses at most one group
* when only some documents changed, only their new chunks are embedded
* a different embedder (another vector space) always triggers a full rebuild
* progress is logged at least every 15 s with rate and ETA
* at the end stale points are removed and the point count is verified
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time

from ..config import Settings
from ..embed.base import Progress, chunk_embed_text, embed_in_batches, get_embedder
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

        embedder = get_embedder(settings)
        try:
            fp_now = embedder.fingerprint()      # cheap: model name, or hash of the fitted LSA file
        except Exception:
            fp_now = None                        # e.g. LSA not fitted yet

        # ---- skip if nothing changed (same chunks, same embedding model, complete collection)
        if not recreate and store.exists() and fp_now:
            try:
                meta = store.read_meta()
                if (meta.get("chunks_sha256") == chunks_sha and meta.get("embed_fingerprint") == fp_now
                        and meta.get("points") == store.count() == len(chunks)):
                    log.info("index up to date - nothing to do", extra={"count": len(chunks)})
                    return {"run_id": current_run_id(), "stage": "index", "status": "up_to_date", "points": len(chunks)}
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
            mode, existing = "full", set()
            avg = sp.average_length(texts)
            if store.exists():
                store.client.delete_collection(store.collection)
        else:
            existing = store.existing_chunk_ids()
            mode = "resume" if prog.get("status") == "building" else "incremental"
            # keep the BM25 length normalisation of the points already written
            avg = prog.get("bm25_avg_len") or old_meta.get("bm25_avg_len") or sp.average_length(texts)
        wanted = {c.chunk_id for c in chunks}
        todo = [i for i, c in enumerate(chunks) if c.chunk_id not in existing]
        already = len(wanted & existing)
        log.info(f"index plan: {mode} - {already} chunks already indexed, {len(todo)} to embed",
                 extra={"mode": mode, "count": len(todo), "already": already, "embed_fingerprint": fp})
        store.write_progress({"status": "building", "embed_fingerprint": fp, "bm25_avg_len": avg,
                              "started": dt.datetime.now().isoformat(timespec="seconds"), "run_id": current_run_id()})

        # ---- embed + upsert group by group (saved continuously)
        group = max(settings.embed_batch_size * 8, 64)
        progress = Progress(total=len(chunks), already_done=already, label="embedding progress")
        dim = old_meta.get("dim") if mode != "full" else None
        written = 0
        with stage("index.embed"), step("embed + upsert", log) as r:
            for g in range(0, len(todo), group):
                idx = todo[g:g + group]
                dense = embed_in_batches(embedder, [texts[i] for i in idx], settings.embed_batch_size, progress)
                sparse = [sp.doc_vector(texts[i], avg) for i in idx]
                if dim is None or not store.exists():
                    dim = int(dense.shape[1])
                    store.ensure_collection(dim)
                written += store.upsert([chunks[i] for i in idx], dense, sparse)
            r.update(count=written, mode=mode)

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
            "chunks_sha256": chunks_sha,
            "embed_backend": settings.embed_backend,
            "embed_fingerprint": fp,
            "dim": int(dim),
            "bm25_avg_len": avg,
            "mode": mode,
            "embedded_this_run": written,
        }
        store.write_meta(meta)
        store.write_progress({"status": "complete", "embed_fingerprint": fp, "bm25_avg_len": avg,
                              "finished": meta["built_at"]})
        summary = {"run_id": current_run_id(), "stage": "index", "status": "built", **meta,
                   "duration_s": round(time.perf_counter() - t_run, 2)}
        (settings.reports_dir / f"index_{current_run_id()}.json").write_text(json.dumps(summary, indent=1))
        log.info("index finished", extra={"count": n, "embedded": written, "duration_s": summary["duration_s"]})
        return summary
