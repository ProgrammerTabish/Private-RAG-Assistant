"""Stage 1 orchestration: manifest -> validate -> parse -> chunk -> chunks.jsonl

Robustness guarantees
* one failing document never stops the run; it is recorded with an error code
* checkpoint per document (data/parsed/<doc_id>.chunks.jsonl + ingest_state.json);
  unchanged documents are skipped on re-run, an interrupted run resumes
* a document whose content hash equals another one is skipped as a duplicate
* if the configured parser fails, PyMuPDF is tried as a fallback
* outputs are written atomically (tmp file + rename), chunk ids are checked unique
* a JSON report per run lists status, pages, chunks, warnings and timings per document
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from ..config import Settings
from ..errors import ChunkError, ParseError, PrivRagError
from ..logging_setup import current_run_id, doc_context, get_logger, stage, step
from ..models import Chunk
from .chunker import ChunkStats, chunk_document
from .manifest import load_manifest, reconcile
from .parsers import PyMuPDFParser, get_parser, sha256_file, validate_pdf_file

log = get_logger("ingest")
PIPELINE_VERSION = "1"


def _code_fingerprint() -> str:
    """Hash of the ingestion source code - a logic change must invalidate checkpoints too,
    otherwise a re-run silently keeps chunks produced by the old code."""
    h = hashlib.sha1()
    for name in ("cleaning.py", "sections.py", "chunker.py", "parsers.py"):
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:10]


def _settings_fingerprint(s: Settings) -> str:
    keys = ("parser_backend", "chunk_size", "chunk_overlap", "min_chunk_chars",
            "header_footer_min_ratio", "min_text_chars_per_page")
    raw = json.dumps({k: getattr(s, k) for k in keys} | {"v": PIPELINE_VERSION, "code": _code_fingerprint()},
                     sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("ingest state file unreadable - starting fresh", extra={"path": str(path)})
        return {}


def run_ingest(settings: Settings, only: list[str] | None = None, force: bool = False) -> dict:
    settings.ensure_dirs()
    state_path = settings.data_dir / "ingest_state.json"
    fp = _settings_fingerprint(settings)
    t_run = time.perf_counter()

    with stage("ingest"):
        log.info("ingest started", extra={"pdf_dir": str(settings.pdf_dir), "parser": settings.parser_backend,
                                          "fingerprint": fp, "force": force})
        if not settings.pdf_dir.exists():
            raise ParseError(f"pdf_dir does not exist: {settings.pdf_dir}", code="PARSE_DIR_MISSING")
        pdfs = sorted(p for p in settings.pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
        if only:
            wanted = {o if o.lower().endswith(".pdf") else f"{o}.pdf" for o in only}
            pdfs = [p for p in pdfs if p.name in wanted or p.stem in only]
        if not pdfs:
            raise ParseError("no PDF files found", code="PARSE_NO_FILES", pdf_dir=str(settings.pdf_dir))

        with stage("ingest.manifest"):
            manifest = load_manifest(settings.manifest_path)
            metas, rec_warnings = reconcile(manifest, pdfs)

        state = {} if force else _load_state(state_path)
        primary = None
        try:
            primary = get_parser(settings.parser_backend)
        except PrivRagError as exc:
            log.error("configured parser unavailable - using pymupdf fallback",
                      extra={"error_code": exc.code, "backend": settings.parser_backend})
        fallback = PyMuPDFParser()

        report_docs: list[dict] = []
        seen_hashes: dict[str, str] = {}
        for meta in metas:
            path = settings.pdf_dir / meta.file
            entry: dict = {"doc_id": meta.doc_id, "file": meta.file, "from_manifest": meta.from_manifest}
            t0 = time.perf_counter()
            with doc_context(meta.doc_id):
                try:
                    with stage("ingest.validate"):
                        validate_pdf_file(path, settings.max_pdf_mb)
                        meta.sha256 = sha256_file(path)
                    if meta.sha256 in seen_hashes:
                        entry.update(status="duplicate", duplicate_of=seen_hashes[meta.sha256])
                        log.warning("duplicate content - skipped", extra={"duplicate_of": seen_hashes[meta.sha256]})
                        report_docs.append(entry)
                        continue
                    seen_hashes[meta.sha256] = meta.doc_id

                    ckpt = settings.parsed_dir / f"{meta.doc_id}.chunks.jsonl"
                    prev = state.get(meta.doc_id, {})
                    if (prev.get("status") == "ok" and prev.get("sha256") == meta.sha256
                            and prev.get("fingerprint") == fp and ckpt.exists()):
                        entry.update(status="skipped_unchanged", chunks=prev.get("chunks"), pages=prev.get("pages"))
                        log.info("unchanged since last run - reused checkpoint", extra={"chunks": prev.get("chunks")})
                        report_docs.append(entry)
                        continue

                    parser_used = None
                    with stage("ingest.parse"), step("parse", log) as r:
                        try:
                            if primary is None:
                                raise ParseError("primary parser unavailable", code="PARSE_BACKEND_MISSING")
                            parsed = primary.parse(path, meta)
                            parser_used = primary.name
                        except ParseError as exc:
                            if exc.code in {"PARSE_ENCRYPTED", "PARSE_NOT_PDF", "PARSE_EMPTY_FILE"} or primary is fallback:
                                raise
                            log.warning("primary parser failed - trying pymupdf fallback",
                                        extra={"error_code": exc.code})
                            parsed = fallback.parse(path, meta)
                            parsed.warnings.append(f"fallback parser used ({exc.code})")
                            parser_used = fallback.name
                        meta.pages = len(parsed.pages)
                        r.update(pages=meta.pages, parser=parser_used)

                    with stage("ingest.chunk"), step("chunk", log) as r:
                        st = ChunkStats()
                        chunks = chunk_document(
                            parsed, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
                            min_chunk_chars=settings.min_chunk_chars,
                            header_footer_min_ratio=settings.header_footer_min_ratio,
                            min_text_chars_per_page=settings.min_text_chars_per_page, stats=st,
                        )
                        r.update(count=len(chunks), sections=st.sections, empty_pages=st.empty_pages)

                    warnings = list(parsed.warnings)
                    if meta.pages and st.empty_pages / meta.pages > 0.3:
                        warnings.append(f"{st.empty_pages}/{meta.pages} pages have almost no text (scanned? OCR needed)")
                    if st.sections == 0:
                        warnings.append("no section headings detected - citations will use page numbers only")
                    _atomic_write(ckpt, "".join(c.model_dump_json() + "\n" for c in chunks))
                    entry.update(status="ok", parser=parser_used, pages=meta.pages, chunks=len(chunks),
                                 sections=st.sections, heading_kinds=st.heading_kinds,
                                 empty_pages=st.empty_pages, toc_pages=st.toc_pages, dropped_small=st.dropped_small,
                                 avg_chunk_chars=round(sum(c.char_len for c in chunks) / len(chunks)),
                                 warnings=warnings)
                    for w in warnings:
                        log.warning(w)
                    state[meta.doc_id] = {"status": "ok", "sha256": meta.sha256, "fingerprint": fp,
                                          "chunks": len(chunks), "pages": meta.pages,
                                          "meta": meta.model_dump(mode="json")}
                except (ParseError, ChunkError) as exc:
                    entry.update(status="failed", error_code=exc.code, error=str(exc))
                    state[meta.doc_id] = {"status": "failed", "sha256": meta.sha256, "error_code": exc.code}
                    log.error("document failed - continuing with next", extra={"error_code": exc.code, "error": str(exc)})
                except Exception as exc:  # unexpected bug: still isolate, but log the traceback
                    entry.update(status="failed", error_code="UNEXPECTED", error=f"{type(exc).__name__}: {exc}")
                    state[meta.doc_id] = {"status": "failed", "sha256": meta.sha256, "error_code": "UNEXPECTED"}
                    log.exception("unexpected error - continuing with next", extra={"error_code": "UNEXPECTED"})
                finally:
                    entry["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    # checkpoint after every document so a crash can resume
                    _atomic_write(state_path, json.dumps(state, indent=1, ensure_ascii=False))
            report_docs.append(entry)

        with stage("ingest.assemble"), step("assemble chunks.jsonl", log) as r:
            total = _assemble(settings, [e for e in report_docs if e.get("status") in ("ok", "skipped_unchanged")])
            r.update(count=total)

        summary = {
            "run_id": current_run_id(),
            "stage": "ingest",
            "fingerprint": fp,
            "documents": len(report_docs),
            "ok": sum(e["status"] == "ok" for e in report_docs),
            "skipped_unchanged": sum(e["status"] == "skipped_unchanged" for e in report_docs),
            "duplicates": sum(e["status"] == "duplicate" for e in report_docs),
            "failed": sum(e["status"] == "failed" for e in report_docs),
            "total_chunks": total,
            "total_pages": sum(e.get("pages") or 0 for e in report_docs),
            "duration_s": round(time.perf_counter() - t_run, 2),
            "manifest_warnings": manifest.warnings + rec_warnings,
            "docs": report_docs,
        }
        rpath = settings.reports_dir / f"ingest_{current_run_id()}.json"
        _atomic_write(rpath, json.dumps(summary, indent=1, ensure_ascii=False))
        _atomic_write(settings.reports_dir / "ingest_latest.json", json.dumps(summary, indent=1, ensure_ascii=False))
        level = log.error if summary["failed"] else log.info
        level("ingest finished", extra={k: summary[k] for k in ("documents", "ok", "skipped_unchanged", "duplicates",
                                                                "failed", "total_chunks", "duration_s")} | {"report": str(rpath)})
        return summary


def _assemble(settings: Settings, entries: list[dict]) -> int:
    """Concatenate per-document checkpoints into chunks.jsonl, validating every record."""
    seen: set[str] = set()
    lines: list[str] = []
    for e in entries:
        ckpt = settings.parsed_dir / f"{e['doc_id']}.chunks.jsonl"
        with open(ckpt, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                c = Chunk.model_validate_json(line)  # raises on a corrupted checkpoint
                if c.chunk_id in seen:
                    raise ChunkError(f"duplicate chunk id {c.chunk_id}", code="CHUNK_DUPLICATE_ID", line=n)
                seen.add(c.chunk_id)
                lines.append(line if line.endswith("\n") else line + "\n")
    _atomic_write(settings.chunks_path, "".join(lines))
    return len(lines)


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise ChunkError(f"{path} not found - run ingest first", code="CHUNKS_MISSING")
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(Chunk.model_validate_json(line))
                except Exception as exc:
                    raise ChunkError(f"bad chunk record at line {n}: {exc}", code="CHUNKS_CORRUPT", line=n) from exc
    return out


def stats_to_dict(st: ChunkStats) -> dict:
    return asdict(st)
