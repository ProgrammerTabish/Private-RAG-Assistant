"""Command line: one command per pipeline stage, each with its own run log.

  privrag doctor            check config, files, index, endpoints
  privrag ingest [--force]  stage 1: PDFs -> chunks.jsonl
  privrag index [--recreate] stage 2: chunks -> Qdrant
  privrag ask "question"    stage 3: one question end to end
  privrag eval              stage 5: run the UC13 question set, write report
  privrag serve             stage 4: start the API
  privrag logs [--errors]   show the last run's log (errors only with --errors)

Exit code is non-zero when a stage fails, so it can run in CI or Azure jobs.
"""
from __future__ import annotations

import json
import sys

import typer

from .config import get_settings
from .errors import PrivRagError
from .logging_setup import new_run_id, setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Self-hosted RAG pipeline (UC13 Variant B)")


def _init(prefix: str):
    s = get_settings()
    s.ensure_dirs()
    rid = setup_logging(s.log_dir, run_id=new_run_id(prefix), level=s.log_level, console=s.log_console)
    typer.secho(f"run_id={rid}  log=logs/{rid}.jsonl", fg="cyan", err=True)
    return s


def _fail(exc: PrivRagError) -> None:
    typer.secho(f"FAILED [{exc.stage}] {exc.code}: {exc}", fg="red", err=True)
    typer.secho("details: privrag logs --errors", fg="red", err=True)
    raise typer.Exit(2)


@app.command()
def ingest(force: bool = typer.Option(False, help="re-process all documents"),
           only: list[str] = typer.Option(None, help="doc_id(s) to process"),
           allow_failures: int = typer.Option(0, help="exit 0 even if up to N documents fail")):
    """Stage 1: parse, clean, section and chunk the PDFs."""
    s = _init("ingest")
    from .ingest.pipeline import run_ingest
    try:
        r = run_ingest(s, only=only or None, force=force)
    except PrivRagError as exc:
        _fail(exc)
    typer.echo(json.dumps({k: r[k] for k in ("documents", "ok", "skipped_unchanged", "duplicates", "failed",
                                              "total_chunks", "total_pages", "duration_s")}, indent=1))
    for d in r["docs"]:
        if d["status"] == "failed":
            typer.secho(f"  {d['file']}: {d['error_code']} - {d['error']}", fg="red")
    if r["failed"] > allow_failures:
        raise typer.Exit(1)


@app.command()
def index(recreate: bool = typer.Option(False, help="drop and rebuild the collection")):
    """Stage 2: embed chunks and load them into Qdrant."""
    s = _init("index")
    from .index.pipeline import run_index
    try:
        r = run_index(s, recreate=recreate)
    except PrivRagError as exc:
        _fail(exc)
    typer.echo(json.dumps(r, indent=1, default=str))


@app.command()
def ask(question: str, doc: list[str] = typer.Option(None, help="restrict to doc_id(s)"),
        as_json: bool = typer.Option(False, "--json")):
    """Stage 3: answer one question (retrieve + generate) and print the citations."""
    s = _init("ask")
    from .generate.service import RagService
    try:
        a = RagService(s).ask(question, doc_ids=doc or None)
    except PrivRagError as exc:
        _fail(exc)
    if as_json:
        typer.echo(a.model_dump_json(indent=1))
        return
    typer.secho(f"[{a.status}] confidence={a.confidence} ({a.confidence_label})  {a.latency_ms}", fg="green")
    typer.echo(a.answer)
    for c in a.citations:
        typer.echo(f"  [{c.n}] {c.file} | {c.section or '-'} | p. {c.page_start}-{c.page_end}")
    for w in a.warnings:
        typer.secho(f"  warning: {w}", fg="yellow")
    if a.status == "error":
        raise typer.Exit(1)


@app.command(name="eval")
def eval_cmd(limit: int = typer.Option(0, help="only the first N questions"),
             no_glossary: bool = typer.Option(False, help="disable the EN->DE glossary expansion")):
    """Stage 5: run the UC13 evaluation question set and write a scored report."""
    s = _init("eval")
    from .eval.runner import run_eval
    try:
        r = run_eval(s, limit=limit or None, use_glossary=not no_glossary)
    except PrivRagError as exc:
        _fail(exc)
    typer.echo(json.dumps(r["summary"], indent=1))


def _wait_for(name: str, check, timeout_s: float, every_s: float = 5.0) -> None:
    import time
    t0 = time.time()
    last = ""
    while True:
        try:
            check()
            typer.secho(f"  {name}: ready ({time.time() - t0:.0f} s)", fg="green", err=True)
            return
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"[:200]
            if msg != last:
                typer.echo(f"  {name}: waiting ... ({msg})", err=True)
                last = msg
            if time.time() - t0 > timeout_s:
                typer.secho(f"FAILED: {name} not ready after {timeout_s:.0f} s", fg="red", err=True)
                raise typer.Exit(3)
            time.sleep(every_s)


@app.command()
def init(only: list[str] = typer.Option(None, help="doc_id(s) to index (default: all PDFs)"),
         wait_s: float = typer.Option(1800, help="max seconds to wait for Qdrant and the embedding service"),
         allow_failures: int = typer.Option(0, help="tolerate up to N PDFs that cannot be parsed")):
    """One-shot container job: wait for Qdrant + embedding service, then ingest and index.

    Safe to run on every start: unchanged documents and an up-to-date index are skipped,
    an interrupted embedding run resumes, so the expensive embedding happens only once.
    """
    s = _init("init")
    from .embed.base import get_embedder
    from .index.pipeline import run_index
    from .index.store import VectorStore, close_clients
    from .ingest.pipeline import run_ingest

    typer.echo("waiting for dependencies ...", err=True)

    def qdrant_ok():
        close_clients()
        VectorStore(s).client.get_collections()
    _wait_for(f"qdrant ({s.qdrant_url or s.qdrant_mode})", qdrant_ok, wait_s)
    if s.embed_backend == "remote":
        # the embedding server may still be downloading the model on its first start
        _wait_for(f"embedding service ({s.embed_url})", lambda: get_embedder(s).embed_query("ping"), wait_s, 10)
    try:
        r = run_ingest(s, only=only or None)
        typer.echo(json.dumps({k: r[k] for k in ("documents", "ok", "skipped_unchanged", "failed", "total_chunks")}))
        if r["failed"] > allow_failures:
            typer.secho(f"FAILED: {r['failed']} documents could not be ingested", fg="red", err=True)
            raise typer.Exit(1)
        r = run_index(s)
        typer.echo(json.dumps({k: r.get(k) for k in ("status", "points", "dense_points", "duration_s")}))
    except PrivRagError as exc:
        _fail(exc)
    typer.secho("init done - index ready", fg="green", err=True)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Stage 4: start the FastAPI backend."""
    import uvicorn
    from .api.app import create_app
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command()
def doctor(before_build: bool = typer.Option(False, "--before-build",
                                             help="only inputs and endpoints; a missing index is not an error")):
    """Check configuration, input files, index and model endpoints."""
    s = _init("doctor")
    ok = True

    def line(good: bool, msg: str, optional: bool = False):
        nonlocal ok
        if optional and not good:
            typer.secho("  TODO " + msg, fg="yellow")
            return
        ok &= good
        typer.secho(("  OK   " if good else "  FAIL ") + msg, fg="green" if good else "red")

    line(s.pdf_dir.exists(), f"pdf_dir {s.pdf_dir} ({len(list(s.pdf_dir.glob('*.pdf'))) if s.pdf_dir.exists() else 0} PDFs)")
    line(s.manifest_path.exists(), f"manifest {s.manifest_path}")
    line(s.eval_path.exists(), f"eval set {s.eval_path}")
    line(s.chunks_path.exists(), f"chunks {s.chunks_path}", optional=before_build)
    try:
        from .index.store import VectorStore
        st = VectorStore(s)
        meta = st.read_meta()
        n = st.count()
        line(n == meta.get("points"), f"qdrant ({s.qdrant_mode}) collection '{s.collection}' points={n}",
             optional=before_build)
        line(meta.get("embed_backend") == s.embed_backend,
             f"index embedder {meta.get('embed_backend')} == configured {s.embed_backend}", optional=before_build)
    except Exception as exc:
        line(False, f"index: {exc}", optional=before_build)
    finally:
        from .index.store import close_clients
        close_clients()   # release the local Qdrant lock for the next command
    import httpx
    for name, url in (("llm", s.llm_base_url and f"{s.llm_base_url.rstrip('/')}/models"),
                      ("embed", s.embed_url and f"{s.embed_url.rstrip('/')}/models"),
                      ("rerank", s.rerank_url and f"{s.rerank_url.rstrip('/')}/health")):
        if url:
            try:
                code = httpx.get(url, timeout=5).status_code
                line(code < 400, f"{name} endpoint {url} -> {code}")
            except Exception as exc:
                line(False, f"{name} endpoint {url}: {exc}")
    typer.echo(f"backends: parser={s.parser_backend} embed={s.embed_backend} rerank={s.rerank_backend} llm={s.llm_backend}")
    if not ok:
        raise typer.Exit(1)


@app.command()
def logs(errors: bool = typer.Option(False, "--errors", help="only WARNING/ERROR lines"),
         tail: int = typer.Option(60), run: str = typer.Option("", help="run_id (default: latest)")):
    """Show the newest run log in readable form."""
    s = get_settings()
    path = s.log_dir / (f"{run}.jsonl" if run else "latest.jsonl")
    if not path.exists():
        typer.echo(f"no log at {path}")
        raise typer.Exit(1)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if errors:
        rows = [r for r in rows if r["level"] in ("WARNING", "ERROR")]
    for r in rows[-tail:]:
        extra = {k: v for k, v in r.items() if k not in ("ts", "level", "run_id", "stage", "event", "logger", "where", "traceback")}
        color = {"ERROR": "red", "WARNING": "yellow"}.get(r["level"])
        typer.secho(f"{r['ts'][11:23]} {r['level']:<7} [{r['stage']}] {r['event']}  {json.dumps(extra, ensure_ascii=False)[:300]}",
                    fg=color)
        if errors and r.get("traceback"):
            typer.echo(r["traceback"])


if __name__ == "__main__":
    sys.exit(app())
