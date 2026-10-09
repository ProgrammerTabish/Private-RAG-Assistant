"""Step logging.

* Every log record is written as one JSON line to ``logs/<run_id>.jsonl`` and to
  ``logs/latest.jsonl`` (a copy of the newest run), plus a short human line on
  the console.
* ``run_id``, ``stage``, ``doc_id`` and ``request_id`` are carried in context
  variables, so every line emitted inside a stage is tagged automatically.
* ``step()`` wraps a unit of work: logs start, end, duration and - on failure -
  the error code, type, message and full traceback, then re-raises.

Trace a failure:  grep '"level": "ERROR"' logs/latest.jsonl  or
                  privrag logs --errors
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import json
import logging
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterator

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")
_stage: contextvars.ContextVar[str] = contextvars.ContextVar("stage", default="-")
_doc_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("doc_id", default=None)
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

LOGGER_NAME = "privrag"
_configured_file: Path | None = None


def new_run_id(prefix: str = "run") -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}-{uuid.uuid4().hex[:6]}"


def current_run_id() -> str:
    return _run_id.get()


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        record.stage = getattr(record, "stage_override", None) or _stage.get()
        record.doc_id = _doc_id.get()
        record.request_id = _request_id.get()
        return True


_RESERVED = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {
    "run_id", "stage", "doc_id", "request_id", "stage_override", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": _dt.datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "run_id": getattr(record, "run_id", "-"),
            "stage": getattr(record, "stage", "-"),
            "event": record.getMessage(),
            "logger": record.name,
            "where": f"{record.module}.{record.funcName}:{record.lineno}",
        }
        if getattr(record, "doc_id", None):
            out["doc_id"] = record.doc_id
        if getattr(record, "request_id", None):
            out["request_id"] = record.request_id
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["traceback"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(out, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        t = _dt.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        extras = []
        if getattr(record, "doc_id", None):
            extras.append(f"doc={record.doc_id}")
        for k in ("duration_ms", "error_code", "count", "status"):
            if k in record.__dict__:
                extras.append(f"{k}={record.__dict__[k]}")
        tail = ("  " + " ".join(extras)) if extras else ""
        return f"{t} {record.levelname:<7} [{getattr(record, 'stage', '-')}] {record.getMessage()}{tail}"


_LOGRECORD_RESERVED = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}


def safe_extra(d: dict | None) -> dict:
    """Rename keys that would collide with LogRecord attributes ('message', 'name', 'args', ...).
    Passing such a key to logging raises KeyError - i.e. the error log itself would crash."""
    if not d:
        return {}
    return {(f"x_{k}" if k in _LOGRECORD_RESERVED else k): v for k, v in d.items()}


class _SafeLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):  # type: ignore[override]
        super()._log(level, msg, args, exc_info=exc_info, extra=safe_extra(extra), stack_info=stack_info,
                     stacklevel=stacklevel)


def setup_logging(log_dir: Path, run_id: str | None = None, level: str = "INFO", console: bool = True) -> str:
    """Configure the privrag logger for one run. Returns the run_id."""
    global _configured_file
    run_id = run_id or new_run_id()
    _run_id.set(run_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    ctx = _ContextFilter()
    run_file = log_dir / f"{run_id}.jsonl"
    for path, mode in ((run_file, "a"), (log_dir / "latest.jsonl", "w")):
        fh = logging.FileHandler(path, mode=mode, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        fh.addFilter(ctx)
        logger.addHandler(fh)
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(ConsoleFormatter())
        ch.addFilter(ctx)
        logger.addHandler(ch)
    _configured_file = run_file
    return run_id


def get_logger(name: str = "") -> logging.Logger:
    full = f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME
    existing = logging.Logger.manager.loggerDict.get(full)
    if isinstance(existing, _SafeLogger):
        return existing
    # create privrag loggers with the safe class without changing the global logger class
    old = logging.getLoggerClass()
    logging.setLoggerClass(_SafeLogger)
    try:
        if isinstance(existing, logging.Logger):     # created earlier as a plain logger
            existing.__class__ = _SafeLogger
            return existing
        return logging.getLogger(full)
    finally:
        logging.setLoggerClass(old)


def log_file() -> Path | None:
    return _configured_file


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    tok = _stage.set(name)
    try:
        yield
    finally:
        _stage.reset(tok)


@contextlib.contextmanager
def doc_context(doc_id: str | None) -> Iterator[None]:
    tok = _doc_id.set(doc_id)
    try:
        yield
    finally:
        _doc_id.reset(tok)


@contextlib.contextmanager
def request_context(request_id: str) -> Iterator[None]:
    tok = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(tok)


@contextlib.contextmanager
def step(name: str, logger: logging.Logger | None = None, *, level: int = logging.INFO, **fields: Any) -> Iterator[dict]:
    """Log start/end/duration of a unit of work. Yields a dict you can add result
    fields to (they are logged on completion)."""
    logger = logger or get_logger()
    result: dict[str, Any] = {}
    logger.log(logging.DEBUG, f"{name}: start", extra={"step": name, **fields})
    t0 = time.perf_counter()
    try:
        yield result
    except Exception as exc:
        dur = round((time.perf_counter() - t0) * 1000, 1)
        code = getattr(exc, "code", type(exc).__name__)
        ctx = getattr(exc, "context", {}) or {}
        logger.error(
            f"{name}: FAILED - {exc}",
            exc_info=True,
            extra={"step": name, "status": "failed", "duration_ms": dur, "error_code": code,
                   "error_type": type(exc).__name__, **fields, **{f"ctx_{k}": v for k, v in ctx.items()}},
        )
        raise
    else:
        dur = round((time.perf_counter() - t0) * 1000, 1)
        logger.log(level, f"{name}: ok", extra={"step": name, "status": "ok", "duration_ms": dur, **fields, **result})
