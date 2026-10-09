"""FastAPI backend for the self-hosted variant.

Endpoints
  GET  /health                liveness + what is loaded (index size, models)
  GET  /ready                 503 until the index is loaded and the LLM endpoint answers
  POST /ask                   question -> cited answer (+ audit row)
  GET  /documents             corpus list for UI filters
  GET  /sources/{chunk_id}    full chunk text for source highlighting
  GET  /audit                 recent audit rows   (?limit=&status=&user=)
  GET  /audit/{request_id}    one audit row

Every request gets an ``X-Request-ID`` (taken from the client or generated),
which appears in every log line and in the audit row.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from .. import __version__
from ..config import Settings, get_settings
from ..errors import PrivRagError
from ..generate.service import RagService, safe_ask
from ..logging_setup import get_logger, new_run_id, request_context, setup_logging, stage
from ..models import Answer
from .audit import AuditLog

log = get_logger("api")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_ids: list[str] | None = None
    regulators: list[str] | None = None
    user: str | None = Field(None, max_length=200)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is empty")
        return v


class _State:
    service: RagService | None = None
    audit: AuditLog | None = None
    startup_error: dict | None = None
    lock = threading.Lock()


def create_app(settings: Settings | None = None, service: RagService | None = None) -> FastAPI:
    s = settings or get_settings()
    st = _State()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s.ensure_dirs()
        setup_logging(s.log_dir, run_id=new_run_id("api"), level=s.log_level, console=s.log_console)
        with stage("api.startup"):
            log.info("starting API", extra={"version": __version__, "settings": s.public_dict()})
            try:
                st.audit = AuditLog(s.audit_url)
            except Exception as exc:
                st.startup_error = {"error_code": "AUDIT_UNAVAILABLE", "message": str(exc)}
                log.exception("audit store unavailable")
            try:
                st.service = service or RagService(s)
                t0 = time.perf_counter()
                st.service.retriever.embedder.embed_query("warmup")   # load models before the first user
                log.info("warmup done", extra={"duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
            except PrivRagError as exc:
                st.startup_error = exc.to_dict()
                log.error("service not ready - API runs in degraded mode", extra=exc.to_dict())
            except Exception as exc:
                st.startup_error = {"error_code": "UNEXPECTED", "message": str(exc)}
                log.exception("service not ready - API runs in degraded mode")
        yield
        log.info("API shutting down")

    app = FastAPI(title="Private RAG Assistant - self-hosted variant", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Request-ID"])

    @app.middleware("http")
    async def request_id_mw(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        with request_context(rid), stage("api"):
            try:
                resp = await call_next(request)
            except Exception:
                log.exception("unhandled error", extra={"path": request.url.path})
                resp = JSONResponse({"error_code": "UNEXPECTED", "request_id": rid}, status_code=500)
            dur = round((time.perf_counter() - t0) * 1000, 1)
            log.info(f"{request.method} {request.url.path} -> {resp.status_code}",
                     extra={"status": resp.status_code, "duration_ms": dur})
        resp.headers["X-Request-ID"] = rid
        return resp

    def check_key(x_api_key: str | None = Header(None)) -> None:
        if s.api_key and x_api_key != s.api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    def need_service() -> RagService:
        if st.service is None:
            raise HTTPException(503, detail={"error": "service not ready", **(st.startup_error or {})})
        return st.service

    @app.get("/", include_in_schema=False)
    def ui():
        """Minimal chat page (static, same origin as the API)."""
        from importlib.resources import files
        return HTMLResponse(files("privrag.api").joinpath("static/index.html").read_text(encoding="utf-8"))

    @app.get("/health")
    def health():
        info = {"status": "ok" if st.service else "degraded", "version": __version__, "variant": "self-hosted",
                "embed_backend": s.embed_backend, "llm_backend": s.llm_backend, "llm_model": s.llm_model,
                "rerank_backend": s.rerank_backend, "parser_backend": s.parser_backend}
        if st.service:
            try:
                info["index_points"] = st.service.retriever.store.count()
                meta = st.service.retriever.store.read_meta()
                info["dense_points"] = meta.get("dense_points", info["index_points"])
                info["dense_coverage"] = meta.get("dense_coverage", 1.0)
            except Exception as exc:
                info["index_error"] = str(exc)
        if st.startup_error:
            info["startup_error"] = st.startup_error
        return info

    @app.get("/ready")
    def ready():
        svc = need_service()
        checks = {"index": svc.retriever.store.count() > 0, "audit": st.audit is not None}
        if s.llm_backend == "openai":
            import httpx
            try:
                checks["llm"] = httpx.get(f"{s.llm_base_url.rstrip('/')}/models", timeout=5).status_code == 200
            except Exception:
                checks["llm"] = False
        ok = all(checks.values())
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    @app.post("/ask", response_model=Answer, dependencies=[Depends(check_key)])
    def ask(req: AskRequest, request: Request):
        svc = need_service()
        if len(req.question) > s.max_question_chars:
            raise HTTPException(422, f"question longer than {s.max_question_chars} characters")
        rid = request.state.request_id
        # local Qdrant (path/memory) is not thread-safe; a Qdrant server is
        if s.qdrant_mode != "server":
            with st.lock:
                ans = safe_ask(svc, req.question, doc_ids=req.doc_ids, regulators=req.regulators, request_id=rid)
        else:
            ans = safe_ask(svc, req.question, doc_ids=req.doc_ids, regulators=req.regulators, request_id=rid)
        if st.audit:
            try:
                st.audit.record(ans, req.user or "anonymous", {"doc_ids": req.doc_ids, "regulators": req.regulators})
            except Exception as exc:
                ans.warnings.append("audit write failed")
                log.exception("audit write failed", extra={"error_code": "AUDIT_WRITE_FAILED", "error": str(exc)})
        else:
            ans.warnings.append("audit store unavailable")
        if ans.status == "error":
            return JSONResponse(ans.model_dump(mode="json"), status_code=503)
        return ans

    @app.get("/documents", dependencies=[Depends(check_key)])
    def documents():
        path = s.data_dir / "ingest_state.json"
        if not path.exists():
            return []
        state = json.loads(path.read_text(encoding="utf-8"))
        return [{"doc_id": k, "chunks": v.get("chunks"), "pages": v.get("pages"), **(v.get("meta") or {})}
                for k, v in sorted(state.items()) if v.get("status") == "ok"]

    @app.get("/sources/{chunk_id}", dependencies=[Depends(check_key)])
    def source(chunk_id: str):
        from ..index.store import point_id

        svc = need_service()
        pts = svc.retriever.store.client.retrieve(s.collection, ids=[point_id(chunk_id)], with_payload=True)
        if not pts:
            raise HTTPException(404, "chunk not found")
        return pts[0].payload

    @app.get("/audit", dependencies=[Depends(check_key)])
    def audit_list(limit: int = Query(50, ge=1, le=1000), status: str | None = None, user: str | None = None):
        if not st.audit:
            raise HTTPException(503, "audit store unavailable")
        return st.audit.recent(limit, status, user)

    @app.get("/audit/{request_id}", dependencies=[Depends(check_key)])
    def audit_get(request_id: str):
        if not st.audit:
            raise HTTPException(503, "audit store unavailable")
        row = st.audit.get(request_id)
        if not row:
            raise HTTPException(404, "unknown request_id")
        return row

    return app
