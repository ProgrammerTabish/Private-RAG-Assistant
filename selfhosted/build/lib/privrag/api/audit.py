"""Audit trail: one immutable row per question (who, when, what was asked, what
was answered, which sources, which model, how long). SQLite locally, PostgreSQL
(open source) on Azure - same code via SQLAlchemy.

Rows are append-only; the API exposes read endpoints only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from ..logging_setup import get_logger
from ..models import Answer

log = get_logger("audit")


class Base(DeclarativeBase):
    pass


class AuditRow(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    user: Mapped[str] = mapped_column(String(200), default="anonymous", index=True)
    question: Mapped[str] = mapped_column(Text)
    question_sha256: Mapped[str] = mapped_column(String(64))
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    model: Mapped[str] = mapped_column(String(200))
    variant: Mapped[str] = mapped_column(String(32), default="self-hosted")
    citations: Mapped[list] = mapped_column(JSON)
    latency_ms: Mapped[dict] = mapped_column(JSON)
    warnings: Mapped[list] = mapped_column(JSON)
    filters: Mapped[dict] = mapped_column(JSON)
    retrieved: Mapped[int] = mapped_column(Integer)

    def to_dict(self) -> dict:
        return {c.name: (getattr(self, c.name).isoformat() if isinstance(getattr(self, c.name), dt.datetime)
                         else getattr(self, c.name)) for c in self.__table__.columns}


class AuditLog:
    def __init__(self, url: str):
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        self.engine = create_engine(url, **kw)
        Base.metadata.create_all(self.engine)
        log.info("audit store ready", extra={"backend": url.split(":", 1)[0]})

    def record(self, ans: Answer, user: str, filters: dict) -> None:
        row = AuditRow(
            request_id=ans.request_id, ts=dt.datetime.now(dt.timezone.utc), user=user or "anonymous",
            question=ans.question, question_sha256=hashlib.sha256(ans.question.encode()).hexdigest(),
            answer=ans.answer, status=ans.status, confidence=ans.confidence, model=ans.model,
            citations=[c.model_dump() for c in ans.citations], latency_ms=ans.latency_ms,
            warnings=ans.warnings, filters=filters, retrieved=ans.retrieved,
        )
        with Session(self.engine) as s:
            s.add(row)
            s.commit()

    def recent(self, limit: int = 50, status: str | None = None, user: str | None = None) -> list[dict]:
        q = select(AuditRow).order_by(AuditRow.id.desc()).limit(limit)
        if status:
            q = q.where(AuditRow.status == status)
        if user:
            q = q.where(AuditRow.user == user)
        with Session(self.engine) as s:
            return [r.to_dict() for r in s.scalars(q)]

    def get(self, request_id: str) -> dict | None:
        with Session(self.engine) as s:
            r = s.scalar(select(AuditRow).where(AuditRow.request_id == request_id))
            return r.to_dict() if r else None

    def export_jsonl(self) -> str:
        with Session(self.engine) as s:
            return "".join(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n"
                           for r in s.scalars(select(AuditRow).order_by(AuditRow.id)))
