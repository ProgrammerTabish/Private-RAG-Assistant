"""Data contracts between stages. Each stage reads and writes these, so a bad
record is rejected at the boundary instead of failing three stages later."""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DocMeta(BaseModel):
    doc_id: str                      # file stem, e.g. "01_GwG"
    file: str                        # file name, e.g. "01_GwG.pdf"
    title: str
    regulator: str = "unknown"
    doc_type: str = "unknown"
    publication_date: str | None = None
    language: str = "unknown"
    sha256: str | None = None
    pages: int | None = None
    from_manifest: bool = True


class PageText(BaseModel):
    page: int = Field(ge=1)          # 1-based, as printed in citations
    text: str


class ParsedDoc(BaseModel):
    meta: DocMeta
    pages: list[PageText]
    parser: str
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str                    # deterministic: <doc_id>:<seq>:<hash8>
    doc_id: str
    file: str
    title: str
    regulator: str
    doc_type: str
    language: str
    publication_date: str | None = None
    section: str | None = None       # e.g. "§ 10 Allgemeine Sorgfaltspflichten", "Artikel 19", "AT 4.4.2"
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    seq: int = Field(ge=0)
    text: str
    char_len: int = 0

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("chunk text is empty")
        return v

    @property
    def pages_label(self) -> str:
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"

    @staticmethod
    def make_id(doc_id: str, seq: int, text: str) -> str:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}:{seq:05d}:{h}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    n: int                           # number used in the answer text: [1], [2]
    chunk_id: str
    doc_id: str
    file: str
    title: str
    section: str | None
    page_start: int
    page_end: int
    quote: str                       # short excerpt for source highlighting in the UI


class Answer(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "insufficient_sources", "out_of_scope", "error"]
    citations: list[Citation]
    confidence: float = Field(ge=0, le=1)
    confidence_label: Literal["high", "medium", "low"]
    retrieved: int
    retrieved_refs: list[str] = Field(default_factory=list)   # "doc_id | section | pages" of all chunks shown to the LLM
    usage: dict[str, int] = Field(default_factory=dict)       # LLM tokens for this question (all calls)
    model: str
    latency_ms: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
    request_id: str | None = None
