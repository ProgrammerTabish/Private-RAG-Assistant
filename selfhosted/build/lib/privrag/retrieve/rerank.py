"""Optional second-stage reranking (cross-encoder bge-reranker-v2-m3).

* ``none``          - keep the fused hybrid order (local default)
* ``cross-encoder`` - model in process (sentence-transformers CrossEncoder)
* ``remote``        - Hugging Face TEI ``/rerank`` endpoint inside the tenant

A reranker failure never fails the request: the fused order is kept and a
warning is attached to the answer.
"""
from __future__ import annotations

from typing import Protocol

from ..config import Settings
from ..errors import RetrievalError
from ..logging_setup import get_logger

log = get_logger("retrieve.rerank")


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RetrievalError("cross-encoder needs sentence-transformers", code="RERANK_BACKEND_MISSING") from exc
        self._m = CrossEncoder(model, max_length=1024)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self._m.predict([(query, t) for t in texts])]


class RemoteReranker:
    name = "remote"

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def score(self, query: str, texts: list[str]) -> list[float]:
        import httpx

        try:
            r = httpx.post(f"{self.url}/rerank", json={"query": query, "texts": texts, "truncate": True}, timeout=self.timeout)
            r.raise_for_status()
            out = [0.0] * len(texts)
            for row in r.json():
                out[int(row["index"])] = float(row["score"])
            return out
        except Exception as exc:
            raise RetrievalError(f"rerank endpoint failed: {exc}", code="RERANK_FAILED") from exc


def get_reranker(s: Settings) -> Reranker | None:
    if s.rerank_backend == "none":
        return None
    if s.rerank_backend == "cross-encoder":
        return CrossEncoderReranker(s.rerank_model)
    return RemoteReranker(s.rerank_url or "")
