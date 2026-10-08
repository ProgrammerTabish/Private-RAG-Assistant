"""Dense embedders behind one interface.

* ``lsa``    - local stand-in: character n-gram TF-IDF + truncated SVD, fitted on
               the corpus. No downloads; good enough to test the pipeline end to end.
               NOT cross-lingual - English questions vs German text is weak here.
* ``bge-m3`` - BAAI/bge-m3 loaded in-process (FlagEmbedding / sentence-transformers),
               multilingual DE/EN, 1024 dims. Production choice.
* ``remote`` - any OpenAI-compatible ``/v1/embeddings`` endpoint (Hugging Face TEI or
               vLLM serving bge-m3 inside the tenant).

Every backend output is validated: shape, dimension and finite values.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import Settings
from ..errors import EmbeddingError
from ..logging_setup import get_logger
from ..models import Chunk

log = get_logger("embed")
MAX_EMBED_CHARS = 8000


def chunk_embed_text(c: Chunk) -> str:
    """What gets embedded: document title + section give the chunk its context."""
    head = c.title + (f" | {c.section}" if c.section else "")
    return f"{head}\n{c.text}"


class Embedder(Protocol):
    name: str
    dim: int

    def fit(self, texts: list[str]) -> None: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def fingerprint(self) -> str: ...


def _validate(vecs: np.ndarray, n: int, dim: int | None, who: str) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] != n:
        raise EmbeddingError(f"{who}: expected {n} vectors, got shape {vecs.shape}", code="EMBED_BAD_SHAPE")
    if dim is not None and vecs.shape[1] != dim:
        raise EmbeddingError(f"{who}: expected dim {dim}, got {vecs.shape[1]}", code="EMBED_DIM_MISMATCH")
    if not np.isfinite(vecs).all():
        raise EmbeddingError(f"{who}: NaN/inf in embeddings", code="EMBED_NOT_FINITE")
    return vecs


def _l2(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _clean_inputs(texts: list[str]) -> list[str]:
    out = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            raise EmbeddingError(f"empty text at position {i}", code="EMBED_EMPTY_INPUT", position=i)
        out.append(t[:MAX_EMBED_CHARS])
    return out


# ------------------------------------------------------------------ LSA (local stand-in)
class LSAEmbedder:
    name = "lsa"

    def __init__(self, dim: int, model_path: Path):
        self.dim = dim
        self.model_path = model_path
        self._vec = None
        self._svd = None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib

        texts = _clean_inputs(texts)
        if len(texts) < 3:
            raise EmbeddingError("LSA needs at least 3 texts to fit", code="EMBED_FIT_TOO_SMALL")
        t0 = time.perf_counter()
        # char n-grams cope with German compounds ("Sorgfaltspflichten" ~ "Sorgfaltspflicht")
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2 if len(texts) > 50 else 1,
                                    max_features=300_000, sublinear_tf=True, lowercase=True, dtype=np.float32)
        X = self._vec.fit_transform(texts)
        k = min(self.dim, X.shape[1] - 1, X.shape[0] - 1)
        if k < self.dim:
            log.warning("LSA dim reduced (corpus too small)", extra={"requested": self.dim, "used": k})
            self.dim = k
        self._svd = TruncatedSVD(n_components=self.dim, random_state=42, algorithm="randomized", n_iter=5)
        self._svd.fit(X)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vec": self._vec, "svd": self._svd, "dim": self.dim}, self.model_path)
        log.info("LSA fitted", extra={"docs": len(texts), "features": X.shape[1], "dim": self.dim,
                                      "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})

    def _load(self) -> None:
        if self._vec is not None:
            return
        import joblib
        if not self.model_path.exists():
            raise EmbeddingError(f"LSA model not found at {self.model_path} - run 'privrag index' first",
                                 code="EMBED_MODEL_MISSING")
        d = joblib.load(self.model_path)
        self._vec, self._svd, self.dim = d["vec"], d["svd"], d["dim"]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._load()
        texts = _clean_inputs(texts)
        v = self._svd.transform(self._vec.transform(texts))
        return _validate(_l2(v), len(texts), self.dim, self.name)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def fingerprint(self) -> str:
        self._load()
        return f"lsa-{self.dim}-{hashlib.sha1(self.model_path.read_bytes()).hexdigest()[:10]}"


# ------------------------------------------------------------------ BGE-M3 in process
class BGEM3Embedder:
    name = "bge-m3"
    dim = 1024

    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(model_name, use_fp16=True)
            self._kind = "flag"
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self._kind = "st"
            except ImportError as exc:
                raise EmbeddingError("bge-m3 needs FlagEmbedding or sentence-transformers (pip install 'privrag[models]')",
                                     code="EMBED_BACKEND_MISSING") from exc
        except Exception as exc:
            raise EmbeddingError(f"cannot load {model_name}: {exc}", code="EMBED_MODEL_LOAD_FAILED") from exc

    def fit(self, texts: list[str]) -> None:  # pre-trained, nothing to fit
        return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = _clean_inputs(texts)
        if self._kind == "flag":
            out = self._model.encode(texts, batch_size=self.batch_size, max_length=8192)["dense_vecs"]
        else:
            out = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        return _validate(_l2(np.asarray(out)), len(texts), self.dim, self.name)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def fingerprint(self) -> str:
        return f"bge-m3-{self.model_name}"


# ------------------------------------------------------------------ remote OpenAI-compatible endpoint
class RemoteEmbedder:
    name = "remote"

    def __init__(self, url: str, model: str, timeout: float = 60.0, retries: int = 3, dim: int | None = None):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.dim = dim or 0

    def fit(self, texts: list[str]) -> None:
        return None

    def _post(self, texts: list[str]) -> np.ndarray:
        import httpx
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(self.retries), wait=wait_exponential(min=1, max=20),
               retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)), reraise=True)
        def call() -> dict:
            r = httpx.post(f"{self.url}/embeddings", json={"model": self.model, "input": texts}, timeout=self.timeout)
            if r.status_code >= 500 or r.status_code == 429:
                r.raise_for_status()
            if r.status_code >= 400:
                raise EmbeddingError(f"embedding endpoint returned {r.status_code}: {r.text[:300]}",
                                     code="EMBED_HTTP_ERROR", status=r.status_code)
            return r.json()

        try:
            data = call()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"embedding endpoint unreachable: {exc}", code="EMBED_ENDPOINT_DOWN", url=self.url) from exc
        try:
            rows = sorted(data["data"], key=lambda d: d["index"])
            vecs = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        except Exception as exc:
            raise EmbeddingError(f"unexpected embedding response: {str(data)[:200]}", code="EMBED_BAD_RESPONSE") from exc
        if not self.dim:
            self.dim = int(vecs.shape[1])
        return _validate(_l2(vecs), len(texts), self.dim, self.name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._post(_clean_inputs(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def fingerprint(self) -> str:
        return f"remote-{self.model}"


def get_embedder(s: Settings) -> Embedder:
    if s.embed_backend == "lsa":
        return LSAEmbedder(dim=s.embed_dim, model_path=s.models_dir / "lsa.joblib")
    if s.embed_backend == "bge-m3":
        return BGEM3Embedder(s.embed_model, s.embed_batch_size)
    if s.embed_backend == "remote":
        return RemoteEmbedder(s.embed_url or "", s.embed_model, timeout=s.embed_timeout_s)
    raise EmbeddingError(f"unknown embed backend {s.embed_backend}", code="EMBED_BACKEND_UNKNOWN")


def embed_in_batches(embedder: Embedder, texts: list[str], batch_size: int) -> np.ndarray:
    """Embed in batches with progress logging (~every 5 %) incl. throughput and ETA -
    embedding 12k chunks on a CPU endpoint takes a while and must not look hung."""
    out = []
    n = len(texts)
    t0 = time.perf_counter()
    next_report = 0.05
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        try:
            out.append(embedder.embed_documents(batch))
        except EmbeddingError as exc:
            exc.context.update(batch_start=i, batch_size=len(batch))
            raise
        done = min(i + batch_size, n)
        if n >= 200 and (done / n >= next_report or done == n):
            el = time.perf_counter() - t0
            rate = done / el if el > 0 else 0.0
            log.info(f"embedding progress {done}/{n} ({100 * done / n:.0f}%)",
                     extra={"count": done, "rate_per_s": round(rate, 1),
                            "eta_s": round((n - done) / rate) if rate else None})
            next_report += 0.05
    return np.vstack(out) if out else np.zeros((0, embedder.dim), dtype=np.float32)


def describe(embedder: Embedder) -> str:
    return json.dumps({"name": embedder.name, "dim": embedder.dim})
