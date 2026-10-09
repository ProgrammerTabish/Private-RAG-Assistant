"""Central configuration.

Every setting can be overridden with an environment variable prefixed ``PRIVRAG_``
(e.g. ``PRIVRAG_LLM_BACKEND=openai``) or via a ``.env`` file. The *backend*
switches are what let the same code run locally with stand-ins and on Azure
with the real open-source models.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# selfhosted/ directory (src/privrag/config.py -> parents[2])
PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = PROJECT_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRIVRAG_",
        env_file=(PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- environment ---------------------------------------------------
    env: Literal["local", "azure", "container", "test"] = "local"

    # ---- paths -----------------------------------------------------------
    pdf_dir: Path = REPO_DIR / "documents" / "spg_compliance"
    manifest_path: Path = REPO_DIR / "documents" / "manifest.csv"
    eval_path: Path = REPO_DIR / "documents" / "UC13_Evaluation_Question_Set_Students.xlsx"
    data_dir: Path = PROJECT_DIR / "data"
    log_dir: Path = PROJECT_DIR / "logs"

    # ---- logging ---------------------------------------------------------
    log_level: str = "INFO"
    log_console: bool = True

    # ---- ingestion -------------------------------------------------------
    parser_backend: Literal["pymupdf", "docling"] = "pymupdf"
    chunk_size: int = Field(1400, ge=200, le=8000, description="target characters per chunk")
    chunk_overlap: int = Field(200, ge=0, le=2000)
    min_chunk_chars: int = Field(80, ge=0)
    max_pdf_mb: float = Field(200.0, gt=0)
    min_text_chars_per_page: int = Field(30, ge=0, description="below this a page counts as empty/scanned")
    header_footer_min_ratio: float = Field(0.5, gt=0, le=1, description="share of pages a line must repeat on to be stripped")

    # ---- embeddings ------------------------------------------------------
    embed_backend: Literal["lsa", "bge-m3", "remote"] = "lsa"
    embed_dim: int = Field(256, ge=8, description="only used by the local LSA stand-in")
    embed_model: str = "BAAI/bge-m3"
    embed_url: str | None = None  # OpenAI-compatible /v1/embeddings (TEI or vLLM)
    embed_batch_size: int = Field(32, ge=1)
    embed_timeout_s: float = Field(120.0, gt=0, description="per embedding request (CPU servers need more)")
    index_time_budget_s: float | None = Field(
        None, ge=0, description="limit dense embedding time per index run; all chunks still get BM25. "
                                "Remaining chunks get dense vectors on later runs. None = embed everything")

    # ---- vector store ----------------------------------------------------
    qdrant_mode: Literal["memory", "path", "server"] = "path"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection: str = "spg_compliance"
    upsert_batch_size: int = Field(128, ge=1)

    # ---- retrieval -------------------------------------------------------
    top_k: int = Field(8, ge=1, le=50, description="chunks handed to the LLM")
    candidate_k: int = Field(40, ge=1, le=500, description="candidates per retriever before fusion")
    max_chunks_per_doc: int = Field(4, ge=1)
    dense_weight: float = Field(2.0, ge=0, description="RRF weight of the dense (BGE-M3) lists vs keyword lists")
    rerank_backend: Literal["none", "cross-encoder", "remote"] = "none"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_url: str | None = None
    refusal_recheck: bool = Field(True, description="if the LLM refuses although sources were found, ask it once "
                                                    "more to re-check them (small models refuse too eagerly)")
    query_rewrite: bool = Field(True, description="ask the LLM for German search terms (cross-lingual sparse search)")

    # ---- generation ------------------------------------------------------
    llm_backend: Literal["fake", "openai"] = "fake"
    llm_base_url: str | None = None  # vLLM OpenAI-compatible endpoint, e.g. http://10.0.0.5:8000/v1
    llm_api_key: str = "not-needed"
    llm_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    llm_temperature: float = Field(0.0, ge=0, le=2)
    llm_max_tokens: int = Field(900, ge=16)
    max_source_chars: int = Field(1800, ge=200, description="characters per source shown to the LLM")
    llm_timeout_s: float = Field(120.0, gt=0)
    llm_max_retries: int = Field(2, ge=0, le=10)
    min_citations: int = Field(2, ge=0, description="KPI: at least 2 verifiable sources per answer")
    min_relevance: float = Field(0.0, description="below this best score -> 'not in corpus' refusal")

    # ---- API / audit -----------------------------------------------------
    audit_db_url: str | None = None  # default: sqlite file under data_dir; postgresql+psycopg://... on Azure
    max_question_chars: int = Field(2000, ge=10)
    api_key: str | None = None       # if set, clients must send header X-API-Key
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])

    # cost model for the benchmark (EUR per GPU hour, set to your VM price)
    gpu_hour_cost: float = 3.67

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"invalid log level {v}")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.llm_backend == "openai" and not self.llm_base_url:
            raise ValueError("llm_backend=openai needs PRIVRAG_LLM_BASE_URL (the vLLM endpoint)")
        if self.embed_backend == "remote" and not self.embed_url:
            raise ValueError("embed_backend=remote needs PRIVRAG_EMBED_URL")
        if self.rerank_backend == "remote" and not self.rerank_url:
            raise ValueError("rerank_backend=remote needs PRIVRAG_RERANK_URL")
        if self.qdrant_mode == "server" and not self.qdrant_url:
            raise ValueError("qdrant_mode=server needs PRIVRAG_QDRANT_URL")
        if self.top_k > self.candidate_k:
            raise ValueError("top_k must be <= candidate_k")
        return self

    # derived paths
    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def audit_url(self) -> str:
        return self.audit_db_url or f"sqlite:///{(self.data_dir / 'audit.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.log_dir, self.parsed_dir, self.reports_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        """Settings safe to log (secrets masked)."""
        d = self.model_dump(mode="json")
        for k in ("llm_api_key", "qdrant_api_key", "api_key"):
            if d.get(k) and d[k] != "not-needed":
                d[k] = "***"
        return d


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
