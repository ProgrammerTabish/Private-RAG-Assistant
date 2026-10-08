"""Typed errors. Every error carries the pipeline stage and a stable code so a
log line like ``code=PARSE_ENCRYPTED stage=ingest.parse doc=05_x.pdf`` is enough
to find the cause without reading a traceback."""
from __future__ import annotations


class PrivRagError(Exception):
    stage: str = "unknown"
    code: str = "ERROR"

    def __init__(self, message: str, *, code: str | None = None, stage: str | None = None, **context):
        super().__init__(message)
        if code:
            self.code = code
        if stage:
            self.stage = stage
        self.context = context

    def to_dict(self) -> dict:
        return {"error_code": self.code, "stage": self.stage, "message": str(self), **self.context}


class ConfigError(PrivRagError):
    stage = "config"
    code = "CONFIG_INVALID"


class ManifestError(PrivRagError):
    stage = "ingest.manifest"
    code = "MANIFEST_INVALID"


class ParseError(PrivRagError):
    stage = "ingest.parse"
    code = "PARSE_FAILED"


class ChunkError(PrivRagError):
    stage = "ingest.chunk"
    code = "CHUNK_FAILED"


class EmbeddingError(PrivRagError):
    stage = "embed"
    code = "EMBED_FAILED"


class IndexError_(PrivRagError):  # avoid clashing with builtin IndexError
    stage = "index"
    code = "INDEX_FAILED"


class RetrievalError(PrivRagError):
    stage = "retrieve"
    code = "RETRIEVE_FAILED"


class LLMError(PrivRagError):
    stage = "generate"
    code = "LLM_FAILED"
