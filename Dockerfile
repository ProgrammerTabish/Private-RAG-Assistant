# Self-hosted RAG backend (UC13 Variant B): FastAPI + ingestion/indexing pipeline.
# The 50 PDFs, the manifest and the evaluation set are baked into the image.
# The LLM and the embedding model run OUTSIDE this container (OpenAI-compatible endpoints).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# dependencies first (better layer caching)
COPY selfhosted/pyproject.toml selfhosted/README.md /app/selfhosted/
COPY selfhosted/src /app/selfhosted/src
RUN pip install "/app/selfhosted[postgres]"

# corpus + evaluation set (read-only inside the image)
COPY documents /app/documents

# non-root user; /data is the persistent volume (chunks, index metadata, audit DB, reports, logs)
RUN useradd --create-home --uid 10001 privrag && mkdir -p /data && chown privrag:privrag /data
USER privrag

ENV PRIVRAG_ENV=container \
    PRIVRAG_PDF_DIR=/app/documents/spg_compliance \
    PRIVRAG_MANIFEST_PATH=/app/documents/manifest.csv \
    PRIVRAG_EVAL_PATH=/app/documents/UC13_Evaluation_Question_Set_Students.xlsx \
    PRIVRAG_DATA_DIR=/data \
    PRIVRAG_LOG_DIR=/data/logs \
    PRIVRAG_QDRANT_MODE=server \
    PRIVRAG_EMBED_BACKEND=remote \
    PRIVRAG_LOG_CONSOLE=true

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# default: API server. The one-shot indexing job runs the same image with `privrag init`.
CMD ["privrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
