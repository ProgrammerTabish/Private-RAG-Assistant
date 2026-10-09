#!/usr/bin/env bash
# install_run_rag.sh - install Ollama + Mistral Small 3.1 (24B) without sudo and keep the RAG
# assistant running on top of the embeddings / vector DB built by ./make_embeddings.sh.
#
#   ./install_run_rag.sh                 # foreground; Ctrl+C stops the API and Ollama
#   nohup ./install_run_rag.sh > rag.log 2>&1 &     # keep it running after logout
#
# * Ollama binary -> ./.runtime/ollama, model weights -> ./db_data/ollama_models
#   (so ./db_data holds everything needed to move the assistant to another machine)
# * Python deps are installed into ./.runtime/venv (same environment as make_embeddings.sh)
# * API: http://127.0.0.1:8000  (POST /ask, GET /health, docs at /docs)
#
# Optional environment variables:
#   DB_DIR=/path/db_data      vector DB folder (default: ./db_data)
#   LLM_MODEL=mistral-small3.1:24b
#   OLLAMA_PORT=11434   API_HOST=127.0.0.1   API_PORT=8000
#   OLLAMA_CONTEXT_LENGTH=16384    GPU_INDEX=0
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="${DB_DIR:-$ROOT/db_data}"
RUNTIME="${RUNTIME_DIR:-$ROOT/.runtime}"
VENV="$RUNTIME/venv"
PY_VERSION="${PY_VERSION:-3.11}"
LLM_MODEL="${LLM_MODEL:-mistral-small3.1:24b}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
OLLAMA_DIR="$RUNTIME/ollama"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"

log()  { echo -e "\033[1;36m[install_run_rag]\033[0m $*"; }
warn() { echo -e "\033[1;33m[install_run_rag] WARNING:\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[install_run_rag] ERROR:\033[0m $*" >&2; exit 1; }
trap 'die "failed at line $LINENO (exit code $?)"' ERR

# ---------------------------------------------------------------- 0. prerequisites
META="$DB_DIR/index_meta.path.spg_compliance.json"
[[ -f "$META" && -d "$DB_DIR/qdrant" ]] || die "no vector database in $DB_DIR - run ./make_embeddings.sh first (or copy db_data here)"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
  [[ -n "${GPU_INDEX:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  log "NVIDIA GPU(s):"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | sed 's/^/    /'
else
  warn "no NVIDIA GPU found by nvidia-smi - Mistral 24B will run on the CPU and be very slow"
fi

download() {  # download <url> <file>
  if command -v curl >/dev/null 2>&1; then curl -fL --progress-bar "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then wget -q --show-progress "$1" -O "$2"
  else die "neither curl nor wget found"; fi
}
url_exists() {
  if command -v curl >/dev/null 2>&1; then curl -fsIL "$1" >/dev/null 2>&1; else wget -q --spider "$1"; fi
}

# ---------------------------------------------------------------- 1. Python + dependencies (no sudo)
setup_python() {
  mkdir -p "$RUNTIME"
  export UV_CACHE_DIR="$RUNTIME/uv-cache" UV_PYTHON_INSTALL_DIR="$RUNTIME/python"
  UV="$RUNTIME/uv/uv"
  if [[ ! -x "$UV" ]]; then
    log "installing uv (Python package manager) into $RUNTIME/uv ..."
    download https://astral.sh/uv/install.sh "$RUNTIME/uv-install.sh"
    UV_UNMANAGED_INSTALL="$RUNTIME/uv" INSTALLER_NO_MODIFY_PATH=1 sh "$RUNTIME/uv-install.sh" >/dev/null
  fi
  [[ -x "$UV" ]] || die "uv installation failed"
  if [[ ! -x "$VENV/bin/python" ]]; then
    log "creating Python $PY_VERSION virtual environment ..."
    "$UV" venv --python "$PY_VERSION" "$VENV" >/dev/null
  fi
  log "installing Python dependencies ..."
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    "$UV" pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX_URL" "torch>=2.2"
  fi
  "$UV" pip install --python "$VENV/bin/python" -e "$ROOT/selfhosted" \
      "torch>=2.2" "sentence-transformers>=3.0" "zstandard>=0.22"
}
setup_python

# ---------------------------------------------------------------- 2. Ollama (user-space install)
install_ollama() {
  local base="https://ollama.com/download/ollama-linux-amd64" tmp="$RUNTIME/ollama-download"
  [[ "$(uname -m)" == "x86_64" ]] || base="https://ollama.com/download/ollama-linux-arm64"
  mkdir -p "$OLLAMA_DIR" "$tmp"
  if url_exists "$base.tar.zst"; then
    log "downloading Ollama ($base.tar.zst) ..."
    download "$base.tar.zst" "$tmp/ollama.tar.zst"
    "$VENV/bin/python" - "$tmp/ollama.tar.zst" "$OLLAMA_DIR" <<'PY'
import sys, tarfile, zstandard
with open(sys.argv[1], "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
    with tarfile.open(fileobj=zr, mode="r|") as tf:
        tf.extractall(sys.argv[2])
PY
  else
    log "downloading Ollama ($base.tgz) ..."
    download "$base.tgz" "$tmp/ollama.tgz"
    tar -xzf "$tmp/ollama.tgz" -C "$OLLAMA_DIR"
  fi
  rm -rf "$tmp"
  [[ -x "$OLLAMA_BIN" ]] || die "Ollama binary not found after extraction ($OLLAMA_BIN)"
}
[[ -x "$OLLAMA_BIN" ]] || install_ollama
log "Ollama: $("$OLLAMA_BIN" --version 2>/dev/null | tail -n1)"

export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
export OLLAMA_MODELS="$DB_DIR/ollama_models"          # LLM weights stay next to the vector DB
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"   # RAG prompts need > 4k tokens
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"   # keep the model loaded in GPU memory
mkdir -p "$OLLAMA_MODELS" "$DB_DIR/logs"

ollama_up() { "$VENV/bin/python" -c "import urllib.request; urllib.request.urlopen('http://$OLLAMA_HOST/api/version', timeout=2)" >/dev/null 2>&1; }

OLLAMA_PID=""
API_PID=""
cleanup() {
  trap - ERR INT TERM EXIT
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
  if [[ -n "$OLLAMA_PID" ]]; then log "stopping Ollama ..."; kill "$OLLAMA_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if ollama_up; then
  warn "an Ollama server is already running on $OLLAMA_HOST - using it (its own model folder applies)"
else
  log "starting Ollama on $OLLAMA_HOST (log: $DB_DIR/logs/ollama.log) ..."
  nohup "$OLLAMA_BIN" serve >> "$DB_DIR/logs/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 60); do ollama_up && break; sleep 1; done
  ollama_up || die "Ollama did not start - see $DB_DIR/logs/ollama.log"
fi

log "pulling $LLM_MODEL (≈15 GB the first time, then cached in $OLLAMA_MODELS) ..."
"$OLLAMA_BIN" pull "$LLM_MODEL"
log "loading $LLM_MODEL into memory ..."
"$OLLAMA_BIN" run "$LLM_MODEL" "Antworte nur mit OK." >/dev/null

# ---------------------------------------------------------------- 3. RAG API on the existing embeddings
export HF_HOME="$DB_DIR/hf_cache"
[[ -d "$HF_HOME/hub/models--BAAI--bge-m3" ]] && export HF_HUB_OFFLINE=1   # model came with db_data
export PRIVRAG_ENV=local
export PRIVRAG_DATA_DIR="$DB_DIR"
export PRIVRAG_LOG_DIR="$DB_DIR/logs"
export PRIVRAG_PDF_DIR="$ROOT/documents/spg_compliance"
export PRIVRAG_MANIFEST_PATH="$ROOT/documents/manifest.csv"
export PRIVRAG_EVAL_PATH="$ROOT/documents/UC13_Evaluation_Question_Set_Students.xlsx"
export PRIVRAG_QDRANT_MODE=path
export PRIVRAG_EMBED_BACKEND=bge-m3                   # must match the index (checked below)
export PRIVRAG_EMBED_MODEL=BAAI/bge-m3
export PRIVRAG_EMBED_DEVICE="${EMBED_DEVICE:-auto}"
export PRIVRAG_LLM_BACKEND=openai
export PRIVRAG_LLM_BASE_URL="http://$OLLAMA_HOST/v1"
export PRIVRAG_LLM_MODEL="$LLM_MODEL"
export PRIVRAG_LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-300}"
PRIVRAG="$VENV/bin/privrag"

log "checking index + LLM endpoint ..."
"$PRIVRAG" doctor || die "privrag doctor failed - see output above"

log "RAG assistant running: http://$API_HOST:$API_PORT  (Ctrl+C to stop)"
log "  try: curl -X POST http://127.0.0.1:$API_PORT/ask -H 'Content-Type: application/json' \\"
log "         -d '{\"question\": \"Innerhalb welcher Frist muss die interne Meldestelle den Eingang einer Meldung bestätigen?\"}'"
"$PRIVRAG" serve --host "$API_HOST" --port "$API_PORT" &
API_PID=$!
wait "$API_PID"
