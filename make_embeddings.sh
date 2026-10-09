#!/usr/bin/env bash
# make_embeddings.sh - build the BGE-M3 embeddings for all PDFs on the NVIDIA GPU and store
# them permanently in a local Qdrant vector database under ./db_data.
#
#   ./make_embeddings.sh
#
# * No Docker, no sudo: Python, uv and all packages are installed into ./.runtime
# * Stops immediately if nvidia-smi / an NVIDIA GPU is not available
# * Everything that has to survive (vector DB, chunks, BGE-M3 model, logs) lives in ./db_data
#   -> copy that one folder to another machine and run ./install_run_rag.sh there
# * Re-running is safe: an existing, up-to-date index is skipped, an interrupted run resumes
#
# Optional environment variables:
#   GPU_INDEX=1            which GPU to use (default: the one with the most free memory)
#   DB_DIR=/path/db_data   where the vector DB lives (default: ./db_data)
#   EMBED_BATCH_SIZE=20    chunks per GPU batch
#   ALLOW_FAILURES=2       PDFs that may fail to parse without aborting
#   TORCH_INDEX_URL=...    e.g. https://download.pytorch.org/whl/cu121 for old drivers
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="${DB_DIR:-$ROOT/db_data}"
RUNTIME="${RUNTIME_DIR:-$ROOT/.runtime}"
VENV="$RUNTIME/venv"
PY_VERSION="${PY_VERSION:-3.11}"

log()  { echo -e "\033[1;36m[make_embeddings]\033[0m $*"; }
die()  { echo -e "\033[1;31m[make_embeddings] ERROR:\033[0m $*" >&2; exit 1; }
trap 'die "failed at line $LINENO (exit code $?)"' ERR

# ---------------------------------------------------------------- 1. NVIDIA GPU (mandatory)
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found - no NVIDIA GPU available. Terminating."
nvidia-smi >/dev/null 2>&1           || die "nvidia-smi failed - NVIDIA driver/GPU not usable. Terminating."
nvidia-smi -L 2>/dev/null | grep -q '^GPU' || die "nvidia-smi reports no GPU. Terminating."

if [[ -z "${GPU_INDEX:-}" ]]; then
  GPU_INDEX="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
               | sort -t, -k2 -nr | head -n1 | cut -d, -f1 | tr -d ' ')"
fi
GPU_INFO="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader)" \
  || die "GPU index $GPU_INDEX not found. Terminating."
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
log "using GPU $GPU_INDEX: $GPU_INFO"

# ---------------------------------------------------------------- 2. Python + dependencies (no sudo)
download() {  # download <url> <file>
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then wget -q "$1" -O "$2"
  else die "neither curl nor wget found"; fi
}

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
    log "creating Python $PY_VERSION virtual environment (downloaded if not on the system) ..."
    "$UV" venv --python "$PY_VERSION" "$VENV" >/dev/null
  fi
  log "installing Python dependencies (torch with CUDA, sentence-transformers, privrag) ..."
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    "$UV" pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX_URL" "torch>=2.2"
  fi
  "$UV" pip install --python "$VENV/bin/python" -e "$ROOT/selfhosted" \
      "torch>=2.2" "sentence-transformers>=3.0" "zstandard>=0.22"
}
setup_python

"$VENV/bin/python" - <<'PY' || die "PyTorch cannot use the NVIDIA GPU (torch.cuda.is_available() is False). Terminating."
import sys, torch
if not torch.cuda.is_available():
    sys.exit(1)
print(f"[make_embeddings] torch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}")
PY

# ---------------------------------------------------------------- 3. configuration -> db_data
mkdir -p "$DB_DIR/logs" "$DB_DIR/hf_cache"
export HF_HOME="$DB_DIR/hf_cache"                     # BGE-M3 weights are kept with the DB (offline reuse)
export PRIVRAG_ENV=local
export PRIVRAG_DATA_DIR="$DB_DIR"
export PRIVRAG_LOG_DIR="$DB_DIR/logs"
export PRIVRAG_PDF_DIR="$ROOT/documents/spg_compliance"
export PRIVRAG_MANIFEST_PATH="$ROOT/documents/manifest.csv"
export PRIVRAG_QDRANT_MODE=path                        # embedded Qdrant, files in db_data/qdrant
export PRIVRAG_EMBED_BACKEND=bge-m3
export PRIVRAG_EMBED_MODEL=BAAI/bge-m3
export PRIVRAG_EMBED_DEVICE=cuda
export PRIVRAG_REQUIRE_GPU=true
export PRIVRAG_EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-20}"
export PRIVRAG_PROGRESS_EVERY=20                       # progress line after every 20th embedding
export PRIVRAG_LLM_BACKEND=fake                        # no LLM needed to build the index
PRIVRAG="$VENV/bin/privrag"

# ---------------------------------------------------------------- 4. PDFs -> chunks -> embeddings -> Qdrant
log "stage 1/2: parsing and chunking the PDFs ..."
"$PRIVRAG" ingest --allow-failures "${ALLOW_FAILURES:-2}"

log "stage 2/2: embedding the chunks with BGE-M3 on the GPU and storing them in Qdrant ($DB_DIR/qdrant) ..."
"$PRIVRAG" index

cat > "$DB_DIR/EMBEDDINGS_INFO.txt" <<EOF
Built:        $(date -Iseconds)
Host:         $(hostname)
GPU:          $GPU_INFO
Embedder:     BAAI/bge-m3 (sentence-transformers, fp16, 1024 dims) + BM25 sparse vectors
Vector DB:    Qdrant local mode, folder qdrant/, collection spg_compliance
Contents:     qdrant/ (vectors)  chunks.jsonl + parsed/ (texts)  hf_cache/ (BGE-M3 model)  logs/
Transfer:     copy this whole db_data folder next to the repo on the target machine,
              then run ./install_run_rag.sh there.
EOF
log "vector database stored in $DB_DIR ($(du -sh "$DB_DIR" | cut -f1))"
echo "DONE!"
