# Running the self-hosted RAG assistant (no Docker)

Two shell scripts in the repo root. Both run as a normal user (no `sudo`) on Linux and install
everything they need into `./.runtime` (uv, Python 3.11 venv, PyTorch with CUDA, Ollama).

| Script | What it does |
|---|---|
| `./make_embeddings.sh` | Checks the NVIDIA GPU with `nvidia-smi` (terminates immediately if none), installs the Python deps, parses + chunks the 50 PDFs, embeds every chunk with **BGE-M3 on the GPU** and stores the vectors permanently in a local **Qdrant** database in `./db_data`. Prints a progress line after every 20th embedding and `DONE!` at the end. |
| `./install_run_rag.sh` | Restores the **prebuilt embeddings** from `rag_db/rag_db_data.tar.gz` into `./db_data` if needed (checksum-verified), installs the Python deps (CPU-only PyTorch when there is no GPU) and **Ollama** (user space), pulls **`mistral-small3.1:24b`**, keeps Ollama running and starts the RAG API. Works on CPU-only machines. |

The repo ships a prebuilt vector DB (`rag_db/`, built with `make_embeddings.sh` on an RTX 2080 SUPER:
50 PDFs, 12,186 chunks, BGE-M3). On a CPU-only machine just run `./install_run_rag.sh`.
To rebuild it on a GPU machine: `./make_embeddings.sh`, then
`tar czf rag_db/rag_db_data.tar.gz db_data/{qdrant,chunks.jsonl,parsed,index_*.json,ingest_state.json,reports,EMBEDDINGS_INFO.txt}`
and `sha256sum rag_db/rag_db_data.tar.gz | sed 's|rag_db/||' > rag_db/rag_db_data.tar.gz.sha256`.

```bash
chmod +x make_embeddings.sh install_run_rag.sh
./make_embeddings.sh          # once (re-runs skip what is already indexed, interrupted runs resume)
./install_run_rag.sh          # foreground, Ctrl+C stops API + Ollama
# or keep it running in the background:
nohup ./install_run_rag.sh > rag.log 2>&1 &

curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Innerhalb welcher Frist muss die interne Meldestelle den Eingang einer Meldung bestätigen?"}'
```

## Where the data lives

```
db_data/                      <- copy this folder to move the assistant to another machine
  qdrant/                     vector database (dense BGE-M3 + BM25 sparse vectors)
  chunks.jsonl, parsed/       chunk texts and metadata
  index_meta.path.*.json      which embedder built the index (checked at query time)
  hf_cache/                   BGE-M3 model (used offline on the target machine)
  ollama_models/              mistral-small3.1:24b weights (~15 GB)
  logs/, audit.db             run logs, ollama.log, audit trail of all questions
.runtime/                     venv, uv, Ollama binary - machine-specific, recreated by the scripts
```

To transfer: `tar czf db_data.tgz db_data`, unpack it next to the repo on the target machine and
run `./install_run_rag.sh` there. Both folders are in `.gitignore`.

## Options (environment variables)

| Variable | Default | Script |
|---|---|---|
| `GPU_INDEX` | GPU with most free memory | both |
| `DB_DIR` | `./db_data` | both |
| `EMBED_BATCH_SIZE` | `20` | make_embeddings |
| `ALLOW_FAILURES` | `2` (PDFs that may fail to parse) | make_embeddings |
| `TORCH_INDEX_URL` | PyPI (CUDA 12 wheels); e.g. `https://download.pytorch.org/whl/cu118` for old drivers | both |
| `LLM_MODEL` | `mistral-small3.1:24b` | install_run_rag |
| `OLLAMA_PORT` / `API_HOST` / `API_PORT` | `11434` / `127.0.0.1` / `8000` | install_run_rag |
| `OLLAMA_CONTEXT_LENGTH` | `16384` | install_run_rag |

Needs: Linux x86_64, `curl` or `wget`, `tar`, NVIDIA driver (`nvidia-smi`), internet on the first run
(PyPI, astral.sh, huggingface.co, ollama.com). A 24 GB GPU fits Mistral Small 3.1 (Q4) plus BGE-M3. CPU-only: ~20 GB free RAM; answers take minutes
(LLM timeout is raised to 30 min automatically, override with `LLM_TIMEOUT_S`).

## Windows: `install_run_rag.exe`

Same steps as `install_run_rag.sh`, as one double-clickable program (no admin rights, no Docker):
restores `rag_db/` into `db_data\`, installs uv + Python 3.11 + packages into `.runtime\`
(CUDA PyTorch if `nvidia-smi` finds a GPU, otherwise CPU), installs Ollama into `.runtime\ollama`
(or reuses a running/installed one), pulls `mistral-small3.1:24b`, starts the API and prints:

```
 This PC:   http://127.0.0.1:8000/ask
 Network:   http://192.168.x.y:8000/ask
 API key:   <generated once, saved in .runtime\api_key.txt>
```

1. Clone or download the repo, keep `install_run_rag.exe` in the repo folder.
2. Double-click it (SmartScreen: "More info" -> "Run anyway"; the exe is not code-signed).
3. When Windows Firewall asks about `python.exe`, allow it so other machines can reach port 8000.

Options (run from a terminal): `install_run_rag.exe --api-key hello123 --host 127.0.0.1 --port 8000
--model mistral:7b`. Close the window or press Ctrl+C to stop API and Ollama.
Source: `launcher/main.go` (Go, stdlib only). Rebuild:
`cd launcher && GOOS=windows GOARCH=amd64 go build -ldflags "-s -w" -o ../install_run_rag.exe .`
