# Deploying the self-hosted RAG backend (Variant B) with Docker

```
                       ┌──────────────── docker compose (privrag) ────────────────┐
  client ──HTTP:8000──▶│  api (FastAPI)  ◀── app_data volume: chunks, audit DB, logs│
                       │     │      │                                              │
                       │     │      └──▶ qdrant   ◀── qdrant_storage volume         │
                       │     └─────────▶ embeddings (BGE-M3, TEI) ◀── hf_models     │
                       │  indexer (one-shot: PDFs → chunks → BGE-M3 → Qdrant)       │
                       └──────────────────────────────┬───────────────────────────┘
                                                      │ OpenAI-compatible /v1
                                                      ▼
                                LLM outside the stack (vLLM / Ollama), swappable
```

* The 50 PDFs, `manifest.csv` and the evaluation set are **baked into the image**.
* Embeddings live in the **`qdrant_storage` volume**; chunks, index metadata and the audit
  trail in **`app_data`**. Stopping, restarting or rebuilding containers keeps them.
  Only `docker compose down -v` deletes them.
* The **indexer** runs on every `docker compose up` but skips everything that is already
  indexed (≈ 1-2 s). The expensive embedding of the 12k chunks happens **once**; if it is
  interrupted, the next start resumes where it stopped.
* The **LLM is not in the stack**. Changing `LLM_BASE_URL` / `LLM_MODEL` in `.env` and
  `docker compose up -d` swaps it; the index is not touched.

## Run

```bash
cp .env.example .env            # set LLM_BASE_URL and LLM_MODEL
docker compose up -d --build    # first start: downloads BGE-M3, embeds all PDFs
docker compose logs -f indexer  # progress with ETA ("embedding progress 2400/12186 ...")
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Innerhalb welcher Frist muss die interne Meldestelle den Eingang einer Meldung bestätigen?"}'
```

First start: the embeddings container downloads BGE-M3 (~2.3 GB) into `hf_models`, then the
indexer embeds ~12,200 chunks. On a GPU VM this takes minutes; on CPU only it takes hours
(use the GPU image, see below). Later starts take seconds.

## Azure - one command (Cloud Shell)

Current subscription setup (checked 2026-10-09): no GPU quota, 10 CPU cores for VMs, AI Foundry
resource with pay-per-token open-weight models. So: **LLM = Llama-3.3-70B-Instruct in AI Foundry**
(Global Standard, billed per token), **stack = CPU VM with Docker**.

```bash
# Azure Cloud Shell (bash)
git clone https://github.com/ProgrammerTabish/private-rag-assistant.git   # asks for GitHub user + token
cd private-rag-assistant
bash deploy/azure/deploy.sh          # SKIP_START=true bash deploy/azure/deploy.sh = prepare only
```

The script creates the Llama deployment, builds the image in `rgcontainerrag`, creates
`vm-privrag` (own VNet, no SSH port open), starts the stack and opens port 8000 only for the
web app's outbound IPs (`ALLOW_IPS=1.2.3.4` adds more). It prints the API URL; the API key is in
`~/.privrag_api_key` in Cloud Shell and in `/opt/privrag/.env` on the VM. Existing resources
are not modified. Note: "Global Standard" means Azure may process LLM requests in any region.

## API (port 8000)

| Method | Path | |
|---|---|---|
| GET | `/health` | status, models, index size and BGE-M3 coverage |
| GET | `/ready` | 200 when index, audit DB and LLM are reachable |
| POST | `/ask` | `{"question", "doc_ids"?, "regulators"?, "user"?}` → answer, citations (file, section, pages, quote), confidence, latency |
| GET | `/documents` | indexed documents |
| GET | `/sources/{chunk_id}` | full source text of a citation |
| GET | `/audit`, `/audit/{request_id}` | audit trail |
| GET | `/docs` | OpenAPI / Swagger |

Set `PRIVRAG_API_KEY` to require the header `X-API-Key` on everything except `/health`.

## Swap the LLM

```bash
# .env
LLM_BASE_URL=http://10.0.1.4:8000/v1        # e.g. vLLM with Llama 3.3 70B on an Azure GPU VM
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
docker compose up -d                         # api restarts, index unchanged
```

Any OpenAI-compatible server works (vLLM, Ollama `…:11434/v1`, TGI, llama.cpp server).

## Azure

1. **VM** (Ubuntu, Docker + compose plugin) in the customer VNet. For fast first indexing use a
   GPU VM (NC-series) and set `EMBED_IMAGE=ghcr.io/huggingface/text-embeddings-inference:1.7`
   plus the `deploy.resources` GPU block in `docker-compose.yml`.
2. **LLM**: vLLM on the same or another GPU VM in the VNet → `LLM_BASE_URL=http://<private-ip>:8000/v1`.
3. Copy the repo (or pull it), `cp .env.example .env`, edit, `docker compose up -d --build`.
4. Expose port 8000 only inside the VNet (NSG) or behind Azure Application Gateway; set
   `PRIVRAG_API_KEY`. Optional: `PRIVRAG_AUDIT_DB_URL` → Azure Database for PostgreSQL.
5. Back up the `qdrant_storage` and `app_data` volumes (or map them to Azure managed disks)
   so a VM rebuild does not require re-embedding.

Data, embeddings, model weights and LLM all stay inside the tenant; nothing calls a public
model API at runtime (the BGE-M3 weights are downloaded once on first start; pre-load
`hf_models` from tenant storage if the VM has no internet access).

## Operations

```bash
docker compose ps                                  # indexer should be "exited (0)"
docker compose logs indexer | tail                 # last index run
docker compose run --rm indexer privrag init       # re-run indexing manually (idempotent)
docker compose run --rm api privrag ask "…"        # one question from the CLI
docker compose run --rm api privrag logs --errors  # recent warnings/errors with tracebacks
docker compose down                                # stop; volumes (embeddings) are kept
```

New or changed PDFs: add them to `documents/spg_compliance/` (and `manifest.csv`), rebuild
(`docker compose up -d --build`); the indexer embeds only the new chunks.
