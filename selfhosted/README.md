# Variant B - self-hosted, open-source RAG pipeline

Our own RAG pipeline for UC13. Every component is open source. On Azure it runs on
tenant-controlled infrastructure only: **documents, embeddings, model weights, prompts and answers
never leave the customer's VNet.** The managed Variant A (Azure AI Search / Document Intelligence /
Azure OpenAI) is not used or touched here.

| Stage | Managed service it replaces | Our open-source component | Local stand-in (for tests) |
|---|---|---|---|
| 1 Parse | Azure AI Document Intelligence | **Docling** (PyMuPDF fallback) | PyMuPDF |
| 1 Chunk | - | own §/Artikel/AT-aware chunker + LangChain splitter | same |
| 2 Embed | Azure OpenAI embeddings | **BGE-M3** (TEI or in-process) | char-n-gram LSA |
| 2 Index | Azure AI Search (hybrid) | **Qdrant** dense + BM25 sparse, RRF fusion | Qdrant local mode |
| 3 Rerank | semantic ranker | **bge-reranker-v2-m3** | off |
| 3 Generate | Azure OpenAI GPT-4o | **Llama 3.3 70B via vLLM**, LangChain `ChatOpenAI` | deterministic extractive fake |
| 4 API + audit | - | **FastAPI**, SQLAlchemy -> **PostgreSQL** | SQLite |
| 5 Evaluate | - | own runner on the UC13 question set | same |

The same code runs in both modes; only `PRIVRAG_*` settings change (`.env.example` vs `azure.env.example`).

## Run it

```bash
cd selfhosted
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

privrag doctor          # checks files, index, endpoints
privrag ingest          # stage 1: 50 PDFs -> data/chunks.jsonl   (~10 s)
privrag index           # stage 2: -> Qdrant                       (~100 s locally)
privrag ask "Welche Fristen gelten für die Meldung schwerwiegender IKT-Vorfälle nach DORA?"
privrag eval            # stage 5: question set -> data/reports/eval_<run>.json + .xlsx
privrag serve           # stage 4: API on http://127.0.0.1:8000  (docs at /docs)
pytest                  # 113 tests, ~15 s
```

Each stage is idempotent: unchanged documents/indexes are skipped; a changed PDF, setting or
ingestion code re-processes only what is affected. `--force` / `--recreate` rebuild everything.

## Local test with the real BGE-M3 (Ollama on a laptop)

`scripts/run_local_ollama.ps1` reproduces the Azure setup on a Windows laptop: Ollama serves
**BGE-M3** on `/v1/embeddings` (same contract as TEI on Azure) and a small open LLM on
`/v1/chat/completions` (same contract as vLLM). Results go to `data_ollama/` so the stand-in
results in `data/` stay untouched.

```powershell
cd D:\Private-RAG-Assistant
powershell -ExecutionPolicy Bypass -File selfhosted\scripts\run_local_ollama.ps1            # full run
powershell -ExecutionPolicy Bypass -File selfhosted\scripts\run_local_ollama.ps1 -Limit 5   # quick check
```

**Laptop profile (CPU only).** BGE-M3 embeds only ~0.3-0.6 chunks/s on a laptop CPU, so a full
index takes hours. The launcher therefore gives every chunk keyword (BM25) search immediately and
adds BGE-M3 vectors for as many chunks as fit in a time budget (default 5 min), spread over all
50 documents. First start ~10 min, later starts ~1-2 min. `start_chat.ps1 -Improve` adds another
5 min of BGE-M3 coverage (resumable, coverage grows each time), `-Full` embeds everything.
Answers use 4 sources x 900 chars and no extra rewrite call (Azure: 8 x 1800 with rewrite).

Options: `-LlmModel qwen2.5:7b` (bigger, slower), `-LlmModel none` (retrieval only, fake LLM),
`-TopK 8 -MaxSourceChars 1800` (Azure prompt size). Compare runs with
`python scripts/compare_evals.py A=data/reports/eval_x.json B=data_ollama/reports/eval_y.json`.

## How retrieval works (and why)

1. The English question is also turned into a **German query** (LLM rewrite on Azure; plus a
   generic EN->DE compliance glossary, `src/privrag/retrieve/glossary_en_de.json`).
2. Original and German query each run through **dense** and **BM25** search (4 ranked lists).
3. Lists are fused with **Reciprocal Rank Fusion**, optionally **reranked**, then capped at
   4 chunks per document so multi-document questions get several sources.
4. The LLM must cite `[n]` after each statement; citations to non-existent sources are removed,
   fewer than 2 citations triggers one retry, no citations -> `insufficient_sources`,
   `NOT_IN_SOURCES` -> `out_of_scope`.

Chunks never cross a section boundary and carry `section` (e.g. `§ 10 Allgemeine Sorgfaltspflichten`,
`Artikel 19 ...`, `AT 4.4.2 Compliance-Funktion`) and exact pages, so every citation reads
`01_GwG.pdf | § 10 ... | p. 17-18`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness, loaded backends, index size, startup error if degraded |
| GET | `/ready` | 200 only when index, audit DB and LLM endpoint are reachable |
| POST | `/ask` | `{"question", "doc_ids"?, "regulators"?, "user"?}` -> answer, citations (+quote), confidence, latency |
| GET | `/documents` | corpus list (for UI filters) |
| GET | `/sources/{chunk_id}` | full chunk text for source highlighting |
| GET | `/audit`, `/audit/{request_id}` | audit trail (append-only) |

`X-Request-ID` is accepted or generated and returned; set `PRIVRAG_API_KEY` to require `X-API-Key`.
When the LLM is down `/ask` returns **503** with a structured error - and the request is still audited.

## Debugging - every step is logged

* Every run writes `logs/<run_id>.jsonl` (+ `logs/latest.jsonl`). Each line has `run_id`, `stage`,
  `doc_id` / `request_id`, `event`, `duration_ms`, and on failure `error_code`, `error_type` and the
  full `traceback`.
* `privrag logs --errors` shows only warnings/errors of the last run, with tracebacks.
* Per-run reports: `data/reports/ingest_<run>.json` (status/pages/chunks/warnings per document),
  `index_<run>.json`, `eval_<run>.json/.xlsx`.
* API: grep the request id - `grep rid-123 logs/latest.jsonl` shows question, retrieval (top
  sources), generation, answer and HTTP status.

| Error code | Stage | Meaning / fix |
|---|---|---|
| `PARSE_EMPTY_FILE`, `PARSE_NOT_PDF`, `PARSE_CORRUPT`, `PARSE_TOO_LARGE` | ingest | bad upload - replace the file |
| `PARSE_ENCRYPTED` | ingest | password-protected PDF |
| `CHUNK_NO_TEXT` | ingest | scanned PDF without text layer - needs OCR (Docling OCR) |
| `MANIFEST_*` | ingest | manifest.csv unreadable / no `file` column |
| `EMBED_*` | index | embedding model/endpoint problem (`EMBED_ENDPOINT_DOWN`, `EMBED_DIM_MISMATCH`, ...) |
| `INDEX_DIM_MISMATCH`, `INDEX_VERIFY_FAILED` | index | rebuild with `privrag index --recreate` |
| `INDEX_NOT_BUILT`, `INDEX_EMBEDDER_MISMATCH` | retrieve | run `privrag index` / fix `PRIVRAG_EMBED_BACKEND` |
| `RETRIEVE_ALL_FAILED` | retrieve | dense and keyword search both down (Qdrant unreachable) |
| `LLM_TIMEOUT`, `LLM_UNREACHABLE`, `LLM_HTTP_ERROR`, `LLM_EMPTY` | generate | vLLM endpoint problem |
| `AUDIT_WRITE_FAILED` | api | answer delivered, audit DB failed - check PostgreSQL |

## Tests (113) - what robustness is checked

* **Stage 1:** heading detection incl. look-alike references (`§ 12 Absatz 4`, `§ 46 Abs. 1`), header/footer
  and hyphenation cleanup, manifest variants (`;` delimiter, BOM, bad dates, duplicates, missing rows),
  empty / non-PDF / truncated / encrypted / image-only / duplicate PDFs isolated with error codes,
  resume after a crash, corrupted state file, settings or code change invalidates checkpoints,
  Docling missing -> PyMuPDF fallback, deterministic chunk ids.
* **Stage 2:** tokenizer keeps `§ 25h`, `Art. 5b`, `AT 4.4.2`, `2022/2554`; embedding output validation
  (shape, dim, NaN); remote embedder unreachable / response format; idempotent upsert, skip-if-unchanged,
  stale points removed, dimension mismatch, length mismatch.
* **Stage 3:** dense side down -> keyword-only; both down -> clear error; reranker down -> fused order kept;
  remote reranker; embedder/index mismatch refused; citation parsing/renumbering, invalid citation removal,
  no-citation answers not marked verified, retry on 1 citation, out-of-scope; the **real LangChain client
  against a mock vLLM server** over HTTP incl. timeout, HTTP 500, HTTP 400, empty answer, unreachable host.
* **Stage 4:** health/ready, audit row per request, source lookup, input validation, filters, LLM down ->
  503 + audited, degraded start without index, API key, audit DB failure, 16 concurrent requests,
  request-id tracing.
* **Stage 5:** expected-source parsing, key-point proxy, end-to-end eval with an out-of-scope question.

## Local results (stand-in models) - see `docs/BUILD_LOG.md`

These validate the pipeline, not answer quality: LSA embeddings are not cross-lingual and the fake
LLM only copies sentences. Accuracy >= 85% has to be measured on Azure with BGE-M3 + Llama 3.3.
