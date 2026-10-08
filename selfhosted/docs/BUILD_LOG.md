# Build log - Variant B (self-hosted) pipeline

Built and tested locally (cloud workspace, 2 CPU / 7 GB RAM, no GPU, no Hugging Face access) on
2026-10-08/09. **Nothing has been deployed to Azure.** Branch: `feature/selfhosted-pipeline`.

## Step 0 - corpus profiling

| Check | Result |
|---|---|
| PDFs | 50, all with a text layer (no OCR needed), none encrypted |
| Pages / characters | 3,387 pages / ~11 M characters |
| Largest | 18_EU_Kommission_Sanktions_FAQ (495 p.), 49_BGB (490 p.), 22_KWG (247 p.) |
| Heading styles found | German statutes `§ 10<NBSP>Titel`; EU acts `Artikel 6` + title on next line; MaRisk/MaComp `AT 4.3.3 Titel`; EBA/ESMA `Title II`, `Guideline 4`, `4.1 Title` |
| Question set | 32 questions (English), 30 in scope, 2 out of scope; expected sources name file + § / Article / AT |

**Corpus finding:** `15_EU_Sanktionen_833_2014.pdf` is the original 2014 text (11 pages). Article 5b,
the expected source of **Q14**, was added by later amendments and is not in this file. Use the
consolidated version of Regulation 833/2014 (EUR-Lex "consolidated text") or Q14 cannot be answered
from the corpus by either variant.

## Step 1 - scaffold, config, logging

* `privrag` package, pydantic settings (`PRIVRAG_*`), typed errors with stable codes, JSON-lines step
  logging with run_id / stage / doc_id / request_id and automatic duration + traceback on failure.

## Step 2 - stage 1 ingestion

Result on the corpus: **50/50 OK, 0 failed, 12,204 chunks, ~10 s**, avg chunk ~890 chars; 32/33
expected sections from the question set exist as labelled chunks (the missing one is Q14, see above).

Bugs found by testing and fixed:
1. PyMuPDF/Docling may turn the NBSP in `§ 10<NBSP>Titel` into a normal space -> added a stricter
   fallback rule (title starts upper-case, not a law abbreviation like `KWG`, no sentence punctuation).
2. `\b` after `Abs.` never matches -> `§ 46 Abs. 1 Satz 5` on the GwG cover page became a section.
   Fixed with a look-ahead; also inflected forms (`Absatzes`).
3. Table-of-contents pages produced junk sections -> pages that are mostly headings are dropped.
4. List markers on their own line (`1.\nDie Identifizierung`) -> joined.
5. EU recitals got no label -> `Erwägungsgründe / Recitals`, but only when articles dominate the
   document (the EU sanctions FAQ mentions two articles and was wrongly labelled at first).
6. **Stale checkpoints after a code change:** re-ingest skipped all documents although the chunking
   code had changed. The checkpoint fingerprint now includes a hash of the ingestion source code.

## Step 3 - stage 2 embeddings + Qdrant

Result: **12,204 points, 256-dim LSA + BM25 sparse, ~100 s, 915 MB peak RAM**, verified count.
Point ids are uuid5(chunk_id) -> idempotent; embedder fingerprint stored; querying with another
embedder than the index was built with is refused.

## Step 4 - stage 3 retrieval + generation

Retrieval experiment on the 31 question set entries with expected sources (local LSA stand-in):

| Strategy | expected doc in top 8 | all expected docs |
|---|---|---|
| dense(q) + BM25(q) - no German terms | 5/31 | 2/31 |
| dense(q) + BM25(q + glossary) | 16/31 | 8/31 |
| **dense(q) + BM25(q) + BM25(DE) + dense(DE)** (chosen, multi-query) | **20/31** | **14/31** |
| same, with *ideal* German terms (simulates the LLM rewrite; uses the set's "Key terms (DE)" column, analysis only) | 27-29/31 | 20/31 |

=> English questions on German law need a German query. On Azure the Llama rewrite + the
cross-lingual BGE-M3 provide it; the generic EN->DE glossary is a cheap extra.

Bug found by testing and fixed: logging a `PrivRagError` with `extra=err.to_dict()` crashed because
`message` is a reserved LogRecord field - the error log itself raised. All privrag loggers now
rename reserved keys (`x_message`).

## Step 5 - stage 4 API + audit

FastAPI with request-id middleware, degraded mode, API key, CORS, SQLite/PostgreSQL audit trail.
Smoke test with the real server (`privrag serve` + curl): `/health` OK, `/ask` answered with 3
citations in ~0.9 s, audit row written, 422 on empty question, 6 log lines for the request id.

## Step 6 - evaluation (local stand-ins)

`privrag eval` on all 32 questions:

| Metric | Local result | Comment |
|---|---|---|
| errors | 0/32 | pipeline stable |
| expected doc retrieved | 20/30 (67 %) | real pipeline component, limited by LSA |
| expected section retrieved | 7/27 (26 %) | same |
| answers with >= 2 citations | 22/30 (73 %) | fake LLM sometimes finds only one supporting sentence |
| expected doc cited | 5/30 | fake LLM picks sentences by English word overlap - not meaningful |
| out-of-scope refused | 0/2 | the fake LLM cannot judge; on Azure: Llama + `NOT_IN_SOURCES` rule + reranker threshold |
| latency p50 / p95 | 0.68 s / 1.1 s | local Qdrant (brute force); a Qdrant server is faster |

Asked in German, the same pipeline goes straight to `27_DORA_2022_2554 Artikel 19` and
`32_DORA_Meldefristen` - the gap is cross-lingual embedding quality, which BGE-M3 addresses.

## Test suite

113 tests, all passing (`pytest`, ~13 s): stage 1: 44, stage 2: 19, stage 3: 30, stage 4: 16, stage 5: 4.

## Not done yet (waiting for guidance)

* Azure deployment: GPU VM / AML endpoint with vLLM (Llama 3.3 70B AWQ or smaller), TEI for BGE-M3 and
  the reranker, Qdrant container, PostgreSQL, API container - see `azure.env.example`.
* Docling, BGE-M3 and reranker code paths are implemented but could only be tested with mocks here
  (no model downloads possible in this workspace). First Azure step: `privrag doctor`, then
  `privrag ingest --force` with Docling and `privrag eval` against the real models.
* Tune `PRIVRAG_MIN_RELEVANCE` (reranker threshold) on the two out-of-scope questions.
