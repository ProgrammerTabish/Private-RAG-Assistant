# RAG database (prebuilt)

`rag_db_data.tar.gz` is the ready-to-use `db_data/` folder for **Private-RAG-Assistant**.
Chunking and embedding are already done, so a new machine can answer questions without
re-ingesting the PDFs.

| | |
|---|---|
| Built | 2026-10-09 09:30 on ki-031 |
| Documents | 50 (spg_compliance corpus) |
| Chunks | 12,186 (`chunks.jsonl`, plus per-document files in `parsed/`) |
| Vector DB | Qdrant local mode, collection `spg_compliance` |
| Vectors | BAAI/bge-m3 dense (1024-dim, cosine) + BM25 sparse (IDF) |
| Payload | chunk_id, doc_id, title, file, regulator, doc_type, language, publication_date, section, page_start/end, seq, char_len, text |

Not included: the BGE-M3 model cache (4.3 GB), which is downloaded automatically
on first run, and the logs.

## Restore on another machine (CPU is fine)

`./install_run_rag.sh` does this automatically: if `db_data/` does not exist yet, it verifies
`rag_db_data.tar.gz` against the `.sha256` file and unpacks it to `./db_data`. No embedding is
computed on the target machine (only the question itself is embedded at query time), so it runs
without a GPU. It then installs Ollama + `mistral-small3.1:24b` and starts the API.

```bash
git clone https://github.com/ProgrammerTabish/private-rag-assistant.git
cd private-rag-assistant
./install_run_rag.sh
```

Do **not** run `make_embeddings.sh` on the target machine; it needs an NVIDIA GPU and rebuilds
everything from the PDFs. The first run needs internet access (Python packages, BGE-M3 for the
query embedding, the LLM).

Manual restore, if needed: `sha256sum -c rag_db_data.tar.gz.sha256 && tar -xzf rag_db_data.tar.gz -C ..`
