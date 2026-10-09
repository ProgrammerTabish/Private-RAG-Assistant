# Private RAG Assistant

**Use Case 13: AI Compliance Document Reader for Banking** (Specific-Group GmbH)

A privacy-preserving, citation-backed AI assistant for regulatory documents (BaFin, MaRisk, EBA), with the LLM and all data running inside infrastructure fully controlled by the bank. Includes a comparative evaluation of **Managed (Azure OpenAI)** vs. **Self-hosted (open-source LLM)** deployment.

## Event

| | |
|---|---|
| Event | [Innovation Challenge & Innovation Fair 2026](https://innovationchallenge2026.smart.niederbayern.de/events/innovation-challenge-2026/) |
| Organizer | Ecosystem Smart Niederbayern |
| Dates | October 8–10, 2026 |
| Location | Stadthalle Landau, Stadtgraben 3, 94405 Landau an der Isar |
| Registration deadline | October 3, 2026 |
| Team size | 5 students per use case |
| Language | Deliverables, documentation and presentations in English |
| Format | Students work independently; results are presented to a professional jury and partners (some use cases run a 24-hour challenge) |

## Motivation

Banks in the DACH region operate under a growing volume of regulatory documentation (BaFin circulars, MaRisk, EBA guidelines, internal policies). Compliance, risk and second-line-of-defence teams spend significant time searching, cross-referencing and quoting these documents. Public LLM services (e.g. ChatGPT) cannot or should not be used because documents and questions are highly confidential and subject to banking secrecy.

## Target

Build an AI assistant that lets a bank employee ask natural-language questions on a curated body of regulatory and internal documents and receive precise, citation-backed answers, with the LLM and all data running inside tenant-controlled infrastructure. A core deliverable is a **comparative evaluation of two deployment architectures (managed vs. self-hosted) with a clear recommendation.**

## Approach

1. **Corpus:** Curate a sample corpus of public regulatory documents (BaFin, MaRisk, EBA) as the demo dataset.
2. **Ingestion pipeline:** PDF parsing via Azure AI Document Intelligence → chunking → embedding → index in Azure AI Search (vector + hybrid).
3. **Two RAG variants in parallel:**
   - **(A) Managed:** Azure OpenAI Service (GPT-4o) inside the customer's Azure tenant via Private Endpoint.
   - **(B) Self-hosted:** Open-source LLM (e.g. Llama 3.3 70B or Mixtral 8x22B) on Azure ML Managed Online Endpoints or AKS with GPU nodes in the customer's tenant. Model weights live on tenant-controlled storage.
4. **UI:** Chat-style web UI with source highlighting, confidence indicator and audit trail.
5. **Benchmark** both variants on the same evaluation set: answer accuracy, latency, cost per 1k queries, regulatory posture (where weights run, where data flows, AVV implications).
6. **Decision matrix** "Managed vs. Self-hosted" as guidance for regulated industries.

## Actors / Stakeholders

- Compliance officers, risk managers, internal audit, IT security and data-protection officers of a typical DACH bank
- Internally: SPG Germany product/sales team driving the "Private AI" offering, SPG nearshore development team

## Business Value

- Tangible proof-of-concept and reference architecture for SPG's "Private/On-Premise AI" go-to-market story in regulated industries, reusable in sales situations with banks and asset-finance institutions
- Defensible answer to the customer question *"why not just use ChatGPT?"*
- Estimated 20–30% productivity gain for compliance teams on document-research tasks

## Definition of Done / Success Metrics (KPIs)

- [ ] Working web UI deployed on Azure
- [ ] Both RAG variants (managed and self-hosted) operational
- [ ] Ingestion of **at least 50** regulatory documents
- [ ] Answers with **at least 2 verifiable source citations** per query
- [ ] Answer accuracy **≥ 85%** on a prepared question set (graded by coaches)
- [ ] Side-by-side benchmark table for both variants
- [ ] Architecture documentation with a clear "data & weights stay in tenant" statement for the self-hosted variant

## Distribution of Activities

| Share | Area |
|---|---|
| 40% | Development and programming (ingestion pipeline, RAG backend, chat UI) |
| 30% | AI training and integration (open-source LLM deployment, prompt engineering, retrieval tuning, benchmark) |
| 15% | Architecture and security documentation (private-deployment story, AVV/MaRisk angle) |
| 15% | User experience design (citation display, audit trail, comparison view) |

## Technical Stack

**Common:** Azure AI Search (vector + hybrid), Azure Blob Storage, Azure AI Document Intelligence, Azure App Service or Container Apps (web UI), Azure Key Vault, Azure Virtual Network with Private Endpoints.

**Variant A (Managed):** Azure OpenAI Service (GPT-4o, embeddings).

**Variant B (Self-hosted):** Azure ML Managed Online Endpoints or AKS with GPU nodes (NC-series), running an open-source LLM (Llama 3.3, Mixtral or similar) plus an open-source embedding model (BGE-M3 or E5).

**Languages:** Python (FastAPI) backend, TypeScript/React frontend.

**Recommended enablement:** Azure, Azure ML, GitHub Copilot. Prior exposure to vector search, RAG or model deployment (Hugging Face, vLLM) is a plus, not required.

## Provided by SPG

- Curated sample document corpus
- Prepared evaluation question set
- Azure subscription with GPU quota

Deliverables (reference architecture and benchmark) feed into SPG's "Private AI" reference architecture for regulated industries.

## Candidate Profile

Bachelor or Master students in Computer Science, Data Science or Information Systems. Useful knowledge: Python, basic NLP/LLM concepts, web development, interest in regulatory/compliance topics and cloud architecture.

## Useful Links

- [Innovation Challenge 2026 event page](https://innovationchallenge2026.smart.niederbayern.de/events/innovation-challenge-2026/)
- [Azure AI Search](https://learn.microsoft.com/azure/search/)
- [Azure AI Document Intelligence](https://learn.microsoft.com/azure/ai-services/document-intelligence/)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-services/openai/)
- [Azure ML Managed Online Endpoints](https://learn.microsoft.com/azure/machine-learning/concept-endpoints-online)
- [Azure Private Endpoints](https://learn.microsoft.com/azure/private-link/private-endpoint-overview)
- [vLLM](https://docs.vllm.ai/)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [FastAPI](https://fastapi.tiangolo.com/)
- Regulatory sources: [BaFin](https://www.bafin.de), [MaRisk](https://www.bafin.de/DE/Aufsicht/BankenFinanzdienstleister/MaRisk/maRisk_node.html), [EBA Guidelines](https://www.eba.europa.eu/regulation-and-policy)

## Run it (self-hosted, no Docker)

```bash
./make_embeddings.sh   # NVIDIA GPU -> BGE-M3 embeddings -> Qdrant in ./db_data, prints DONE!
./install_run_rag.sh   # restores rag_db/ embeddings (CPU ok) + Ollama + mistral-small3.1:24b + API on :8000
```

Details: [RUN.md](RUN.md)
