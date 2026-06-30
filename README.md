# RAG LLMOps

Production-grade Retrieval Augmented Generation system with a full LLMOps observability stack — built entirely on local infrastructure, zero API costs.

## Architecture

```
PDF Documents
     ↓
 Ingestion (LangChain + nomic-embed-text)
     ↓
 Qdrant Vector Store (HNSW, cosine similarity)
     ↓
 Two-Stage Retrieval (vector search → cross-encoder rerank)
     ↓
 RAG Pipeline (prompt → Ollama)
     ↓
 FastAPI + BentoML (serving)
     ↓
 Prometheus + Grafana (observability)
     ↓
 MLflow (experiment tracking + evaluation)
```

## Stack

| Layer                 | Tool                                                           |
| --------------------- | -------------------------------------------------------------- |
| LLM                   | Ollama (`qwen3.5:9b`) — local, no API cost                     |
| Embeddings            | `nomic-embed-text` via Ollama (768-dim)                        |
| Vector Store          | Qdrant (HNSW approximate nearest neighbor)                     |
| Reranker              | `cross-encoder/ms-marco-MiniLM-L-6-v2` (two-stage retrieval)   |
| RAG Orchestration     | LangChain LCEL                                                 |
| API                   | FastAPI                                                        |
| Serving               | BentoML                                                        |
| Evaluation            | Custom LLM-as-judge (RAGAS-style metrics, no RAGAS dependency) |
| Experiment Tracking   | MLflow                                                         |
| Metrics               | Prometheus + Grafana                                           |
| Dependency Management | `uv` + `pyproject.toml`                                        |
| Infra                 | Docker Compose                                                 |

## Project Structure

```
rag-llmops/
├── src/
│   ├── ingest.py               # PDF → chunk → embed → Qdrant
│   ├── rag_chain.py            # retriever + reranker + Ollama generation
│   ├── reranker.py             # cross-encoder reranking module
│   ├── api.py                  # FastAPI endpoints + Prometheus metrics
│   ├── evaluate.py             # LLM-as-judge evaluation + MLflow logging
│   ├── log_retrieval.py        # isolated retrieval-quality experiments
│   └── diagnose_retrieval.py   # chunk-level retrieval inspection tool
├── bento/
│   └── service.py              # BentoML service definition
├── observability/
│   ├── prometheus.yml
│   ├── loki-config.yml
│   └── grafana/
│       └── provisioning/
│           └── datasources/
├── docker-compose.yml
├── bentofile.yaml
├── pyproject.toml              # uv-managed dependencies
└── .env
```

## Quickstart

### Prerequisites

- [Ollama](https://ollama.ai) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [uv](https://docs.astral.sh/uv/)

### 1. Pull models

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Start infrastructure

```bash
docker compose up qdrant mlflow prometheus loki grafana -d
```

### 4. Add documents

Drop PDF files into `data/papers/`:

```bash
curl -L "https://arxiv.org/pdf/1706.03762" -o data/papers/attention.pdf
curl -L "https://arxiv.org/pdf/2106.09685" -o data/papers/lora.pdf
curl -L "https://arxiv.org/pdf/2005.11401" -o data/papers/rag.pdf
```

### 5. Ingest documents

```bash
PYTHONPATH=src uv run python src/ingest.py
```

### 6. Start the API

```bash
PYTHONPATH=src uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 7. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the attention mechanism in transformers?"}'
```

## BentoML Serving

```bash
# Serve via BentoML
PYTHONPATH=src uv run bentoml serve bento.service:RAGService --port 8051

# Query BentoML endpoint
curl -X POST http://localhost:8051/query \
  -H "Content-Type: application/json" \
  -d '{"req": {"question": "What is LoRA?"}}'
```

## Evaluation

### Why not RAGAS?

RAGAS (the standard RAG evaluation library) has a known breaking incompatibility with LangChain v0.3+. It internally imports `langchain_community.chat_models.vertexai`, which was removed in v0.3, making it impossible to use alongside a modern LangChain stack without downgrading the entire dependency tree.

### Custom LLM-as-Judge Implementation

Instead of RAGAS, we implement the same 4 metrics from scratch using direct Ollama API calls — the LLM itself acts as the judge. Same mathematical definitions, zero external dependency, full control over prompts.

**Faithfulness** — are answers grounded in retrieved context?

```
Faithfulness = supported_claims / total_claims
```

The LLM decomposes the answer into atomic claims, then checks each claim against the retrieved context (YES/NO per claim).

**Answer Relevancy** — does the answer address the question?

```
Answer Relevancy = cosine_sim(embed(answer), embed(question))
```

Embeds both question and answer via `nomic-embed-text`, computes cosine similarity. High score = answer is semantically on-topic.

**Context Recall** — does retrieved context cover the ground truth?

```
Context Recall = ground_truth_claims_found_in_context / total_ground_truth_claims
```

The LLM scores what fraction of the ground truth is present in the retrieved chunks (0.0–1.0).

**Context Precision** — are retrieved chunks actually relevant?

```
Context Precision = relevant_chunks / total_chunks_retrieved
```

Each retrieved chunk is individually judged for relevance to the question.

### Diagnosing and Fixing a Context Precision Regression

Initial evaluation surfaced a weak Context Precision score (0.40) — meaning 60% of retrieved chunks were irrelevant noise. Rather than guessing at a fix, each hypothesis was tested in isolation with results logged to MLflow for direct before/after comparison.

**Step 1 — Diagnose with raw chunk inspection.** Built `diagnose_retrieval.py` to print every retrieved chunk alongside an LLM relevance judgment. This surfaced two distinct problems: a missing source document (the RAG paper had never been successfully ingested, so RAG-related queries scored 0/5 relevant chunks), and chunk granularity (512-token chunks mixed multiple ideas — definitions, formulas, implementation details — into the same chunk, diluting relevance per chunk).

**Step 2 — Add a cross-encoder reranker.** Implemented two-stage retrieval: a fast vector search returns 15 candidates, then `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores each (query, chunk) pair jointly for true relevance. Vector similarity alone (bi-encoder) can only capture topical overlap; a cross-encoder can distinguish "topically related" from "directly answers the question." Result: 0.40 → 0.60.

**Step 3 — Test recall depth.** Hypothesized that widening the candidate pool (`RETRIEVE_K` from 15 to 25) would surface more relevant chunks for the reranker to choose from. Logged as a separate MLflow run for direct comparison — result was unchanged (0.60 → 0.60), ruling out recall depth as the bottleneck and pointing back to chunk quality as the real constraint.

**Step 4 — Fix chunking granularity.** Reduced `CHUNK_SIZE` from 512 to 256 tokens (overlap 100). Smaller chunks isolate single ideas instead of mixing definition, formula, and implementation detail in one block. Result: 0.60 → 0.667.

**Step 5 — Tighten `TOP_K` using reranker confidence.** Reranker scores revealed a real confidence cliff per query — e.g. one query's top-3 candidates scored `6.08, -0.22, -2.64` — only the top chunk was strongly relevant, yet a fixed `TOP_K=5` was force-including weak, low-confidence chunks just to hit a count. Tested `TOP_K=3`: context precision jumped to 0.778, but a full 4-metric evaluation (not just the isolated precision check) revealed a real cost — faithfulness dropped from 0.80 to 0.56 and context recall dropped from 0.75 to 0.58. Retrieving fewer, narrower chunks gave the model less material to construct a complete, well-grounded answer from — a textbook precision/recall tradeoff.

**Step 6 — Find the balance point.** Tested `TOP_K=4` as a middle ground between the original 5 and the over-aggressive 3. This recovered faithfulness to 0.73 (close to the 0.80 baseline) while keeping context precision nearly double the original (0.75 vs 0.40), with answer relevancy unchanged. Context recall stayed flat at 0.58 between `top_k=3` and `top_k=4`, indicating recall is currently bottlenecked by chunk size, not top_k — a separate, known limitation rather than something tuning `top_k` further would fix.

### Results Summary

| Stage                      | Configuration                      | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
| -------------------------- | ---------------------------------- | ------------ | ---------------- | -------------- | ----------------- |
| Baseline                   | chunk=512/64, top_k=5, no reranker | 0.80         | 0.81             | 0.75           | 0.40              |
| + Reranker, smaller chunks | chunk=256/100, top_k=5             | —            | —                | —              | 0.667             |
| Over-aggressive top_k      | chunk=256/100, top_k=3             | 0.56         | 0.79             | 0.58           | 0.778             |
| **Final (balanced)**       | **chunk=256/100, top_k=4**         | **0.73**     | **0.81**         | **0.58**       | **0.75**          |

The final configuration nearly doubles context precision (0.40 → 0.75) while keeping faithfulness and answer relevancy close to their original levels. Context recall (0.75 → 0.58) is the one metric that didn't fully recover — it's bottlenecked by the smaller chunk size rather than `top_k`, and is documented here as a known, deliberate tradeoff rather than an unexamined regression.

This process is the actual point: the first "improvement" (`top_k=3`, precision 0.778) looked like a win on a single metric but was a regression once measured against the full evaluation suite. Tracking all 4 metrics together — not optimizing one in isolation — is what caught it.

### Run evaluation

Full RAG evaluation (generation + all 4 metrics):

```bash
PYTHONPATH=src uv run python src/evaluate.py
```

Isolated retrieval-quality experiments (compare retrieval configs without re-running generation):

```bash
PYTHONPATH=src uv run python src/log_retrieval.py --tag my_experiment
PYTHONPATH=src uv run python src/log_retrieval.py --tag my_baseline --no-reranker
```

View results at `http://localhost:5000`

## Observability

| Dashboard  | URL                                 |
| ---------- | ----------------------------------- |
| Grafana    | http://localhost:3000 (admin/admin) |
| MLflow     | http://localhost:5000               |
| Prometheus | http://localhost:9090               |
| Qdrant UI  | http://localhost:6333/dashboard     |

Custom Prometheus metrics:

- `rag_query_latency_seconds` — end-to-end latency histogram (p50, p95, p99)
- `rag_queries_total` — query count by status (success/error)
- `rag_context_chunks` — chunks retrieved per query
- `rag_retrieval_latency_seconds` — Qdrant retrieval latency

## How It Works

### Retrieval — Two-Stage: HNSW + Cross-Encoder Reranking

**Stage 1 (recall):** Query text is embedded into a 768-dim vector via `nomic-embed-text`. Qdrant finds the top-15 candidate chunks using HNSW (Hierarchical Navigable Small World graphs):

```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||)
```

HNSW navigates a layered graph — O(log n) vs O(n) brute force. This stage is fast but approximate — it embeds the query and each chunk _separately_, so it can only capture topical similarity, not whether a chunk actually answers the question.

**Stage 2 (precision):** The top-15 candidates are rescored by a cross-encoder, which takes (query, chunk) _jointly_ as input and outputs a single relevance logit. This is slower per-pair but far more accurate, so it's only applied to the small candidate set from stage 1, not the whole corpus. The top 4 by cross-encoder score become the final context — tuned down from an initial top 5, see Evaluation section for why 4 (not 3 or 5) is the balance point.

### Generation — Grounded Prompting

Retrieved chunks are injected into a prompt that instructs the model to answer only from context — maximizing faithfulness and minimizing hallucination.

### Chunking Strategy

Documents split with `RecursiveCharacterTextSplitter` (chunk_size=256, overlap=100), trying separators in order: `\n\n → \n → " " → ""` — preserving semantic boundaries before hard character splits. Chunk size was tuned down from an initial 512 after evaluation showed larger chunks diluted relevance by mixing multiple ideas per chunk (see Evaluation section above).

## Tech Decisions

**Why Qdrant over ChromaDB?** Production-grade server with REST + gRPC API, proper HNSW tuning, and filtering on metadata payloads. ChromaDB is in-process only.

**Why a cross-encoder reranker?** Bi-encoder vector search alone plateaued at 0.40 context precision. A reranker that jointly scores (query, chunk) pairs, combined with chunking and top_k tuning, lifted this to 0.75 while keeping faithfulness and answer relevancy near baseline — see the Evaluation section for the full validated experiment trail, including a precision-only "improvement" that was caught and reverted after checking the full metric suite.

**Why direct Ollama API over LangChain's OllamaLLM?** Qwen3's `think=False` parameter (disables chain-of-thought, cuts latency significantly) is only respected at the raw API level — LangChain's wrapper doesn't pass it through.

**Why uv over pip/poetry?** Rust-based resolver — 10-100x faster installs, deterministic lockfile (`uv.lock`), no dependency conflicts.

**Why custom evaluation over RAGAS?** RAGAS is broken with LangChain v0.3+ due to a removed Vertex AI import. Custom implementation gives identical metrics, no version constraints, and full control over judge prompts.

## License

MIT
