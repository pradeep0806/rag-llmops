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
 RAG Pipeline (top-k retrieval → prompt → Ollama)
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
│   ├── ingest.py        # PDF → chunk → embed → Qdrant
│   ├── rag_chain.py     # retriever + Ollama generation
│   ├── api.py           # FastAPI endpoints + Prometheus metrics
│   └── evaluate.py      # LLM-as-judge evaluation + MLflow logging
├── bento/
│   └── service.py       # BentoML service definition
├── observability/
│   ├── prometheus.yml
│   ├── loki-config.yml
│   └── grafana/
│       └── provisioning/
│           └── datasources/
├── docker-compose.yml
├── bentofile.yaml
├── pyproject.toml       # uv-managed dependencies
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

RAGAS (the standard RAG evaluation library) has a known breaking incompatibility with LangChain v0.3+. It internally imports `langchain_community.chat_models.vertexai` which was removed in v0.3, making it impossible to use alongside a modern LangChain stack without downgrading the entire dependency tree.

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

### Baseline Results (qwen3.5:9b, top_k=5, chunk_size=512)

| Metric            | Score | Interpretation                                                |
| ----------------- | ----- | ------------------------------------------------------------- |
| Faithfulness      | 0.80  | 80% of answer claims grounded in context                      |
| Answer Relevancy  | 0.81  | Answers semantically close to questions                       |
| Context Recall    | 0.75  | 75% of ground truth covered by retrieval                      |
| Context Precision | 0.40  | 40% of retrieved chunks are relevant — retriever over-fetches |

**Key insight from evaluation:** Context Precision at 0.40 indicates the retriever is pulling irrelevant chunks into the top-5. Actionable fix: reduce `TOP_K` from 5 to 3, or add a cross-encoder reranker (e.g. `ms-marco-MiniLM`) as a second-stage filter.

### Run evaluation

```bash
PYTHONPATH=src uv run python src/evaluate.py
```

Results are logged to MLflow at `http://localhost:5000` — every run tracked with model config, chunk settings, and all 4 metric scores for comparison across experiments.

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

### Retrieval — HNSW Approximate Nearest Neighbor

Query text is embedded into a 768-dim vector via `nomic-embed-text`. Qdrant finds the top-k most similar chunks using HNSW (Hierarchical Navigable Small World graphs):

```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||)
```

HNSW navigates a layered graph — O(log n) vs O(n) brute force.

### Generation — Grounded Prompting

Retrieved chunks are injected into a prompt that instructs the model to answer only from context — maximizing faithfulness and minimizing hallucination.

### Chunking Strategy

Documents split with `RecursiveCharacterTextSplitter` (chunk_size=512, overlap=64), trying separators in order: `\n\n → \n → " " → ""` — preserving semantic boundaries before hard character splits.

## Tech Decisions

**Why Qdrant over ChromaDB?** Production-grade server with REST + gRPC API, proper HNSW tuning, and filtering on metadata payloads. ChromaDB is in-process only.

**Why direct Ollama API over LangChain's OllamaLLM?** Qwen3's `think=False` parameter (disables chain-of-thought, cuts latency from ~60s to ~20s) is only respected at the raw API level — LangChain's wrapper doesn't pass it through.

**Why uv over pip/poetry?** Rust-based resolver — 10-100x faster installs, deterministic lockfile (`uv.lock`), no dependency conflicts.

**Why custom evaluation over RAGAS?** RAGAS is broken with LangChain v0.3+ due to a removed Vertex AI import. Custom implementation gives identical metrics, no version constraints, and full control over judge prompts.

## License

MIT
