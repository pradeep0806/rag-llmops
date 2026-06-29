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
 MLflow (experiment tracking)
```

## Stack

| Layer                 | Tool                                                                      |
| --------------------- | ------------------------------------------------------------------------- |
| LLM                   | Ollama (`qwen3.5:9b`) — local, no API cost                                |
| Embeddings            | `nomic-embed-text` via Ollama (768-dim)                                   |
| Vector Store          | Qdrant (HNSW approximate nearest neighbor)                                |
| RAG Orchestration     | LangChain LCEL                                                            |
| API                   | FastAPI                                                                   |
| Serving               | BentoML                                                                   |
| Evaluation            | RAGAS (faithfulness, answer relevancy, context precision, context recall) |
| Experiment Tracking   | MLflow                                                                    |
| Metrics               | Prometheus + Grafana                                                      |
| Dependency Management | `uv` + `pyproject.toml`                                                   |
| Infra                 | Docker Compose                                                            |

## Project Structure

```
rag-llmops/
├── src/
│   ├── ingest.py        # PDF → chunk → embed → Qdrant
│   ├── rag_chain.py     # retriever + Ollama generation
│   ├── api.py           # FastAPI endpoints + Prometheus metrics
│   └── evaluate.py      # RAGAS evaluation + MLflow logging
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

Drop PDF files into `data/papers/`. The repo includes scripts to download AI papers:

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

Run RAGAS evaluation and log results to MLflow:

```bash
PYTHONPATH=src uv run python src/evaluate.py
```

View results at `http://localhost:5000`

Metrics tracked:

- **Faithfulness** — are answers grounded in retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are relevant chunks ranked first?
- **Context Recall** — does retrieved context cover the ground truth?

## Observability

| Dashboard  | URL                                 |
| ---------- | ----------------------------------- |
| Grafana    | http://localhost:3000 (admin/admin) |
| MLflow     | http://localhost:5000               |
| Prometheus | http://localhost:9090               |
| Qdrant UI  | http://localhost:6333/dashboard     |

Custom Prometheus metrics:

- `rag_query_latency_seconds` — end-to-end latency histogram
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

**Why uv over pip/poetry?** Rust-based resolver — 10-100x faster installs, deterministic lockfile, no dependency conflicts.

## License

MIT
