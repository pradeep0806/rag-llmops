"""
api.py — FastAPI application

Endpoints:
  POST /query     — run a RAG query
  POST /ingest    — trigger document ingestion
  GET  /health    — health check
  GET  /metrics   — Prometheus scrape endpoint (auto-exposed)

Observability wired in:
  - Prometheus: request count, latency histogram, token estimates
  - Loki: structured log lines per request
"""

import os
import time
import logging
import logging_loki
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
from pydantic import BaseModel

from rag_chain import query as rag_query
from ingest import ingest

# ── Loki logging setup ──────────────────────────────────────────────────────
# Every log.info() call gets shipped to Loki with these labels
# so you can filter in Grafana by app, environment, etc.

loki_handler = logging_loki.LokiHandler(
    url=os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push"),
    tags={"app": "rag-llmops", "env": "dev"},
    version="1",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), loki_handler],
)
log = logging.getLogger("rag-api")

# ── Custom Prometheus metrics ────────────────────────────────────────────────
# prometheus_fastapi_instrumentator auto-tracks: request count, latency, status codes
# We add LLM-specific metrics on top:

from prometheus_client import Histogram, Counter, REGISTRY


def _get_or_create_metric(metric_class, name, description, **kwargs):
    try:
        return metric_class(name, description, **kwargs)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


rag_latency = _get_or_create_metric(
    Histogram,
    "rag_query_latency_seconds",
    "End-to-end RAG query latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
query_counter = _get_or_create_metric(
    Counter, "rag_queries_total", "Total RAG queries", labelnames=["status"]
)
context_chunks_used = _get_or_create_metric(
    Histogram,
    "rag_context_chunks",
    "Number of context chunks retrieved per query",
    buckets=[1, 2, 3, 4, 5, 8, 10],
)
retrieval_latency = _get_or_create_metric(
    Histogram,
    "rag_retrieval_latency_seconds",
    "Qdrant retrieval latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0],
)


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("RAG API starting up")
    yield
    log.info("RAG API shutting down")


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG LLMOps API",
    description="Production RAG system with Qdrant + Ollama + full observability",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auto-exposes /metrics endpoint for Prometheus to scrape
Instrumentator().instrument(app).expose(app)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    latency_ms: float


class IngestRequest(BaseModel):
    data_dir: str = "data/papers"


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL")}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    log.info(f"Query received: {req.question[:80]}")
    start = time.perf_counter()

    try:

        result = rag_query(req.question)
        latency = time.perf_counter() - start

        # Record metrics
        rag_latency.observe(latency)
        query_counter.labels(status="success").inc()
        context_chunks_used.observe(len(result["context"]))

        log.info(
            f"Query completed | latency={latency*1000:.1f}ms | "
            f"chunks={len(result['context'])} | "
            f"question={req.question[:60]}"
        )

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            latency_ms=result["latency_ms"],
        )

    except Exception as e:
        query_counter.labels(status="error").inc()
        log.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    log.info(f"Ingestion triggered for: {req.data_dir}")
    try:
        ingest(req.data_dir)
        return {"status": "success", "data_dir": req.data_dir}
    except Exception as e:
        log.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
