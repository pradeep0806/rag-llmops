"""
bento/service.py — BentoML service definition

BentoML wraps the RAG pipeline into a self-contained deployable unit.

Key concepts:
  @bentoml.service  — declares a class as a BentoML service
                      handles lifecycle, batching, async, health checks
  @bentoml.api      — declares a method as an HTTP endpoint
                      auto-generates OpenAPI spec from type hints
  runners           — BentoML's abstraction for model inference
                      handles batching and async under the hood

Deployment flow:
  bentoml build → creates a Bento (versioned snapshot)
  bentoml containerize → builds a Docker image from the Bento
  docker push → ship to registry
"""

import os
import sys
import logging
from pathlib import Path
import time
from rag_chain import format_docs, build_prompt, call_ollama

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import bentoml
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    latency_ms: float


class IngestRequest(BaseModel):
    data_dir: str = "data/papers"


# ── BentoML Service ───────────────────────────────────────────────────────────
@bentoml.service(
    name="rag-llmops",
    traffic={"timeout": 120, "concurrency": 4},
    resources={"cpu": "2", "memory": "4Gi"},
)
class RAGService:
    """
    Production RAG service with Qdrant + Ollama + full observability.

    BentoML handles:
      - Health checks (/healthz, /readyz)
      - Metrics endpoint (/metrics)
      - OpenAPI docs (/docs)
      - Graceful shutdown
      - Docker packaging (bentoml containerize)
    """

    def __init__(self):
        """
        Initialize once when the service starts.
        BentoML calls __init__ once per worker — not per request.
        This is where we build the retriever (expensive operation).
        """
        log.info("Initializing RAG service...")
        from rag_chain import get_retriever

        self._retriever = get_retriever()
        log.info("RAG service ready")

    @bentoml.api(route="/query")
    async def query(self, req: QueryRequest) -> QueryResponse:
        """Run a RAG query — retrieval + generation."""

        if not req.question.strip():
            raise ValueError("Question cannot be empty")

        start = time.perf_counter()
        context_docs = self._retriever.invoke(req.question)
        context_str = format_docs(context_docs)
        prompt = build_prompt(req.question, context_str)
        answer = call_ollama(prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        log.info(
            f"Query completed | latency={latency_ms:.0f}ms | chunks={len(context_docs)}"
        )

        return QueryResponse(
            question=req.question,
            answer=answer,
            sources=[d.metadata for d in context_docs],
            latency_ms=round(latency_ms, 2),
        )

    @bentoml.api(route="/ingest")
    async def ingest(self, req: IngestRequest) -> dict:
        """Trigger document ingestion into Qdrant."""
        from ingest import ingest as run_ingest

        log.info(f"Ingestion triggered: {req.data_dir}")
        run_ingest(req.data_dir)
        return {"status": "success", "data_dir": req.data_dir}

    @bentoml.api(route="/health")
    async def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "ok",
            "model": os.getenv("LLM_MODEL", "qwen3.5:9b"),
            "collection": os.getenv("QDRANT_COLLECTION", "ai_papers"),
        }
