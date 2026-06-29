"""
rag_chain.py — RAG pipeline

Flow per query:
  query → embed → ANN search in Qdrant (top-k chunks)
        → build prompt → Ollama API (direct) → answer

Why direct Ollama API instead of LangChain's OllamaLLM wrapper?
  Qwen3's thinking mode is controlled via "think": False in the request
  body. LangChain's wrapper doesn't expose this parameter, so we bypass
  it and call /api/generate directly via httpx. This cuts latency from
  ~60s (thinking mode on) to ~5-10s (thinking mode off).

Retrieval — Approximate Nearest Neighbor via HNSW:
  Given query vector q, find k vectors {v₁...vₖ} where
  cosine_sim(q, vᵢ) is maximized — without scanning every vector.

  cosine_sim(a, b) = (a · b) / (||a|| × ||b||)   range: [-1, 1]

  HNSW builds a layered graph: upper layers coarse, lower layers dense.
  Search starts at top and greedily navigates down — O(log n) vs O(n)
  for brute force. Qdrant uses this internally for all similarity search.

Caching:
  _retriever is initialized once at module load and reused across
  requests — avoids rebuilding the Qdrant client + embedding model
  on every query.
"""

import os
import time
import logging
import httpx
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()
log = logging.getLogger(__name__)

# ── Module-level cache — initialized once, reused across all requests ─────────
_retriever = None


def get_retriever():
    """Return cached retriever, building it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


# ── Retriever ─────────────────────────────────────────────────────────────────
def build_retriever():
    """
    Build the Qdrant retriever with nomic-embed-text embeddings.

    nomic-embed-text produces 768-dim vectors via a transformer encoder
    trained specifically for retrieval tasks (not generation).
    """
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION", "ai_papers"),
        embedding=embeddings,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": int(os.getenv("TOP_K", 5))},
    )


# ── LLM — direct Ollama API call ──────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    """
    Call Ollama /api/generate directly via httpx.

    Key parameter: "think": False
      Qwen3 models support an explicit thinking mode where the model
      reasons step-by-step before answering (like chain-of-thought).
      This adds 40-60s of latency with no benefit for RAG tasks where
      context is already provided. Setting think=False disables it.

      This parameter is only respected at the raw API level —
      LangChain's OllamaLLM wrapper does not pass it through.
    """
    response = httpx.post(
        f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate",
        json={
            "model": os.getenv("LLM_MODEL", "qwen3.5:9b"),
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 256,
                "num_ctx": 4096,
            },
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["response"]


# ── Prompt builder ─────────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    """Concatenate retrieved chunks with source metadata."""
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def build_prompt(question: str, context: str) -> str:
    return f"""You are an expert AI/ML research assistant.
Use the provided context to answer the question thoroughly.
If the context is partially relevant, use what's available and say what's missing.

Context:
{context}

Question: {question}

Answer:"""


# ── Main query function ────────────────────────────────────────────────────────
def query(question: str) -> dict:
    """
    Run a query through the RAG pipeline.

    Steps:
      1. Embed query → ANN search in Qdrant → top-k chunks
      2. Format chunks into context string
      3. Build prompt and call Ollama directly
      4. Return answer + sources + latency

    Returns:
      dict with keys: question, answer, context, sources, latency_ms
    """
    retriever = get_retriever()
    start = time.perf_counter()

    context_docs = retriever.invoke(question)
    context_str = format_docs(context_docs)
    prompt = build_prompt(question, context_str)
    answer = call_ollama(prompt)

    latency_ms = (time.perf_counter() - start) * 1000
    log.info(f"Query done | latency={latency_ms:.0f}ms | chunks={len(context_docs)}")

    return {
        "question": question,
        "answer": answer,
        "context": [d.page_content for d in context_docs],
        "sources": [d.metadata for d in context_docs],
        "latency_ms": round(latency_ms, 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = query("What is the attention mechanism in transformers?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nLatency: {result['latency_ms']}ms")
    print(f"\nSources: {result['sources']}")
