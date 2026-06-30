"""
log_retrieval.py — structured retrieval quality logging

Logs every retrieval configuration change as an MLflow run, so you can
compare chunking strategies, top_k, reranker on/off, etc. with real
numbers instead of eyeballing terminal output.

Usage:
  PYTHONPATH=src uv run python src/log_retrieval.py --tag baseline
  PYTHONPATH=src uv run python src/log_retrieval.py --tag semantic --collection ai_papers_semantic
  PYTHONPATH=src uv run python src/log_retrieval.py --tag structured --collection ai_papers_structured
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

import mlflow
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rag_chain import call_ollama

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

TEST_QUESTIONS = [
    "What is the attention mechanism in transformers?",
    "What is LoRA and how does it reduce parameters?",
    "How does RAG improve factual accuracy of LLMs?",
    "What is multi-head attention and why use multiple heads?",
    "How does LoRA compare to full fine-tuning in performance?",
    "What is the encoder-decoder architecture in transformers?",
    "What are the limitations of retrieval-augmented generation?",
    "What is positional encoding and why is it needed?",
]


def get_retriever_for_collection(collection_name: str, retrieve_k: int):
    """
    Build a retriever pointed at a specific Qdrant collection — lets us
    A/B test chunking strategies (recursive vs semantic vs structured)
    which each live in their own collection.
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
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieve_k},
    )


def judge_relevance(question: str, chunk: str) -> bool:
    """LLM judges YES/NO relevance — returns bool for easy aggregation."""
    prompt = f"""Question: {question}
Context chunk: {chunk}

Is this chunk relevant to answering the question? Answer YES or NO only.
Answer:"""
    result = call_ollama(prompt).strip().upper()
    return result.startswith("YES")


def evaluate_retrieval(use_reranker: bool, tag: str, collection_name: str):
    """
    Run retrieval-only evaluation (no generation) — isolates retrieval
    quality from generation quality, so we know exactly what we're
    measuring when we change retrieval config.
    """
    retrieve_k = int(os.getenv("RETRIEVE_K", 15))
    top_k = int(os.getenv("TOP_K", 4))

    retriever = get_retriever_for_collection(collection_name, retrieve_k)

    print(
        f"\n[CONFIG] collection={collection_name} | retrieve_k={retrieve_k} | top_k={top_k} | reranker={'ON' if use_reranker else 'OFF'}\n"
    )

    per_question_results = []

    for question in TEST_QUESTIONS:
        candidates = retriever.invoke(question)

        if use_reranker:
            from reranker import rerank

            final_docs = rerank(question, candidates, top_k=top_k)
        else:
            final_docs = candidates[:top_k]

        relevance_flags = [
            judge_relevance(question, d.page_content) for d in final_docs
        ]
        precision = (
            sum(relevance_flags) / len(relevance_flags) if relevance_flags else 0.0
        )

        sources = [d.metadata.get("source", "?").split("/")[-1] for d in final_docs]

        log.info(
            f"  [{tag}] '{question[:50]}' precision={precision:.2f} sources={set(sources)}"
        )

        per_question_results.append(
            {
                "question": question,
                "precision": precision,
                "relevant_count": sum(relevance_flags),
                "total_count": len(relevance_flags),
                "sources": sources,
            }
        )

    avg_precision = sum(r["precision"] for r in per_question_results) / len(
        per_question_results
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("retrieval-quality")

    run_name = f"{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "tag": tag,
                "collection": collection_name,
                "use_reranker": use_reranker,
                "retrieve_k": retrieve_k,
                "top_k": top_k,
                "embed_model": os.getenv("EMBED_MODEL"),
                "reranker_model": (
                    "cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else "none"
                ),
            }
        )
        mlflow.log_metric("avg_context_precision", avg_precision)
        for i, r in enumerate(per_question_results):
            mlflow.log_metric(f"precision_q{i+1}", r["precision"])

        log.info(f"Logged run '{run_name}' — avg_context_precision={avg_precision:.4f}")

    return avg_precision, per_question_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run", help="Label for this run")
    parser.add_argument(
        "--no-reranker", action="store_true", help="Disable reranker for this run"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection to query (defaults to QDRANT_COLLECTION env var)",
    )
    args = parser.parse_args()

    use_reranker = not args.no_reranker
    collection_name = args.collection or os.getenv("QDRANT_COLLECTION", "ai_papers")

    print(
        f"\nRunning retrieval quality eval | tag={args.tag} | collection={collection_name} | reranker={'ON' if use_reranker else 'OFF'}\n"
    )
    avg_precision, results = evaluate_retrieval(
        use_reranker=use_reranker, tag=args.tag, collection_name=collection_name
    )

    print(f"\n=== Results ({args.tag}) ===")
    print(f"  Average Context Precision: {avg_precision:.4f}")
    for r in results:
        print(f"  - {r['question'][:60]:60s} {r['relevant_count']}/{r['total_count']}")
