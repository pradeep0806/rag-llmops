"""
log_retrieval_quality.py — structured retrieval quality logging

Logs every retrieval configuration change as an MLflow run, so you can
compare "before reranker" vs "after reranker" with real numbers instead
of eyeballing terminal output.

This is the proper LLMOps way to validate a change: run the same
evaluation suite before and after, log both as MLflow runs, compare.

Usage:
  PYTHONPATH=src uv run python src/log_retrieval_quality.py --tag baseline
  PYTHONPATH=src uv run python src/log_retrieval_quality.py --tag with_reranker
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
from rag_chain import get_retriever, call_ollama

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

TEST_QUESTIONS = [
    "What is the attention mechanism in transformers?",
    "What is LoRA and how does it reduce parameters?",
    "How does RAG improve factual accuracy of LLMs?",
]


def judge_relevance(question: str, chunk: str) -> bool:
    """LLM judges YES/NO relevance — returns bool for easy aggregation."""
    prompt = f"""Question: {question}
Context chunk: {chunk}

Is this chunk relevant to answering the question? Answer YES or NO only.
Answer:"""
    result = call_ollama(prompt).strip().upper()
    return result.startswith("YES")


def evaluate_retrieval(use_reranker: bool, tag: str):
    """
    Run retrieval-only evaluation (no generation) — isolates retrieval
    quality from generation quality, so we know exactly what we're
    measuring when we change retrieval config.
    """
    retriever = get_retriever()
    retrieve_k = int(os.getenv("RETRIEVE_K", 15))
    top_k = int(os.getenv("TOP_K", 3))

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

    # Log to MLflow — every config change becomes a comparable run
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("retrieval-quality")

    run_name = f"{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "tag": tag,
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
    parser.add_argument(
        "--tag",
        default="run",
        help="Label for this run, e.g. 'baseline' or 'with_reranker'",
    )
    parser.add_argument(
        "--no-reranker", action="store_true", help="Disable reranker for this run"
    )
    args = parser.parse_args()

    use_reranker = not args.no_reranker

    print(
        f"\nRunning retrieval quality eval | tag={args.tag} | reranker={'ON' if use_reranker else 'OFF'}\n"
    )
    avg_precision, results = evaluate_retrieval(use_reranker=use_reranker, tag=args.tag)

    print(f"\n=== Results ({args.tag}) ===")
    print(f"  Average Context Precision: {avg_precision:.4f}")
    for r in results:
        print(f"  - {r['question'][:60]:60s} {r['relevant_count']}/{r['total_count']}")
