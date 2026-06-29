"""
evaluate.py — RAGAS evaluation pipeline with MLflow tracking

Runs a test set through the RAG chain, computes 4 RAGAS metrics,
and logs everything to MLflow for experiment comparison.

RAGAS Metrics (recap):
  Faithfulness     = supported_claims / total_claims         (hallucination check)
  Answer Relevancy = avg cosine_sim(embed(generated_q), embed(original_q))
  Context Precision = weighted precision@k for retrieved chunks
  Context Recall   = ground_truth_claims_in_context / total_gt_claims
"""

import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    _faithfulness,
    _answer_relevance,
    _answer_correctness,
    _context_precision,
    _context_recall,
)
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from rag_chain import query as rag_query

load_dotenv()
log = logging.getLogger(__name__)

# ── Test set ──────────────────────────────────────────────────────────────────
# ground_truth = what a perfect answer would say.
# RAGAS uses this for context_recall and answer_correctness.
# For your own papers, replace these with real Q&A pairs.
TEST_SET = [
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": "The attention mechanism allows the model to weigh the importance of different tokens when encoding a sequence, computing a weighted sum of values based on query-key dot product similarities.",
    },
    {
        "question": "What is LoRA and how does it reduce parameters?",
        "ground_truth": "LoRA decomposes the weight update matrix into two low-rank matrices A and B, so instead of updating the full d×d matrix, only r×d + d×r parameters are trained, where r << d.",
    },
    {
        "question": "How does RAG improve factual accuracy of LLMs?",
        "ground_truth": "RAG retrieves relevant documents at inference time and grounds the LLM's response in that context, reducing hallucinations caused by relying solely on parametric memory.",
    },
]


def run_evaluation(
    test_set: list[dict] = None,
    experiment_name: str = "rag-llmops-eval",
    run_name: str = None,
):
    """
    Run RAGAS evaluation and log results to MLflow.

    Args:
        test_set: List of {question, ground_truth} dicts
        experiment_name: MLflow experiment to log under
        run_name: Name for this specific run (defaults to timestamp)
    """
    test_set = test_set or TEST_SET
    run_name = run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log.info(f"Running RAG chain on {len(test_set)} test questions...")
    results = []
    for item in test_set:
        result = rag_query(item["question"])
        results.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "contexts": result["context"],  # list of retrieved chunks
                "ground_truth": item["ground_truth"],
            }
        )
        log.info(f"  ✓ {item['question'][:60]}")
    # ── Step 2: Build RAGAS dataset ──────────────────────────────────────────
    eval_dataset = Dataset.from_list(results)

    # ── Step 3: Configure RAGAS to use Ollama (no OpenAI needed) ─────────────
    ollama_llm = OllamaLLM(
        model=os.getenv("LLM_MODEL", "qwen3.5:9b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
        num_predict=512,
        num_ctx=4096,  # add this
        extra_body={"think": False},
    )
    ollama_embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    # ── Step 4: Run RAGAS ────────────────────────────────────────────────────
    log.info("Running RAGAS evaluation...")
    scores = evaluate(
        dataset=eval_dataset,
        metrics=[
            _faithfulness,
            _answer_relevance,
            _context_precision,
            _context_recall,
        ],
        llm=ollama_llm,
        embeddings=ollama_embeddings,
    )

    scores_dict = scores.to_pandas().mean().to_dict()
    log.info(f"RAGAS scores: {json.dumps(scores_dict, indent=2)}")

    # ── Step 5: Log everything to MLflow ─────────────────────────────────────
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "llm_model": os.getenv("LLM_MODEL"),
                "embed_model": os.getenv("EMBED_MODEL"),
                "chunk_size": os.getenv("CHUNK_SIZE"),
                "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
                "top_k": os.getenv("TOP_K"),
            }
        )
        # Log RAGAS metrics
        mlflow.log_metrics(
            {
                "faithfulness": scores_dict.get("faithfulness", 0),
                "answer_relevancy": scores_dict.get("answer_relevancy", 0),
                "context_precision": scores_dict.get("context_precision", 0),
                "context_recall": scores_dict.get("context_recall", 0),
            }
        )
        # Log full results as artifact
        scores_df = scores.to_pandas()
        scores_df.to_csv("/tmp/ragas_results.csv", index=False)
        mlflow.log_artifact("/tmp/ragas_results.csv", "evaluation")

        log.info(f"MLflow run logged: {run_name}")
    return scores_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scores = run_evaluation()
    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in scores.items():
        print(f"  {metric:25s}: {score:.4f}")
