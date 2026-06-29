"""
evaluate.py — RAG evaluation pipeline with MLflow tracking

Implements RAGAS-style metrics from scratch using direct Ollama calls.
No RAGAS dependency — avoids langchain version conflicts entirely.

Metrics implemented:
  Faithfulness     = supported_claims / total_claims
  Answer Relevancy = cosine_sim(embed(answer), embed(question))
  Context Recall   = ground_truth_claims_found_in_context / total_gt_claims
  Context Precision = relevant_chunks_in_top_k / k
"""

import os
import json
import logging
import httpx
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

import mlflow

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Test set ──────────────────────────────────────────────────────────────────
TEST_SET = [
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": "The attention mechanism allows the model to weigh the importance of different tokens when encoding a sequence, computing a weighted sum of values based on query-key dot product similarities scaled by sqrt(dk).",
    },
    {
        "question": "What is LoRA and how does it reduce parameters?",
        "ground_truth": "LoRA decomposes the weight update matrix into two low-rank matrices A and B, so instead of updating the full d×d matrix, only r×d + d×r parameters are trained, where r is much smaller than d.",
    },
    {
        "question": "How does RAG improve factual accuracy of LLMs?",
        "ground_truth": "RAG retrieves relevant documents at inference time and grounds the LLM response in that context, reducing hallucinations caused by relying solely on parametric memory.",
    },
]


# ── Ollama helpers ────────────────────────────────────────────────────────────
def ollama_generate(prompt: str) -> str:
    response = httpx.post(
        f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate",
        json={
            "model": os.getenv("LLM_MODEL", "qwen3.5:9b"),
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        },
        timeout=120.0,
    )
    return response.json()["response"].strip()


def ollama_embed(text: str) -> list[float]:
    response = httpx.post(
        f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/embed",
        json={"model": os.getenv("EMBED_MODEL", "nomic-embed-text"), "input": text},
        timeout=30.0,
    )
    return response.json()["embeddings"][0]


def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Metric implementations ────────────────────────────────────────────────────
def compute_faithfulness(answer: str, context: str) -> float:
    """
    Faithfulness = supported_claims / total_claims

    Ask LLM to decompose answer into atomic claims,
    then check each claim against context.
    """
    decompose_prompt = f"""Break this answer into atomic factual claims.
Return ONLY a JSON array of strings, one claim per item.
Answer: {answer}
Claims:"""

    try:
        claims_raw = ollama_generate(decompose_prompt)
        # strip markdown fences if present
        claims_raw = claims_raw.replace("```json", "").replace("```", "").strip()
        claims = json.loads(claims_raw)
    except Exception:
        # fallback: split by sentences
        claims = [s.strip() for s in answer.split(".") if s.strip()]

    if not claims:
        return 0.0

    supported = 0
    for claim in claims:
        check_prompt = f"""Context: {context}

Is this claim supported by the context above? Answer only YES or NO.
Claim: {claim}
Answer:"""
        result = ollama_generate(check_prompt).upper()
        if "YES" in result:
            supported += 1

    return supported / len(claims)


def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy = cosine_sim(embed(answer), embed(question))

    High score = answer is semantically close to the question.
    """
    q_embed = ollama_embed(question)
    a_embed = ollama_embed(answer)
    return cosine_sim(q_embed, a_embed)


def compute_context_recall(ground_truth: str, context: str) -> float:
    """
    Context Recall = gt_claims_found_in_context / total_gt_claims

    Checks how much of the ground truth is covered by retrieved context.
    """
    prompt = f"""Context: {context}

Ground truth: {ground_truth}

What fraction of the ground truth information is present in the context?
Answer with a decimal between 0.0 and 1.0 only, nothing else.
Score:"""

    try:
        score = float(ollama_generate(prompt).strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def compute_context_precision(question: str, context_chunks: list[str]) -> float:
    """
    Context Precision = relevant_chunks / total_chunks

    Checks how many of the retrieved chunks are actually relevant.
    """
    if not context_chunks:
        return 0.0

    relevant = 0
    for chunk in context_chunks:
        prompt = f"""Question: {question}
Context chunk: {chunk}

Is this chunk relevant to answering the question? Answer YES or NO only.
Answer:"""
        result = ollama_generate(prompt).upper()
        if "YES" in result:
            relevant += 1

    return relevant / len(context_chunks)


# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation(
    test_set: list[dict] = None,
    experiment_name: str = "rag-llmops-eval",
    run_name: str = None,
):
    from rag_chain import query as rag_query

    test_set = test_set or TEST_SET
    run_name = run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    log.info(f"Running evaluation on {len(test_set)} questions...")
    all_scores = []

    for item in test_set:
        log.info(f"  Evaluating: {item['question'][:60]}")
        result = rag_query(item["question"])

        context_str = "\n\n".join(result["context"])

        scores = {
            "faithfulness": compute_faithfulness(result["answer"], context_str),
            "answer_relevancy": compute_answer_relevancy(
                item["question"], result["answer"]
            ),
            "context_recall": compute_context_recall(item["ground_truth"], context_str),
            "context_precision": compute_context_precision(
                item["question"], result["context"]
            ),
            "latency_ms": result["latency_ms"],
        }

        log.info(f"    scores: { {k: round(v,3) for k,v in scores.items()} }")
        all_scores.append(scores)

    # Average across test set
    avg_scores = {
        metric: round(sum(s[metric] for s in all_scores) / len(all_scores), 4)
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "latency_ms",
        ]
    }

    # Log to MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "llm_model": os.getenv("LLM_MODEL"),
                "embed_model": os.getenv("EMBED_MODEL"),
                "chunk_size": os.getenv("CHUNK_SIZE"),
                "top_k": os.getenv("TOP_K"),
                "num_questions": len(test_set),
            }
        )
        mlflow.log_metrics({k: v for k, v in avg_scores.items()})
        log.info(f"MLflow run logged: {run_name}")

    return avg_scores


if __name__ == "__main__":
    scores = run_evaluation()
    print("\n=== Evaluation Results ===")
    for metric, score in scores.items():
        print(f"  {metric:25s}: {score:.4f}")
