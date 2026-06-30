"""
reranker.py — Cross-encoder reranking for retrieval precision

Why rerank?
  Vector similarity search (bi-encoder) embeds the query and each chunk
  SEPARATELY, then compares vectors. This is fast but loses information —
  it can only capture "is this topically related," not "does this
  specifically answer the question."

  A cross-encoder instead takes (query, chunk) TOGETHER as joint input
  and outputs a single relevance score. This is much more accurate
  because the model can directly compare query terms against chunk
  content with full attention — but it's slower, so we only apply it
  to the top-k candidates from the fast vector search (not the whole
  corpus).

Two-stage retrieval pattern:
  Stage 1 (recall):    vector search → top 20 candidates (fast, approximate)
  Stage 2 (precision):  cross-encoder reranks → top 5 final (slow, accurate)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Trained on MS MARCO passage ranking — a query/passage relevance
  dataset. Outputs a single relevance logit per (query, passage) pair.

Implementation note:
  We use transformers' AutoModelForSequenceClassification directly
  instead of sentence_transformers.CrossEncoder. The CrossEncoder
  wrapper probes for a "modules.json" file that doesn't exist in this
  model's repo (it's a transformers-native model, not a
  sentence-transformers-native one), causing a confusing 404 -> SSL
  retry loop. Going through transformers directly avoids that probe
  entirely — same model weights, same math, no wrapper quirk.
"""

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

log = logging.getLogger(__name__)

_model = None
_tokenizer = None

# If RERANKER_MODEL_PATH is set (e.g. a manually downloaded local folder),
# use that instead of fetching from the HF Hub. Useful when corporate
# VPN/firewall SSL inspection blocks huggingface_hub's download client.
MODEL_NAME = os.getenv("RERANKER_MODEL_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_reranker():
    """Load the cross-encoder once, reuse across requests."""
    global _model, _tokenizer
    if _model is None:
        log.info(f"Loading cross-encoder reranker model: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        _model.to(
            "cpu"
        )  # MPS (Apple GPU) produces NaN logits for this model — force CPU
        _model.eval()
        log.info("Reranker loaded")
    return _model, _tokenizer


def rerank(question: str, docs: list, top_k: int = 5) -> list:
    """
    Rerank retrieved documents by cross-encoder relevance score.

    Args:
        question: the user's query
        docs: list of LangChain Document objects from vector search
        top_k: how many to keep after reranking

    Returns:
        Reordered + filtered list of docs, most relevant first.
    """
    if not docs:
        return docs

    model, tokenizer = get_reranker()

    pairs = [(question, doc.page_content) for doc in docs]
    queries = [p[0] for p in pairs]
    passages = [p[1] for p in pairs]

    features = tokenizer(
        queries,
        passages,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    features = {k: v.to("cpu") for k, v in features.items()}

    with torch.no_grad():
        scores = model(**features).logits.squeeze(-1).tolist()

    if isinstance(scores, float):
        scores = [scores]

    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    log.info("Rerank scores: " + ", ".join(f"{s:.2f}" for _, s in scored_docs))

    return [doc for doc, _ in scored_docs[:top_k]]
