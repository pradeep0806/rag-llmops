"""Standalone reranker test — isolate the NaN bug outside the full pipeline."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_NAME = os.getenv("RERANKER_MODEL_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Model device: {next(model.parameters()).device}")
model.eval()

query = "How many people live in Berlin?"
passages = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
]

features = tokenizer(
    [query, query],
    passages,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(f"Input ids dtype: {features['input_ids'].dtype}")
print(f"Input ids: {features['input_ids']}")

with torch.no_grad():
    output = model(**features)
    print(f"Raw logits: {output.logits}")
    scores = output.logits.squeeze(-1).tolist()

print(f"Scores: {scores}")
