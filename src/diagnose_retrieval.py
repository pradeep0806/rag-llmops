"""
diagnose_retrieval.py — inspect what's actually being retrieved

For each test question, prints each retrieved chunk with:
  - similarity score
  - which source paper it came from
  - first 150 chars of content
  - LLM judgment: relevant or not (and why)

This tells us WHY context_precision is low — too low a similarity
threshold? Wrong paper getting mixed in? Chunks too generic?
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from rag_chain import get_retriever, call_ollama

TEST_QUESTIONS = [
    "What is the attention mechanism in transformers?",
    "What is LoRA and how does it reduce parameters?",
    "How does RAG improve factual accuracy of LLMs?",
]


def judge_relevance(question: str, chunk: str) -> str:
    prompt = f"""Question: {question}
Context chunk: {chunk}

Is this chunk relevant to answering the question? Answer YES or NO, then one short reason.
Answer:"""
    return call_ollama(prompt)


def main():
    retriever = get_retriever()

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 100)
        print(f"QUESTION: {question}")
        print("=" * 100)

        docs = retriever.invoke(question)

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown").split("/")[-1]
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:150].replace("\n", " ")

            judgment = judge_relevance(question, doc.page_content)

            print(f"\n  [{i}] source={source} page={page}")
            print(f"      preview: {preview}...")
            print(f"      judge: {judgment}")


if __name__ == "__main__":
    main()
