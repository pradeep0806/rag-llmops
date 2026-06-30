"""
ingest_semantic.py — Semantic chunking variant

Instead of fixed-size character splitting, this embeds consecutive
sentences and splits where embedding similarity between adjacent
sentences drops sharply — i.e. where the topic actually shifts.

How it works:
  1. Split document into sentences
  2. Embed each sentence
  3. Compute cosine distance between consecutive sentence embeddings
  4. Where distance exceeds a percentile threshold, insert a chunk break
  5. Group sentences between breaks into chunks (variable length)

This targets the exact problem found in evaluation: fixed 256-token
chunks sometimes still mixed a definition with an unrelated formula
or result, because chunk boundaries were positional, not topical.
Semantic chunking boundaries follow actual meaning shifts instead.

Tradeoff: slower at ingestion time (one embedding call per sentence,
not per chunk), and chunk sizes become unpredictable (some very
short, some long) — but each chunk should be more topically coherent.

Stored in a SEPARATE Qdrant collection (ai_papers_semantic) so the
existing ai_papers collection (recursive splitting, our current
production config) is untouched and both can be evaluated side by side.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

COLLECTION_NAME = "ai_papers_semantic"


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )


def ensure_collection(
    client: QdrantClient, collection_name: str, vector_size: int = 768
):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        log.info(f"Created collection: {collection_name}")
    else:
        log.info(f"Collection already exists: {collection_name}")


def load_documents(data_dir: str = "data/papers"):
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    loader = DirectoryLoader(
        str(path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    log.info(f"Loaded {len(docs)} pages from {data_dir}")
    return docs


def ingest_semantic(data_dir: str = "data/papers"):
    """Full semantic-chunking ingestion pipeline."""
    docs = load_documents(data_dir)

    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    # SemanticChunker splits at embedding-distance breakpoints.
    # breakpoint_threshold_type="percentile" means: split wherever the
    # similarity drop is in the top X% of drops seen across the document
    # (i.e. the most significant topic shifts), rather than a fixed
    # numeric threshold that would need per-document tuning.
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,  # only split at the most significant 15% of shifts
    )

    log.info(
        "Running semantic chunking (this embeds every sentence — slower than fixed-size)..."
    )
    chunks = splitter.split_documents(docs)
    log.info(
        f"Semantic chunking produced {len(chunks)} chunks (vs fixed-size for comparison)"
    )

    client = get_qdrant_client()
    ensure_collection(client, COLLECTION_NAME, vector_size=768)

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', 6333)}",
        collection_name=COLLECTION_NAME,
    )

    log.info(
        f"Ingested {len(chunks)} semantic chunks into Qdrant collection '{COLLECTION_NAME}'"
    )
    return vectorstore


if __name__ == "__main__":
    ingest_semantic()
