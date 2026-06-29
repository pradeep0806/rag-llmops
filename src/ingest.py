"""
ingest.py — Document ingestion pipeline

Flow:
  PDF files → load → chunk → embed (Ollama) → store (Qdrant)

Chunking strategy:
  We split by RecursiveCharacterTextSplitter which tries to split on
  ["\n\n", "\n", " ", ""] in order — preserving semantic boundaries
  (paragraphs first, then lines, then words) before hard character splits.

  chunk_size=512 tokens is a sweet spot for RAG:
  - too small → context lost, faithfulness drops
  - too large → noise injected, precision drops
  chunk_overlap=64 ensures sentences cut at boundaries aren't lost.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )


def ensure_collection(
    client: QdrantClient, collection_name: str, vector_size: int = 768
):
    """
    Create collection if it doesn't exist.
    nomic-embed-text produces 768-dim vectors.
    We use COSINE distance — standard for semantic similarity.

    Cosine similarity: sim(a,b) = (a·b) / (||a|| × ||b||)
    Range: [-1, 1], higher = more similar
    """
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
    """Load all PDFs from the data directory."""
    path = Path(data_dir)
    if not path.exists:
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


def chunk_documents(docs, chunk_size: int = None, chunk_overlap: int = None):
    """
    Split documents into chunks using recursive character splitting.

    The splitter tries each separator in order:
      \n\n → paragraph boundary (best)
      \n   → line boundary
      " "  → word boundary
      ""   → character boundary (last resort)
    """

    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 64))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info(
        f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def ingest(data_dir: str = "data/papers"):
    """Full ingestion pipeline: load → chunk → embed → store."""

    collection_name = os.getenv("QDRANT_COLLECTION", "ai_papers")
    # 1. Load
    docs = load_documents(data_dir)
    # 2. Chunk
    chunks = chunk_documents(docs)

    # 3. Embeddings — nomic-embed-text via Ollama
    #    Produces 768-dim dense vectors via a transformer encoder
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    client = get_qdrant_client()
    ensure_collection(client, collection_name, vector_size=768)
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', 6333)}",
        collection_name=collection_name,
    )
    log.info(
        f"Ingested {len(chunks)} chunks into Qdrant collection '{collection_name}'"
    )
    return vectorstore


if __name__ == "__main__":
    ingest()
