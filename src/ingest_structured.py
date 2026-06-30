"""
ingest_structured.py — Structure-aware chunking variant

Academic papers have real document structure: Abstract, Introduction,
Related Work, Methods, Results, Conclusion. Flat character-based
splitting (recursive or semantic) ignores this entirely and can split
a sentence away from the section it belongs to.

This variant:
  1. Detects section headers using a regex pattern matched against
     common academic paper conventions (numbered sections, all-caps
     headers, standard section names)
  2. Splits documents into per-section chunks first
  3. Within each section, applies RecursiveCharacterTextSplitter if
     the section is still too large for one chunk
  4. Tags each chunk's metadata with its section name — this is a
     bonus: you can now filter retrieval by section (e.g. "only
     search Methods sections") which neither recursive nor semantic
     chunking can do, since they don't track section identity.

Stored in collection: ai_papers_structured — separate from both
ai_papers (recursive, production) and ai_papers_semantic, so all
three can be evaluated side by side without interference.

Tradeoff: regex-based header detection is brittle — it won't catch
every paper's header formatting perfectly. For a production system
at scale, this is where you'd reach for a proper PDF layout parser
(e.g. unstructured.io) instead of regex. This implementation is a
reasonable first pass for the arXiv-style papers in our corpus.
"""

import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

COLLECTION_NAME = "ai_papers_structured"

# Matches common academic section header patterns:
#   "1 Introduction", "1. Introduction", "I. INTRODUCTION", "Abstract"
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*((\d+\.?\d*\.?)\s+)?"
    r"(Abstract|Introduction|Related Work|Background|Method(?:ology|s)?|"
    r"Experiments?|Results?|Discussion|Conclusion|References|"
    r"Acknowledgi?e?ments?|Appendix|Limitations?|Future Work)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


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


def split_by_section(docs: list[Document]) -> list[Document]:
    """
    Group pages by source document, concatenate text, split on detected
    section headers, and tag each resulting chunk with its section name.
    """
    # Group pages by source file (PyPDFLoader gives one Document per page)
    by_source: dict[str, list[Document]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        by_source.setdefault(source, []).append(doc)

    section_chunks = []

    for source, pages in by_source.items():
        full_text = "\n\n".join(p.page_content for p in pages)

        # Find all header matches and their positions
        matches = list(SECTION_HEADER_PATTERN.finditer(full_text))

        if not matches:
            # No detected structure — fall back to treating whole doc as one section
            section_chunks.append(
                Document(
                    page_content=full_text,
                    metadata={"source": source, "section": "unknown"},
                )
            )
            continue

        for i, match in enumerate(matches):
            section_name = match.group(3)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            section_text = full_text[start:end].strip()

            if section_text:
                section_chunks.append(
                    Document(
                        page_content=section_text,
                        metadata={"source": source, "section": section_name},
                    )
                )

    log.info(
        f"Split into {len(section_chunks)} sections across {len(by_source)} documents"
    )
    return section_chunks


def chunk_within_sections(section_docs: list[Document]) -> list[Document]:
    """
    Apply recursive splitting WITHIN each section if it's still too
    large for one chunk — preserves section metadata on every sub-chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 256)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
        separators=["\n\n", "\n", " ", ""],
    )
    final_chunks = splitter.split_documents(section_docs)
    log.info(f"After within-section splitting: {len(final_chunks)} final chunks")
    return final_chunks


def ingest_structured(data_dir: str = "data/papers"):
    """Full structure-aware ingestion pipeline."""
    docs = load_documents(data_dir)
    section_docs = split_by_section(docs)
    chunks = chunk_within_sections(section_docs)

    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
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
        f"Ingested {len(chunks)} structured chunks into Qdrant collection '{COLLECTION_NAME}'"
    )
    return vectorstore


if __name__ == "__main__":
    ingest_structured()
