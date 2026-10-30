"""Build the search index from the documents in rag/documents.   python -m rag.ingest"""
import time

from rag.chunking import load_chunks
from rag.embedder import SentenceTransformerEmbedder
from rag.store import DEFAULT_PATH, Index
from pathlib import Path

DOCUMENTS = Path(__file__).with_name("documents")


def main() -> None:
    chunks = load_chunks(DOCUMENTS)
    print(f"{len(chunks)} chunks from {len(list(DOCUMENTS.glob('*.md')))} documents. Loading the embedding model "
          "(about 90 MB download the first time)...")
    start = time.time()
    count = Index(SentenceTransformerEmbedder(), DEFAULT_PATH).build(chunks)
    print(f"Indexed {count} chunks in {time.time() - start:.0f}s into {DEFAULT_PATH}")


if __name__ == "__main__":
    main()
