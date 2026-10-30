"""The search index: ChromaDB holds the chunks and their embeddings and finds the closest ones to a question."""
import uuid
from pathlib import Path
from typing import List, Optional

import chromadb

from rag.chunking import Chunk
from rag.embedder import Embedder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "chroma"
COLLECTION = "shop_docs"


class Index:
    def __init__(self, embedder: Embedder, path: Optional[Path] = None, collection: str = COLLECTION):
        """path=None keeps everything in memory (used for tests and for the evaluation's extra test corpus)."""
        self.embedder = embedder
        if path is None:
            self.client = chromadb.EphemeralClient()
            collection = f"{collection}_{uuid.uuid4().hex[:8]}"   # in-memory clients share state, so keep names unique
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path))
        self.name = collection
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def build(self, chunks: List[Chunk]) -> int:
        """Replaces the collection's content with these chunks."""
        existing = self.collection.get()["ids"]
        if existing:
            self.collection.delete(ids=existing)
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embedder([c.text for c in chunks]),
            metadatas=[{"source": c.source, "title": c.title, "section": c.section} for c in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def search(self, question: str, k: int = 4) -> List[dict]:
        """The k closest chunks, best first. similarity is cosine similarity: 1.0 = same direction, 0 = unrelated."""
        if self.count() == 0:
            return []
        found = self.collection.query(query_embeddings=self.embedder([question]), n_results=min(k, self.count()))
        results = []
        for cid, text, meta, dist in zip(found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]):
            results.append({"id": cid, "source": meta["source"], "section": meta["section"], "text": text,
                            "similarity": round(1.0 - float(dist), 4)})
        return results
