"""Turns text into numbers that capture meaning, so similar texts end up close together.

An 'embedder' here is just a function: list of texts in, list of vectors out. The real one runs locally on the CPU
(no API calls, no quota). HashingEmbedder is a crude word-overlap stand-in used by the tests.
"""
import hashlib
import math
import re
from typing import Callable, List

Embedder = Callable[[List[str]], List[List[float]]]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # downloads about 90 MB the first time

        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False).tolist()


class HashingEmbedder:
    """Bag-of-words vectors from hashed words. Only for tests and offline checks: it matches words, not meaning."""

    STOP = {"the", "a", "an", "of", "to", "is", "are", "and", "or", "in", "on", "for", "do", "i", "you", "we", "it",
            "my", "me", "can", "how", "what", "be", "if", "at", "by", "with", "your", "our", "that", "this", "as"}

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9£]+", text.lower()):
            if word in self.STOP:
                continue
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]
