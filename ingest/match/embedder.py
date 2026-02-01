import math
import re


class HashEmbedder:
    """Cheap, dependency-free embedding using hashing trick."""

    def __init__(self, dims: int = 256):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in self._tokenize(text):
            idx = hash(token) % self.dims
            vec[idx] += 1.0
        return self._normalize(vec)

    def similarity(self, a: list[float], b: list[float]) -> float:
        return self._cosine(a, b)

    def _normalize(self, vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())
