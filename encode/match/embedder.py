import hashlib
import math
import re


class HashEmbedder:
    """Cheap, dependency-free embedding using deterministic hashing."""

    def __init__(self, dims: int = 256, seed: str = "hippo"):
        self.dims = dims
        self._seed = seed.encode("utf-8")

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in self._tokenize(text):
            idx = self._hash_index(token)
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

    def _hash_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=self._seed)
        value = int.from_bytes(digest.digest(), "big", signed=False)
        return value % self.dims
