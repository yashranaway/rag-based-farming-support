from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Protocol


class Embeddings(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover (interface)
        ...


@dataclass
class EmbeddingResult:
    vector: List[float]


class SimpleTokenizerEmbeddings:
    """
    Very lightweight embedding: bag-of-words hash into fixed-size vectors.
    - Not production-ready. Replace with Granite embeddings later.
    """

    def __init__(self, dim: int = 256, seed: int = 1337) -> None:
        self.dim = dim
        self.seed = seed

    def _hash(self, token: str) -> int:
        # Simple deterministic hash bounded by dim
        h = 2166136261
        for ch in token:
            h ^= ord(ch)
            h *= 16777619
            h &= 0xFFFFFFFF
        return (h ^ self.seed) % self.dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in t.lower().split():
                idx = self._hash(tok)
                vec[idx] += 1.0
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
