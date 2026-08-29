import hashlib
import time

import numpy as np


class TribeMemory:
    """A lightweight episodic memory store per tribe.

    Uses a deterministic hash-based pseudo-embedding so it works fully offline with
    no extra model pulls. If you want real semantic recall, swap `_embed` to call
    Ollama's /api/embeddings with a model like nomic-embed-text.
    """

    def __init__(self, tribe_id: str, dim: int = 256, max_episodes: int = 40):
        self.tribe_id = tribe_id
        self.dim = dim
        self.max_episodes = max_episodes
        self.vectors: list[np.ndarray] = []
        self.entries: list[dict] = []
        self.taboos: list[str] = []

    def _embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def remember(self, text: str, cycle: int, weight: float = 0.5) -> None:
        self.vectors.append(self._embed(text))
        self.entries.append({"text": text, "cycle": cycle, "weight": weight, "ts": time.time()})
        if len(self.vectors) > self.max_episodes * 2:
            self.consolidate()

    def recall(self, query: str, top_k: int = 2) -> list[dict]:
        if not self.vectors:
            return []
        q = self._embed(query)
        sims = np.array([float(np.dot(q, v)) for v in self.vectors])
        idx = np.argsort(sims)[-top_k:][::-1]
        return [self.entries[i] for i in idx]

    def consolidate(self) -> None:
        """Distills high-weight memories into permanent taboos, then trims the log."""
        ranked = sorted(self.entries, key=lambda e: e["weight"], reverse=True)
        for e in ranked[:3]:
            if e["weight"] >= 0.75 and e["text"] not in self.taboos:
                self.taboos.append(e["text"])
        self.vectors = self.vectors[-self.max_episodes:]
        self.entries = self.entries[-self.max_episodes:]
