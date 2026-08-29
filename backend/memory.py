import re
import time


class TribeMemory:
    """A lightweight episodic memory store per tribe.

    `recall` previously scored relevance with a hash-seeded pseudo-random vector per
    text -- since a cryptographic hash is deliberately decorrelated from meaning, two
    memories about the exact same topic ("wolf attack in the forest" vs "forest hunting
    danger") got essentially random similarity scores. It looked like semantic search
    but was closer to retrieving noise. Replaced with token-overlap (Jaccard) scoring:
    cruder than real embeddings, but it actually correlates with what the text is
    about, which the hash version never did. A real upgrade path is Ollama's
    /api/embeddings with a model like nomic-embed-text, but that's an async network
    call and today's remember()/recall() call sites are synchronous -- left as a
    follow-up, not bundled into this fix.
    """

    _WORD_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, tribe_id: str, max_episodes: int = 40):
        self.tribe_id = tribe_id
        self.max_episodes = max_episodes
        self.entries: list[dict] = []
        self.taboos: list[str] = []

    def _tokenize(self, text: str) -> set[str]:
        return set(self._WORD_RE.findall(text.lower()))

    def remember(self, text: str, cycle: int, weight: float = 0.5) -> None:
        self.entries.append({
            "text": text,
            "tokens": self._tokenize(text),
            "cycle": cycle,
            "weight": weight,
            "ts": time.time(),
        })
        if len(self.entries) > self.max_episodes * 2:
            self.consolidate()

    def recall(self, query: str, top_k: int = 2) -> list[dict]:
        """Returns up to `top_k` past entries that actually share vocabulary with
        `query`, ranked by Jaccard overlap. Entries with zero shared tokens are
        excluded rather than padded in -- no match is a more honest answer than a
        random one."""
        if not self.entries:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for entry in self.entries:
            tokens = entry["tokens"]
            if not tokens:
                continue
            overlap = len(query_tokens & tokens) / len(query_tokens | tokens)
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def consolidate(self) -> None:
        """Distills high-weight memories into permanent taboos, then trims the log."""
        ranked = sorted(self.entries, key=lambda e: e["weight"], reverse=True)
        for e in ranked[:3]:
            if e["weight"] >= 0.75 and e["text"] not in self.taboos:
                self.taboos.append(e["text"])
        self.entries = self.entries[-self.max_episodes:]
