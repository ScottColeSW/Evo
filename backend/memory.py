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

    Found and fixed 2026-09-12: every memory in this game is phrased from a small
    set of narrative templates ("Scouts confirmed X at (x,y)", "X near (x,y) is
    known dangerous ground") that all share common connector words -- "at", "near",
    "toward". A live call site (Simulation._prepare_turn's `tribe.memory.recall(
    f"{biome} at {tribe.x},{tribe.y}")`) proved this concretely: querying a location
    with zero real relation to any stored memory still matched one, purely because
    both phrases contained the word "at". Jaccard overlap on raw tokens can't tell a
    genuine topical match from two sentences sharing only grammatical scaffolding --
    worse than the hash-seeded bug this class already replaced once, since it looks
    like a real match instead of obviously not one. _STOPWORDS strips exactly that
    scaffolding before scoring, so overlap only ever counts substantive words (biome
    names, event nouns, and real coordinates, which still tokenize and match
    correctly -- only the connectors around them are filtered)."""

    _WORD_RE = re.compile(r"[a-z0-9]+")
    _STOPWORDS = frozenset({
        "a", "an", "the", "at", "in", "on", "of", "to", "is", "are", "was", "were",
        "and", "or", "near", "toward", "off", "our", "there", "one", "for", "with",
        "it", "this", "that",
    })

    # Explicit request, 2026-09-18: "give him the signal" for a reflection that
    # keeps recurring, not just a one-off. Same Jaccard-overlap scoring recall()
    # already uses (cruder than real embeddings, but it correlates with real
    # topical similarity -- see this class's own docstring), just checked at
    # store time instead of query time: a new kind="reflection" text that
    # substantially overlaps an existing one is the SAME recurring thought
    # showing up again, not a new, unrelated one. First-cut threshold, no real
    # run data behind it yet (this mechanic is brand new) -- worth revisiting
    # once real reflection text exists to check it against.
    REFLECTION_REINFORCEMENT_OVERLAP_THRESHOLD = 0.3

    def __init__(self, tribe_id: str, max_episodes: int = 40):
        self.tribe_id = tribe_id
        self.max_episodes = max_episodes
        self.entries: list[dict] = []
        # {"text", "weight", "cycle"} per entry -- weight/cycle carried forward from
        # the original remembered episode (see consolidate()) specifically so
        # top_taboos() can rank by real importance, not just insertion order. Plain
        # strings before 2026-09-13; see that method's own docstring for why.
        self.taboos: list[dict] = []

    def _tokenize(self, text: str) -> set[str]:
        return set(self._WORD_RE.findall(text.lower())) - self._STOPWORDS

    def remember(self, text: str, cycle: int, weight: float = 0.5, kind: str = "episode") -> dict:
        # `kind` (added 2026-09-18): defaults to "episode" so every existing call
        # site -- geographic discoveries, hazard warnings -- is untouched. A chief's
        # own private night-cycle reflection (Simulation._run_night_cycle) is stored
        # here too, tagged kind="reflection", so it rides the same weighted,
        # self-pruning store instead of a second structure -- see consolidate()'s
        # own comment for why reflections are excluded from ever becoming a taboo.
        #
        # Reflections alone also check for reinforcement first (below) instead of
        # always appending fresh -- a recurring conviction (the same private
        # thought, worded differently or the same, coming back night after
        # night) is real signal a one-off musing isn't, and Simulation.
        # _run_night_cycle uses the returned entry's "reinforced" count to decide
        # whether it's earned real standing-decree weight. Restricted to
        # kind="reflection" only -- episode entries (hazards/discoveries) are
        # each tied to their own real coordinates and were never meant to merge.
        tokens = self._tokenize(text)
        if kind == "reflection":
            for entry in self.entries:
                if entry.get("kind") != "reflection" or not entry["tokens"] or not tokens:
                    continue
                overlap = len(tokens & entry["tokens"]) / len(tokens | entry["tokens"])
                if overlap >= self.REFLECTION_REINFORCEMENT_OVERLAP_THRESHOLD:
                    entry["reinforced"] = entry.get("reinforced", 0) + 1
                    entry["cycle"] = cycle
                    entry["weight"] = max(entry["weight"], weight)
                    entry["ts"] = time.time()
                    return entry

        entry = {
            "text": text,
            "tokens": tokens,
            "cycle": cycle,
            "weight": weight,
            "ts": time.time(),
            "kind": kind,
            "reinforced": 0,
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_episodes * 2:
            self.consolidate()
        return entry

    def recall(self, query: str, top_k: int = 2, kind: str | None = None) -> list[dict]:
        """Returns up to `top_k` past entries that actually share vocabulary with
        `query`, ranked by Jaccard overlap. Entries with zero shared tokens are
        excluded rather than padded in -- no match is a more honest answer than a
        random one. `kind`, when given, restricts the search to that one category
        (e.g. "reflection") so a geographic/hazard memory and a private thought can
        never surface in each other's place just because they happen to share a
        few words."""
        if not self.entries:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for entry in self.entries:
            if kind is not None and entry.get("kind", "episode") != kind:
                continue
            tokens = entry["tokens"]
            if not tokens:
                continue
            overlap = len(query_tokens & tokens) / len(query_tokens | tokens)
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def consolidate(self) -> None:
        """Distills high-weight memories into permanent taboos, then trims the log.

        Restricted to kind="episode" (added 2026-09-18): a taboo is framed
        elsewhere (_build_visible_entities: "taboo: ...") as a real danger to
        avoid -- a chief's own private reflection is self-knowledge, not a
        hazard, and would read as a non-sequitur next to "the volcano near
        (x,y) is deadly." Reflections still get pruned by the plain entries
        trim below same as everything else; they just never graduate into this
        specific, danger-framed permanent slot."""
        candidates = [e for e in self.entries if e.get("kind", "episode") == "episode"]
        ranked = sorted(candidates, key=lambda e: e["weight"], reverse=True)
        known_texts = {t["text"] for t in self.taboos}
        for e in ranked[:3]:
            if e["weight"] >= 0.75 and e["text"] not in known_texts:
                self.taboos.append({"text": e["text"], "weight": e["weight"], "cycle": e["cycle"]})
                known_texts.add(e["text"])
        self.entries = self.entries[-self.max_episodes:]

    def top_taboos(self, n: int = 3) -> list[str]:
        """The n most important taboos, not the n most recently learned.

        Found while grounding a live "learning about this over and over seems
        hindering" report, 2026-09-13: the caller (Simulation._prepare_turn) used to
        slice tribe.memory.taboos[-3:] -- most-recently-added, since consolidate()
        only ever appends. That itself replaced an even older "first 3 ever" bug
        (see this method's own comment history in git blame), but recency isn't
        importance either: a genuinely critical early lesson (a volcano that's
        killed twice) could silently lose its permanent slot to three newer, lower-
        stakes taboos, and never surface again. Ranked by weight first (a real,
        already-tracked severity signal from remember()'s own callers), cycle as
        the tiebreak among equally-weighted ones -- so recency still wins between
        two equally important facts, but no longer overrides importance itself."""
        ranked = sorted(self.taboos, key=lambda t: (t["weight"], t["cycle"]), reverse=True)
        return [t["text"] for t in ranked[:n]]
