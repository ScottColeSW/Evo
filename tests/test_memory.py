from backend.memory import TribeMemory


def test_recall_returns_nothing_when_memory_is_empty():
    memory = TribeMemory("tribe_0")
    assert memory.recall("forest wolf attack") == []


def test_recall_prefers_entries_that_share_vocabulary_with_the_query():
    """Regression test: the old hash-seeded pseudo-embedding scored relevance with a
    vector deliberately decorrelated from meaning, so two memories about the same
    topic got essentially random similarity. This checks recall actually surfaces the
    memory that shares words with the query over one that shares none."""
    memory = TribeMemory("tribe_0")
    memory.remember("A wolf pack struck the hunting party in the forest", cycle=1, weight=0.5)
    memory.remember("Gathered stone at the mountain quarry", cycle=2, weight=0.5)

    results = memory.recall("forest wolf danger while hunting")

    assert len(results) == 1
    assert "wolf" in results[0]["text"].lower()


def test_recall_excludes_entries_with_zero_word_overlap():
    memory = TribeMemory("tribe_0")
    memory.remember("Gathered stone at the mountain quarry", cycle=1, weight=0.5)

    assert memory.recall("river fish drowning hazard") == []


def test_recall_ranks_stronger_overlap_above_weaker_overlap():
    memory = TribeMemory("tribe_0")
    memory.remember("forest danger wolf hunting party struck", cycle=1, weight=0.5)
    memory.remember("forest quiet nothing happened today", cycle=2, weight=0.5)

    results = memory.recall("forest danger wolf struck")

    assert results[0]["text"].startswith("forest danger wolf")


def test_recall_does_not_match_on_shared_connector_words_alone():
    """Regression, 2026-09-12: every memory in this game is phrased from a small
    set of templates ("X at (x,y)", "X near (x,y)") that all share words like "at"/
    "near" -- Jaccard overlap on raw tokens used to score a nonzero, spurious match
    between two sentences that share nothing but that grammatical scaffolding.
    Reproduced live via the real Simulation._prepare_turn query shape
    (f"{biome} at {x},{y}") before this fix: a location with zero real relation to
    the stored memory still matched it, purely on the shared word "at"."""
    memory = TribeMemory("tribe_0")
    memory.remember("Scouts confirmed fresh water at (52,58).", cycle=10, weight=0.9)

    assert memory.recall("mountains at 10,10") == []


def test_recall_still_matches_on_a_shared_real_coordinate():
    """The fix must not throw out genuine signal along with the noise -- an exact
    coordinate match is real, substantive overlap, not grammatical scaffolding."""
    memory = TribeMemory("tribe_0")
    memory.remember("Scouts confirmed fresh water at (52,58).", cycle=10, weight=0.9)

    results = memory.recall("plains at 52,58")

    assert len(results) == 1
    assert "52,58" in results[0]["text"] or "(52,58)" in results[0]["text"]


def test_consolidate_distills_high_weight_memories_into_taboos():
    memory = TribeMemory("tribe_0")
    memory.remember("a catastrophic flood destroyed the settlement", cycle=1, weight=0.9)
    memory.remember("gathered a little wood", cycle=2, weight=0.2)

    memory.consolidate()

    texts = [t["text"] for t in memory.taboos]
    assert "a catastrophic flood destroyed the settlement" in texts
    assert "gathered a little wood" not in texts


def test_consolidate_does_not_duplicate_an_already_known_taboo():
    memory = TribeMemory("tribe_0")
    memory.remember("a catastrophic flood destroyed the settlement", cycle=1, weight=0.9)
    memory.consolidate()
    memory.remember("a catastrophic flood destroyed the settlement", cycle=50, weight=0.9)

    memory.consolidate()

    texts = [t["text"] for t in memory.taboos]
    assert texts.count("a catastrophic flood destroyed the settlement") == 1


def test_top_taboos_ranks_by_weight_not_recency():
    """Explicit fix, 2026-09-13: a genuinely critical early lesson must not lose
    its slot to newer but lower-stakes ones -- weight is the real severity signal
    remember()'s own callers already provide, recency is only the tiebreak."""
    memory = TribeMemory("tribe_0")
    memory.taboos = [
        {"text": "an old but critical volcano warning", "weight": 0.9, "cycle": 5},
        {"text": "a newer, lower-stakes warning", "weight": 0.75, "cycle": 500},
        {"text": "an even newer, lower-stakes warning", "weight": 0.75, "cycle": 600},
        {"text": "the newest warning of all", "weight": 0.75, "cycle": 700},
    ]

    top = memory.top_taboos(3)

    assert top[0] == "an old but critical volcano warning"  # wins on weight despite being oldest by far
    assert "a newer, lower-stakes warning" not in top  # the lowest-cycle of the tied-weight group loses the last slot


def test_top_taboos_breaks_a_weight_tie_by_recency():
    memory = TribeMemory("tribe_0")
    memory.taboos = [
        {"text": "older, same weight", "weight": 0.8, "cycle": 5},
        {"text": "newer, same weight", "weight": 0.8, "cycle": 500},
    ]

    top = memory.top_taboos(2)

    assert top[0] == "newer, same weight"
