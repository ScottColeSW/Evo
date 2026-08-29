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


def test_consolidate_distills_high_weight_memories_into_taboos():
    memory = TribeMemory("tribe_0")
    memory.remember("a catastrophic flood destroyed the settlement", cycle=1, weight=0.9)
    memory.remember("gathered a little wood", cycle=2, weight=0.2)

    memory.consolidate()

    assert "a catastrophic flood destroyed the settlement" in memory.taboos
    assert "gathered a little wood" not in memory.taboos
