"""2026-10-03: RESEARCH is offered, and counts toward the era discount, only when it files something the Library does not hold."""
from backend import config
from backend.actions import ACTION_REGISTRY, research_candidates
from backend.simulation import AFFORDABILITY_CHECKS, Tribe
from tests.test_actions import _NO_TARGET, _bare_simulation


def _tribe():
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.library_built = True
    tribe.wood = 100
    return tribe


def _offered(tribe):
    return AFFORDABILITY_CHECKS["RESEARCH"](tribe, None)


def test_not_offered_with_nothing_to_file():
    assert not _offered(_tribe())


def test_a_repeat_is_not_offered_and_does_not_count():
    sim, tribe = _bare_simulation(), _tribe()
    tribe.memory.remember("a wolf attacked the herd near the river ford, 30 deer lost", cycle=1, weight=0.9)
    assert _offered(tribe)
    ACTION_REGISTRY["RESEARCH"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.research_completed == 1
    tribe.memory.remember("a wolf attacked the herd near the river ford, 66 deer lost", cycle=2, weight=0.95)
    assert not _offered(tribe)
    wood = tribe.wood
    result = ACTION_REGISTRY["RESEARCH"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.research_completed == 1 and tribe.wood == wood and "nothing new" in result


def test_something_genuinely_new_is_offered_again():
    sim, tribe = _bare_simulation(), _tribe()
    tribe.memory.remember("a wolf attacked near the river", cycle=1, weight=0.9)
    ACTION_REGISTRY["RESEARCH"](sim, tribe, "plains", _NO_TARGET)
    tribe.memory.remember("the northern wall section was finished before the raiders came", cycle=2, weight=0.8)
    assert research_candidates(tribe) == ["the northern wall section was finished before the raiders came"]
    assert _offered(tribe)


def test_the_librarys_own_entries_are_never_refiled():
    tribe = _tribe()
    tribe.memory.remember('chose RESEARCH. the library records a new insight: "a wolf attacked"', cycle=3, weight=0.9)
    assert research_candidates(tribe) == []


def test_routine_action_logs_are_not_candidates():
    tribe = _tribe()
    tribe.memory.remember("At (30,64) in lake, chose GATHER_STONE. 30 stone gathered.", cycle=1, weight=0.95)
    assert research_candidates(tribe) == [] and not _offered(tribe)


def test_reflections_and_evidence_are_candidates_with_their_source():
    from backend.actions import research_candidates_with_source
    tribe = _tribe()
    tribe.memory.remember("the river clan can be trusted in lean years", cycle=5, weight=0.5, kind="reflection")
    tribe.decision_journal.append({"cycle": 40, "action": "BUILD_WALL", "delta": {"wood": -30}, "built": {"wall_built": [0, 1]},
                                   "moved": False, "stock_before": {}, "population": 100, "note": None})
    tribe.peace_gate["cost_events"].append({"cycle": 50, "kind": "war", "cause": "raid_losses", "amount": 20, "population": 200})
    picks = research_candidates_with_source(tribe)
    assert [c["source"] for c in picks] == ["belief", "evidence", "evidence"]
    assert any("changed: wall" in c["text"] for c in picks) and any("cost 20 of 200 people" in c["text"] for c in picks)


def test_a_routine_journal_entry_is_not_evidence():
    tribe = _tribe()
    tribe.decision_journal.append({"cycle": 41, "action": "GATHER_WOOD", "delta": {"wood": 20}, "built": {}, "moved": False,
                                   "stock_before": {}, "population": 100, "note": None})
    assert research_candidates(tribe) == []
