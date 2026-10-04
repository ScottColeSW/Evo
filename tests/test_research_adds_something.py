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
    tribe.memory.remember("At (30,64) in lake, chose GATHER_STONE. 30 stone gathered.", cycle=1, weight=0.9)
    assert _offered(tribe)
    ACTION_REGISTRY["RESEARCH"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.research_completed == 1
    tribe.memory.remember("At (30,64) in lake, chose GATHER_STONE. 66 stone gathered.", cycle=2, weight=0.95)
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
