"""Phase 1 of docs/PALIMPSEST-REFLECTIONS-DESIGN.md: an optional judge decides what a new reflection is to the held ones.
The judge here is a fake with a simple rule, so these tests check Evo's wiring (storage, conflicts, promotion, fallback), not
the quality of Palimpsest's real judge."""
from unittest import mock

from backend import config
from backend.memory import TribeMemory
from backend.simulation import Simulation
from tests.conftest import run_async

TRUST = "We should trust the river clan."
NEVER = "We should never trust the river clan."


def fake_judge(new_text, held):
    for h in held:
        if ("never" in new_text) != ("never" in h["text"]):
            return {"relation": "collides", "related_id": h["id"], "reason": "opposite convictions"}
        return {"relation": "reinforces", "related_id": h["id"], "reason": "same conviction"}
    return {"relation": "new", "related_id": None, "reason": "nothing held"}


def test_a_contradiction_is_stored_on_its_own_and_linked_not_merged():
    m = TribeMemory("t")
    m.judge = fake_judge
    first = m.remember_reflection(TRUST, 1, 0.5, None)
    second = m.remember_reflection(NEVER, 2, 0.5, None)
    assert second is not first and len(m.entries) == 2
    assert first["reinforced"] == 0 and second["reinforced"] == 0
    assert m.open_conflicts(first) == [second] and m.open_conflicts(second) == [first]
    trace = m.last_reflection_trace
    assert trace["method"] == "judge" and trace["relation"] == "collides" and trace["reinforced"] is False


def test_a_restatement_still_reinforces_and_a_failing_judge_falls_back_to_the_rules():
    m = TribeMemory("t")
    m.judge = fake_judge
    first = m.remember_reflection(TRUST, 1, 0.5, None)
    assert m.remember_reflection("We ought to trust the river clan.", 2, 0.5, None) is first and first["reinforced"] == 1

    broken = TribeMemory("t")
    broken.judge = mock.Mock(side_effect=RuntimeError("judge down"))
    a = broken.remember_reflection(TRUST, 1, 0.5, None)
    b = broken.remember_reflection(NEVER, 2, 0.5, None)
    assert b is a                                   # exactly today's behavior: the reversal is merged by the token rule
    assert broken.open_conflicts(a) == []


def test_with_no_judge_nothing_changes_and_no_conflict_keys_appear():
    m = TribeMemory("t")
    first = m.remember_reflection(TRUST, 1, 0.5, None)
    assert m.remember_reflection(NEVER, 2, 0.5, None) is first
    assert "conflicts_with" not in first and m.open_conflicts(first) == []


async def _run_nights(sim, tribe, thoughts):
    queue = iter(thoughts)

    async def fake_reflect(*args, **kwargs):
        return {"private_thoughts": next(queue), "revised_philosophy": args[3], "changed": False, "reasoning": "ok"}

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect), \
         mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        for _ in thoughts:
            await sim._run_night_cycle(tribe)


@run_async
async def test_without_the_judge_a_flip_flop_is_promoted_to_a_decree_as_it_is_today():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    await _run_nights(sim, tribe, [TRUST, NEVER, TRUST])
    assert tribe.chief_decree == TRUST          # the flaw this work exists to remove


@run_async
async def test_with_the_judge_a_belief_in_open_disagreement_is_not_promoted():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    sim._reflection_judge_cache = fake_judge
    await _run_nights(sim, tribe, [TRUST, NEVER, TRUST, TRUST, TRUST])
    assert tribe.chief_decree == ""             # reinforced enough, but still in open conflict with the contrary reflection
    assert max(e["reinforced"] for e in tribe.memory.entries) >= config.REFLECTION_STABILIZED_REINFORCEMENT_COUNT


@run_async
async def test_with_the_judge_an_uncontested_belief_is_still_promoted():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    sim._reflection_judge_cache = fake_judge
    await _run_nights(sim, tribe, [TRUST, TRUST, TRUST])
    assert tribe.chief_decree == TRUST


@run_async
async def test_the_judge_setting_is_off_by_default_and_builds_nothing(monkeypatch):
    monkeypatch.delenv("REFLECTION_JUDGE", raising=False)
    assert config.REFLECTION_JUDGE == "off"
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    assert await sim._reflection_judge() is None


@run_async
async def test_the_run_reports_whether_the_judge_is_on_off_or_unavailable():
    import backend.reflection_judge as rj

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}], reflection_judge="nli")
    with mock.patch.object(rj, "build_judge", return_value=None):
        assert await sim._reflection_judge() is None
    assert sim.snapshot()["reflection_judge"] == "unavailable"     # asked for, but not installed: say so, do not hide it

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}], reflection_judge="nli")
    with mock.patch.object(rj, "build_judge", return_value=fake_judge):
        assert await sim._reflection_judge() is fake_judge
    assert sim.snapshot()["reflection_judge"] == "on"

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}], reflection_judge="off")
    assert await sim._reflection_judge() is None
    assert sim.snapshot()["reflection_judge"] == "off"
