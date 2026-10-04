"""2026-10-03 (the owner's request): overcrowding is culled once, at the start of the night, as one event the Chief reflects on,
not a little every cycle."""
from unittest import mock

from backend import config
from backend.simulation import Simulation, _sustainable_population
from tests.conftest import run_async


def _overcrowded(excess=500):
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    sim.cycle = 60
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    target = round(_sustainable_population(tribe) * config.POPULATION_CARRYING_CAPACITY_TARGET_FRACTION)
    tribe.population = target + excess
    return sim, tribe, target


def test_the_night_cull_brings_the_tribe_back_to_the_line_in_one_blow():
    sim, tribe, target = _overcrowded(500)
    sim._advance_population_pressure(tribe, night=True)
    assert tribe.population == target
    # The cull may also kill the Chief (a separate line), so look through the history rather than at its last entry.
    assert any(h.startswith("as the night begins, the population is culled back to what the land can support -- 500 lost") for h in tribe.history)


def test_the_cull_lands_in_what_the_chief_reads_at_night():
    sim, tribe, target = _overcrowded(300)
    seen = {}

    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, *a, **k):
        seen["events"] = list(recent_events)
        return {"private_thoughts": "", "revised_philosophy": current_philosophy, "changed": False, "reasoning": "ok"}

    @run_async
    async def go():
        with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
            await sim._run_night_cycle(tribe)
    go()
    assert tribe.population == target
    assert any("culled back to what the land can support -- 300 lost" in e for e in seen["events"])


def test_the_step_loop_no_longer_trims_a_little_each_cycle():
    import inspect
    from backend import simulation
    source = inspect.getsource(simulation.Simulation.step)
    assert "_advance_population_pressure" not in source
