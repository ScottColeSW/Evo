"""2026-10-03: a tribe must live config.ERA_MIN_CYCLES cycles in an era before the next opens (the owner's guide: no less than 50
cycles in any era). Population jumps and a fully ready tribe cannot skip it."""
from backend import config
from backend.eras import ERAS
from backend.simulation import Simulation
from tests.era_helpers import make_ready_for_every_era


def _rich_tribe(era="cognitive_horizon"):
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.era = era
    tribe.population = 10_000_000
    tribe.water = tribe.stone = tribe.wood = tribe.food = 10_000
    tribe.unique_resources["Fur"] = 10_000
    make_ready_for_every_era(tribe)
    return sim, tribe


def test_a_tribe_that_clears_every_threshold_still_waits_out_the_floor():
    sim, tribe = _rich_tribe()
    sim.cycle = config.ERA_MIN_CYCLES - 1
    sim._advance_era_if_ready(tribe)
    assert tribe.era == "cognitive_horizon"
    sim.cycle = config.ERA_MIN_CYCLES
    sim._advance_era_if_ready(tribe)
    assert tribe.era == "tribal_synapse" and tribe.era_entered_cycle == config.ERA_MIN_CYCLES


def test_the_floor_restarts_in_each_new_era_so_none_is_a_one_cycle_pass_through():
    sim, tribe = _rich_tribe()
    sim.cycle = config.ERA_MIN_CYCLES
    for _ in range(5):
        sim._advance_era_if_ready(tribe)                     # only one step per call, and only once the floor has passed
    assert tribe.era == "tribal_synapse"
    sim.cycle = config.ERA_MIN_CYCLES + config.ERA_MIN_CYCLES - 1
    sim._advance_era_if_ready(tribe)
    assert tribe.era == "tribal_synapse"
    sim.cycle = 2 * config.ERA_MIN_CYCLES
    sim._advance_era_if_ready(tribe)
    assert tribe.era == "monolithic_era"


def test_the_population_lines_were_widened_and_still_rise_with_each_era():
    lines = [e.requires_population for e in ERAS]
    assert lines == sorted(lines) and lines[-1] == 20000 and lines[1] == 60
