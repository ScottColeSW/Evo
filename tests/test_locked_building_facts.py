"""2026-10-07: a building the era has unlocked but the situation does not yet allow is explained as a plain fact (what is missing, what the tribe has), not a nudge.
From a live run: Flinx held another tribe's ore, wanted a forge for ~180 cycles, and nothing said the forge needs ore from its own mine."""
from backend import config
from backend.simulation import Simulation, locked_building_facts
from backend.world import Landscape


def _tribe():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}, {"name": "B", "model": "gemma2:2b", "x": 70, "y": 70}])
    t = sim.tribes["tribe_0"]
    t.has_ever_settled = True
    sim._found_territory(t)
    t.wood = t.stone = 500
    return sim, t


UNLOCKED = ("BUILD_FORGE", "BUILD_MINE")


def test_a_forge_without_ore_from_its_own_mine_says_so_and_names_the_ore_it_has_that_does_not_count():
    sim, t = _tribe()
    t.mine_built, t.mine_resource_name = True, "Whisperwood Amber"
    t.unique_resources = {"Orosite Ore": 251}  # looted from a rival: not this mine's
    facts = locked_building_facts(t, sim.world, UNLOCKED)
    assert len(facts) == 1 and "Whisperwood Amber" in facts[0] and "you have 0" in facts[0] and "has not been worked yet" in facts[0]


def test_nothing_is_said_once_the_forge_is_possible_or_built():
    sim, t = _tribe()
    t.mine_built, t.mine_resource_name = True, "Whisperwood Amber"
    t.unique_resources = {"Whisperwood Amber": config.FORGE_ITEM_ORE_COST}
    assert locked_building_facts(t, sim.world, UNLOCKED) == []
    t.unique_resources = {}
    t.forge_built = True
    assert locked_building_facts(t, sim.world, UNLOCKED) == []


def test_a_mine_without_a_known_vein_or_materials_lists_each_missing_piece():
    sim, t = _tribe()
    t.wood = 0
    facts = locked_building_facts(t, sim.world, UNLOCKED)
    assert len(facts) == 1 and "a known vein" in facts[0] and f"{config.MINE_WOOD_COST} wood (you have 0)" in facts[0]


def test_nothing_is_said_for_a_building_the_era_has_not_unlocked():
    sim, t = _tribe()
    t.mine_built, t.mine_resource_name = True, "Whisperwood Amber"
    assert locked_building_facts(t, sim.world, ()) == []


def test_the_fact_reaches_the_prompt_even_with_nudges_off(monkeypatch):
    monkeypatch.setenv("NUDGES", "off")
    sim, t = _tribe()
    t.era = "monolithic_era"
    t.mine_built, t.mine_resource_name = True, "Whisperwood Amber"
    request, _ctx = sim._prepare_turn(t)
    assert "A forge cannot be built yet" in request["prompt"]
