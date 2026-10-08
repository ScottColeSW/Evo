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


CHAIN = ("BUILD_KEEP", "BUILD_BARRACKS", "BUILD_LIBRARY", "BUILD_FORTRESS", "BUILD_CASTLE")


def test_a_keep_the_next_era_needs_says_what_it_lacks_and_what_it_would_open():
    """2026-10-08: a tribe was told "still short on: a barracks, a keep" for 340 turns and nothing said what a keep needs."""
    sim, t = _tribe()
    t.era = "tribal_synapse"  # the next era asks for a barracks, a keep and a library
    t.long_houses_built, t.wood, t.stone = 1, 12, 500
    facts = locked_building_facts(t, sim.world, CHAIN)
    keep = next(f for f in facts if f.startswith("A keep"))
    assert f"{config.KEEP_LONG_HOUSES_REQUIRED} long houses, built or upgraded (you have 1)" in keep
    assert f"{config.KEEP_WOOD_COST} wood (you have 12)" in keep and "Once built it opens BUILD_BARRACKS and BUILD_FORTRESS." in keep


def test_a_barracks_names_the_kitchen_as_well_as_the_keep():
    sim, t = _tribe()
    t.era = "tribal_synapse"
    t.keep_built, t.kitchen_built = True, False
    barracks = next(f for f in locked_building_facts(t, sim.world, CHAIN) if f.startswith("A barracks"))
    assert "a kitchen (none built yet)" in barracks and "a keep" not in barracks and "opens TRAIN_BATTALION" in barracks


def test_distant_buildings_and_possible_ones_are_not_explained():
    sim, t = _tribe()
    t.era = "tribal_synapse"
    t.long_houses_built = 5
    facts = locked_building_facts(t, sim.world, CHAIN)
    assert not any(f.startswith(("A fortress", "A castle")) for f in facts)  # the next era does not ask for them
    assert not any(f.startswith("A keep") for f in facts)  # a keep is possible with 5 long houses and 500 of each: nothing to explain
    assert not any(f.startswith("A library") for f in facts)


def test_the_explanation_matches_the_menu_rule_for_every_chain_building():
    """The two are written separately (the menu in AFFORDABILITY_CHECKS, the missing pieces in _chain_gaps); this keeps them in step."""
    from backend.simulation import AFFORDABILITY_CHECKS, _chain_gaps
    sim, t = _tribe()
    actions = CHAIN + ("BUILD_MINE", "BUILD_FORGE")
    checked = 0
    for long_houses in (0, 1, 3, 7, 8, 11, 12, 13):
        for keep in (False, True):
            for fortress in (False, True):
                for kitchen in (False, True):
                    for mine, ore in ((False, 0), (True, 0), (True, 3)):
                        for wood, stone in ((0, 0), (500, 0), (0, 500), (500, 500)):
                            t.long_houses_built, t.keep_built, t.fortress_built, t.kitchen_built = long_houses, keep, fortress, kitchen
                            t.mine_built, t.mine_resource_name, t.mine_sites = mine, "Amber", [{"x": 1, "y": 1, "resource": "Amber"}]
                            t.unique_resources, t.wood, t.stone = {"Amber": ore}, wood, stone
                            for action in actions:
                                assert (not _chain_gaps(action, t)) == bool(AFFORDABILITY_CHECKS[action](t, sim.world)), (action, long_houses, keep, fortress, kitchen, mine, ore, wood, stone)
                                checked += 1
    assert checked > 1000


def test_the_keep_fact_reaches_the_prompt_even_with_nudges_off(monkeypatch):
    monkeypatch.setenv("NUDGES", "off")
    sim, t = _tribe()
    t.era = "tribal_synapse"
    t.long_houses_built, t.wood = 1, 500
    request, _ctx = sim._prepare_turn(t)
    assert "A keep cannot be built yet" in request["prompt"]
