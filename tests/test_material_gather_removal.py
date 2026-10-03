"""2026-10-03: while a build is on offer, GATHER_WOOD / GATHER_STONE are removed for a resource already at
config.MATERIAL_SURPLUS_THRESHOLD ("more than any real use"), instead of only being pushed to the back of the menu. Below the
threshold they stay, so a young tribe with a cheap BUILD_FIRE or BUILD_ROAD on offer can still gather. Food and water gathers
are never removed."""
from backend import config
from backend.simulation import Simulation, _is_construction_action


def _camped_tribe(wood, stone):
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    sim._found_territory(tribe)
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.era = "cognitive_horizon"
    tribe.wood, tribe.stone = wood, stone
    return sim, tribe


def test_a_surplus_material_gather_is_removed_while_a_build_is_on_offer():
    threshold = config.MATERIAL_SURPLUS_THRESHOLD
    sim, tribe = _camped_tribe(wood=threshold, stone=threshold)
    _, ctx = sim._prepare_turn(tribe)
    actions = ctx["available_actions"]
    assert any(_is_construction_action(a) for a in actions)
    assert "GATHER_WOOD" not in actions and "GATHER_STONE" not in actions
    assert "GATHER_FOOD" in actions                      # food is never removed, only demoted


def test_each_material_is_judged_on_its_own_stock():
    threshold = config.MATERIAL_SURPLUS_THRESHOLD
    sim, tribe = _camped_tribe(wood=threshold - 1, stone=threshold)
    _, ctx = sim._prepare_turn(tribe)
    actions = ctx["available_actions"]
    assert any(_is_construction_action(a) for a in actions)
    assert "GATHER_WOOD" in actions                      # just under the line: still offered, pushed behind the builds
    assert "GATHER_STONE" not in actions                 # at the line: removed


def test_a_poor_tribe_keeps_both_material_gathers_even_with_a_cheap_build_on_offer():
    sim, tribe = _camped_tribe(wood=20, stone=20)
    _, ctx = sim._prepare_turn(tribe)
    actions = ctx["available_actions"]
    assert "GATHER_WOOD" in actions and "GATHER_STONE" in actions
