"""2026-10-03: while a build is on offer, GATHER_WOOD / GATHER_STONE are removed for a resource already at the gather floor
(simulation._material_gather_floor: the largest ordinary build cost, or the vessel's cost while a departure-era tribe still
needs one), instead of only being pushed to the back of the menu. Below the floor they stay, so a tribe with a cheap BUILD_FIRE
on offer can still gather and save up. Food and water gathers are never removed."""
from backend import config
from backend.simulation import Simulation, _is_construction_action, _material_gather_floor


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
    sim, tribe = _camped_tribe(wood=1, stone=1)
    threshold = _material_gather_floor(tribe)
    tribe.wood = tribe.stone = threshold
    _, ctx = sim._prepare_turn(tribe)
    actions = ctx["available_actions"]
    assert any(_is_construction_action(a) for a in actions)
    assert "GATHER_WOOD" not in actions and "GATHER_STONE" not in actions
    assert "GATHER_FOOD" in actions                      # food is never removed, only demoted


def test_each_material_is_judged_on_its_own_stock():
    sim, tribe = _camped_tribe(wood=1, stone=1)
    threshold = _material_gather_floor(tribe)
    tribe.wood, tribe.stone = threshold - 1, threshold
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


def test_a_departure_era_tribe_can_always_gather_toward_the_vessel():
    """The bug this floor fixes: with the old fixed line (50), a departure-era tribe holding 100 or 1,200 wood and stone was not
    offered the gathers (cheap builds are always on its menu). 2026-10-04: the vessel is built in stages, so the gathers stay on only until
    a stage (VESSEL_MAX_CONTRIBUTION_PER_ACTION) can be paid, then narrow like any other build."""
    sim, tribe = _camped_tribe(wood=100, stone=100)
    tribe.era = "departure_era"
    tribe.departure_dreamed = True
    tribe.warehouses_built = config.WAREHOUSE_MAX_COUNT  # a real departure-era tribe can store the vessel; stock above the cap cannot exist
    for held in (100, 300, config.VESSEL_MAX_CONTRIBUTION_PER_ACTION - 1):
        tribe.wood = tribe.stone = held
        _, ctx = sim._prepare_turn(tribe)
        assert "GATHER_WOOD" in ctx["available_actions"] and "GATHER_STONE" in ctx["available_actions"], held
    tribe.wood = tribe.stone = config.VESSEL_MAX_CONTRIBUTION_PER_ACTION
    _, ctx = sim._prepare_turn(tribe)
    assert "BUILD_VESSEL" in ctx["available_actions"]                      # a stage is payable now, and the gathers can go
    assert "GATHER_WOOD" not in ctx["available_actions"]


def test_the_floor_is_the_largest_ordinary_cost_and_rises_only_for_an_unbuilt_vessel():
    sim, tribe = _camped_tribe(wood=0, stone=0)
    ordinary = _material_gather_floor(tribe)
    assert ordinary >= config.DECLARE_CONQUEST_WOOD_COST and ordinary < config.VESSEL_MAX_CONTRIBUTION_PER_ACTION
    tribe.era = "departure_era"
    assert _material_gather_floor(tribe) == config.VESSEL_MAX_CONTRIBUTION_PER_ACTION
    tribe.vessel_built = True
    assert _material_gather_floor(tribe) == ordinary
