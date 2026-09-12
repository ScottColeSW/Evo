from backend.benchmark_scenarios import SCENARIOS
from backend.world import biome_at


def test_every_scenario_has_a_matching_spawn_count():
    for key, scenario in SCENARIOS.items():
        assert scenario.tribe_count == len(scenario.spawn_positions), key
        assert scenario.cycle_budget > 0, key
        assert scenario.key == key


def test_survival_spawns_somewhere_genuinely_harsh():
    scenario = SCENARIOS["survival"]
    x, y = scenario.spawn_positions[0]
    assert biome_at(x, y) in ("desert", "mountains")


def test_settlement_spawns_on_favorable_ground():
    scenario = SCENARIOS["settlement"]
    x, y = scenario.spawn_positions[0]
    assert biome_at(x, y) in ("plains", "forest", "river")


def test_cooperation_and_conflict_spawn_two_tribes_within_discovery_range():
    from backend import config
    import math

    for key in ("cooperation", "conflict"):
        (x1, y1), (x2, y2) = SCENARIOS[key].spawn_positions
        assert math.hypot(x2 - x1, y2 - y1) <= config.RIVAL_PRECISE_AWARENESS_RADIUS + 5, key
