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


def test_survival_starts_with_deliberately_scarce_resources():
    """2026-09-12 finding from a real batch run: every survival trial scored
    100/100 (zero extinctions) even on the harsh desert spawn, because
    Tribe.__init__'s normal starting stock already buys ~30-40 cycles of
    upkeep before a single gather is required -- a real ceiling effect, not a
    meaningful test of anything. starting_resources exists specifically to
    force a real early decision instead of a long, risk-free runway."""
    scenario = SCENARIOS["survival"]
    assert scenario.starting_resources is not None
    assert all(amount <= 10 for amount in scenario.starting_resources.values())


def test_other_scenarios_leave_starting_resources_at_the_game_defaults():
    for key in ("settlement", "cooperation", "conflict"):
        assert SCENARIOS[key].starting_resources is None, key


def test_cooperation_and_conflict_spawn_two_tribes_within_discovery_range():
    from backend import config
    import math

    for key in ("cooperation", "conflict"):
        (x1, y1), (x2, y2) = SCENARIOS[key].spawn_positions
        assert math.hypot(x2 - x1, y2 - y1) <= config.RIVAL_PRECISE_AWARENESS_RADIUS + 5, key


def test_war_ready_5000_ships_two_real_fixtures():
    """Explicit request, 2026-09-16: "we are going to want to start a war
    starting from around 600" -- corrected to 5000 once real board_history.db
    data showed no run has ever built a Barracks below the low thousands."""
    scenario = SCENARIOS["war_ready_5000"]
    assert scenario.starting_fixtures == ("war_ready_a", "war_ready_b")
    assert len(scenario.starting_fixtures) == scenario.tribe_count
    assert scenario.disabled_actions is None


def test_war_ready_5000_no_alliance_reuses_the_same_fixtures_without_alliance():
    """Explicit request, 2026-09-16: "build a fixture that takes alliance off
    the table" -- follow-up to two real war_ready_5000 trials that both ended
    in an early DECLARE_ALLIANCE rather than any conflict."""
    scenario = SCENARIOS["war_ready_5000_no_alliance"]
    assert scenario.starting_fixtures == ("war_ready_a", "war_ready_b")
    assert scenario.disabled_actions == ("DECLARE_ALLIANCE",)
    assert scenario.cycle_budget == SCENARIOS["war_ready_5000"].cycle_budget
