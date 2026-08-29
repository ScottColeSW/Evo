from unittest import mock

from backend.ancestral_matrix import AncestralTraumaMatrix
from backend.simulation import Simulation, Tribe
from backend.world import Landscape


def _bare_simulation():
    """A Simulation with no network-touching state, for testing pure logic."""
    sim = Simulation.__new__(Simulation)
    sim.world = Landscape(100)
    sim.trauma = AncestralTraumaMatrix(100)
    sim.cycle = 1
    return sim


def test_hunting_hazard_applies_all_effects_when_triggered():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 85, 85, "#c084fc")
    tribe.food = 40
    tribe.population = 10

    with mock.patch("backend.actions.random.random", return_value=0.01):
        note = sim._apply_action(tribe, "HUNT_DEER", "forest")

    assert note == "a wolf pack struck the hunting party"
    assert tribe.food == 30
    assert tribe.population == 9
    assert float(sim.trauma.ghost_tensor[85, 85]) < 0
    assert "DREAD" in sim.trauma.bias_string(85, 85)


def test_hunting_succeeds_when_hazard_roll_misses():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 85, 85, "#c084fc")
    tribe.food = 40

    with mock.patch("backend.actions.random.random", return_value=0.99):
        note = sim._apply_action(tribe, "HUNT_DEER", "forest")

    assert note is None
    assert tribe.food == 55


def test_hunting_hazard_never_fires_outside_forest():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "qwen2.5:3b", 10, 45, "#fb923c")
    tribe.food = 40

    with mock.patch("backend.actions.random.random", return_value=0.01):
        note = sim._apply_action(tribe, "HUNT_DEER", "mountains")

    assert note is None
    assert tribe.food == 55


def test_build_fire_radiates_pride_not_dread():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50

    sim._apply_action(tribe, "BUILD_FIRE", "forest")

    assert tribe.wood == 40
    assert "PRIDE" in sim.trauma.bias_string(50, 50)


def test_overheard_broadcast_reaches_other_tribes_prompt():
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b"},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b"},
        ]
    )
    forest = sim.tribes["tribe_0"]
    mountain = sim.tribes["tribe_1"]
    mountain.last_broadcast = "KRA-ZUL"
    mountain.last_action = "HUNT_DEER"

    request, _ctx = sim._prepare_turn(forest)

    assert "overheard: Mountain Tribe broadcasted 'KRA-ZUL' while performing HUNT_DEER" in request["prompt"]


def test_translation_matrix_is_updated_on_apply_turn():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    tribe_a = sim.tribes["tribe_0"]
    tribe_b = sim.tribes["tribe_1"]
    ctx = {"biome": "forest", "available_actions": ["BUILD_FIRE", "IDLE"]}

    sim._apply_turn(tribe_a, {"visual_action": "BUILD_FIRE", "synthetic_language_broadcast": "VASH-TA"}, 100.0, ctx)
    sim._apply_turn(tribe_b, {"visual_action": "BUILD_FIRE", "synthetic_language_broadcast": "VASH-TA"}, 100.0, ctx)

    summary = sim.translation.pair_summary("tribe_0", "tribe_1")
    assert summary["tracked_tokens"] == 1


def test_gather_water_yields_more_on_river_than_elsewhere():
    sim = _bare_simulation()
    river_tribe = Tribe("tribe_0", "River Tribe", "gemma2:2b", 50, 50, "#60a5fa")
    plains_tribe = Tribe("tribe_1", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")

    # Force the drowning roll to miss so this test isn't flaky against the ~8% hazard.
    with mock.patch("backend.actions.random.random", return_value=0.99):
        sim._apply_action(river_tribe, "GATHER_WATER", "river")
        sim._apply_action(plains_tribe, "GATHER_WATER", "plains")

    assert river_tribe.water > plains_tribe.water


def test_action_outside_current_era_is_rejected_to_idle():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    assert tribe.era == "stone_age"

    request, ctx = sim._prepare_turn(tribe)
    assert "CONSTRUCT_WALL" not in ctx["available_actions"]

    sim._apply_turn(tribe, {"visual_action": "CONSTRUCT_WALL"}, 50.0, ctx)
    assert "IDLE" in tribe.history[-1]


def test_era_advances_once_population_and_resources_are_met():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 20
    tribe.water = 40
    tribe.stone = 40
    tribe.wood = 50

    sim._advance_era_if_ready(tribe)

    assert tribe.era == "bronze_age"
    assert tribe.wood == 20  # 50 - 30 advancement cost
    assert tribe.stone == 10  # 40 - 30 advancement cost
    assert tribe.water == 20  # 40 - 20 advancement cost
    assert "Bronze Age" in tribe.history[-1]
    assert "PRIDE" in sim.trauma.bias_string(50, 50)


def test_era_does_not_advance_without_meeting_resource_requirements():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 20
    tribe.water = 5  # below the bronze_age requirement

    sim._advance_era_if_ready(tribe)

    assert tribe.era == "stone_age"


def test_upkeep_consumes_food_and_water_proportional_to_population():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 25  # upkeep = max(1, 25 // 10) = 2
    tribe.food = 40
    tribe.water = 40

    sim._apply_upkeep(tribe)

    assert tribe.food == 38
    assert tribe.water == 38


def test_unpaid_food_upkeep_causes_starvation():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    tribe.food = 0
    tribe.water = 40

    sim._apply_upkeep(tribe)

    assert tribe.food == 0  # floored, not negative
    assert tribe.population == 9
    assert "starvation" in tribe.history[-1]
    assert "DREAD" in sim.trauma.bias_string(50, 50)


def test_unpaid_water_upkeep_causes_dehydration():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    tribe.food = 40
    tribe.water = 0

    sim._apply_upkeep(tribe)

    assert tribe.water == 0
    assert tribe.population == 9
    assert "thirst" in tribe.history[-1]
    assert "DREAD" in sim.trauma.bias_string(50, 50)


def test_population_never_drops_below_one_from_starvation():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 1
    tribe.food = 0
    tribe.water = 40

    sim._apply_upkeep(tribe)

    assert tribe.population == 1


def test_drowning_hazard_on_river_tile():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "River Tribe", "gemma2:2b", 50, 50, "#60a5fa")
    tribe.population = 10
    tribe.water = 30

    with mock.patch("backend.actions.random.random", return_value=0.01):
        note = sim._apply_action(tribe, "GATHER_WATER", "river")

    assert note == "the river's current pulled someone under"
    assert tribe.population == 9
    assert tribe.water == 30  # no gain on a drowning turn
    assert "DREAD" in sim.trauma.bias_string(50, 50)


def test_drowning_hazard_never_fires_off_river():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.population = 10
    tribe.water = 30

    with mock.patch("backend.actions.random.random", return_value=0.01):
        note = sim._apply_action(tribe, "GATHER_WATER", "plains")

    assert note is None
    assert tribe.population == 10
    assert tribe.water == 33


def test_reaching_classical_age_marks_founded_city():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.era = "bronze_age"
    tribe.population = 40
    tribe.water = 60
    tribe.stone = 40
    tribe.wood = 40

    sim._advance_era_if_ready(tribe)

    assert tribe.era == "classical_age"
    assert tribe.founded_city is True
