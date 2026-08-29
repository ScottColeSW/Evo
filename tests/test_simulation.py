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

    with mock.patch("backend.simulation.random.random", return_value=0.01):
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

    with mock.patch("backend.simulation.random.random", return_value=0.99):
        note = sim._apply_action(tribe, "HUNT_DEER", "forest")

    assert note is None
    assert tribe.food == 55


def test_hunting_hazard_never_fires_outside_forest():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "qwen2.5:3b", 10, 45, "#fb923c")
    tribe.food = 40

    with mock.patch("backend.simulation.random.random", return_value=0.01):
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

    sim._apply_turn(tribe_a, {"visual_action": "BUILD_FIRE", "synthetic_language_broadcast": "VASH-TA"}, 100.0, {"biome": "forest"})
    sim._apply_turn(tribe_b, {"visual_action": "BUILD_FIRE", "synthetic_language_broadcast": "VASH-TA"}, 100.0, {"biome": "forest"})

    summary = sim.translation.pair_summary("tribe_0", "tribe_1")
    assert summary["tracked_tokens"] == 1
