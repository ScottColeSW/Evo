from unittest import mock

from backend.ancestral_matrix import AncestralTraumaMatrix
from backend.simulation import SPAWN_POINTS, Simulation, Tribe
from backend.world import Landscape
from tests.conftest import run_async


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
        note = sim._apply_action(tribe, "HUNT_DEER", "forest", (0, 0))

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
        note = sim._apply_action(tribe, "HUNT_DEER", "forest", (0, 0))

    assert note is None
    assert tribe.food == 55


def test_hunting_hazard_never_fires_outside_forest():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "qwen2.5:3b", 10, 45, "#fb923c")
    tribe.food = 40

    with mock.patch("backend.actions.random.random", return_value=0.01):
        note = sim._apply_action(tribe, "HUNT_DEER", "mountains", (0, 0))

    assert note is None
    assert tribe.food == 55


def test_build_fire_radiates_pride_not_dread():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50

    sim._apply_action(tribe, "BUILD_FIRE", "forest", (0, 0))

    assert tribe.wood == 40
    assert "PRIDE" in sim.trauma.bias_string(50, 50)


def test_overheard_broadcast_reaches_other_tribes_prompt_when_nearby():
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b"},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b"},
        ]
    )
    forest = sim.tribes["tribe_0"]
    mountain = sim.tribes["tribe_1"]
    mountain.x, mountain.y = forest.x - 5, forest.y  # within BROADCAST_HEARING_RADIUS
    mountain.last_broadcast = "KRA-ZUL"
    mountain.last_action = "HUNT_DEER"

    request, _ctx = sim._prepare_turn(forest)

    assert "overheard: Mountain Tribe broadcasted 'KRA-ZUL' while performing HUNT_DEER" in request["prompt"]


def test_broadcast_not_overheard_beyond_hearing_radius():
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b"},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b"},
        ]
    )
    forest = sim.tribes["tribe_0"]
    mountain = sim.tribes["tribe_1"]  # default spawns are far apart (different biomes)
    mountain.last_broadcast = "KRA-ZUL"
    mountain.last_action = "HUNT_DEER"

    request, _ctx = sim._prepare_turn(forest)

    assert "overheard" not in request["prompt"]


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
        sim._apply_action(river_tribe, "GATHER_WATER", "river", (0, 0))
        sim._apply_action(plains_tribe, "GATHER_WATER", "plains", (0, 0))

    assert river_tribe.water > plains_tribe.water


def test_action_outside_current_era_is_rejected_to_idle():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    assert tribe.era == "stone_age"

    request, ctx = sim._prepare_turn(tribe)
    assert "CONSTRUCT_WALL" not in ctx["available_actions"]

    sim._apply_turn(tribe, {"visual_action": "CONSTRUCT_WALL"}, 50.0, ctx)
    assert "IDLE" in tribe.history[-1]


def test_apply_turn_records_last_target_only_for_relocate():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    ctx = {"biome": "forest", "available_actions": ["RELOCATE", "IDLE"]}

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [20, 30]}, 10.0, ctx)

    assert tribe.last_target == [20, 30]


def test_apply_turn_does_not_record_last_target_for_non_relocate_actions():
    """Only RELOCATE actually moves the tribe; other actions' target_vector shouldn't
    create a phantom "journey" reminder for a trip the tribe never intended to take."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    ctx = {"biome": "forest", "available_actions": ["IDLE"]}

    sim._apply_turn(tribe, {"visual_action": "IDLE", "target_vector": [20, 30]}, 10.0, ctx)

    assert tribe.last_target is None


def test_only_relocate_moves_the_tribe_via_apply_turn():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    start = (tribe.x, tribe.y)
    ctx = {"biome": "forest", "available_actions": ["GATHER_WOOD", "RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "GATHER_WOOD", "target_vector": [90, 90]}, 10.0, ctx)
    assert (tribe.x, tribe.y) == start  # gathering never moves the tribe

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [90, 90]}, 10.0, ctx)
    assert (tribe.x, tribe.y) != start  # relocating does


def test_prepare_turn_reminds_tribe_of_unfinished_journey():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.last_target = [tribe.x - 20, tribe.y]  # far off, not yet arrived

    request, _ctx = sim._prepare_turn(tribe)

    assert "you have not yet arrived there" in request["prompt"]


def test_prepare_turn_has_no_journey_note_once_arrived():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.last_target = [tribe.x, tribe.y]  # arrived

    request, _ctx = sim._prepare_turn(tribe)

    assert "you have not yet arrived there" not in request["prompt"]


def test_explicit_spawn_coordinates_override_the_default_spawn_points():
    sim = Simulation([
        {"name": "A", "model": "gemma2:2b", "x": 40, "y": 35},
        {"name": "B", "model": "qwen2.5:3b", "x": 45, "y": 40},
    ])
    assert (sim.tribes["tribe_0"].x, sim.tribes["tribe_0"].y) == (40, 35)
    assert (sim.tribes["tribe_1"].x, sim.tribes["tribe_1"].y) == (45, 40)


def test_omitting_spawn_coordinates_still_falls_back_to_spawn_points():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    assert (sim.tribes["tribe_0"].x, sim.tribes["tribe_0"].y) == SPAWN_POINTS[0]


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


def test_starvation_can_cause_real_extinction():
    """The old behavior floored population at 1 forever (a permanent "walking dead"
    state); a tribe can now actually go extinct."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 1
    tribe.food = 0
    tribe.water = 40

    sim._apply_upkeep(tribe)

    assert tribe.population == 0
    assert tribe.extinct is True
    assert any("gone extinct" in entry for entry in tribe.history)
    assert "DREAD" in sim.trauma.bias_string(50, 50)


def test_extinct_tribe_loses_no_further_population():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 0
    tribe.extinct = True

    sim._lose_population(tribe, 5)

    assert tribe.population == 0


@run_async
async def test_step_skips_extinct_tribes_entirely():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    dead = sim.tribes["tribe_0"]
    dead.extinct = True
    dead.population = 0
    frozen_history = list(dead.history)

    async def fake_run_batch(requests):
        # only the living tribe ("B") should ever be asked for a turn
        assert [r["id"] for r in requests] == ["tribe_1"]
        return {"tribe_1": {"intent": {"visual_action": "IDLE"}, "latency_ms": 0.0}}

    with mock.patch.object(sim.scheduler, "run_batch", fake_run_batch):
        await sim.step()

    assert dead.history == frozen_history  # untouched -- no turn was ever prepared for it


def test_drowning_hazard_on_river_tile():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "River Tribe", "gemma2:2b", 50, 50, "#60a5fa")
    tribe.population = 10
    tribe.water = 30

    with mock.patch("backend.actions.random.random", return_value=0.01):
        note = sim._apply_action(tribe, "GATHER_WATER", "river", (0, 0))

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
        note = sim._apply_action(tribe, "GATHER_WATER", "plains", (0, 0))

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


_FAKE_CHIEF = {"chief_name": "Test Chief", "victory_method": "a coin flip", "guiding_philosophy": "test philosophy"}


@run_async
async def test_add_tribe_appends_with_unique_spawn_and_color():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        error = await sim.add_tribe("B", "qwen2.5:3b")

    assert error is None
    assert len(sim.tribes) == 2
    new_tribe = sim.tribes["tribe_1"]
    assert new_tribe.name == "B"
    assert new_tribe.model == "qwen2.5:3b"
    assert (new_tribe.x, new_tribe.y) != (sim.tribes["tribe_0"].x, sim.tribes["tribe_0"].y)
    assert new_tribe.color != sim.tribes["tribe_0"].color


@run_async
async def test_add_tribe_rejects_beyond_max_tribes():
    sim = Simulation([{"name": f"T{i}", "model": "gemma2:2b"} for i in range(4)])
    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls:
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        error = await sim.add_tribe("Overflow", "gemma2:2b")

    assert error is not None
    assert len(sim.tribes) == 4


@run_async
async def test_add_tribe_records_vram_warning_in_new_tribes_history():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(False, "too big"))
        error = await sim.add_tribe("B", "gemma4:26b")

    assert error is None  # the tribe is still added, just warned
    assert any("VRAM WARNING: too big" in entry for entry in sim.tribes["tribe_1"].history)


@run_async
async def test_add_tribe_installs_a_chief():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        await sim.add_tribe("B", "qwen2.5:3b")

    new_tribe = sim.tribes["tribe_1"]
    assert new_tribe.chief_name == "Test Chief"
    assert new_tribe.chief_philosophy == "test philosophy"
    assert any("Test Chief has become chief" in entry for entry in new_tribe.history)
