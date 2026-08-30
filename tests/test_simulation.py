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
    assert tribe.food == 42  # 40 + round(15 * 0.15 mountains game multiplier)


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


def test_prepare_turn_mentions_an_expedition_already_in_the_field():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.expedition = {
        "pos": [tribe.x, tribe.y], "origin": [tribe.x, tribe.y], "target": [tribe.x + 10, tribe.y],
        "day": 1, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    request, _ctx = sim._prepare_turn(tribe)

    assert "are still in the field" in request["prompt"]
    assert "day 1/" in request["prompt"]


def test_expedition_succeeds_immediately_on_reaching_real_river_water():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 40, 30, "#c084fc")
    # From (40, 30) toward (40, 37), one EXPEDITION_SPEED (6) step lands at (40, 36),
    # which world.py's river geography places on the river -- verified directly against
    # biome_at before trusting it, same discipline as world.py's nearest_water fix.
    tribe.expedition = {
        "pos": [40, 30], "origin": [40, 30], "target": [40, 37],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    assert tribe.expedition["phase"] == "returning"
    assert tribe.expedition["found"] == [40, 36]
    assert any("found fresh water" in entry for entry in tribe.history)


def test_expedition_reaching_its_target_without_water_pushes_onward_if_days_remain():
    """Regression test: a model's own target_vector is usually close (one
    EXPEDITION_SPEED step away), so treating "arrived at the declared spot" as "search
    over" meant max_days and the scout's determination trait almost never actually
    mattered -- live runs showed parties turning back on day one nearly every time.
    Reaching a non-water target with days left should extend the search outward along
    the same heading, not end it."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [56, 50],  # one step away, not water
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    assert tribe.expedition["phase"] == "outbound"  # not turned back
    assert tribe.expedition["terrain_report"] is not None  # still noted what's there
    assert tribe.expedition["found"] is None
    assert tribe.expedition["target"] != [56, 50]  # extended past the original spot
    assert any("pushes onward" in entry for entry in tribe.history)


def test_expedition_gives_up_at_a_pushed_onward_target_once_days_run_out():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [56, 50],
        "day": 3, "phase": "outbound", "found": None, "terrain_report": None,  # already at max_days
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    assert tribe.expedition["phase"] == "returning"


def test_expedition_gives_up_after_max_days_without_success():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    # Far enough away that EXPEDITION_MAX_DAYS worth of travel never arrives or finds water.
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [99, 99],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    from backend import config
    for _ in range(config.EXPEDITION_MAX_DAYS):
        sim._advance_expedition(tribe)

    assert tribe.expedition["phase"] == "returning"
    assert tribe.expedition["found"] is None
    assert any("calls off the search after" in entry for entry in tribe.history)


def test_expedition_arrival_home_delivers_water_finding_to_memory_and_clears_state():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    assert tribe.expedition is None
    assert any("(40,37)" in entry for entry in tribe.history)
    assert any("fresh water at (40,37)" in m["text"] for m in tribe.memory.entries)


def test_expedition_arrival_delivers_foraged_food_and_water_to_the_tribe():
    """The trip isn't a pure resource black hole -- a traveling party forages and
    hunts along the way, more on the outbound leg than the hurried trip home, and
    brings it back regardless of whether the search itself succeeded."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food, tribe.water = 10, 10
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 7, "water_gathered": 5,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    expected_food = 10 + 7 + config.EXPEDITION_RETURN_DAILY_FOOD
    expected_water = 10 + 5 + config.EXPEDITION_RETURN_DAILY_WATER
    assert tribe.food == expected_food
    assert tribe.water == expected_water


def test_expedition_report_is_attributed_to_the_chief_when_one_exists():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    assert any("Chief Ashgar" in entry for entry in tribe.history)


def test_expedition_report_falls_back_to_the_tribe_when_there_is_no_chief():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    assert any("gives the tribe a full report" in entry for entry in tribe.history)


def test_expedition_arrival_home_empty_handed_clears_state_without_a_water_memory():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [99, 99],
        "day": 3, "phase": "returning", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    assert tribe.expedition is None
    assert any("empty-handed" in entry for entry in tribe.history)


def test_expedition_records_every_tile_it_walks_as_a_breadcrumb_path():
    """The persistent world-trail mechanic (Landscape.trails) only lights up once a
    route gets reused, so a single fresh journey barely shows anything even while it's
    actively happening. This is the per-expedition breadcrumb line instead: everywhere
    this one party has actually walked, regardless of reuse."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [80, 80],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [[50, 50]],
    }

    sim._advance_expedition(tribe)
    sim._advance_expedition(tribe)

    assert tribe.expedition["path"][0] == [50, 50]
    assert len(tribe.expedition["path"]) == 3
    assert tribe.expedition["path"][-1] == tribe.expedition["pos"]


def test_expedition_wears_a_trail_on_the_tile_it_moves_into():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [80, 80],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    from backend import config
    landed = tuple(tribe.expedition["pos"])
    assert sim.world.trails.get(landed) == config.TRAIL_WEAR_PER_PASS


def test_expedition_travels_farther_along_an_already_worn_trail():
    """The point of trail wear applying to expeditions too: a route worn down by
    earlier trips lets a later expedition cover more ground per day, potentially
    reaching a destination that was out of reach on the first attempt."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.world.wear_trail(50, 50, 1.0)  # fully worn starting tile
    tribe.expedition = {
        "pos": [50, 50], "origin": [50, 50], "target": [99, 50],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }

    sim._advance_expedition(tribe)

    expected_speed = config.EXPEDITION_SPEED + config.MAX_TRAIL_BONUS_SPEED
    assert tribe.expedition["pos"] == [50 + expected_speed, 50]
    assert not any("fresh water" in m["text"] for m in tribe.memory.entries)


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


def test_every_spawn_point_is_within_a_single_expeditions_reach_of_water():
    """Regression test: the original spawn points were picked purely to land in the
    right-named biome and turned out to be 36-42 tiles from any river -- unreachable
    within EXPEDITION_MAX_DAYS at EXPEDITION_SPEED no matter how well a tribe reasoned.
    Every default spawn should be close enough that a genuine, well-aimed expedition can
    actually succeed."""
    from backend import config
    from backend.world import Landscape

    land = Landscape(100)
    max_reach = config.EXPEDITION_SPEED * config.EXPEDITION_MAX_DAYS
    for x, y in SPAWN_POINTS:
        if land.biome(x, y) == "river":
            continue
        nx, ny = land.nearest_water(x, y, kinds=("river",))
        dist = ((nx - x) ** 2 + (ny - y) ** 2) ** 0.5
        assert dist <= max_reach, f"({x},{y}) is {dist:.1f} tiles from water, beyond a {max_reach}-tile expedition"


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


def test_snapshot_includes_worn_trails_for_the_frontend_to_render():
    sim = _bare_simulation()
    sim.world.wear_trail(12, 34, 0.5)
    sim.tribes = {}
    sim.status = "OPERATIONAL"

    trails = sim.snapshot()["trails"]

    assert {"x": 12, "y": 34, "wear": 0.5} in trails


def test_population_grows_once_food_clears_the_threshold():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 8
    tribe.food = config.POPULATION_GROWTH_FOOD_THRESHOLD + 1

    sim._grow_population(tribe)

    assert tribe.population == 9
    assert tribe.food == config.POPULATION_GROWTH_FOOD_THRESHOLD + 1 - config.POPULATION_GROWTH_FOOD_COST


def test_population_does_not_grow_below_the_food_threshold():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 8
    tribe.food = config.POPULATION_GROWTH_FOOD_THRESHOLD

    sim._grow_population(tribe)

    assert tribe.population == 8


def test_population_growth_threshold_is_reachable_by_realistic_sustained_play():
    """Regression test: the original threshold (food > 80, costing 30) was verified
    live to be unreachable -- a real 79-cycle run under realistic mixed play never got
    food above ~38 for either tribe, starting from 40. A tribe that hunts successfully
    in forest a few cycles in a row (the single best-case income action) should be able
    to clear the threshold well within a normal run, not require inhuman optimization."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 65, 85, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.99):  # no wolf hazard
        for _ in range(6):  # HUNT_DEER in forest, undepleted, nets +14/cycle after upkeep
            sim._apply_action(tribe, "HUNT_DEER", "forest", (0, 0))
            sim._apply_upkeep(tribe)

    assert tribe.food > config.POPULATION_GROWTH_FOOD_THRESHOLD


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


def test_a_population_loss_can_claim_the_chief():
    """Previously a chief, once elected, was permanent flavor text no matter what
    happened to the population underneath them -- the population was already mortal,
    the chief was the one thing exempt from it. Any survived loss now carries a real
    chance of claiming the chief specifically."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 8
    tribe.chief_name = "Ashgar"
    tribe.chief_philosophy = "aggressive expansion"
    tribe.chief_decree = "seek water"

    with mock.patch("backend.simulation.random.random", return_value=0.0):  # below any positive chance
        sim._lose_population(tribe, 1)

    assert tribe.chief_name == ""
    assert tribe.chief_philosophy == ""
    assert tribe.chief_decree == ""
    assert any("Chief Ashgar has died" in entry for entry in tribe.history)


def test_a_population_loss_does_not_always_claim_the_chief():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 8
    tribe.chief_name = "Ashgar"

    with mock.patch("backend.simulation.random.random", return_value=0.999):  # above any plausible chance
        sim._lose_population(tribe, 1)

    assert tribe.chief_name == "Ashgar"


def test_extinction_does_not_also_report_a_separate_chief_death():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 1
    tribe.chief_name = "Ashgar"

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._lose_population(tribe, 1)

    assert tribe.extinct is True
    assert not any("Chief Ashgar has died" in entry for entry in tribe.history)


@run_async
async def test_step_installs_a_successor_chief_when_one_is_missing():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = ""  # a fallen chief, mid-run

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})), \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)):
        await sim.step()

    assert tribe.chief_name == "Test Chief"


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


@run_async
async def test_step_triggers_game_over_and_unloads_models_when_all_tribes_die():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    for tribe in sim.tribes.values():
        tribe.population = 1
        tribe.food = 0  # starves to extinction on this tick's upkeep

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})), \
         mock.patch.object(sim.client, "unload_model", mock.AsyncMock()) as mock_unload:
        await sim.step()

    assert sim.game_over is True
    assert sim.status == "GAME OVER"
    assert {c.args[0] for c in mock_unload.call_args_list} == {"gemma2:2b", "qwen2.5:3b"}


@run_async
async def test_step_does_nothing_once_game_over():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    sim.tribes["tribe_0"].extinct = True
    sim.game_over = True
    cycle_before = sim.cycle

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock()) as mock_run_batch:
        await sim.step()

    mock_run_batch.assert_not_called()
    assert sim.cycle == cycle_before


@run_async
async def test_add_tribe_after_game_over_resumes_the_simulation():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    sim.tribes["tribe_0"].extinct = True
    sim.game_over = True
    sim.status = "GAME OVER"

    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        await sim.add_tribe("B", "qwen2.5:3b")

    assert sim.game_over is False
    assert sim.status == "OPERATIONAL"


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


@run_async
async def test_install_chief_records_decree_when_decreed_and_not_on_water():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.x, tribe.y = 10, 10  # mountains spawn, not on water
    fake_result = {
        "chief_name": "Ashgar",
        "victory_method": "endurance",
        "guiding_philosophy": "expansion",
        "water_decision": {"decreed": True, "reason": "our people need water"},
    }
    with mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=fake_result)):
        await sim._install_chief(tribe)

    assert "dispatching scouts" in tribe.chief_decree
    assert any("decrees" in entry for entry in tribe.history)


@run_async
async def test_install_chief_no_decree_when_chief_declines():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.x, tribe.y = 10, 10
    fake_result = {
        "chief_name": "Ashgar",
        "guiding_philosophy": "expansion",
        "water_decision": {"decreed": False, "reason": "our shelter here is strong"},
    }
    with mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=fake_result)):
        await sim._install_chief(tribe)

    assert tribe.chief_decree == ""


@run_async
async def test_install_chief_skips_water_fact_when_already_on_water():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.x, tribe.y = 40, 37  # on the river
    captured = {}

    async def fake_elect(client, model, name, water_needed=False):
        captured["water_needed"] = water_needed
        return {"chief_name": "Ashgar", "guiding_philosophy": "x", "water_decision": {"decreed": False, "reason": ""}}

    with mock.patch("backend.simulation.elect_chief", fake_elect):
        await sim._install_chief(tribe)

    assert captured["water_needed"] is False
    assert tribe.chief_decree == ""


@run_async
async def test_install_chief_does_not_treat_ocean_as_solving_the_water_need():
    """Standing on the coast doesn't mean the tribe has drinking water -- seawater isn't
    a substitute for a river, so being on the ocean must not skip the water fact."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.x, tribe.y = 95, 50  # ocean, not river
    captured = {}

    async def fake_elect(client, model, name, water_needed=False):
        captured["water_needed"] = water_needed
        return {"chief_name": "Ashgar", "guiding_philosophy": "x", "water_decision": {"decreed": False, "reason": ""}}

    with mock.patch("backend.simulation.elect_chief", fake_elect):
        await sim._install_chief(tribe)

    assert captured["water_needed"] is True
