from unittest import mock

from backend.actions import GAME_SPECIES_BY_BIOME
from backend.ancestral_matrix import AncestralTraumaMatrix
from backend.simulation import SPAWN_POINTS, Simulation, Tribe, _celebration_shout, _guess_intended_action, _resolve_action
from backend.world import Landscape
from tests.conftest import run_async


def _bare_simulation():
    """A Simulation with no network-touching state, for testing pure logic."""
    sim = Simulation.__new__(Simulation)
    sim.world = Landscape(100)
    sim.trauma = AncestralTraumaMatrix(100)
    sim.cycle = 1
    sim.immortality_cycles = 0
    sim.storm_cloud = None
    sim.lightning_strike = None
    sim.recent_encounters = []
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


def test_declared_stance_is_surfaced_as_a_fact_regardless_of_distance():
    """Explicit follow-up from the Agentic Evolution spec reconciliation: a declared
    geopolitical stance is known policy, not something that needs proximity to
    remember -- unlike overheard broadcasts/sightings, which do."""
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b"},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b"},
        ]
    )
    forest = sim.tribes["tribe_0"]
    mountain = sim.tribes["tribe_1"]  # default spawns are far apart (different biomes)
    forest.stance_toward["tribe_1"] = "WAR"

    request, _ctx = sim._prepare_turn(forest)

    assert f"Currently war with {mountain.name}." in request["prompt"]


def test_threat_assessment_layer_appears_in_the_prompt_for_a_declared_enemy():
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b", "x": 50, "y": 50},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b", "x": 51, "y": 51},
        ]
    )
    forest = sim.tribes["tribe_0"]
    forest.stance_toward["tribe_1"] = "WAR"

    request, _ctx = sim._prepare_turn(forest)

    assert "THREAT ASSESSMENT LAYER" in request["prompt"]
    assert "declared enemy" in request["prompt"]


def test_threat_assessment_layer_shows_the_neutral_placeholder_without_a_declared_enemy():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]

    request, _ctx = sim._prepare_turn(tribe)

    assert "NO DECLARED ENEMY WITHIN ASSESSABLE RANGE" in request["prompt"]


def test_prepare_turn_caches_wellbeing_on_the_tribe_and_injects_its_summary_into_the_prompt():
    """See backend/wellbeing.py -- per explicit design decision, the Maslow's-ladder
    read isn't viewer-only: _prepare_turn must both cache it on the tribe (so the
    frontend renders the same numbers) and inject its summary text into the actual
    prompt the chief reasons from, the same way survival_bias already does."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]

    request, _ctx = sim._prepare_turn(tribe)

    assert tribe.wellbeing.get("tiers")
    assert tribe.wellbeing.get("focus")
    assert "COMMUNITY WELL-BEING LAYER" in request["prompt"]
    assert tribe.wellbeing["summary"] in request["prompt"]


def test_wall_fraction_helper_reused_by_wellbeing_matches_raider_defense_lookup():
    """_wall_fraction is the single source of truth both _resolve_raider_attack and
    the wellbeing safety tier read from -- a half-built wall should report the same
    0.5 to both, not two independently-computed numbers that could drift apart."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    sim.world.add_construction(tribe.x, tribe.y, "wall", sim.cycle, progress=50)

    assert sim._wall_fraction(tribe) == 0.5


def test_settling_near_water_succeeds_within_a_confirmed_sites_territory_radius():
    """Explicit request: "the proposed settlement sites, water found, are making
    it hard to Settle. I think we can make this an initial territory with a
    bounding area around it that is larger than the Discovery." A single
    confirmed water tile was too fragile a RELOCATE target -- one tile off onto
    non-qualifying ground meant never settling despite being right next to real
    water."""
    from backend import config

    sim = Simulation([{"name": "Mountain Tribe", "model": "gemma2:2b", "x": 5, "y": 55}])  # mountains, not farmable
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.confirmed_water_sites = [(5, 55)]  # exactly on the tribe's own tile

    assert sim._is_settled(tribe) is True
    assert sim._is_settled_near_water(tribe) is True


def test_settling_near_water_still_fails_outside_the_territory_radius():
    from backend import config

    sim = Simulation([{"name": "Mountain Tribe", "model": "gemma2:2b", "x": 5, "y": 55}])
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    far = config.SETTLEMENT_WATER_TERRITORY_RADIUS + 1
    tribe.confirmed_water_sites = [(5 + far, 55)]

    assert sim._is_settled_near_water(tribe) is False


def test_not_settled_yet_names_ground_already_qualifying_and_warns_against_relocating():
    """Bug report: "look at the Mountain Tribe and tell me why they aren't
    Settled and fix it." They were standing on a lake tile (real, qualifying
    ground) only 3/10 cycles into the stability window -- the old fact always
    said "on farmable ground" regardless of whether the current tile actually
    qualified, easy to misread as needing to move somewhere else rather than
    just wait it out without relocating again."""
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river, qualifies
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = 3

    request, _ctx = sim._prepare_turn(tribe)

    assert "This ground already qualifies for settling -- 3/10 cycles" in request["prompt"]
    assert "relocating somewhere that no longer qualifies resets this progress back to 0" in request["prompt"]


def test_not_settled_yet_names_ground_that_doesnt_qualify_at_all():
    from backend import config

    sim = Simulation([{"name": "Mountain Tribe", "model": "gemma2:2b", "x": 5, "y": 55}])  # mountains, doesn't qualify
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = 3

    request, _ctx = sim._prepare_turn(tribe)

    assert "this ground doesn't qualify for settling at all" in request["prompt"]
    assert "This ground already qualifies" not in request["prompt"]


def test_resolve_toll_ignores_a_tile_that_isnt_a_toll_road_yet():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]

    assert sim._resolve_toll(tribe, 0, 0, 10, 10) == (10, 10)


def test_resolve_toll_free_on_your_own_road():
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    for _ in range(config.ROAD_EVOLVE_CROSSINGS + 1):
        sim.world.wear_trail(10, 10, 0.01, tribe_id=tribe.id)
    wood_before = tribe.wood

    assert sim._resolve_toll(tribe, 0, 0, 10, 10) == (10, 10)
    assert tribe.wood == wood_before


def test_resolve_toll_charges_a_different_tribe_and_pays_the_owner():
    """Explicit request: "The first trailblazer gets the ownership and tolls
    (automatically collected when used or crossed)." """
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    owner, traveler = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    for _ in range(config.ROAD_EVOLVE_CROSSINGS + 1):
        sim.world.wear_trail(10, 10, 0.01, tribe_id=owner.id)
    traveler.wood = config.TOLL_FEE_WOOD + 20
    owner_wood_before = owner.wood

    result = sim._resolve_toll(traveler, 0, 0, 10, 10)

    assert result == (10, 10)
    assert traveler.wood == 20
    assert owner.wood == owner_wood_before + config.TOLL_FEE_WOOD


def test_resolve_toll_blocks_a_tribe_that_cant_afford_it():
    """Explicit request: "can't pay, can't cross." Blocked means staying at
    the mover's own current position (cx, cy) -- not the tribe's home camp,
    since an expedition's own field position is what's actually passed in."""
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    owner, traveler = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    for _ in range(config.ROAD_EVOLVE_CROSSINGS + 1):
        sim.world.wear_trail(10, 10, 0.01, tribe_id=owner.id)
    traveler.wood = config.TOLL_FEE_WOOD - 1

    result = sim._resolve_toll(traveler, 3, 4, 10, 10)

    assert result == (3, 4)  # blocked -- stayed at its own current position
    assert traveler.wood == config.TOLL_FEE_WOOD - 1  # untouched, toll never charged


def test_resolve_toll_free_passage_once_the_owner_is_extinct():
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    owner, traveler = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    for _ in range(config.ROAD_EVOLVE_CROSSINGS + 1):
        sim.world.wear_trail(10, 10, 0.01, tribe_id=owner.id)
    owner.extinct = True
    traveler.wood = 0

    assert sim._resolve_toll(traveler, 0, 0, 10, 10) == (10, 10)
    assert traveler.wood == 0  # no one left to collect from


def test_unfinished_wall_nudges_against_a_premature_long_house():
    """Explicit bug report: live logs showed the chief repeatedly choosing
    BUILD_LONG_HOUSE against an unfinished wall, over and over, each attempt
    silently rejected inside _build_long_house -- CONSTRUCT_WALL and
    BUILD_LONG_HOUSE both unlock at the same era, so nothing ever told the chief
    the wall wasn't done."""
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains, farmable
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    sim.world.add_construction(tribe.x, tribe.y, "wall", sim.cycle, progress=40)

    request, ctx = sim._prepare_turn(tribe)

    assert "CONSTRUCT_WALL" in ctx["available_actions"]
    assert "40% complete" in request["prompt"]
    assert "a long house is not worth attempting until the wall is finished" in request["prompt"]


def test_no_wall_started_yet_nudges_toward_construct_wall():
    """Bug report: "wall building is not coming up for them" -- a tribe sat at
    Tribal Synapse for many cycles with a wall never even started, buried among
    a dozen other newly-unlocked actions with nothing calling it out
    specifically."""
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    request, ctx = sim._prepare_turn(tribe)

    assert "CONSTRUCT_WALL" in ctx["available_actions"]
    assert "No wall has been started here yet" in request["prompt"]


def test_confirmed_water_nudge_says_scouting_for_it_is_no_longer_needed():
    """Bug report: "they still search for water even after they found it." The
    fact used to only name the benefit of relocating -- never said outright that
    further scouting for water specifically was pointless once already found."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])  # forest, not water
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites = [(40, 37)]

    request, _ctx = sim._prepare_turn(tribe)

    assert "Water has already been found at (40,37)" in request["prompt"]


def test_confirmed_water_nudge_recognizes_already_being_there():
    """Bug report: "it looks like they want to consider relocating when they
    are on top of the water discovery site." The old fact always said
    "RELOCATE there" regardless of whether the tribe's current position
    already qualified -- confusing when settled_near_water was still False for
    some other reason (not enough cycles yet) while standing right on the
    confirmed site."""
    from backend import config

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites = [(40, 37)]  # exactly where the tribe already stands
    tribe.cycles_since_relocate = 3  # not yet enough to count as settled_near_water

    request, _ctx = sim._prepare_turn(tribe)

    assert "The tribe is already at or near the confirmed water site (40,37)" in request["prompt"]
    assert "Water has already been found at" not in request["prompt"]


def test_quarry_nudge_requires_a_scouted_site_not_just_the_other_prerequisites():
    """Explicit correction: "once they know where a quarry is, they need to just
    use it to get stone... they might consider building one closer to their
    establishment." A quarry is only worth building once a real stone-rich site
    has actually been scouted (tribe.quarry_sites), the same real-discovery gate
    _build_mine already used -- built at the settlement, not requiring travel to
    the exact discovered tile."""
    sim = Simulation([{"name": "Mountain Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.has_ever_settled = True
    tribe.long_houses_built = 1
    tribe.fishing_learned = True

    request, _ctx = sim._prepare_turn(tribe)
    assert "no stone-rich site has been scouted yet" in request["prompt"]

    tribe.quarry_sites.append((12, 34))
    request, _ctx = sim._prepare_turn(tribe)
    assert "A stone-rich site is known at (12,34)" in request["prompt"]


def test_sawmill_nudge_requires_a_scouted_site_not_just_the_other_prerequisites():
    """Explicit correction: "if they have found a... Stand of Trees to
    Harvest, these are collectables that must be fetched" -- mirrors the
    quarry nudge above exactly, for lumber_sites instead of quarry_sites."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.has_ever_settled = True
    tribe.long_houses_built = 1
    tribe.fishing_learned = True

    request, _ctx = sim._prepare_turn(tribe)
    assert "no stand of trees has been scouted yet" in request["prompt"]

    tribe.lumber_sites.append((12, 34))
    request, _ctx = sim._prepare_turn(tribe)
    assert "A stand of trees is known at (12,34)" in request["prompt"]


def test_finished_wall_nudges_toward_a_long_house_then_reinforcement():
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains, farmable
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    sim.world.add_construction(tribe.x, tribe.y, "wall", sim.cycle, progress=100)

    request, _ctx = sim._prepare_turn(tribe)
    assert "a long house is now worth building for real, lasting shelter" in request["prompt"]

    tribe.long_houses_built = 1
    request, _ctx = sim._prepare_turn(tribe)
    assert "it can be reinforced with" in request["prompt"]


def test_long_house_count_nudges_toward_keep_then_fortress_then_castle():
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])
    tribe = sim.tribes["tribe_0"]
    tribe.era = "monolithic_era"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    sim.world.add_construction(tribe.x, tribe.y, "wall", sim.cycle, progress=100)

    tribe.long_houses_built = config.KEEP_LONG_HOUSES_REQUIRED
    request, _ctx = sim._prepare_turn(tribe)
    assert "a keep is now worth building" in request["prompt"]

    tribe.keep_built = True
    tribe.long_houses_built = config.FORTRESS_LONG_HOUSES_REQUIRED
    request, _ctx = sim._prepare_turn(tribe)
    assert "a fortress is now worth building" in request["prompt"]

    tribe.fortress_built = True
    tribe.long_houses_built = config.CASTLE_LONG_HOUSES_REQUIRED
    request, _ctx = sim._prepare_turn(tribe)
    assert "a castle is now worth building" in request["prompt"]


def test_torches_and_moat_nudge_once_wall_is_fully_reinforced():
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.fire_ever_built = True
    sim.world.add_construction(tribe.x, tribe.y, "wall", sim.cycle, progress=100)
    tribe.wall_layers = config.WALL_MAX_LAYERS

    request, _ctx = sim._prepare_turn(tribe)

    assert "torches now line it for free" in request["prompt"]
    assert "a moat is now available" in request["prompt"]


def test_compass_direction_matches_the_map_convention():
    from backend.simulation import _compass_direction

    assert _compass_direction(10, 0) == "east"
    assert _compass_direction(-10, 0) == "west"
    assert _compass_direction(0, 10) == "south"  # y increases southward on this map
    assert _compass_direction(0, -10) == "north"


def test_interpolated_path_covers_every_tile_between_two_points():
    from backend.simulation import _interpolated_path

    path = _interpolated_path(0, 0, 4, 0)

    assert path == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_interpolated_path_is_a_single_point_when_already_there():
    from backend.simulation import _interpolated_path

    assert _interpolated_path(5, 5, 5, 5) == [(5, 5)]


def test_advance_resource_trails_wears_a_route_to_each_locked_in_site():
    """Explicit request: "if they have found a Quarry, Mine, Stand of Trees to
    Harvest, these are collectables that must be fetched and so trails/roads
    to them should be established naturally.\""""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.quarry_site = (52, 50)

    sim._advance_resource_trails(tribe)

    assert (50, 50) in sim.world.trails
    assert (51, 50) in sim.world.trails
    assert (52, 50) in sim.world.trails
    assert sim.world.trails[(52, 50)]["owner"] == tribe.id


def test_advance_resource_trails_does_nothing_without_any_locked_in_sites():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    sim._advance_resource_trails(tribe)

    assert sim.world.trails == {}


def test_precise_rival_awareness_within_radius_gives_exact_coordinates():
    """Regression test: real data this session showed 25/25 tribe-reports with zero
    trades and zero raids, ever -- the default ~62-tile spawn distance meant tribes had
    no way to notice each other at all, broadcast or not."""
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b"},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b"},
        ]
    )
    forest = sim.tribes["tribe_0"]
    mountain = sim.tribes["tribe_1"]
    mountain.x, mountain.y = forest.x - 10, forest.y  # within RIVAL_PRECISE_AWARENESS_RADIUS

    request, _ctx = sim._prepare_turn(forest)

    assert f"Mountain Tribe is nearby at ({mountain.x},{mountain.y})" in request["prompt"]


def test_distant_rival_sighting_gives_a_direction_not_coordinates():
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b"},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b"},
        ]
    )
    forest = sim.tribes["tribe_0"]
    mountain = sim.tribes["tribe_1"]
    mountain.x, mountain.y = forest.x - 40, forest.y  # beyond precise, within distant sighting

    request, _ctx = sim._prepare_turn(forest)

    assert "distant signs of Mountain Tribe somewhere to the west" in request["prompt"]
    assert f"({mountain.x},{mountain.y})" not in request["prompt"]  # no exact coordinates from this far


def test_no_rival_awareness_beyond_the_distant_sighting_radius():
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b"},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b"},
        ]
    )
    forest = sim.tribes["tribe_0"]
    forest.x, forest.y = 0, 0
    mountain = sim.tribes["tribe_1"]
    mountain.x, mountain.y = 99, 99  # far beyond RIVAL_DISTANT_SIGHTING_RADIUS

    request, _ctx = sim._prepare_turn(forest)

    assert "Mountain Tribe" not in request["prompt"]


def test_extinct_rival_produces_no_awareness_fact():
    sim = Simulation(
        [
            {"name": "Forest Tribe", "model": "gemma2:2b"},
            {"name": "Mountain Tribe", "model": "qwen2.5:3b"},
        ]
    )
    forest = sim.tribes["tribe_0"]
    mountain = sim.tribes["tribe_1"]
    mountain.x, mountain.y = forest.x - 10, forest.y
    mountain.extinct = True

    request, _ctx = sim._prepare_turn(forest)

    assert "Mountain Tribe" not in request["prompt"]


def test_visible_taboos_show_the_most_recent_not_the_oldest():
    """Regression test: taboos accumulates for a tribe's whole lifetime, and slicing
    the first 3 meant a fact learned later (e.g. a hard-won water location) could never
    surface again once 3 earlier ones already existed."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.memory.taboos = ["oldest taboo", "second taboo", "third taboo", "newest taboo"]

    request, _ctx = sim._prepare_turn(tribe)

    assert "taboo: newest taboo" in request["prompt"]
    assert "taboo: third taboo" in request["prompt"]
    assert "taboo: second taboo" in request["prompt"]
    assert "taboo: oldest taboo" not in request["prompt"]


def test_material_surplus_is_surfaced_alongside_a_real_food_or_water_warning():
    """Regression test: real runs showed tribes starving/dehydrating while sitting on
    100+ wood or stone -- gathering more of a resource that was never the bottleneck,
    with no fact anywhere telling them the stockpile was already well past any
    near-term use."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.food = 0  # triggers the starvation warning
    tribe.wood = 132
    tribe.stone = 20  # below MATERIAL_SURPLUS_THRESHOLD -- should not be mentioned

    request, _ctx = sim._prepare_turn(tribe)

    assert "132 wood" in request["prompt"]
    assert "20 stone" not in request["prompt"]


def test_water_warning_mentions_settling_at_a_confirmed_site_as_the_real_fix():
    """Regression: a live-run complaint -- "the warnings do not mention settling as
    an alternative to low water." A tribe already chronically short on water may
    already know exactly where real water is (a confirmed site) without ever having
    relocated there; the warning used to only ever say "gather more" or "scout for
    more," never that settling at an already-known site would fix this for good."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])  # forest, not water
    tribe = sim.tribes["tribe_0"]
    tribe.water = 0  # triggers the thirst warning
    tribe.confirmed_water_sites = [(40, 37)]

    request, _ctx = sim._prepare_turn(tribe)

    assert "Settling at the confirmed water source (40,37) would fix this for good" in request["prompt"]


def test_water_warning_stops_suggesting_relocate_once_already_at_the_confirmed_site():
    """Live bug, confirmed via decision_log analysis of a real run: this used to gate
    on _is_settled_near_water, which stays False for the tribe's entire
    SETTLEMENT_STABILITY_CYCLES wait even after physically arriving at the site --
    so a starving tribe already parked on its own confirmed water tile kept getting
    told "relocate there to fix this" every cycle, and it kept RELOCATE-ing in place
    instead of switching to GATHER_FOOD, losing over half its population before
    finally pivoting. The suggestion should stop the moment arrival happens, not
    once settling officially finishes."""
    from backend import config

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.water = 0  # triggers the thirst warning
    tribe.confirmed_water_sites = [(40, 37)]  # already standing exactly on it
    tribe.cycles_since_relocate = 0  # still well short of settling -- would be
    assert tribe.cycles_since_relocate < config.SETTLEMENT_STABILITY_CYCLES
    assert sim._is_settled_near_water(tribe) is False  # the old (buggy) gate condition

    request, _ctx = sim._prepare_turn(tribe)

    assert "Settling at the confirmed water source" not in request["prompt"]


def test_water_warning_says_nothing_extra_without_a_confirmed_site():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.water = 0  # triggers the thirst warning, but no confirmed site exists yet

    request, _ctx = sim._prepare_turn(tribe)

    assert "would fix this for good" not in request["prompt"]


def test_water_warning_omits_the_settling_suggestion_once_already_settled_there():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.water = 0
    tribe.confirmed_water_sites = [(40, 37)]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES  # already settled here

    request, _ctx = sim._prepare_turn(tribe)

    assert "would fix this for good" not in request["prompt"]


def test_material_surplus_is_not_surfaced_without_a_real_survival_warning():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.wood = 132  # plenty of food/water by default -- no warning should fire

    request, _ctx = sim._prepare_turn(tribe)

    assert "well beyond any near-term building need" not in request["prompt"]


def test_storm_spawns_when_the_rare_roll_succeeds():
    sim = _bare_simulation()
    sim.tribes = {}

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_weather()

    assert sim.storm_cloud is not None


def test_storm_does_not_spawn_when_the_roll_fails():
    sim = _bare_simulation()
    sim.tribes = {}

    with mock.patch("backend.simulation.random.random", return_value=0.99):
        sim._advance_weather()

    assert sim.storm_cloud is None


def test_storm_expires_after_its_lifespan():
    sim = _bare_simulation()
    sim.tribes = {}
    sim.storm_cloud = {"x": 50, "y": 50, "heading": 0.0, "cycles_left": 1}

    with mock.patch("backend.simulation.random.uniform", return_value=0.0), \
         mock.patch("backend.simulation.random.random", return_value=0.99):  # no strike this cycle
        sim._advance_weather()

    assert sim.storm_cloud is None


def test_lightning_strike_is_cleared_every_cycle():
    sim = _bare_simulation()
    sim.tribes = {}
    sim.lightning_strike = (10, 10)  # leftover from a previous cycle
    sim.storm_cloud = None

    with mock.patch("backend.simulation.random.random", return_value=0.99):  # no new spawn either
        sim._advance_weather()

    assert sim.lightning_strike is None


def test_lightning_strike_directly_on_a_tribe_causes_population_loss():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 53, 50, "#c084fc")
    tribe.population = 8
    sim.tribes = {"tribe_0": tribe}
    sim.storm_cloud = {"x": 50, "y": 50, "heading": 0.0, "cycles_left": 5}

    with mock.patch("backend.simulation.random.uniform", return_value=0.0), \
         mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_weather()

    assert sim.lightning_strike == (53, 50)  # heading 0 + STORM_SPEED moves it exactly onto the tribe
    assert tribe.population == 7
    assert any("lightning struck the heart of camp" in e for e in tribe.history)


def test_lightning_strike_respects_immortality():
    sim = _bare_simulation()
    sim.immortality_cycles = 200
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 53, 50, "#c084fc")
    tribe.population = 8
    sim.tribes = {"tribe_0": tribe}
    sim.storm_cloud = {"x": 50, "y": 50, "heading": 0.0, "cycles_left": 5}

    with mock.patch("backend.simulation.random.uniform", return_value=0.0), \
         mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_weather()

    assert tribe.population == 8  # protected, same channel as every other hazard


def test_visible_entities_reports_a_direct_lightning_hit():
    sim = _bare_simulation()
    sim.tribes = {}
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.lightning_strike = (50, 50)

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert "lightning just struck directly at your camp" in entities


def test_visible_entities_reports_a_nearby_forest_lightning_strike():
    sim = _bare_simulation()
    sim.tribes = {}
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 80, 10, "#c084fc")  # deep in the forest band
    sim.lightning_strike = (82, 10)  # 2 tiles away -- nearby, not a direct hit

    entities, _ = sim._build_visible_entities(tribe, "forest", [], [], [])

    assert "lightning struck a tree near (82,10) -- it looks like it's burning" in entities


def test_visible_entities_ignores_a_distant_lightning_strike():
    sim = _bare_simulation()
    sim.tribes = {}
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.lightning_strike = (90, 90)

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert not any("lightning" in e for e in entities)


def test_wildlife_sighting_appears_when_roll_succeeds_in_game_rich_terrain():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])  # default spawn is forest
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True  # HUNT_DEER isn't in the pre-settlement action set

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        request, _ctx = sim._prepare_turn(tribe)

    # The species named is picked by random.choice (a separate roll from the sighting
    # chance above, and not controlled by the mocked random.random) from whichever
    # biome actually produced the sighting -- forest's own pool, here.
    assert any(f"wildlife sighting: signs of {species} nearby" in request["prompt"]
               for species in GAME_SPECIES_BY_BIOME["forest"])


def test_wildlife_sighting_absent_when_roll_fails():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]

    with mock.patch("backend.simulation.random.random", return_value=0.99):
        request, _ctx = sim._prepare_turn(tribe)

    assert "wildlife sighting" not in request["prompt"]


def test_wildlife_sighting_never_appears_where_no_game_is_within_range():
    sim = Simulation([{"name": "Ocean Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.x, tribe.y = 95, 50  # deep ocean; zero game multiplier for miles around

    with mock.patch("backend.simulation.random.random", return_value=0.0):  # would trigger if unguarded
        request, _ctx = sim._prepare_turn(tribe)

    assert "wildlife sighting" not in request["prompt"]


def test_wildlife_sighting_names_a_species_from_the_richest_nearby_biomes_pool():
    """Regression guard: the sighting used to always say 'deer' regardless of where the
    tribe actually stood, keyed only off whichever hunting action was unlocked. It
    should now reflect the biome that actually produced the sighting."""
    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True  # HUNT_DEER isn't in the pre-settlement action set

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        request, _ctx = sim._prepare_turn(tribe)

    assert any(f"wildlife sighting: signs of {species} nearby" in request["prompt"]
               for species in GAME_SPECIES_BY_BIOME["plains"])
    assert "signs of deer nearby" not in request["prompt"]


def test_unsettled_tribe_cannot_gather_wood_or_stone():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    assert tribe.cycles_since_relocate == 0  # freshly founded, hasn't settled anywhere

    _request, ctx = sim._prepare_turn(tribe)

    assert "GATHER_WOOD" not in ctx["available_actions"]
    assert "GATHER_STONE" not in ctx["available_actions"]
    assert "GATHER_WATER" in ctx["available_actions"]  # survival actions untouched


def test_settled_tribe_on_farmable_ground_can_gather_wood_and_stone():
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.has_ever_settled = True  # isolate this test from the pre-settlement gate

    _request, ctx = sim._prepare_turn(tribe)

    assert "GATHER_WOOD" in ctx["available_actions"]
    assert "GATHER_STONE" in ctx["available_actions"]


def test_settled_long_enough_but_on_unfarmable_ground_still_cannot_gather():
    from backend import config

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])  # forest, not farmable
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES + 50

    _request, ctx = sim._prepare_turn(tribe)

    assert "GATHER_WOOD" not in ctx["available_actions"]


def test_farming_and_eggs_available_once_settled_even_away_from_water():
    """Explicit correction: PLANT_CROP/GATHER_EGGS used to require the stricter
    settled_near_water gate -- "the requirement of 'real' water is bogus, this is a
    Settled gate," same general condition GATHER_WOOD/STONE already use."""
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains, not water
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    _request, ctx = sim._prepare_turn(tribe)

    assert "PLANT_CROP" in ctx["available_actions"]
    assert "GATHER_EGGS" in ctx["available_actions"]


def test_farming_and_eggs_locked_before_any_settling():
    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains, not water
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    assert tribe.cycles_since_relocate == 0  # freshly founded, not yet settled

    _request, ctx = sim._prepare_turn(tribe)

    assert "PLANT_CROP" not in ctx["available_actions"]
    assert "GATHER_EGGS" not in ctx["available_actions"]


def test_cook_food_unavailable_without_either_prerequisite():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True  # bypass the pre-settlement narrowing, unrelated gate

    _, ctx = sim._prepare_turn(tribe)

    assert "COOK_FOOD" not in ctx["available_actions"]


def test_cook_food_unavailable_with_only_one_of_the_two_prerequisites():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.hunt_ever_succeeded = True
    # fire_ever_built stays False

    _, ctx = sim._prepare_turn(tribe)

    assert "COOK_FOOD" not in ctx["available_actions"]


def test_cook_food_available_once_hunted_and_fire_built():
    """Explicit request: "if you learn to hunt successfully and you learn to build
    fire successfully, you should get the chance to learn cooking... this can
    happen early." """
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.hunt_ever_succeeded = True
    tribe.fire_ever_built = True

    request, ctx = sim._prepare_turn(tribe)

    assert "COOK_FOOD" in ctx["available_actions"]
    assert "learning to cook would make stored food go much further" in request["prompt"]


def test_cook_food_retires_once_learned():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.hunt_ever_succeeded = True
    tribe.fire_ever_built = True
    tribe.cooking_learned = True

    _, ctx = sim._prepare_turn(tribe)

    assert "COOK_FOOD" not in ctx["available_actions"]


def test_build_fire_available_before_it_is_ever_built():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True

    _, ctx = sim._prepare_turn(tribe)

    assert "BUILD_FIRE" in ctx["available_actions"]


def test_build_fire_retires_once_ever_built():
    """Explicit request: "they do not have to build_fire after they have it once.
    it should leave the action list after discovered and be known ubiquitously.\""""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.fire_ever_built = True

    _, ctx = sim._prepare_turn(tribe)

    assert "BUILD_FIRE" not in ctx["available_actions"]


def test_farming_and_eggs_available_once_settled_next_to_real_water():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    request, ctx = sim._prepare_turn(tribe)

    assert "PLANT_CROP" in ctx["available_actions"]
    assert "GATHER_EGGS" in ctx["available_actions"]
    assert "this ground could support a farm plot" in request["prompt"]
    assert "gathering their eggs here could begin one" in request["prompt"]


def test_hunting_party_nudge_names_a_confirmed_wildlife_site():
    """Explicit request: "scouts have to evolve so they can inform the hunters and
    gatherers." A confirmed wildlife-rich area used to only ever surface as a bare
    coordinate, same gap water had before its own relocate nudge -- nothing connected
    it to the hunting action that could actually use it."""
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES  # HUNTING_PARTY needs has_ever_settled
    tribe.wildlife_sites = [{"x": 60, "y": 60, "type": "Deer Stand"}]

    request, ctx = sim._prepare_turn(tribe)

    assert "HUNTING_PARTY" in ctx["available_actions"]
    assert "A deer stand was confirmed at (60,60)" in request["prompt"]


def test_no_hunting_party_nudge_before_settling():
    """HUNTING_PARTY isn't in available_actions pre-settlement (config.
    PRE_SETTLEMENT_ACTIONS), so the nudge shouldn't suggest an action the tribe
    couldn't actually take yet."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.wildlife_sites = [{"x": 60, "y": 60, "type": "Deer Stand"}]

    request, ctx = sim._prepare_turn(tribe)

    assert "HUNTING_PARTY" not in ctx["available_actions"]
    assert "was confirmed at (60,60)" not in request["prompt"]


def test_no_farming_nudge_once_a_plot_and_flock_already_exist():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.farm_plots = 1
    tribe.flock = 3

    request, _ctx = sim._prepare_turn(tribe)

    assert "this ground could support a farm plot" not in request["prompt"]
    assert "gathering their eggs here could begin one" not in request["prompt"]
    assert "1 farm plot(s) growing" in request["prompt"]
    assert "A flock of 3 is being kept" in request["prompt"]


def test_fresh_tribe_has_only_pre_settlement_actions_available():
    """Explicit request: a weak model faced with the full Stone Age action list from
    cycle one has no structural push toward the single most important early decision
    -- settling somewhere real. Before it's ever settled next to real water, its
    choices are narrowed to just enough to survive and go find a home."""
    from backend import config

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    assert tribe.has_ever_settled is False

    request, ctx = sim._prepare_turn(tribe)

    # RELOCATE is further gated behind having confirmed a real water source (see
    # test_unsettled_tribe_cannot_relocate_without_confirmed_water) -- not offered
    # yet for a brand-new tribe that hasn't scouted anything.
    assert set(ctx["available_actions"]) == set(config.PRE_SETTLEMENT_ACTIONS) - {"RELOCATE"}
    assert "HUNT_DEER" not in ctx["available_actions"]
    # BREED and RAID are explicitly never locked behind settling -- see the set
    # equality assertion above, which already accounts for both being present.
    assert "only survival and exploration actions are available" in request["prompt"]


def test_settling_near_water_permanently_unlocks_the_full_action_set():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    _request, ctx = sim._prepare_turn(tribe)

    assert tribe.has_ever_settled is True
    assert "HUNT_DEER" in ctx["available_actions"]
    assert "BREED" in ctx["available_actions"]


def test_has_ever_settled_does_not_relock_after_relocating_away_again():
    """A one-way unlock -- proving the tribe CAN settle properly shouldn't be undone
    by later choosing to move on again."""
    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.cycles_since_relocate = 0  # just relocated away, no longer currently settled

    _request, ctx = sim._prepare_turn(tribe)

    assert tribe.has_ever_settled is True
    assert "HUNT_DEER" in ctx["available_actions"]


def test_unsettled_tribe_cannot_relocate_without_confirmed_water():
    """Explicit request: "RELOCATE should not show until they find water and the
    place to settle" -- relocating with no known destination wasn't meaningfully
    different from wandering at random."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    assert tribe.confirmed_water_sites == []

    _request, ctx = sim._prepare_turn(tribe)

    assert "RELOCATE" not in ctx["available_actions"]


def test_unsettled_tribe_can_relocate_once_water_is_confirmed():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites.append((40, 37))

    _request, ctx = sim._prepare_turn(tribe)

    assert "RELOCATE" in ctx["available_actions"]


def test_settled_tribe_can_no_longer_relocate():
    """A tribe that has genuinely settled next to real water -- stable long enough to
    be gathering wood/stone and farming -- shouldn't be one whim away from uprooting
    the whole settlement. Uses the stricter settled_near_water condition, not the
    looser GATHER_WOOD gate -- see test_settled_but_not_near_water_can_still_relocate
    for why the looser one would trap a tribe permanently."""
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    _request, ctx = sim._prepare_turn(tribe)

    assert "RELOCATE" not in ctx["available_actions"]
    assert "no longer considering relocating" in _request["prompt"]


def test_settled_but_not_near_water_can_still_relocate():
    """Regression: RELOCATE used to lock out on the same looser check GATHER_WOOD
    uses (any farmable ground, long enough) -- a live run caught a tribe that settled
    on plains (farmable, but not river/lake) getting permanently stuck there once
    RELOCATE disappeared, unable to ever reach real water and actually farm. A tribe
    settled on merely-farmable, non-water ground must keep RELOCATE available so it
    can still choose to move on."""
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains, not water
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.has_ever_settled = True  # isolate this test from the pre-settlement gate

    request, ctx = sim._prepare_turn(tribe)

    assert "RELOCATE" in ctx["available_actions"]
    assert "GATHER_WOOD" in ctx["available_actions"]  # still generally settled
    assert "relocating toward confirmed water is still an option" in request["prompt"]


def test_choosing_relocate_resets_settlement_progress():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = 8

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [60, 50]}, 100.0,
                     {"biome": "plains", "available_actions": ["RELOCATE"]})

    assert tribe.cycles_since_relocate == 0


def test_relocating_to_your_own_current_tile_does_not_reset_settlement_progress():
    """Regression: choosing RELOCATE used to reset cycles_since_relocate to 0 purely
    because it was the *chosen action*, even when target_vector pointed at the tribe's
    own current tile and terrain_aware_step was a genuine no-op. A model that keeps
    re-issuing RELOCATE toward an already-reached confirmed water site (the fact
    recommending it stays true after arrival) could never accumulate any settlement
    progress at all, standing right on the water forever."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = 8

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [40, 37]}, 100.0,
                     {"biome": "river", "available_actions": ["RELOCATE"]})

    assert (tribe.x, tribe.y) == (40, 37)  # genuinely didn't move
    assert tribe.cycles_since_relocate == 9  # advanced, same as any other non-moving action


def test_choosing_a_non_relocate_action_advances_settlement_progress():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = 3

    sim._apply_turn(tribe, {"visual_action": "GATHER_FOOD"}, 100.0, {"biome": "plains", "available_actions": ["GATHER_FOOD"]})

    assert tribe.cycles_since_relocate == 4


def test_apply_turn_records_the_full_rationale_not_a_60_char_fragment():
    """Regression: the chronicle/sidebar history line used to hard-cut a chief's
    metacognitive_rationale at 60 characters with no ellipsis, silently chopping it off
    mid-word most of the time -- a live-run complaint ('thinking output is cut off')
    even though the prompt already asks the model for brief reasoning."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    rationale = (
        "Water reserves are healthy for now, but food is running low and no one is "
        "out gathering, so the priority this cycle is securing a fresh food source."
    )
    assert len(rationale) > 60  # the old cap would have mangled this

    sim._apply_turn(
        tribe,
        {"visual_action": "GATHER_FOOD", "metacognitive_rationale": rationale},
        100.0,
        {"biome": "plains", "available_actions": ["GATHER_FOOD"]},
    )

    assert rationale in tribe.history[-1]


def test_apply_turn_still_caps_an_extremely_long_rationale():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    rationale = "word " * 100  # 500 chars, well past any real "brief" reasoning

    sim._apply_turn(
        tribe,
        {"visual_action": "GATHER_FOOD", "metacognitive_rationale": rationale},
        100.0,
        {"biome": "plains", "available_actions": ["GATHER_FOOD"]},
    )

    assert tribe.history[-1].endswith("…")
    assert len(tribe.history[-1]) < len(rationale)


def test_unsettled_fact_reports_real_progress():
    from backend import config

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = 4

    request, _ctx = sim._prepare_turn(tribe)

    assert f"4/{config.SETTLEMENT_STABILITY_CYCLES}" in request["prompt"]


def test_era_progress_fact_names_the_specific_shortfalls():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.era = "cognitive_horizon"  # so the gap fact names Tribal Synapse specifically
    tribe.population = 15
    tribe.water = 10
    tribe.stone = 50  # already meets the tribal_synapse stone requirement
    tribe.wood = 5

    request, _ctx = sim._prepare_turn(tribe)

    assert "To reach Tribal Synapse, still short on:" in request["prompt"]
    assert "population 15/20" in request["prompt"]
    assert "water 10/40" in request["prompt"]
    assert "wood 5/40" in request["prompt"]
    assert "stone" not in request["prompt"].split("To reach Tribal Synapse, still short on:")[1].split(".")[0]


def test_era_progress_fact_absent_once_the_next_era_is_fully_met():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.population = 20
    tribe.water = 40
    tribe.stone = 40
    tribe.wood = 40

    request, _ctx = sim._prepare_turn(tribe)

    assert "To reach Cognitive Horizon" not in request["prompt"]


def test_translation_matrix_is_updated_on_apply_turn():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    tribe_a = sim.tribes["tribe_0"]
    tribe_b = sim.tribes["tribe_1"]
    ctx = {"biome": "forest", "available_actions": ["BUILD_FIRE"]}

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


def test_gather_water_on_a_lake_matches_river_yield_with_no_drowning_risk():
    from backend import config

    sim = _bare_simulation()
    lake_tribe = Tribe("tribe_0", "Lake Tribe", "gemma2:2b", 50, 50, "#60a5fa")

    # A drowning roll that would definitely fire on a river must not fire on a lake.
    with mock.patch("backend.actions.random.random", return_value=0.0):
        note = sim._apply_action(lake_tribe, "GATHER_WATER", "lake", (0, 0))

    assert note is None  # no drowning note
    assert lake_tribe.water > config.STARTING_WATER  # river-level yield, not the off-water rate


def test_action_outside_current_era_falls_back_to_a_real_available_action():
    """A real, well-formed action name that's just not unlocked yet (wrong era) is a
    legitimate "can't do that here" case, not a parse failure -- IDLE's removal means
    this now falls back to a genuinely available action (the turn still does
    something real) rather than a no-op, and still gets no confusion nudge since the
    tribe wasn't actually confused."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    assert tribe.era == "primitive_dawn"

    request, ctx = sim._prepare_turn(tribe)
    assert "CONSTRUCT_WALL" not in ctx["available_actions"]

    sim._apply_turn(tribe, {"visual_action": "CONSTRUCT_WALL"}, 50.0, ctx)
    assert ctx["available_actions"][0] in tribe.history[-1]
    assert tribe.last_confusion is None


def test_idle_does_not_exist_anywhere_in_the_action_system():
    """Explicit request: "IDLE needs to be removed altogether, we should never need
    this." Not just absent from what a tribe is offered (that was already true) --
    ACTION_REGISTRY itself no longer has an IDLE entry at all, and _resolve_action
    can no longer return it under any circumstance."""
    from backend import config
    from backend.actions import ACTION_REGISTRY
    from backend.eras import ERAS

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    _, ctx = sim._prepare_turn(tribe)
    assert "IDLE" not in ctx["available_actions"]
    assert "IDLE" not in config.PRE_SETTLEMENT_ACTIONS
    assert "IDLE" not in ACTION_REGISTRY
    for era in ERAS:
        assert "IDLE" not in era.unlocks_actions


def test_resolve_action_exact_and_normalized_and_out_of_context_cases():
    avail = ["GATHER_FOOD", "GATHER_WATER", "SCOUT", "RELOCATE"]
    assert _resolve_action("GATHER_FOOD", avail) == ("GATHER_FOOD", None)
    assert _resolve_action("gather-food", avail) == ("GATHER_FOOD", None)
    assert _resolve_action("gather food", avail) == ("GATHER_FOOD", None)
    # A real, globally-known action that's simply not in this tribe's current
    # available_actions (wrong era/context) -- not a parse failure. Falls back to
    # the first available action rather than a no-op (see IDLE's removal).
    assert _resolve_action("PLANT_CROP", avail) == (avail[0], None)
    # Nothing recognizable at all -- a genuine confusion case. Still falls back to a
    # real action (no fuzzy guess matches this gibberish, so the first available
    # one), but the raw text is preserved so a correction nudge can fire.
    action, unresolved = _resolve_action("xyzzy nonsense", avail)
    assert action == avail[0]
    assert unresolved == "xyzzy nonsense"


def test_guess_intended_action_is_display_only_and_best_effort():
    avail = ["GATHER_FOOD", "GATHER_WATER", "SCOUT", "RELOCATE", "CATCH_FISH"]
    assert _guess_intended_action("catch some fish", avail) == "CATCH_FISH"
    assert _guess_intended_action("xyzzy nonsense", avail) is None


def test_gibberish_action_text_falls_back_to_a_real_action_and_records_confusion():
    """A decision that matches nothing real at all -- not even a formatting variant
    of a real action -- is a genuine parse failure. It still resolves to a real,
    currently-available action (never a no-op, see IDLE's removal) -- here, the
    loose display-only fuzzy guess actually lands on GATHER_WATER given the raw
    text's own wording -- and is recorded distinctly from a deliberate choice so
    _prepare_turn can surface a correction fact next cycle."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    ctx = {"biome": "plains", "available_actions": ["GATHER_FOOD", "GATHER_WATER", "SCOUT", "RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "PONDER THE MEANING OF WATER"}, 50.0, ctx)

    assert tribe.last_action == "GATHER_WATER"
    assert tribe.last_confusion is not None
    assert tribe.last_confusion["raw"] == "PONDER THE MEANING OF WATER"
    assert "unrecognized decision text" in tribe.history[-1]


def test_case_and_spacing_variants_resolve_cleanly_without_confusion():
    """Cheap normalization (case, spaces/hyphens for underscores) should recover the
    intended action with no correction nudge needed -- these aren't confusion, just
    formatting noise."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    ctx = {"biome": "plains", "available_actions": ["GATHER_FOOD", "GATHER_WATER", "SCOUT", "RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "gather-food"}, 50.0, ctx)

    assert tribe.last_action == "GATHER_FOOD"
    assert tribe.last_confusion is None


def test_confusion_nudge_appears_once_then_clears():
    """The 'Instant Enlightenment' correction fact should show up in the very next
    cycle's facts after a genuine parse failure, then not repeat once addressed."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.last_confusion = {"raw": "PONDER THE MEANING OF WATER", "guess": None, "fallback": "GATHER_FOOD"}

    request, ctx = sim._prepare_turn(tribe)

    assert "PONDER THE MEANING OF WATER" in request["prompt"]
    assert "did not match any valid action" in request["prompt"]
    assert "GATHER_FOOD was taken instead" in request["prompt"]
    assert tribe.last_confusion is None  # cleared after being surfaced once

    request2, _ = sim._prepare_turn(tribe)
    assert "PONDER" not in request2["prompt"]  # doesn't repeat on the following cycle


def test_apply_turn_records_last_target_only_for_relocate():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    ctx = {"biome": "forest", "available_actions": ["RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [20, 30]}, 10.0, ctx)

    assert tribe.last_target == [20, 30]


def test_apply_turn_does_not_record_last_target_for_non_relocate_actions():
    """Only RELOCATE actually moves the tribe; other actions' target_vector shouldn't
    create a phantom "journey" reminder for a trip the tribe never intended to take."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    ctx = {"biome": "forest", "available_actions": ["GATHER_FOOD"]}

    sim._apply_turn(tribe, {"visual_action": "GATHER_FOOD", "target_vector": [20, 30]}, 10.0, ctx)

    assert tribe.last_target is None


def test_relocate_corrects_a_self_targeting_no_op_toward_confirmed_water():
    """Live bug, confirmed via last_decision_target on a real run: llama3.2:1b kept
    submitting its own current position as target_vector for RELOCATE, every single
    cycle, despite a confirmed water site being named explicitly in its prompt --
    a guaranteed no-op that left it standing still while it starved to death. Same
    class of small-model target-parameter failure as SCOUT's; RELOCATE should now
    substitute the real confirmed site when the model's target is this degenerate."""
    sim = Simulation([{"name": "A", "model": "llama3.2:1b", "x": 80, "y": 38}])
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites = [(83, 58)]
    tribe.food = 100
    tribe.water = 100
    ctx = {"biome": "forest", "available_actions": ["RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [80, 38]}, 10.0, ctx)

    assert (tribe.x, tribe.y) != (80, 38)  # actually moved, toward the real site
    assert tribe.last_target != [80, 38]
    assert tribe.last_decision_target == [80, 38]  # raw model output preserved for analysis


def test_relocate_leaves_a_genuine_self_target_alone_once_already_at_the_site():
    """A tribe that has actually arrived at its confirmed water site legitimately
    submits its own position as target_vector -- that's not the bug, and shouldn't
    get redirected anywhere."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 83, "y": 58}])
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites = [(83, 58)]
    tribe.food = 100
    tribe.water = 100
    ctx = {"biome": "forest", "available_actions": ["RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [83, 58]}, 10.0, ctx)

    assert (tribe.x, tribe.y) == (83, 58)
    assert tribe.last_target == [83, 58]


def test_relocate_leaves_a_real_different_target_alone():
    """A model that successfully produces a real, different target shouldn't be
    second-guessed -- the correction only fires for the specific degenerate case."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 80, "y": 38}])
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites = [(83, 58)]
    tribe.food = 100
    tribe.water = 100
    ctx = {"biome": "forest", "available_actions": ["RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [90, 90]}, 10.0, ctx)

    assert tribe.last_target == [90, 90]


def test_settlement_ground_ok_true_on_farmable_biome_or_near_confirmed_water():
    sim = Simulation([{"name": "Mountain Tribe", "model": "gemma2:2b", "x": 5, "y": 55}])
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites = [(19, 62)]

    assert sim._settlement_ground_ok(tribe, 19, 62) is True  # exactly on the site
    assert sim._settlement_ground_ok(tribe, 20, 61) is True  # a tile away, still in radius
    assert sim._settlement_ground_ok(tribe, 90, 90) is False  # nowhere near it, not farmable


def test_apply_turn_does_not_reset_relocate_clock_when_still_within_qualifying_territory():
    """Live bug: a tribe with a confirmed water site kept RELOCATE-jittering between
    nearby tiles that all already satisfied _is_settled's ground check --
    (19,62) -> (20,61) -> (20,62) -- and the old "any position change resets the
    clock" rule meant cycles_since_relocate restarted at 0 every single hop, so the
    tribe could never accumulate enough cycles to actually settle despite already
    standing on qualifying ground the whole time."""
    sim = Simulation([{"name": "Mountain Tribe", "model": "gemma2:2b", "x": 19, "y": 62}])
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites = [(19, 62)]
    tribe.cycles_since_relocate = 5
    tribe.food = 100
    tribe.water = 100
    ctx = {"biome": "mountains", "available_actions": ["RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [20, 61]}, 10.0, ctx)

    assert (tribe.x, tribe.y) == (20, 61)  # confirms the move actually happened
    assert tribe.cycles_since_relocate == 6  # kept climbing instead of resetting to 0


def test_apply_turn_still_resets_relocate_clock_when_leaving_qualifying_territory():
    """A single RELOCATE step (speed 4) starting exactly on a confirmed water site can
    never actually leave its radius-6 territory in one hop -- so this places the tribe
    near the edge of the territory instead, confirming a hop that crosses out of the
    radius still resets the clock as before. Column x=5 stays real mountains (not
    farmable) through y=59, same terrain used by the existing settlement-radius tests
    above, so biome alone can't accidentally satisfy _settlement_ground_ok here."""
    sim = Simulation([{"name": "Mountain Tribe", "model": "gemma2:2b", "x": 5, "y": 55}])
    tribe = sim.tribes["tribe_0"]
    tribe.confirmed_water_sites = [(5, 50)]  # distance 5 from tribe -- within radius 6
    tribe.cycles_since_relocate = 5
    tribe.food = 100
    tribe.water = 100
    ctx = {"biome": "mountains", "available_actions": ["RELOCATE"]}

    sim._apply_turn(tribe, {"visual_action": "RELOCATE", "target_vector": [5, 99]}, 10.0, ctx)

    assert tribe.y > 55  # confirms the move actually happened, away from the site
    assert sim._settlement_ground_ok(tribe) is False  # now outside the radius, still not farmable
    assert tribe.cycles_since_relocate == 0


def test_apply_turn_records_last_decision_target_for_every_action():
    """Unlike last_target, last_decision_target exists purely for decision_log.py's
    offline analysis and should capture the submitted target_vector regardless of
    which action was chosen."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    ctx = {"biome": "forest", "available_actions": ["GATHER_FOOD"]}

    sim._apply_turn(tribe, {"visual_action": "GATHER_FOOD", "target_vector": [20, 30]}, 10.0, ctx)

    assert tribe.last_decision_target == [20, 30]


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
    tribe.expeditions = [{
        "pos": [tribe.x, tribe.y], "origin": [tribe.x, tribe.y], "target": [tribe.x + 10, tribe.y],
        "day": 1, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    request, _ctx = sim._prepare_turn(tribe)

    assert "Still in the field" in request["prompt"]


def test_expedition_in_the_field_names_its_actual_target_coordinate():
    """Bug report: "2 scouts going same direction still." The field report
    used to name who was out and what day/phase they were on, but never where
    they were actually headed -- a second SCOUT call had no way to tell it
    would just cover the same ground again."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.expeditions = [{
        "pos": [tribe.x, tribe.y], "origin": [tribe.x, tribe.y], "target": [70, 30],
        "day": 1, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    request, _ctx = sim._prepare_turn(tribe)

    assert "headed toward (70,30)" in request["prompt"]
    assert "Test Scout" in request["prompt"]
    assert "You could send out 1 more at once" in request["prompt"]
    assert "day 1" in request["prompt"]


def test_expedition_succeeds_immediately_on_reaching_real_river_water():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 40, 30, "#c084fc")
    # From (40, 30) toward (40, 37), one EXPEDITION_SPEED (10) step lands at (40, 37),
    # which world.py's river geography places on the river -- verified directly against
    # biome_at before trusting it, same discipline as world.py's nearest_water fix.
    tribe.expeditions = [{
        "pos": [40, 30], "origin": [40, 30], "target": [40, 37],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["found"] == [40, 37]
    assert any("found fresh water" in entry for entry in tribe.history)


def test_expedition_succeeds_on_reaching_the_lake_same_as_the_river():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 25, 55, "#c084fc")
    # From (25, 55) toward (25, 65) -- the lake center -- one EXPEDITION_SPEED (10)
    # step lands exactly there, verified directly against biome_at first.
    tribe.expeditions = [{
        "pos": [25, 55], "origin": [25, 55], "target": [25, 65],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["found"] == [25, 65]
    assert any("found fresh water" in entry for entry in tribe.history)


def test_expedition_senses_nearby_water_without_stepping_onto_it():
    """Regression: a scout used to have to land on the exact water tile to report
    anything, so a party could pass within a tile or two of a lake and come home
    empty-handed -- unrealistic (running water carries; a lake is visible from its
    shore) and the source of a live-run complaint ('missed water by a hair'). A step
    that lands within WATER_SENSING_RADIUS of water, but not on it, should now count
    as a find -- and, since the party never actually touched the water, it should
    carry none of the on-tile drowning risk."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 25, 44, "#c084fc")
    tribe.expeditions = [{
        "pos": [25, 44], "origin": [25, 44], "target": [25, 65],  # the lake, far off yet
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.0):  # would drown if on-tile
        sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["pos"] == [25, 54]  # landed short of the lake itself
    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["found"] == [27, 54]  # the actual water tile it sensed, not its own position
    assert tribe.population == 8  # no drowning -- never touched the water
    assert any("hears water nearby" in entry for entry in tribe.history)


def test_settled_tribe_scouts_no_longer_turn_back_for_more_water():
    """Explicit request: "the find water scouting needs to be removed from
    available actions after they Settle. The scouts can still explore and
    report sightings and discoveries." Once already settled near water,
    sensing more water nearby should no longer cut a scouting trip short --
    the same step should instead fall through to the ordinary
    arrived-at-target/push-onward handling, the same as if no water were
    nearby at all."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 25, 44, "#c084fc")
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.confirmed_water_sites = [(25, 44)]  # already settled near water, right here
    tribe.expeditions = [{
        "pos": [25, 44], "origin": [25, 44], "target": [25, 65],  # the lake, far off yet
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "outbound"  # kept going, not turned back for water
    assert tribe.expeditions[0]["found"] is None
    assert not any("hears water nearby" in entry for entry in tribe.history)


def test_expedition_can_drown_reaching_river_water_but_still_reports_the_find():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 40, 30, "#c084fc")
    tribe.population = 10
    tribe.expeditions = [{
        "pos": [40, 30], "origin": [40, 30], "target": [40, 37],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_expeditions(tribe)

    assert tribe.population == 9
    assert tribe.expeditions[0]["found"] == [40, 37]  # the crossing still pays off despite the loss
    assert any("pulled someone under" in entry for entry in tribe.history)
    assert any("drowned one of our own" in m["text"] for m in tribe.memory.entries)


def test_expedition_senses_water_crossed_mid_step_even_if_the_landing_tile_misses_it():
    """Bug report: "clearly they see water, the scout walked right through
    it." A single EXPEDITION_SPEED step (up to 10 tiles) can be wider than
    WATER_SENSING_RADIUS (6) -- sensing only at the final landing tile let a
    party leap clean over water narrower than the step itself. Mocks
    terrain_aware_step directly so this doesn't depend on exactly where the
    procedural river happens to run: (40,25) and (40,50) are both real,
    confirmed-dry tiles more than WATER_SENSING_RADIUS from the river, but the
    straight line between them crosses it (see backend.world._river_center_y)
    -- before this fix, only the destination (40,50) would ever be checked,
    and this trip would have come home with nothing found."""
    from backend.world import biome_at

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 40, 25, "#c084fc")
    tribe.expeditions = [{
        "pos": [40, 25], "origin": [40, 25], "target": [40, 60],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    with mock.patch("backend.physics.terrain_aware_step", return_value=(40, 50)):
        sim._advance_expeditions(tribe)

    found = tribe.expeditions[0]["found"]
    assert found is not None
    assert biome_at(*found) in ("river", "lake")


def test_expedition_reaching_its_target_without_water_pushes_onward_if_days_remain():
    """Regression test: a model's own target_vector is usually close (one
    EXPEDITION_SPEED step away), so treating "arrived at the declared spot" as "search
    over" meant max_days and the scout's determination trait almost never actually
    mattered -- live runs showed parties turning back on day one nearly every time.
    Reaching a non-water target with days left should extend the search outward along
    the same heading, not end it."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 60, 10, "#c084fc")
    tribe.expeditions = [{
        # (60,10) is well clear of the river/lake -- WATER_SENSING_RADIUS must not
        # fire here, or this would test the wrong mechanic.
        "pos": [60, 10], "origin": [60, 10], "target": [66, 10],  # one step away, not water
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "outbound"  # not turned back
    assert tribe.expeditions[0]["terrain_report"] is not None  # still noted what's there
    assert tribe.expeditions[0]["found"] is None
    assert tribe.expeditions[0]["target"] != [66, 10]  # extended past the original spot
    assert any("pushes onward" in entry for entry in tribe.history)


def test_expedition_does_not_give_up_from_day_count_alone():
    """Regression test: an arbitrary day-count cutoff used to end a search regardless
    of whether the party still had somewhere left to look. Elapsed days, alone,
    should never end an outbound search anymore -- only running out of world to
    search does."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 60, 10, "#c084fc")
    tribe.expeditions = [{
        # Already pushed onward once (terrain_report set) and nowhere near the
        # (extended) target yet -- a huge day count should still change nothing.
        # (60,10) is well clear of the river/lake -- WATER_SENSING_RADIUS must not
        # fire here, or this would test the wrong mechanic.
        "pos": [60, 10], "origin": [60, 10], "target": [99, 10],
        "day": 50, "phase": "outbound", "found": None, "terrain_report": "plains",
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_one_expedition(tribe, tribe.expeditions[0])

    assert tribe.expeditions[0]["phase"] == "outbound"


def test_build_road_speeds_up_expeditions():
    """See actions.py._build_road -- a flat, always-on version of the same trail
    bonus a well-worn path already grants, since a deliberately-built road doesn't
    need to wear in from repeated travel."""
    import math

    def _expedition(tribe):
        return {
            "pos": [tribe.x, tribe.y], "origin": [tribe.x, tribe.y], "target": [tribe.x + 40, tribe.y],
            "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
            "food_gathered": 0, "water_gathered": 0,
            "lead_scout": "Test Scout", "determination": 0.5, "max_days": 10, "path": [[tribe.x, tribe.y]],
        }

    sim = _bare_simulation()
    plain_tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 60, 10, "#c084fc")
    exp = _expedition(plain_tribe)
    sim._advance_one_expedition(plain_tribe, exp)
    plain_distance = math.hypot(exp["pos"][0] - 60, exp["pos"][1] - 10)

    sim2 = _bare_simulation()
    road_tribe = Tribe("tribe_1", "Road Tribe", "gemma2:2b", 60, 10, "#fb923c")
    road_tribe.road_built = True
    exp2 = _expedition(road_tribe)
    sim2._advance_one_expedition(road_tribe, exp2)
    road_distance = math.hypot(exp2["pos"][0] - 60, exp2["pos"][1] - 10)

    assert road_distance > plain_distance


def test_expedition_gives_up_when_physically_boxed_in_by_ocean():
    """Regression: a live run caught a scouting party stuck at the same tile for 400+
    days. physics.terrain_aware_step falls back to "stay put" when every candidate step
    toward the target is ocean (boxed in on every axis) -- a pushed-onward target
    (extend_ray_to_grid_edge) can land past the actual coastline in open water, and the
    old logic only ever gave up on reaching the target exactly, which an unreachable
    target never does. Being physically unable to advance at all must count as
    "nowhere left to search," same as reaching the grid's literal edge."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 90, 84, "#c084fc")
    tribe.expeditions = [{
        # (90, 84) is coastal; every candidate step toward (99, 84) is ocean, verified
        # directly against physics.terrain_aware_step first.
        "pos": [90, 84], "origin": [50, 84], "target": [99, 84],
        "day": 5, "phase": "outbound", "found": None, "terrain_report": "forest",
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_one_expedition(tribe, tribe.expeditions[0])

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["pos"] == [90, 84]  # didn't silently teleport anywhere
    assert any("can go no further" in entry for entry in tribe.history)


def test_expedition_gives_up_upon_reaching_the_edge_of_the_world():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expeditions = [{
        # Already pushed onward once (terrain_report set); one EXPEDITION_SPEED (10)
        # step from here lands exactly on the extended target -- the edge itself.
        # South (increasing y) rather than east: x=99 is past OCEAN_X_START and would
        # be deflected by the impassable-ocean physics, never actually arriving.
        "pos": [50, 89], "origin": [50, 50], "target": [50, 99],
        "day": 4, "phase": "outbound", "found": None, "terrain_report": "plains",
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_one_expedition(tribe, tribe.expeditions[0])

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["found"] is None
    assert any("reaches the edge of explored land" in entry for entry in tribe.history)


def test_expedition_arrival_home_delivers_water_finding_to_memory_and_clears_state():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions == []
    assert any("(40,37)" in entry for entry in tribe.history)
    assert any("fresh water at (40,37)" in m["text"] for m in tribe.memory.entries)


def test_water_discovery_does_not_stack_a_second_celebration_the_same_cycle():
    """Explicit question: "are celebrations stomping on each other?" Yes, in one real
    case -- _celebrate_water_discovery/_celebrate_game_discovery run from
    _advance_expeditions, a separate, later loop than the one _celebrate_settling/
    _advance_farming's harvest celebration run from, so nothing previously stopped a
    second celebration (spending food, resetting last_celebration_cycle again) from
    firing in the exact same cycle as one that already happened."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 100
    tribe.last_celebration_cycle = sim.cycle  # a celebration already fired this exact cycle
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    # Only the ordinary arrival-home food delivery -- no extra feast cost stacked on top.
    assert tribe.food == 100 + config.EXPEDITION_RETURN_DAILY_FOOD
    assert not any("celebrates the discovery of water" in entry for entry in tribe.history)
    # The underlying discovery itself still lands -- only the party is suppressed.
    assert tribe.confirmed_water_sites == [(40, 37)]


def test_water_bringer_trophy_credits_the_scout_who_confirmed_it_not_the_chief():
    """Explicit request: crediting Water Bringer only ever to the chief (the original
    design) meant a young tribe could go a long time with just one named individual,
    leaving _eligible_breeding_pair permanently empty until a much higher-threshold
    trophy (Master Pathfinder/Master Hunter) came in. The scout who actually found the
    water is the one credited here -- _check_chief_trophies' separate river/lake-
    standing case still credits the chief, a genuinely different circumstance."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert any(t["name"] == "Water Bringer" and t["chief"] == "Test Scout" for t in tribe.trophies)


def test_expedition_arrival_delivers_foraged_food_and_water_to_the_tribe():
    """The trip isn't a pure resource black hole -- a traveling party forages and
    hunts along the way, more on the outbound leg than the hurried trip home, and
    brings it back regardless of whether the search itself succeeded."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food, tribe.water = 10, 10
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 7, "water_gathered": 5,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    # Confirming water for the first time now throws a celebration (explicit request)
    # -- a real food cost on top of the delivered/foraged amounts, not a separate
    # confound to work around.
    delivered_food = 10 + 7 + config.EXPEDITION_RETURN_DAILY_FOOD
    spent_on_celebration = round(delivered_food * config.CELEBRATION_RESOURCE_COST_FRACTION)
    expected_food = delivered_food - spent_on_celebration
    expected_water = 10 + 5 + config.EXPEDITION_RETURN_DAILY_WATER
    assert tribe.food == expected_food
    assert tribe.water == expected_water
    assert any("celebrates the discovery of water" in entry for entry in tribe.history)


def test_expedition_report_is_attributed_to_the_chief_when_one_exists():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert any("Chief Ashgar" in entry for entry in tribe.history)


def test_expedition_report_falls_back_to_the_tribe_when_there_is_no_chief():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert any("gives the tribe a full report" in entry for entry in tribe.history)


def test_expedition_arrival_home_empty_handed_clears_state_without_a_water_memory():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [99, 99],
        "day": 3, "phase": "returning", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions == []
    assert any("empty-handed" in entry for entry in tribe.history)


def test_hunting_party_catches_something_and_heads_home():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 80, 38, "#c084fc")  # forest, game=1.0
    tribe.expeditions = [{
        "kind": "hunt", "pos": [80, 38], "origin": [80, 38], "target": [80, 38],
        "day": 0, "phase": "outbound", "food_caught": 0,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", side_effect=[0.99, 0.0]):  # miss hazard, hit catch
        sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["food_caught"] > 0
    assert any("made a catch" in entry for entry in tribe.history)


def test_hunting_party_hazard_ends_the_hunt_and_costs_population():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 80, 38, "#c084fc")
    tribe.population = 10
    tribe.expeditions = [{
        "kind": "hunt", "pos": [80, 38], "origin": [80, 38], "target": [80, 38],
        "day": 0, "phase": "outbound", "food_caught": 0,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.0):  # hazard roll checked first
        sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["food_caught"] == 0
    assert tribe.population == 9
    assert any("wolf pack struck" in entry for entry in tribe.history)


def test_expedition_raider_ambush_ends_the_trip_and_costs_population():
    """Explicit request: "It would be interesting to see a Scout encounter a RAIDER
    group" -- a real, in-the-field ambush distinct from the settlement-level attack
    and the report-based sighting roll."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.has_ever_settled = True
    tribe.population = 10
    exp = {"lead_scout": "Test Scout"}

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        ambushed = sim._expedition_raider_ambush(tribe, exp, 60, 60)

    assert ambushed is True
    assert tribe.population == 9
    assert tribe.raider_sightings == [(60, 60)]
    assert any("ambushed by raiders" in entry for entry in tribe.history)
    assert "DREAD" in sim.trauma.bias_string(60, 60)
    assert sim.recent_encounters and sim.recent_encounters[0]["label"] == "Scouts ambushed"


def test_expedition_raider_ambush_never_fires_before_has_ever_settled():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.has_ever_settled = False
    exp = {"lead_scout": "Test Scout"}

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        ambushed = sim._expedition_raider_ambush(tribe, exp, 60, 60)

    assert ambushed is False
    assert tribe.raider_sightings == []


def test_outbound_expedition_flees_home_immediately_when_ambushed():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.has_ever_settled = True
    tribe.expeditions = [{
        "kind": "hunt", "pos": [50, 50], "origin": [50, 50], "target": [60, 60],
        "day": 0, "phase": "outbound", "food_caught": 0,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert any("ambushed by raiders" in entry for entry in tribe.history)


def test_hunting_party_drowns_if_its_daily_step_lands_on_a_river_tile():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 40, 30, "#c084fc")
    tribe.population = 10
    tribe.expeditions = [{
        "kind": "hunt", "pos": [40, 30], "origin": [40, 30], "target": [40, 37],
        "day": 0, "phase": "outbound", "food_caught": 0,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_expeditions(tribe)

    assert tribe.population == 9
    assert tribe.expeditions[0]["phase"] == "returning"
    assert any("pulled someone under" in entry for entry in tribe.history)


def test_hunting_party_does_not_give_up_from_day_count_alone():
    """Regression test: an arbitrary day-count cutoff used to end a hunt regardless of
    whether there was still ground worth covering. Elapsed days, alone, should never
    end an outbound hunt anymore -- only running out of world to search does."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 80, 38, "#c084fc")
    tribe.expeditions = [{
        # Already pushed onward once and nowhere near the (extended) target yet -- a
        # huge day count should still change nothing.
        "kind": "hunt", "pos": [80, 38], "origin": [80, 38], "target": [99, 38],
        "day": 50, "phase": "outbound", "food_caught": 0, "pushed_onward": True,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.99):  # no hazard, no catch
        sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "outbound"


def test_hunting_party_gives_up_when_physically_boxed_in_by_ocean():
    """Regression: the ocean-boxed-in stuck-forever fix (see the scout version of this
    test) originally only lived in the scout branch -- a hunting party's own push-onward
    target can land in open water exactly the same way, and hunting parties are
    dispatched to a branch that returned before ever reaching that check."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 90, 84, "#c084fc")
    tribe.expeditions = [{
        "kind": "hunt", "pos": [90, 84], "origin": [50, 84], "target": [99, 84],
        "day": 5, "phase": "outbound", "food_caught": 0, "pushed_onward": True,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.99):  # no hazard, no catch
        sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["pos"] == [90, 84]
    assert any("can go no further" in entry for entry in tribe.history)


def test_hunting_party_gives_up_upon_reaching_the_edge_of_the_hunting_grounds():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 80, 38, "#c084fc")
    tribe.expeditions = [{
        # Already pushed onward once; one day's step (forest's 0.8x terrain multiplier
        # applies to EXPEDITION_SPEED here, so 8 tiles not 10) lands exactly on the
        # extended target -- the edge itself.
        "kind": "hunt", "pos": [80, 38], "origin": [80, 38], "target": [88, 38],
        "day": 5, "phase": "outbound", "food_caught": 0, "pushed_onward": True,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.99):  # no hazard, no catch
        sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["food_caught"] == 0
    assert any("reaches the edge of the hunting grounds" in entry for entry in tribe.history)


def test_hunting_party_arrival_home_delivers_caught_food_and_clears_state():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10
    tribe.expeditions = [{
        "kind": "hunt", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
        "day": 2, "phase": "returning", "food_caught": 25,
        "food_gathered": 7, "water_gathered": 5,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    sim._advance_expeditions(tribe)

    expected_food = 10 + 25 + 7 + config.EXPEDITION_RETURN_DAILY_FOOD
    assert tribe.food == expected_food
    assert tribe.expeditions == []
    assert any("25 food caught" in entry for entry in tribe.history)


def test_hunting_party_arrival_home_empty_handed_still_delivers_forage_and_clears_state():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10
    tribe.expeditions = [{
        "kind": "hunt", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
        "day": 4, "phase": "returning", "food_caught": 0,
        "food_gathered": 7, "water_gathered": 5,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions == []
    assert any("nothing caught, though not empty-handed" in entry for entry in tribe.history)


def test_trade_emissary_finds_a_partner_and_trades_immediately():
    """Explicit request: build SEND_TRADE_EMISSARY like HUNTING_PARTY -- finding a
    rival executes the exchange at the point of contact, not once the emissary
    walks home."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 100
    partner = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    partner.wood = 200
    sim.tribes = {"tribe_0": tribe, "tribe_1": partner}
    tribe.expeditions = [{
        # target == pos/origin so the day's movement is a no-op -- isolates the
        # partner-find check from needing to model movement math, same convention
        # the wolf-hazard/catch-chance tests already use for HUNTING_PARTY.
        "kind": "trade", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
        "day": 0, "phase": "outbound", "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Emissary", "determination": 0.5, "max_days": 4, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert tribe.expeditions[0]["trade_partner"] == "Mountain Tribe"
    assert tribe.wood != 100 and partner.wood != 200  # goods already moved, not waiting on the walk home
    assert any("finds Mountain Tribe and opens trade" in entry for entry in tribe.history)


def test_trade_emissary_pushes_onward_then_gives_up_with_no_partner():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.expeditions = [{
        # Already pushed onward once, with the (extended) target already reached
        # (pos == target, well within a single day's reach on plains, not ocean) --
        # this cycle should give up for good rather than push onward a second time.
        "kind": "trade", "pos": [52, 50], "origin": [50, 50], "target": [52, 50],
        "day": 5, "phase": "outbound", "food_gathered": 0, "water_gathered": 0,
        "pushed_onward": True,
        "lead_scout": "Test Emissary", "determination": 0.5, "max_days": 4, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["phase"] == "returning"
    assert any("no one to trade with" in entry for entry in tribe.history)


def test_trade_emissary_arrival_home_reports_success():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10
    tribe.expeditions = [{
        "kind": "trade", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
        "day": 4, "phase": "returning", "food_gathered": 7, "water_gathered": 5,
        "trade_partner": "Mountain Tribe",
        "lead_scout": "Test Emissary", "determination": 0.5, "max_days": 4, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions == []
    assert any("traded with Mountain Tribe" in entry for entry in tribe.history)


def test_trade_emissary_arrival_home_reports_failure():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10
    tribe.expeditions = [{
        "kind": "trade", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
        "day": 4, "phase": "returning", "food_gathered": 7, "water_gathered": 5,
        "lead_scout": "Test Emissary", "determination": 0.5, "max_days": 4, "path": [],
    }]

    sim._advance_expeditions(tribe)

    assert tribe.expeditions == []
    assert any("found no one to trade with" in entry for entry in tribe.history)


def test_send_trade_emissary_unlocked_from_primitive_dawn():
    from backend.eras import unlocked_actions_through

    assert "SEND_TRADE_EMISSARY" in unlocked_actions_through("primitive_dawn")


def test_expedition_records_every_tile_it_walks_as_a_breadcrumb_path():
    """The persistent world-trail mechanic (Landscape.trails) only lights up once a
    route gets reused, so a single fresh journey barely shows anything even while it's
    actively happening. This is the per-expedition breadcrumb line instead: everywhere
    this one party has actually walked, regardless of reuse."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [80, 80],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [[50, 50]],
    }]

    sim._advance_expeditions(tribe)
    sim._advance_expeditions(tribe)

    assert tribe.expeditions[0]["path"][0] == [50, 50]
    assert len(tribe.expeditions[0]["path"]) == 3
    assert tribe.expeditions[0]["path"][-1] == tribe.expeditions[0]["pos"]


def test_expedition_wears_a_trail_on_the_tile_it_moves_into():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [80, 80],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    from backend import config
    landed = tuple(tribe.expeditions[0]["pos"])
    assert sim.world.trails.get(landed)["wear"] == config.TRAIL_WEAR_PER_PASS
    assert sim.world.trails.get(landed)["color"] == tribe.color


def test_expedition_travels_farther_along_an_already_worn_trail():
    """The point of trail wear applying to expeditions too: a route worn down by
    earlier trips lets a later expedition cover more ground per day, potentially
    reaching a destination that was out of reach on the first attempt."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.world.wear_trail(50, 50, 1.0)  # fully worn starting tile
    tribe.expeditions = [{
        "pos": [50, 50], "origin": [50, 50], "target": [99, 50],
        "day": 0, "phase": "outbound", "found": None, "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Test Scout", "determination": 0.5, "max_days": 3, "path": [],
    }]

    sim._advance_expeditions(tribe)

    expected_speed = config.EXPEDITION_SPEED + config.MAX_TRAIL_BONUS_SPEED
    assert tribe.expeditions[0]["pos"] == [50 + expected_speed, 50]
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


def test_water_bringer_trophy_is_awarded_to_the_current_chief_on_reaching_river():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 40, 37, "#c084fc")  # (40,37) is river
    tribe.chief_name = "Ashgar"

    sim._check_chief_trophies(tribe)

    assert any(t["name"] == "Water Bringer" and t["chief"] == "Ashgar" for t in tribe.trophies)
    assert any("Water Bringer" in entry for entry in tribe.history)


def test_water_bringer_trophy_is_also_awarded_on_reaching_the_lake():
    from backend.world import LAKE_CENTER

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", *LAKE_CENTER, "#c084fc")
    tribe.chief_name = "Ashgar"

    sim._check_chief_trophies(tribe)

    assert any(t["name"] == "Water Bringer" for t in tribe.trophies)


def test_trophies_are_only_awarded_once_per_tribe_lifetime():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 40, 37, "#c084fc")
    tribe.chief_name = "Ashgar"

    sim._check_chief_trophies(tribe)
    tribe.chief_name = "Successor"  # a later chief shouldn't steal an already-earned trophy
    sim._check_chief_trophies(tribe)

    water_trophies = [t for t in tribe.trophies if t["name"] == "Water Bringer"]
    assert len(water_trophies) == 1
    assert water_trophies[0]["chief"] == "Ashgar"


def test_master_pathfinder_trophy_credited_to_the_specific_scout_at_the_milestone():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    for i in range(config.MILESTONE_SCOUT_SUCCESSES):
        exp = {
            "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
            "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
            "food_gathered": 0, "water_gathered": 0,
            "lead_scout": f"Scout{i}", "determination": 0.5, "max_days": 3, "path": [],
        }
        tribe.expeditions = [exp]
        sim._advance_one_expedition(tribe, exp)

    assert tribe.scout_successes == config.MILESTONE_SCOUT_SUCCESSES
    pathfinder = [t for t in tribe.trophies if t["name"] == "Master Pathfinder"]
    assert len(pathfinder) == 1
    assert pathfinder[0]["chief"] == f"Scout{config.MILESTONE_SCOUT_SUCCESSES - 1}"


def test_confirmed_water_site_persists_after_a_successful_scout():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    exp = {
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Ashgar", "determination": 0.5, "max_days": 3, "path": [],
    }
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    assert tribe.confirmed_water_sites == [(40, 37)]


def test_confirmed_water_sites_are_deduplicated():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(2):
        exp = {
            "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
            "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
            "food_gathered": 0, "water_gathered": 0,
            "lead_scout": "Ashgar", "determination": 0.5, "max_days": 3, "path": [],
        }
        tribe.expeditions = [exp]
        sim._advance_one_expedition(tribe, exp)

    assert tribe.confirmed_water_sites == [(40, 37)]


def test_confirmed_water_sites_are_surfaced_as_a_durable_fact():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.confirmed_water_sites = [(12, 34), (40, 37)]

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert "confirmed water source at (12,34)" in entities
    assert "confirmed water source at (40,37)" in entities


def test_all_confirmed_sites_are_remembered_not_just_the_most_recent_three():
    """Explicit request: "make sure they remember all the important discover
    sites when they are making decisions." These lists used to be sliced to
    the 3 most recent, so a 4th+ genuinely distinct discovery silently
    disappeared from the tribe's own facts."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.confirmed_water_sites = [(1, 1), (2, 2), (3, 3), (4, 4)]
    tribe.quarry_sites = [(5, 5), (6, 6), (7, 7), (8, 8)]

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert "confirmed water source at (1,1)" in entities
    assert "confirmed water source at (4,4)" in entities
    assert "confirmed stone-rich area at (5,5)" in entities
    assert "confirmed stone-rich area at (8,8)" in entities


def test_no_discoveries_at_all_is_named_as_a_real_fact():
    """Bug report: "one is not exploring and one only explores in one
    direction.\""""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert any("wider world beyond home remains completely unknown" in e for e in entities)


def test_discoveries_clustered_in_one_direction_are_named_as_a_real_fact():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.lumber_sites = [(60, 50), (70, 50), (80, 50)]  # all due east of (50, 50)

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert any("Every confirmed discovery so far lies to the east" in e for e in entities)


def test_discoveries_spread_across_directions_dont_trigger_the_one_direction_fact():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.lumber_sites = [(60, 50)]  # east
    tribe.wildlife_sites = [{"x": 50, "y": 40, "type": "Deer Stand"}]  # north
    tribe.quarry_sites = [(40, 50)]  # west

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert not any("remains completely unexplored" in e for e in entities)
    assert not any("wider world beyond home remains completely unknown" in e for e in entities)


def test_confirmed_water_sites_are_exposed_to_the_frontend_as_landmarks():
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.confirmed_water_sites = [(12, 34)]

    assert tribe.to_dict()["confirmed_water_sites"] == [(12, 34)]


def _returning_scout_exp(target, terrain_report, lead_scout="Ashgar"):
    return {
        "pos": [50, 50], "origin": [50, 50], "target": list(target),
        "day": 2, "phase": "returning", "found": None, "terrain_report": terrain_report,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": lead_scout, "determination": 0.5, "max_days": 3, "path": [],
    }


def test_forest_terrain_report_confirms_both_a_lumber_and_a_wildlife_site():
    from backend.world import WILDLIFE_SITE_TYPES

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    exp = _returning_scout_exp((60, 60), "forest")
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    assert tribe.lumber_sites == [(60, 60)]
    assert len(tribe.wildlife_sites) == 1
    site = tribe.wildlife_sites[0]
    assert (site["x"], site["y"]) == (60, 60)
    assert site["type"] in WILDLIFE_SITE_TYPES
    assert tribe.quarry_sites == []


def test_celebration_shout_reuses_the_tribes_own_last_broadcast():
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.last_broadcast = "KRA-ZUL"

    assert _celebration_shout(tribe) == ' -- "KRA-ZUL!"'


def test_celebration_shout_is_silent_with_no_prior_broadcast():
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.last_broadcast = ""

    assert _celebration_shout(tribe) == ""


def test_harvest_celebration_includes_the_tribes_own_shout():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 100
    tribe.last_broadcast = "VASH-TA"

    sim._celebrate_harvest(tribe)

    assert any('holds a harvest festival' in e and '"VASH-TA!"' in e for e in tribe.history)


def test_new_wildlife_site_throws_a_game_discovery_celebration():
    """Explicit request: "the celebration label should say celebrating... the
    discovery of a small-game site." Terrain reports only ever carry memory weight
    0.6, below CELEBRATION_DISCOVERY_WEIGHT, so this would otherwise never trigger
    the generic _check_for_celebration path at all."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 100
    exp = _returning_scout_exp((60, 60), "forest")
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    assert any("celebrates the discovery of a game-rich site at (60,60)" in entry for entry in tribe.history)
    assert tribe.food < 100
    assert tribe.last_celebration_cycle == sim.cycle


def test_revisiting_an_already_known_wildlife_site_does_not_re_celebrate():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wildlife_sites = [{"x": 60, "y": 60, "type": "Deer Stand"}]
    tribe.food = 100
    exp = _returning_scout_exp((60, 60), "forest")
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    # Only the ordinary arrival-home food delivery -- no feast cost, not a new site.
    assert tribe.food == 100 + config.EXPEDITION_RETURN_DAILY_FOOD
    assert not any("celebrates the discovery of a game-rich site" in entry for entry in tribe.history)


def test_mountains_terrain_report_confirms_a_quarry_site():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "qwen2.5:3b", 50, 50, "#fb923c")
    exp = _returning_scout_exp((15, 20), "mountains")
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    assert tribe.quarry_sites == [(15, 20)]
    assert tribe.lumber_sites == []
    assert tribe.wildlife_sites == []


def test_plains_terrain_report_confirms_no_resource_landmark():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    exp = _returning_scout_exp((60, 60), "plains")
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    assert tribe.lumber_sites == tribe.wildlife_sites == tribe.quarry_sites == []


def test_resource_landmarks_are_surfaced_as_durable_facts_and_exposed_to_the_frontend():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.lumber_sites = [(60, 60)]
    tribe.wildlife_sites = [{"x": 60, "y": 60, "type": "Deer Stand"}]
    tribe.quarry_sites = [(15, 20)]

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert "confirmed lumber-rich area at (60,60)" in entities
    assert "a Deer Stand was found at (60,60)" in entities
    assert "confirmed stone-rich area at (15,20)" in entities
    d = tribe.to_dict()
    assert d["lumber_sites"] == [(60, 60)]
    assert d["wildlife_sites"] == [{"x": 60, "y": 60, "type": "Deer Stand"}]
    assert d["quarry_sites"] == [(15, 20)]


def test_raider_sighting_is_recorded_and_radiates_dread_at_the_sighting_not_the_camp():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    exp = _returning_scout_exp((60, 60), "plains")
    tribe.expeditions = [exp]

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_one_expedition(tribe, exp)

    assert tribe.raider_sightings == [(60, 60)]
    assert "DREAD" in sim.trauma.bias_string(60, 60)
    assert "DREAD" not in sim.trauma.bias_string(50, 50)  # not at the tribe's own camp
    assert any("signs of raiders near (60,60)" in entry for entry in tribe.history)


def test_revisiting_a_known_raider_sighting_does_not_duplicate_the_list():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.raider_sightings = [(60, 60)]
    exp = _returning_scout_exp((60, 60), "plains")
    tribe.expeditions = [exp]

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_one_expedition(tribe, exp)

    assert tribe.raider_sightings == [(60, 60)]


def test_hunting_party_never_rolls_a_raider_sighting():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10
    tribe.expeditions = [{
        "kind": "hunt", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
        "day": 2, "phase": "returning", "food_caught": 0,
        "food_gathered": 7, "water_gathered": 5,
        "lead_scout": "Test Hunter", "determination": 0.5, "max_days": 4, "path": [],
    }]

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_expeditions(tribe)

    assert tribe.raider_sightings == []


def test_raider_sightings_are_surfaced_as_a_durable_fact():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.raider_sightings = [(60, 60)]

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert "raiders reported near (60,60)" in entities
    assert tribe.to_dict()["raider_sightings"] == [(60, 60)]


def test_raider_attack_never_fires_before_a_tribe_has_ever_settled():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.has_ever_settled = False
    tribe.population = 100  # would otherwise guarantee the max attack chance

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._check_raider_attack(tribe)

    assert tribe.last_raider_attack_cycle < 0
    assert sim.recent_encounters == []


def test_raider_attack_respects_cooldown():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.has_ever_settled = True
    tribe.population = 100
    tribe.last_raider_attack_cycle = sim.cycle  # just attacked this exact cycle

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._check_raider_attack(tribe)

    assert sim.recent_encounters == []


def test_raider_attack_chance_scales_with_population():
    from backend import config

    sim = _bare_simulation()
    low = Tribe("tribe_0", "Small Tribe", "gemma2:2b", 50, 50, "#c084fc")
    low.has_ever_settled = True
    low.population = 5
    high = Tribe("tribe_1", "Big Tribe", "gemma2:2b", 60, 60, "#f97316")
    high.has_ever_settled = True
    high.population = config.RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE

    # A roll that clears the max chance but not a small tribe's much lower chance.
    roll = config.RAIDER_HAZARD_MAX_CHANCE * (low.population / config.RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE) + 0.001
    with mock.patch("backend.simulation.random.random", return_value=roll):
        sim._check_raider_attack(low)
        sim._check_raider_attack(high)

    assert low.last_raider_attack_cycle < 0  # never triggered
    assert high.last_raider_attack_cycle == sim.cycle  # triggered


def test_raider_attack_never_resolves_directly_only_starts_an_approach():
    """Explicit request: "I do want to see RAIDERs ride in over time" -- a triggered
    attack no longer resolves in the same instant it's rolled."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.has_ever_settled = True
    tribe.population = 100

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._check_raider_attack(tribe)

    assert tribe.raiders_approaching is not None
    assert tribe.raiders_approaching["cycles_left"] == config.RAIDER_APPROACH_CYCLES
    assert sim.recent_encounters == []  # nothing resolved yet


def test_raider_approach_counts_down_and_resolves_on_arrival():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.has_ever_settled = True
    tribe.population = 100

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._check_raider_attack(tribe)
    for _ in range(config.RAIDER_APPROACH_CYCLES):
        with mock.patch("backend.simulation.random.random", return_value=0.0):  # defense holds each time
            sim._advance_raider_approach(tribe)

    assert tribe.raiders_approaching is None
    assert sim.recent_encounters  # resolution actually ran


def test_raider_approach_moves_toward_the_settlement_over_time():
    import math

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.has_ever_settled = True
    tribe.population = 100

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._check_raider_attack(tribe)
    start = (tribe.raiders_approaching["x"], tribe.raiders_approaching["y"])
    start_dist = math.hypot(start[0] - tribe.x, start[1] - tribe.y)

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_raider_approach(tribe)
    mid = (tribe.raiders_approaching["x"], tribe.raiders_approaching["y"])
    mid_dist = math.hypot(mid[0] - tribe.x, mid[1] - tribe.y)

    assert mid_dist < start_dist


def test_raider_attack_failed_defense_steals_resources_and_costs_population():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    tribe.food = tribe.water = tribe.wood = tribe.stone = 100

    with mock.patch("backend.simulation.random.random", return_value=0.99):  # defense fails
        sim._resolve_raider_attack(tribe)

    assert tribe.food < 100
    assert tribe.population < 10
    assert "DREAD" in sim.trauma.bias_string(50, 50)
    assert any("raiders struck the camp" in entry for entry in tribe.history)
    assert sim.recent_encounters and sim.recent_encounters[0]["outcome"] == "struck"


def test_raider_attack_successful_defense_radiates_pride_and_increments_raids_defended():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    tribe.food = tribe.water = tribe.wood = tribe.stone = 100

    with mock.patch("backend.simulation.random.random", return_value=0.0):  # defense holds
        sim._resolve_raider_attack(tribe)

    assert tribe.raids_defended == 1
    assert "PRIDE" in sim.trauma.bias_string(50, 50)
    assert any("repelled" in entry for entry in tribe.history)
    assert sim.recent_encounters and sim.recent_encounters[0]["outcome"] == "repelled"
    assert any(t["name"] == "Raid Breaker" for t in tribe.trophies)


def test_successful_defense_awards_loot_scaled_by_raider_strength():
    """Explicit request: "the repelling Tribe better get some good rewards from
    that. it's huge for them!" -- a successful defense used to yield only pride."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = config.RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE  # max raider_strength
    tribe.food = tribe.wood = tribe.stone = 100

    with mock.patch("backend.simulation.random.random", return_value=0.0):  # defense holds
        sim._resolve_raider_attack(tribe)

    assert tribe.food > 100
    assert tribe.wood > 100
    assert tribe.stone > 100


def test_partial_wall_gives_partial_defense_bonus_not_full_or_zero():
    from backend import config

    no_wall = Tribe("tribe_0", "A", "gemma2:2b", 50, 50, "#c084fc")
    no_wall.population = 10

    no_wall_defense = (
        config.RAIDER_DEFENSE_BASE_CHANCE + (no_wall.population // 10) * config.RAIDER_DEFENSE_POPULATION_BONUS_PER_10
    )
    half_wall_defense = no_wall_defense + config.RAIDER_DEFENSE_WALL_BONUS_AT_FULL_PROGRESS * 0.5
    full_wall_defense = no_wall_defense + config.RAIDER_DEFENSE_WALL_BONUS_AT_FULL_PROGRESS * 1.0

    assert no_wall_defense < half_wall_defense < full_wall_defense


def test_settled_near_water_gives_a_defense_bonus_even_without_a_wall():
    from backend import config

    on_water = Tribe("tribe_0", "A", "gemma2:2b", 40, 37, "#c084fc")  # river
    on_water.population = 10
    on_water.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    sim_water = _bare_simulation()

    off_water = Tribe("tribe_1", "B", "gemma2:2b", 80, 38, "#fb923c")  # forest, not water
    off_water.population = 10
    off_water.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    sim_land = _bare_simulation()

    base_defense = (
        config.RAIDER_DEFENSE_BASE_CHANCE
        + (10 // 10) * config.RAIDER_DEFENSE_POPULATION_BONUS_PER_10
        - config.RAIDER_STRENGTH_DEFENSE_PENALTY_AT_MAX * min(1.0, 10 / config.RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE)
    )
    roll = base_defense + config.RAIDER_DEFENSE_WATER_BONUS / 2  # clears on-water only

    with mock.patch("backend.simulation.random.random", return_value=roll):
        sim_land._resolve_raider_attack(off_water)
    with mock.patch("backend.simulation.random.random", return_value=roll):
        sim_water._resolve_raider_attack(on_water)

    assert off_water.raids_defended == 0
    assert on_water.raids_defended == 1


def test_raider_strength_scales_with_population_and_can_outweigh_its_own_defense_bonus():
    """Explicit finding: raiders were being repelled too consistently because the
    raiding force never scaled with what it was attacking -- a bigger tribe's own
    population defense bonus could reliably clear the cap. Now a big enough
    population's raider-strength penalty outweighs that same population's defense
    bonus, so a wall/water genuinely matters instead of population alone being
    enough."""
    from backend import config

    small = Tribe("tribe_0", "A", "gemma2:2b", 50, 50, "#c084fc")
    small.population = 5
    sim_small = _bare_simulation()

    big = Tribe("tribe_1", "B", "gemma2:2b", 50, 50, "#fb923c")
    big.population = config.RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE  # raider_strength = 1.0
    sim_big = _bare_simulation()

    small_defense = (
        config.RAIDER_DEFENSE_BASE_CHANCE
        + (5 // 10) * config.RAIDER_DEFENSE_POPULATION_BONUS_PER_10
        - config.RAIDER_STRENGTH_DEFENSE_PENALTY_AT_MAX * min(1.0, 5 / config.RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE)
    )
    big_defense = (
        config.RAIDER_DEFENSE_BASE_CHANCE
        + (big.population // 10) * config.RAIDER_DEFENSE_POPULATION_BONUS_PER_10
        - config.RAIDER_STRENGTH_DEFENSE_PENALTY_AT_MAX * 1.0
    )
    assert big_defense < small_defense  # the whole point of the fix

    roll = (small_defense + big_defense) / 2
    with mock.patch("backend.simulation.random.random", return_value=roll):
        sim_small._resolve_raider_attack(small)
    with mock.patch("backend.simulation.random.random", return_value=roll):
        sim_big._resolve_raider_attack(big)

    assert small.raids_defended == 1
    assert big.raids_defended == 0


def test_full_wall_reduces_population_loss_more_than_a_partial_wall():
    sim_partial = _bare_simulation()
    partial = Tribe("tribe_0", "A", "gemma2:2b", 50, 50, "#c084fc")
    partial.population = 10
    partial.food = partial.water = partial.wood = partial.stone = 100
    sim_partial.world.constructions[(50, 50)] = {"type": "wall", "cycle": 0, "progress": 30}

    sim_full = _bare_simulation()
    full = Tribe("tribe_1", "B", "gemma2:2b", 50, 50, "#c084fc")
    full.population = 10
    full.food = full.water = full.wood = full.stone = 100
    sim_full.world.constructions[(50, 50)] = {"type": "wall", "cycle": 0, "progress": 100}

    with mock.patch("backend.simulation.random.random", return_value=0.99):
        sim_partial._resolve_raider_attack(partial)
    with mock.patch("backend.simulation.random.random", return_value=0.99):
        sim_full._resolve_raider_attack(full)

    assert (10 - full.population) < (10 - partial.population)


def test_failed_wall_defense_destroys_the_wall_forcing_a_rebuild():
    """Explicit request: a wall that fails to stop a raid doesn't stay standing --
    the tribe has to rebuild it, the same as any real defensive structure that gets
    breached."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    tribe.food = tribe.water = tribe.wood = tribe.stone = 100
    sim.world.constructions[(50, 50)] = {"type": "wall", "cycle": 0, "progress": 80}

    with mock.patch("backend.simulation.random.random", return_value=0.99):  # defense fails
        sim._resolve_raider_attack(tribe)

    assert (50, 50) not in sim.world.constructions


def test_successful_wall_defense_leaves_the_wall_standing():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    tribe.food = tribe.water = tribe.wood = tribe.stone = 100
    sim.world.constructions[(50, 50)] = {"type": "wall", "cycle": 0, "progress": 80}

    with mock.patch("backend.simulation.random.random", return_value=0.0):  # defense holds
        sim._resolve_raider_attack(tribe)

    assert sim.world.constructions[(50, 50)] == {"type": "wall", "cycle": 0, "progress": 80}


def test_recent_encounters_cleared_each_cycle_and_populated_on_a_raider_attack():
    sim = _bare_simulation()
    sim.recent_encounters = [{"x": 1, "y": 1, "kind": "raider_attack", "label": "stale", "outcome": "struck"}]
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    tribe.food = tribe.water = tribe.wood = tribe.stone = 100

    sim.recent_encounters = []  # mirrors step()'s per-cycle reset before _resolve_raider_attack runs
    with mock.patch("backend.simulation.random.random", return_value=0.99):  # defense fails
        sim._resolve_raider_attack(tribe)

    assert len(sim.recent_encounters) == 1
    assert sim.recent_encounters[0]["x"] == 50 and sim.recent_encounters[0]["y"] == 50


def test_scouting_custom_award_goes_to_the_scout_who_first_confirms_water():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.custom_awards = [{"name": "Keeper of the Trails", "category": "scouting", "cycle": 1}]
    exp = {
        "pos": [50, 50], "origin": [50, 50], "target": [40, 37],
        "day": 2, "phase": "returning", "found": [40, 37], "terrain_report": None,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "Ashgar", "determination": 0.5, "max_days": 3, "path": [],
    }
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    award = [t for t in tribe.trophies if t["name"] == "Keeper of the Trails"]
    assert len(award) == 1
    assert award[0]["chief"] == "Ashgar"


def test_hunting_custom_award_goes_to_the_hunter_who_first_makes_a_catch():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.custom_awards = [{"name": "Wolf's Bane", "category": "hunting", "cycle": 1}]
    exp = {
        "kind": "hunt", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
        "day": 2, "phase": "returning", "food_caught": 20,
        "food_gathered": 0, "water_gathered": 0,
        "lead_scout": "BriMir", "determination": 0.5, "max_days": 4, "path": [],
    }
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    award = [t for t in tribe.trophies if t["name"] == "Wolf's Bane"]
    assert len(award) == 1
    assert award[0]["chief"] == "BriMir"


def test_check_custom_awards_only_matches_its_own_category():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.custom_awards = [{"name": "Keeper of the Trails", "category": "scouting", "cycle": 1}]

    sim._check_custom_awards(tribe, "hunting", individual="BriMir")

    assert tribe.trophies == []


def test_check_custom_awards_only_grants_each_award_once():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.custom_awards = [{"name": "Keeper of the Trails", "category": "scouting", "cycle": 1}]

    sim._check_custom_awards(tribe, "scouting", individual="Ashgar")
    sim._check_custom_awards(tribe, "scouting", individual="Someone Later")

    matches = [t for t in tribe.trophies if t["name"] == "Keeper of the Trails"]
    assert len(matches) == 1
    assert matches[0]["chief"] == "Ashgar"


def test_master_hunter_trophy_credited_to_the_specific_hunter_at_the_milestone():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    for i in range(config.MILESTONE_HUNT_SUCCESSES):
        exp = {
            "kind": "hunt", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
            "day": 2, "phase": "returning", "food_caught": 20,
            "food_gathered": 0, "water_gathered": 0,
            "lead_scout": f"Hunter{i}", "determination": 0.5, "max_days": 4, "path": [],
        }
        tribe.expeditions = [exp]
        sim._advance_one_expedition(tribe, exp)

    assert tribe.hunt_successes == config.MILESTONE_HUNT_SUCCESSES
    hunter_trophy = [t for t in tribe.trophies if t["name"] == "Master Hunter"]
    assert len(hunter_trophy) == 1
    assert hunter_trophy[0]["chief"] == f"Hunter{config.MILESTONE_HUNT_SUCCESSES - 1}"


@run_async
async def test_resolve_birth_grows_population_and_records_lineage():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.population = 8
    tribe.pending_birth = {"parent_a": "Ashgar", "parent_b": "BriMir"}

    async def fake_breed(client, model, tribe_name, parent_a, parent_b):
        return {"child_name": "Toka", "note": "born under a clear sky"}

    with mock.patch("backend.simulation.breed_individuals", fake_breed):
        await sim._resolve_birth(tribe)

    assert tribe.population == 9
    assert tribe.pending_birth is None
    assert tribe.lineage == [{"child_name": "Toka", "parents": ["Ashgar", "BriMir"], "cycle": sim.cycle}]
    assert any("Toka" in entry and "born under a clear sky" in entry for entry in tribe.history)


@run_async
async def test_resolve_birth_falls_back_to_a_generic_name_if_the_llm_call_fails():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.pending_birth = {"parent_a": "Ashgar", "parent_b": "BriMir"}

    async def fake_breed(client, model, tribe_name, parent_a, parent_b):
        return {}

    with mock.patch("backend.simulation.breed_individuals", fake_breed):
        await sim._resolve_birth(tribe)

    assert tribe.lineage[0]["child_name"] == "child of Ashgar and BriMir"


def test_tribal_gathering_reports_new_trophies_unclaimed_awards_and_population_change():
    sim = _bare_simulation()
    sim.cycle = 20
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.chief_philosophy = "strength through unity"
    tribe.population_at_last_gathering = 8
    tribe.population = 10
    tribe.trophies = [{"name": "Water Bringer", "chief": "Ashgar", "cycle": 5}]
    tribe.custom_awards = [{"name": "Keeper of the Trails", "category": "scouting", "cycle": 5}]

    sim._hold_tribal_gathering(tribe)

    assert "Ashgar earned the 'Water Bringer' honor" in tribe.gathering_brief
    assert "unclaimed: the 'Keeper of the Trails'" in tribe.gathering_brief
    assert "grown by 2" in tribe.gathering_brief
    assert "strength through unity" in tribe.gathering_brief
    assert tribe.history[-1].startswith("The tribe gathers as the sun rises.")
    assert tribe.last_gathering_cycle == 20
    assert tribe.population_at_last_gathering == 10


def test_tribal_gathering_omits_a_trophy_already_reported_at_a_prior_gathering():
    sim = _bare_simulation()
    sim.cycle = 40
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.last_gathering_cycle = 20
    tribe.trophies = [{"name": "Water Bringer", "chief": "Ashgar", "cycle": 5}]

    sim._hold_tribal_gathering(tribe)

    assert "Water Bringer" not in tribe.gathering_brief


def test_tribal_gathering_does_not_report_an_already_claimed_custom_award():
    sim = _bare_simulation()
    sim.cycle = 20
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.custom_awards = [{"name": "Keeper of the Trails", "category": "scouting", "cycle": 5}]
    tribe.trophies = [{"name": "Keeper of the Trails", "chief": "Ashgar", "cycle": 6}]

    sim._hold_tribal_gathering(tribe)

    assert "unclaimed" not in tribe.gathering_brief


def test_tribal_gathering_says_something_even_with_nothing_new():
    sim = _bare_simulation()
    sim.cycle = 20
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    sim._hold_tribal_gathering(tribe)

    assert tribe.gathering_brief == "a quiet gathering -- nothing new to report"


def test_gathering_brief_is_surfaced_into_the_next_turns_visible_entities():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.gathering_brief = "the tribe has grown by 2 since the last gathering"

    entities, _ = sim._build_visible_entities(tribe, "plains", [], [], [])

    assert any("this morning's gathering: the tribe has grown by 2" in e for e in entities)


@run_async
async def test_step_holds_the_tribal_gathering_only_on_its_own_interval():
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    sim.cycle = config.DAY_LENGTH_CYCLES - 1  # step() increments before checking

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})):
        await sim.step()

    assert tribe.last_gathering_cycle == config.DAY_LENGTH_CYCLES


@run_async
async def test_step_does_not_hold_the_tribal_gathering_off_its_interval():
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    sim.cycle = config.DAY_LENGTH_CYCLES - 2

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})):
        await sim.step()

    assert tribe.last_gathering_cycle == 0


@run_async
async def test_night_cycle_updates_philosophy_when_the_reviewer_calls_for_a_change():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    tribe.chief_philosophy = "expand aggressively"
    tribe.history.append("starvation claimed lives")

    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory=""):
        return {"revised_philosophy": "caution and hoarding", "changed": True, "reasoning": "too many losses"}

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
        await sim._run_night_cycle(tribe)

    assert tribe.chief_philosophy == "caution and hoarding"
    assert any("reconsiders the tribe's philosophy" in entry and "too many losses" in entry for entry in tribe.history)


@run_async
async def test_night_cycle_leaves_philosophy_and_history_untouched_when_nothing_changed():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    tribe.chief_philosophy = "expand aggressively"
    history_before = list(tribe.history)

    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory=""):
        return {"revised_philosophy": "expand aggressively", "changed": False, "reasoning": "still working"}

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
        await sim._run_night_cycle(tribe)

    assert tribe.chief_philosophy == "expand aggressively"
    assert list(tribe.history) == history_before  # no new chronicle noise on an unremarkable review


@run_async
async def test_night_cycle_records_the_reasoning_even_when_nothing_changed():
    """The frontend's night-time thought bubble needs something real to show even on
    an unremarkable review -- reasoning is captured regardless of whether the
    philosophy itself changed."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    sim.cycle = 30

    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory=""):
        return {"revised_philosophy": current_philosophy, "changed": False, "reasoning": "still working"}

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
        await sim._run_night_cycle(tribe)

    assert tribe.last_reflection == "still working"
    assert tribe.last_reflection_cycle == 30


@run_async
async def test_night_cycle_passes_the_tribes_own_recent_history_and_philosophy():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    tribe.chief_philosophy = "expand aggressively"
    tribe.history.append("starvation claimed lives")
    captured = {}

    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory=""):
        captured["reviewer_model"] = reviewer_model
        captured["tribe_name"] = tribe_name
        captured["current_philosophy"] = current_philosophy
        captured["recent_events"] = recent_events
        captured["inventory"] = inventory
        return {"revised_philosophy": current_philosophy, "changed": False, "reasoning": ""}

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
        await sim._run_night_cycle(tribe)

    from backend import config
    assert captured["reviewer_model"] == config.NIGHT_CYCLE_REVIEWER_MODEL
    assert captured["tribe_name"] == "Forest Tribe"
    assert captured["current_philosophy"] == "expand aggressively"
    assert "starvation claimed lives" in captured["recent_events"]
    assert f"Population: {tribe.population}." in captured["inventory"]


def test_night_inventory_reports_the_real_state_of_affairs():
    """The night-cycle reviewer used to only ever see prose chronicle entries, which
    tend to just echo whatever the tribe has been doing turn after turn -- a real
    mismatch (surplus water, zero food) was easy to miss reading prose alone. This is
    the structured "state of affairs" the chief takes stock of before sleeping on it."""
    from backend import config

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.population = 5
    tribe.food = 0
    tribe.water = 200
    tribe.wood = 10
    tribe.stone = 10

    inventory = sim._build_night_inventory(tribe)

    assert "Population: 5." in inventory
    assert "10 wood, 10 stone, 0 food, 200 water" in inventory
    assert "starving" in inventory.lower()
    assert f"not settled anywhere farmable yet (0/{config.SETTLEMENT_STABILITY_CYCLES}" in inventory


def test_night_inventory_reports_settlement_when_actually_settled():
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    inventory = sim._build_night_inventory(tribe)

    assert "settled on farmable ground" in inventory


@run_async
async def test_night_cycle_captures_a_valid_proposed_award():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"

    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory=""):
        return {
            "revised_philosophy": current_philosophy, "changed": False, "reasoning": "",
            "proposed_award": {"name": "Keeper of the Trails", "category": "scouting"},
        }

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
        await sim._run_night_cycle(tribe)

    assert tribe.custom_awards == [{"name": "Keeper of the Trails", "category": "scouting", "cycle": sim.cycle}]
    assert any("establishes a new honor" in entry and "Keeper of the Trails" in entry for entry in tribe.history)


@run_async
async def test_night_cycle_survives_a_non_dict_proposed_award():
    """Regression test, same failure class as test_install_chief_survives_a_non_dict_
    water_decision: a weak model can put a bare string/bool where proposed_award's
    nested object was asked for."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"

    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory=""):
        return {
            "revised_philosophy": current_philosophy, "changed": False, "reasoning": "",
            "proposed_award": "Keeper of the Trails",
        }

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
        await sim._run_night_cycle(tribe)

    assert tribe.custom_awards == []


@run_async
async def test_night_cycle_ignores_a_proposed_award_outside_the_real_categories():
    """STUB constraint: the chief can name anything, but the category has to be one
    this simulation actually measures (see reflection.py's AWARD_CATEGORIES) -- an
    invented category the simulation can't honestly judge is dropped, not stored."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"

    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory=""):
        return {
            "revised_philosophy": current_philosophy, "changed": False, "reasoning": "",
            "proposed_award": {"name": "Master of Dreams", "category": "dreaming"},
        }

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
        await sim._run_night_cycle(tribe)

    assert tribe.custom_awards == []


async def _night_cycle_no_change(sim, tribe):
    async def fake_reflect(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory=""):
        return {"revised_philosophy": current_philosophy, "changed": False, "reasoning": ""}

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect):
        await sim._run_night_cycle(tribe)


@run_async
async def test_night_cycle_can_start_a_family_when_the_roll_succeeds_and_a_pair_is_eligible():
    """Explicit request: "can we have some random breeding in the over-night cycle?"
    -- an occasional chance encounter, independent of any specific celebration
    milestone, using the same $0-cost eligibility rule every other breeding path
    already uses."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    tribe.trophies.append({"name": "Growing Legacy", "chief": "Mira", "cycle": 1})

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        await _night_cycle_no_change(sim, tribe)

    assert tribe.pending_birth == {"parent_a": "Ashgar", "parent_b": "Mira"}
    assert any("in the quiet of the night" in e and "Ashgar" in e and "Mira" in e for e in tribe.history)


@run_async
async def test_night_cycle_breeding_roll_failing_starts_no_family():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    tribe.trophies.append({"name": "Growing Legacy", "chief": "Mira", "cycle": 1})

    with mock.patch("backend.simulation.random.random", return_value=0.99):
        await _night_cycle_no_change(sim, tribe)

    assert tribe.pending_birth is None


@run_async
async def test_night_cycle_breeding_skipped_without_an_eligible_pair():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"  # only one named individual -- no trophy holder yet

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        await _night_cycle_no_change(sim, tribe)

    assert tribe.pending_birth is None


@run_async
async def test_night_cycle_breeding_skipped_at_population_cap():
    from backend import config

    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    tribe.trophies.append({"name": "Growing Legacy", "chief": "Mira", "cycle": 1})
    tribe.population = config.POPULATION_GROWTH_CAP

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        await _night_cycle_no_change(sim, tribe)

    assert tribe.pending_birth is None


@run_async
async def test_night_cycle_breeding_does_not_override_an_already_pending_birth():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    tribe.trophies.append({"name": "Growing Legacy", "chief": "Mira", "cycle": 1})
    tribe.pending_birth = {"parent_a": "Someone", "parent_b": "Else"}

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        await _night_cycle_no_change(sim, tribe)

    assert tribe.pending_birth == {"parent_a": "Someone", "parent_b": "Else"}


def test_well_fed_and_growing_legacy_trophies_have_their_own_thresholds():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")  # plains, not river
    tribe.chief_name = "Ashgar"
    tribe.food = config.FOOD_TROPHY_THRESHOLD
    tribe.population = 9  # above the starting 8

    sim._check_chief_trophies(tribe)

    names = {t["name"] for t in tribe.trophies}
    assert "Well Fed" in names
    assert "Growing Legacy" in names
    assert "Water Bringer" not in names


def test_celebration_fires_on_food_surplus_and_spends_a_real_fraction():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = config.FOOD_TROPHY_THRESHOLD

    sim._check_for_celebration(tribe)

    expected_spent = round(config.FOOD_TROPHY_THRESHOLD * config.CELEBRATION_RESOURCE_COST_FRACTION)
    assert tribe.food == config.FOOD_TROPHY_THRESHOLD - expected_spent
    assert any("holds a celebration" in entry and "season of plenty" in entry for entry in tribe.history)
    assert float(sim.trauma.ghost_tensor[50, 50]) > 0


def test_celebration_fires_on_a_fresh_high_weight_discovery_without_food_surplus():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10  # well under the surplus threshold
    sim.cycle = 5
    tribe.memory.remember("Scouts confirmed fresh water at (40,37).", 5, weight=0.9)

    sim._check_for_celebration(tribe)

    assert any("holds a celebration" in entry and "fresh discovery" in entry for entry in tribe.history)


def test_celebration_does_not_fire_without_surplus_or_discovery():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10

    sim._check_for_celebration(tribe)

    assert not any("holds a celebration" in entry for entry in tribe.history)


def test_celebration_respects_its_own_cooldown():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = config.FOOD_TROPHY_THRESHOLD * 2

    sim._check_for_celebration(tribe)
    first_count = sum("holds a celebration" in e for e in tribe.history)
    sim.cycle += 1
    sim._check_for_celebration(tribe)
    second_count = sum("holds a celebration" in e for e in tribe.history)

    assert first_count == 1
    assert second_count == 1  # unchanged -- still inside the cooldown window


def test_celebration_cost_is_capped_regardless_of_wealth():
    """Explicit finding: 30% of current food with no ceiling gets more expensive in
    absolute terms the wealthier a tribe gets -- "we spend a lot of time on
    Parties." """
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10_000

    sim._check_for_celebration(tribe)

    assert tribe.food == 10_000 - config.CELEBRATION_MAX_COST


def test_celebration_costs_less_once_cooking_is_learned():
    """Explicit request: "Celebrations can even be cheaper if they learn how to
    cook food.\""""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 10_000
    tribe.cooking_learned = True

    sim._check_for_celebration(tribe)

    expected_cost = round(config.CELEBRATION_MAX_COST * config.CELEBRATION_COOKING_COST_MULTIPLIER)
    assert tribe.food == 10_000 - expected_cost


def test_celebration_chronicle_calls_it_a_potluck_once_cooking_is_learned():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = config.FOOD_TROPHY_THRESHOLD
    tribe.cooking_learned = True

    sim._check_for_celebration(tribe)

    assert any("potluck feast" in e for e in tribe.history)


def test_surplus_only_celebration_retires_after_the_configured_count():
    """A tribe that has already celebrated mere plenty a few times shouldn't keep
    paying for it forever -- the same 'generalist narrows to specialist' shape
    GATHER_FOOD's own retirement uses. Discovery-based celebrations never retire."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    for i in range(config.CELEBRATION_SURPLUS_RETIREMENT_COUNT):
        tribe.food = config.FOOD_TROPHY_THRESHOLD * 2
        sim.cycle += config.CELEBRATION_COOLDOWN_CYCLES
        sim._check_for_celebration(tribe)

    assert tribe.surplus_celebrations == config.CELEBRATION_SURPLUS_RETIREMENT_COUNT
    assert any("no longer celebrates mere plenty" in e for e in tribe.history)
    count_before = sum("holds a celebration" in e for e in tribe.history)

    tribe.food = config.FOOD_TROPHY_THRESHOLD * 2
    sim.cycle += config.CELEBRATION_COOLDOWN_CYCLES
    sim._check_for_celebration(tribe)

    assert sum("holds a celebration" in e for e in tribe.history) == count_before  # did not fire again


def test_discovery_celebration_never_retires():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.surplus_celebrations = config.CELEBRATION_SURPLUS_RETIREMENT_COUNT  # already retired
    tribe.food = 10  # no surplus at all
    tribe.memory.remember("Scouts confirmed fresh water at (40,37).", sim.cycle, weight=0.9)

    sim._check_for_celebration(tribe)

    assert any("fresh discovery" in e for e in tribe.history)


def test_celebration_triggers_breeding_when_two_named_individuals_are_eligible():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = config.FOOD_TROPHY_THRESHOLD
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]

    sim._check_for_celebration(tribe)

    assert tribe.pending_birth == {"parent_a": "Ashgar", "parent_b": "BriMir"}
    assert any("decide to start a family" in entry for entry in tribe.history)


def test_celebration_does_not_start_a_second_family_while_one_is_already_pending():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = config.FOOD_TROPHY_THRESHOLD
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]
    tribe.pending_birth = {"parent_a": "Someone", "parent_b": "Else"}

    sim._check_for_celebration(tribe)

    assert tribe.pending_birth == {"parent_a": "Someone", "parent_b": "Else"}  # untouched


def test_era_advances_once_population_and_resources_are_met():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.era = "cognitive_horizon"  # one step below tribal_synapse
    tribe.population = 20
    tribe.water = 40
    tribe.stone = 40
    tribe.wood = 50

    sim._advance_era_if_ready(tribe)

    assert tribe.era == "tribal_synapse"
    assert tribe.wood == 20  # 50 - 30 advancement cost
    assert tribe.stone == 10  # 40 - 30 advancement cost


def test_era_advances_one_step_at_a_time_even_if_stats_clear_a_later_era_too():
    """_advance_era_if_ready only ever checks next_era(tribe.era) -- meeting a later
    era's thresholds too doesn't let a tribe skip the one immediately ahead."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 20
    tribe.water = 40
    tribe.stone = 40
    tribe.wood = 50

    sim._advance_era_if_ready(tribe)

    assert tribe.era == "cognitive_horizon"
    assert tribe.water == 30  # 40 - 10 advancement cost
    assert "Cognitive Horizon" in tribe.history[-1]
    assert "PRIDE" in sim.trauma.bias_string(50, 50)


def test_era_does_not_advance_without_meeting_resource_requirements():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 20
    tribe.water = 5  # below the tribal_synapse requirement

    sim._advance_era_if_ready(tribe)

    assert tribe.era == "primitive_dawn"


def test_snapshot_includes_worn_trails_for_the_frontend_to_render():
    sim = _bare_simulation()
    sim.world.wear_trail(12, 34, 0.5, color="#c084fc")
    sim.tribes = {}
    sim.status = "OPERATIONAL"

    trails = sim.snapshot()["trails"]
    entry = next(t for t in trails if t["x"] == 12 and t["y"] == 34)

    assert entry["wear"] == 0.5
    assert entry["color"] == "#c084fc"
    assert entry["is_toll_road"] is False


def test_snapshot_marks_an_evolved_road_and_its_owner():
    from backend import config

    sim = _bare_simulation()
    sim.tribes = {}
    sim.status = "OPERATIONAL"
    for _ in range(config.ROAD_EVOLVE_CROSSINGS + 1):
        sim.world.wear_trail(12, 34, 0.01, tribe_id="tribe_0")

    entry = next(t for t in sim.snapshot()["trails"] if t["x"] == 12 and t["y"] == 34)

    assert entry["is_toll_road"] is True
    assert entry["owner"] == "tribe_0"


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


def test_advance_farming_grows_the_plot_and_consumes_water():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.farm_plots = 1
    tribe.water = 100

    sim._advance_farming(tribe)

    assert tribe.crop_growth == config.CROP_GROWTH_PER_CYCLE
    assert tribe.water == 100 - config.CROP_WATER_PER_PLOT_PER_CYCLE


def test_advance_farming_harvests_food_once_growth_matures_and_resets():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.farm_plots = 2
    tribe.crop_growth = 100 - config.CROP_GROWTH_PER_CYCLE
    tribe.water = 100
    tribe.food = 0

    sim._advance_farming(tribe)

    harvested = config.CROP_HARVEST_YIELD * 2
    # A harvest now also throws a celebration (explicit request: "a grand harvest is
    # a real celebration"), spending a fraction of the freshly-harvested food.
    spent_on_celebration = round(harvested * config.CELEBRATION_RESOURCE_COST_FRACTION)
    assert tribe.crop_growth == 0
    assert tribe.food == harvested - spent_on_celebration
    assert tribe.last_harvest_cycle == sim.cycle
    assert any("harvest" in entry for entry in tribe.history)
    assert any(t["name"] == "Harvester" for t in tribe.trophies)
    assert any("harvest festival" in entry for entry in tribe.history)


def test_advance_farming_harvest_scales_with_population():
    """Explicit finding: a harvest used to be a flat CROP_HARVEST_YIELD per plot
    regardless of population -- every other resource-producing mechanic already
    scales with _labor_multiplier ("more hands get more done"), but farming never
    did, making a single GATHER_FOOD action out-yield an entire farming cycle for
    any tribe past starting size."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.farm_plots = 1
    tribe.population = 40  # well past POPULATION_YIELD_BASELINE (8)
    tribe.crop_growth = 100 - config.CROP_GROWTH_PER_CYCLE
    tribe.water = 100
    tribe.food = 0
    tribe.last_celebration_cycle = sim.cycle  # skip the celebration cost for a clean read

    sim._advance_farming(tribe)

    assert tribe.food > config.CROP_HARVEST_YIELD


def test_advance_farming_harvest_celebration_respects_the_cooldown():
    """Unlike the water-discovery/settling celebrations (each essentially one-time),
    a harvest recurs every ~10 cycles per plot -- an uncapped feast on every single
    one would drain food faster than farming produces it."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.farm_plots = 1
    tribe.crop_growth = 100 - config.CROP_GROWTH_PER_CYCLE
    tribe.water = 100
    tribe.food = 0
    tribe.last_celebration_cycle = sim.cycle  # just celebrated this exact cycle

    sim._advance_farming(tribe)

    assert tribe.food == config.CROP_HARVEST_YIELD  # no celebration cost taken
    assert not any("harvest festival" in entry for entry in tribe.history)


def test_advance_farming_withers_a_plot_without_enough_water():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.farm_plots = 1
    tribe.crop_growth = 40
    tribe.water = 0

    sim._advance_farming(tribe)

    assert tribe.farm_plots == 0
    assert tribe.crop_growth == 0
    assert any("withers" in entry for entry in tribe.history)


def test_advance_farming_does_nothing_with_no_plots():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.water = 100

    sim._advance_farming(tribe)

    assert tribe.crop_growth == 0
    assert tribe.water == 100


def test_advance_water_supply_flows_in_once_settled_near_water():
    """Explicit request: "like relocate, gather water becomes irrelevant once they
    have settled." A tribe genuinely settled next to real water shouldn't need to
    keep manually choosing GATHER_WATER every cycle just to stand still."""
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.water = 10

    sim._advance_water_supply(tribe)

    upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
    assert tribe.water == 10 + round(upkeep * config.SETTLED_WATER_SUPPLY_MULTIPLIER)


def test_advance_water_supply_scales_with_population_not_a_flat_amount():
    """Bug report: "we have hit a food and water scaling problem... I'm not
    sure why water is still a problem when they are settled." A flat per-cycle
    supply couldn't keep pace with population-scaled upkeep past a certain
    tribe size -- confirmed by comparing a small and a large settled tribe."""
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.population = 200
    tribe.water = 0

    sim._advance_water_supply(tribe)

    upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
    assert tribe.water == round(upkeep * config.SETTLED_WATER_SUPPLY_MULTIPLIER)
    assert tribe.water > upkeep  # a real surplus, not just barely keeping pace


def test_advance_water_supply_does_nothing_before_settling():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])  # forest, not settled
    tribe = sim.tribes["tribe_0"]
    tribe.water = 10

    sim._advance_water_supply(tribe)

    assert tribe.water == 10


def test_advance_fish_supply_flows_in_once_fishing_is_learned_and_settled():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.fishing_learned = True
    tribe.food = 10

    sim._advance_fish_supply(tribe)

    upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
    assert tribe.food == 10 + round(upkeep * config.FISHING_SUPPLY_MULTIPLIER)


def test_advance_fish_supply_does_nothing_until_fishing_is_learned():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river, settled
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.food = 10

    sim._advance_fish_supply(tribe)

    assert tribe.food == 10


def test_advance_fish_supply_does_nothing_before_settling_even_if_learned():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])  # forest, not settled
    tribe = sim.tribes["tribe_0"]
    tribe.fishing_learned = True
    tribe.food = 10

    sim._advance_fish_supply(tribe)

    assert tribe.food == 10


def test_catch_fish_gated_the_same_as_farming_and_eggs():
    """Explicit correction: PLANT_CROP/GATHER_EGGS/CATCH_FISH used to require the
    stricter settled_near_water check -- "the requirement of 'real' water is bogus,
    this is a Settled gate," same general condition as GATHER_WOOD/STONE."""
    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains, not water
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    assert tribe.cycles_since_relocate == 0  # freshly founded, not yet settled

    _request, ctx = sim._prepare_turn(tribe)

    assert "CATCH_FISH" not in ctx["available_actions"]


def test_catch_fish_available_once_settled_even_away_from_water():
    from backend import config

    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 85}])  # plains, not water
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    request, ctx = sim._prepare_turn(tribe)

    assert "CATCH_FISH" in ctx["available_actions"]
    assert "a single successful catch would make fishing a permanent" in request["prompt"]


def test_no_fishing_nudge_once_already_learned():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.fishing_learned = True

    request, _ctx = sim._prepare_turn(tribe)

    assert "a single successful catch would make fishing a permanent" not in request["prompt"]


def test_fishing_vs_hunting_party_comparison_once_fishing_is_learned():
    """Explicit observation: "it should be an easy choice, fish locally, no
    travel time, or send a hunting party taking an indefinate amount of
    time... still travel vs. home." Both action descriptions already say this
    independently, but nothing ever put the two side by side for the model."""
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.fishing_learned = True

    request, ctx = sim._prepare_turn(tribe)

    assert "HUNTING_PARTY" in ctx["available_actions"]
    assert "Fishing here pays out food immediately with no travel time" in request["prompt"]
    assert "Fishing has been mastered here" in request["prompt"]


def test_gather_food_stays_available_before_any_real_food_experience():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True  # bypass the pre-settlement narrowing, unrelated gate

    _, ctx = sim._prepare_turn(tribe)

    assert "GATHER_FOOD" in ctx["available_actions"]
    assert tribe.foraging_retired is False


def test_gather_food_retires_once_fishing_is_learned():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.fishing_learned = True

    request, ctx = sim._prepare_turn(tribe)

    assert "GATHER_FOOD" not in ctx["available_actions"]
    assert tribe.foraging_retired is True
    assert any("GATHER_FOOD is retired" in e and "fishing" in e for e in tribe.history)


def test_gather_food_retires_once_a_harvest_completes():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.last_harvest_cycle = 5  # a real harvest already landed

    _, ctx = sim._prepare_turn(tribe)

    assert "GATHER_FOOD" not in ctx["available_actions"]
    assert any("GATHER_FOOD is retired" in e and "farming" in e for e in tribe.history)


def test_gather_food_retirement_is_not_re_archived_every_cycle():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.fishing_learned = True

    sim._prepare_turn(tribe)
    sim._prepare_turn(tribe)

    assert sum("GATHER_FOOD is retired" in e for e in tribe.history) == 1


def test_gather_water_stays_available_before_settling_near_water():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True  # bypass the pre-settlement narrowing, unrelated gate

    _, ctx = sim._prepare_turn(tribe)

    assert "GATHER_WATER" in ctx["available_actions"]
    assert tribe.watering_retired is False


def test_gather_water_retires_once_settled_near_water():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    request, ctx = sim._prepare_turn(tribe)

    assert "GATHER_WATER" not in ctx["available_actions"]
    assert tribe.watering_retired is True
    assert any("GATHER_WATER is retired" in e for e in tribe.history)


def test_gather_water_retirement_is_not_re_archived_every_cycle():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    sim._prepare_turn(tribe)
    sim._prepare_turn(tribe)

    assert sum("GATHER_WATER is retired" in e for e in tribe.history) == 1


def test_gather_food_retirement_survives_fishing_being_learned_later_checked():
    """A tribe that already planted a farm plot before ever fishing shouldn't need to
    wait on fishing specifically -- either real experience retires it."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.farm_plots = 1
    tribe.last_harvest_cycle = 0  # planted, but nothing has actually been harvested yet

    _, ctx = sim._prepare_turn(tribe)

    assert "GATHER_FOOD" in ctx["available_actions"]  # not retired on planting alone


def test_diversification_note_suggests_farming_once_fishing_alone_is_proven():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.fishing_learned = True

    request, _ctx = sim._prepare_turn(tribe)

    assert "no crop has ever been planted" in request["prompt"]


def test_diversification_note_suggests_fishing_once_farming_alone_is_proven():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.last_harvest_cycle = 5

    request, _ctx = sim._prepare_turn(tribe)

    assert "fishing has never been tried" in request["prompt"]


def test_diversification_note_absent_once_both_are_proven():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.era = "tribal_synapse"
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.fishing_learned = True
    tribe.last_harvest_cycle = 5

    request, _ctx = sim._prepare_turn(tribe)

    assert "no crop has ever been planted" not in request["prompt"]
    assert "fishing has never been tried" not in request["prompt"]


def test_settled_near_water_fact_mentions_passive_water_supply():
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES

    request, _ctx = sim._prepare_turn(tribe)

    assert "Water now flows in on its own each cycle" in request["prompt"]


def test_resource_priority_fact_ranks_stockpiles_lowest_to_highest():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.wood, tribe.stone, tribe.food, tribe.water = 50, 20, 5, 30

    request, _ctx = sim._prepare_turn(tribe)

    assert "Resource priority, lowest to highest: food (5), stone (20), water (30), wood (50)" in request["prompt"]


def test_advance_flock_consumes_feed():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.flock = 3
    tribe.food = 100

    with mock.patch("backend.simulation.random.random", return_value=0.999):  # no natural hatch
        sim._advance_flock(tribe)

    assert tribe.food == 100 - config.FLOCK_UPKEEP_FOOD_PER_MEMBER * 3
    assert tribe.flock == 3


def test_advance_flock_shrinks_without_enough_feed():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.flock = 2
    tribe.food = 0

    sim._advance_flock(tribe)

    assert tribe.flock == 1
    assert any("lost for lack of feed" in entry for entry in tribe.history)


def test_advance_flock_can_naturally_hatch_once_established_and_fed():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.flock = config.FLOCK_MIN_SIZE_TO_BREED
    tribe.food = 1000

    with mock.patch("backend.simulation.random.random", return_value=0.0):  # below any chance
        sim._advance_flock(tribe)

    assert tribe.pending_hatch == {"parents": None}


def test_advance_flock_does_nothing_with_an_empty_flock():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 100

    sim._advance_flock(tribe)

    assert tribe.food == 100
    assert tribe.pending_hatch is None


def test_advance_city_growth_adds_buildings_as_population_climbs():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.founded_city = True
    tribe.population = config.CITY_BUILDING_POPULATION_STEP * 2

    sim._advance_city_growth(tribe)

    assert tribe.city_buildings == 2
    assert any("new building rises" in entry for entry in tribe.history)


def test_advance_city_growth_does_nothing_before_a_city_is_founded():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 1000

    sim._advance_city_growth(tribe)

    assert tribe.city_buildings == 0


def test_advance_city_growth_caps_at_max_buildings():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.founded_city = True
    tribe.population = config.CITY_BUILDING_POPULATION_STEP * (config.MAX_CITY_BUILDINGS + 10)

    sim._advance_city_growth(tribe)

    assert tribe.city_buildings == config.MAX_CITY_BUILDINGS


def test_celebrate_settling_fires_once_a_tribe_settles_near_real_water():
    """Explicit request: settling somewhere for good deserves its own celebration,
    not just whatever unrelated surplus/discovery celebration happens to fire next."""
    from backend import config

    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])  # river
    tribe = sim.tribes["tribe_0"]
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.food = 100

    if not tribe.settlement_name and not tribe.pending_settlement_naming and sim._is_settled_near_water(tribe):
        sim._celebrate_settling(tribe)

    assert tribe.pending_settlement_naming is True
    assert any("celebrates settling here for good" in entry for entry in tribe.history)
    assert tribe.food < 100  # a real food cost, not a free flourish


@run_async
async def test_resolve_settlement_naming_names_the_place_via_a_real_llm_call():
    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    tribe.pending_settlement_naming = True

    async def fake_name_settlement(client, model, tribe_name, chief_name, biome):
        return {"settlement_name": "Rivergate", "note": "named for the river's mouth"}

    with mock.patch("backend.simulation.name_settlement", fake_name_settlement):
        await sim._resolve_settlement_naming(tribe)

    assert tribe.settlement_name == "Rivergate"
    assert tribe.pending_settlement_naming is False
    assert any("Rivergate" in entry and "named for the river's mouth" in entry for entry in tribe.history)


@run_async
async def test_resolve_settlement_naming_falls_back_gracefully_if_the_llm_call_fails():
    sim = Simulation([{"name": "River Tribe", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.pending_settlement_naming = True

    async def fake_name_settlement(client, model, tribe_name, chief_name, biome):
        return {}

    with mock.patch("backend.simulation.name_settlement", fake_name_settlement):
        await sim._resolve_settlement_naming(tribe)

    assert tribe.settlement_name == "River Tribe's Settlement"


@run_async
async def test_resolve_hatch_founding_egg_hatches_without_crossing():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.pending_hatch = {"parents": None}

    await sim._resolve_hatch(tribe)

    assert tribe.flock == 1
    assert tribe.pending_hatch is None
    assert tribe.flock_lineage[0]["parents"] == []
    assert any("hatches" in entry for entry in tribe.history)
    assert any(t["name"] == "Flock Keeper" for t in tribe.trophies)


@run_async
async def test_resolve_hatch_crosses_two_existing_parents_via_hatch():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.flock = 2
    parents = [
        {"trait": "hardy", "parents": [], "cycle": 1, "note": ""},
        {"trait": "quick to forage", "parents": [], "cycle": 2, "note": ""},
    ]
    tribe.pending_hatch = {"parents": parents}
    captured = {}

    async def fake_hatch(client, model, parent_a, parent_b, era):
        captured["parent_a"] = parent_a
        captured["parent_b"] = parent_b
        return {"trait": "hardy forager", "note": "a promising hatchling"}

    with mock.patch("backend.simulation.hatch", fake_hatch):
        await sim._resolve_hatch(tribe)

    assert captured["parent_a"] == parents[0]
    assert captured["parent_b"] == parents[1]
    assert tribe.flock == 3
    assert tribe.flock_lineage[-1]["trait"] == "hardy forager"
    assert tribe.flock_lineage[-1]["parents"] == ["hardy", "quick to forage"]
    assert any("hardy forager" not in entry for entry in tribe.history)  # trait itself isn't echoed
    assert any("a promising hatchling" in entry for entry in tribe.history)


@run_async
async def test_resolve_hatch_falls_back_gracefully_if_the_llm_call_fails():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.flock = 2
    parents = [
        {"trait": "hardy", "parents": [], "cycle": 1, "note": ""},
        {"trait": "quick to forage", "parents": [], "cycle": 2, "note": ""},
    ]
    tribe.pending_hatch = {"parents": parents}

    async def fake_hatch(client, model, parent_a, parent_b, era):
        return {}

    with mock.patch("backend.simulation.hatch", fake_hatch):
        await sim._resolve_hatch(tribe)

    assert tribe.flock == 3
    assert tribe.flock_lineage[-1]["trait"] == "unremarkable but hardy"


def test_upkeep_consumes_food_and_water_proportional_to_population():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 25  # upkeep = max(1, 25 // 10) = 2
    tribe.food = 40
    tribe.water = 40

    sim._apply_upkeep(tribe)

    assert tribe.food == 38
    assert tribe.water == 38


def test_upkeep_food_drain_reduced_once_cooking_is_learned():
    """Explicit request: "cooked food is worth 3 raw food." Water is unaffected --
    only the food side of upkeep is reduced."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 25  # upkeep = max(1, 25 // 10) = 2
    tribe.food = 40
    tribe.water = 40
    tribe.cooking_learned = True

    sim._apply_upkeep(tribe)

    assert tribe.food == 40 - max(1, round(2 / config.COOKING_UPKEEP_DIVISOR))
    assert tribe.water == 38  # unchanged formula


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


def test_immortality_suppresses_extinction_within_the_window():
    sim = _bare_simulation()
    sim.cycle = 50
    sim.immortality_cycles = 200
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 1
    tribe.food = 0
    tribe.water = 40

    sim._apply_upkeep(tribe)

    assert tribe.population == 1  # unchanged -- the hazard is suppressed, not the tribe's state
    assert tribe.extinct is False
    assert any("starvation claimed lives" in entry for entry in tribe.history)  # the event still happened


def test_immortality_stops_protecting_once_the_window_passes():
    sim = _bare_simulation()
    sim.cycle = 201
    sim.immortality_cycles = 200
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 1
    tribe.food = 0
    tribe.water = 40

    sim._apply_upkeep(tribe)

    assert tribe.population == 0
    assert tribe.extinct is True


def test_immortality_does_not_apply_when_disabled():
    sim = _bare_simulation()
    sim.cycle = 5
    assert sim.immortality_cycles == 0  # disabled by default
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 1
    tribe.food = 0
    tribe.water = 40

    sim._apply_upkeep(tribe)

    assert tribe.extinct is True


def test_immortality_still_allows_chief_succession():
    sim = _bare_simulation()
    sim.cycle = 5
    sim.immortality_cycles = 200
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 8
    tribe.chief_name = "Ashgar"

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._lose_population(tribe, 3)

    assert tribe.population == 8  # protected
    assert tribe.chief_name == ""  # but succession still plays out


def test_simulation_create_defaults_immortality_to_disabled():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    assert sim.immortality_cycles == 0


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


def test_chief_death_carries_the_fallen_chiefs_legacy_into_pending_chief_context():
    """Regression: a fallen chief's philosophy and decree used to just vanish, with
    nothing carried into the next election -- a live-run complaint ('new chiefs aren't
    inheriting the old chief's knowledge'). _install_chief already has a real mechanism
    for exactly this (pending_chief_context, previously wired up only for the conquest
    merge case) -- this doesn't force the new chief to keep anything, it just makes sure
    the next election is actually told what the predecessor believed."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 8
    tribe.chief_name = "Ashgar"
    tribe.chief_philosophy = "aggressive expansion"
    tribe.chief_decree = "seek water"

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._lose_population(tribe, 1)

    assert "Ashgar" in tribe.pending_chief_context
    assert "aggressive expansion" in tribe.pending_chief_context
    assert "seek water" in tribe.pending_chief_context


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
        return {"tribe_1": {"intent": {"visual_action": "GATHER_FOOD"}, "latency_ms": 0.0}}

    with mock.patch.object(sim.scheduler, "run_batch", fake_run_batch):
        await sim.step()

    assert dead.history == frozen_history  # untouched -- no turn was ever prepared for it


@run_async
async def test_step_runs_the_night_cycle_only_on_its_own_interval():
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    sim.cycle = config.NIGHT_CYCLE_EVERY_N_CYCLES - 1  # step() increments before checking

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})), \
         mock.patch("backend.simulation.reflect_on_history", mock.AsyncMock(
             return_value={"revised_philosophy": "x", "changed": False, "reasoning": ""}
         )) as mock_reflect:
        await sim.step()

    mock_reflect.assert_called_once()


@run_async
async def test_step_does_not_run_the_night_cycle_off_its_interval():
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    sim.cycle = config.NIGHT_CYCLE_EVERY_N_CYCLES - 2  # will not land on the interval after +1

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})), \
         mock.patch("backend.simulation.reflect_on_history", mock.AsyncMock()) as mock_reflect:
        await sim.step()

    mock_reflect.assert_not_called()


@run_async
async def test_step_does_not_run_the_night_cycle_for_a_chiefless_tribe():
    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = ""  # between chiefs -- election below fails, so it stays that way
    sim.cycle = config.NIGHT_CYCLE_EVERY_N_CYCLES - 1

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})), \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value={})), \
         mock.patch("backend.simulation.reflect_on_history", mock.AsyncMock()) as mock_reflect:
        await sim.step()

    assert tribe.chief_name == ""  # confirms the test setup actually held
    mock_reflect.assert_not_called()


@run_async
async def test_step_triggers_game_over_and_unloads_models_when_all_tribes_die():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    # Set directly rather than relying on a starvation tick: IDLE's removal means an
    # unresolved turn now always applies a real fallback action (see
    # Simulation._resolve_action), which could itself add resources back before
    # upkeep runs -- this test only cares about the all-extinct game-over/unload
    # behavior, not the starvation mechanic that used to get it there.
    for tribe in sim.tribes.values():
        tribe.population = 0
        tribe.extinct = True

    with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})), \
         mock.patch.object(sim.client, "unload_model", mock.AsyncMock()) as mock_unload:
        await sim.step()

    assert sim.game_over is True
    assert sim.status == "GAME OVER"
    assert {c.args[0] for c in mock_unload.call_args_list} == {"gemma2:2b", "qwen2.5:3b"}


@run_async
async def test_step_unloads_a_single_tribes_model_when_it_alone_goes_extinct():
    """Explicit request: "when a tribe dies off are we unloading the model" --
    previously only the ALL-tribes-extinct case (_trigger_game_over) ever unloaded
    anything; a lone tribe going extinct while another tribe (on a different model)
    plays on left its model resident in Ollama's VRAM for no reason."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    dying, surviving = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    dying.population = 1
    dying.food = 0  # starves to extinction on this tick's upkeep

    async def fake_run_batch(requests):
        # SCOUT dispatches on the expedition's own separate supply -- "no drain on
        # the tribe's stockpile" per its own docstring -- keeping the starvation path
        # below deterministic regardless of IDLE's removal (a fallback action now
        # always does something real, see _resolve_action). Both tribes are still
        # pre-settlement here, so this also has to be one of config.
        # PRE_SETTLEMENT_ACTIONS or _resolve_action would substitute something else.
        return {
            r["id"]: {"intent": {"visual_action": "SCOUT", "target_vector": [55, 55]}, "latency_ms": 0.0}
            for r in requests
        }

    with mock.patch.object(sim.scheduler, "run_batch", fake_run_batch), \
         mock.patch.object(sim.client, "unload_model", mock.AsyncMock()) as mock_unload:
        await sim.step()

    assert dying.extinct is True
    assert surviving.extinct is False
    assert sim.game_over is False
    mock_unload.assert_called_once_with("gemma2:2b")


@run_async
async def test_step_does_not_unload_a_model_still_used_by_a_surviving_tribe():
    """Two tribes sharing the same model -- one going extinct shouldn't unload it out
    from under the tribe still using it."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    dying = sim.tribes["tribe_0"]
    dying.population = 1
    dying.food = 0

    async def fake_run_batch(requests):
        return {
            r["id"]: {"intent": {"visual_action": "SCOUT", "target_vector": [55, 55]}, "latency_ms": 0.0}
            for r in requests
        }

    with mock.patch.object(sim.scheduler, "run_batch", fake_run_batch), \
         mock.patch.object(sim.client, "unload_model", mock.AsyncMock()) as mock_unload:
        await sim.step()

    assert dying.extinct is True
    mock_unload.assert_not_called()


@run_async
async def test_shutdown_unloads_every_model_in_play():
    """Regression test: PAUSE (Simulation.toggle_pause) only ever stopped stepping --
    a browser tab closing or reloading mid-game (not every tribe reaching extinction)
    left that session's models resident in Ollama's VRAM until their keep_alive
    window expired on its own. shutdown() is the explicit cleanup path for that,
    called from both an explicit STOP command and app.py's disconnect handler."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])

    with mock.patch.object(sim.client, "unload_model", mock.AsyncMock()) as mock_unload:
        await sim.shutdown()

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
    assert tribe.water == 34  # 30 + round(3 * 1.25 labor multiplier at population 10)


def test_reaching_monolithic_era_marks_founded_city():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.era = "tribal_synapse"
    tribe.population = 40
    tribe.water = 60
    tribe.stone = 40
    tribe.wood = 50

    sim._advance_era_if_ready(tribe)

    assert tribe.era == "monolithic_era"
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
async def test_install_chief_records_victory_story_for_the_leadership_block():
    """Explicit request: the tribe's own standing prompt context should carry the
    chief's origin story (see prompts.py's LEADERSHIP - ACTIVE CHIEF block), not just
    a distilled philosophy the human sidebar happens to also show."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    fake_result = {
        "chief_name": "Ashgar", "victory_method": "won a wrestling match",
        "guiding_philosophy": "expansion",
    }
    with mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=fake_result)):
        await sim._install_chief(tribe)

    assert tribe.chief_victory == "won a wrestling match"
    request, ctx = sim._prepare_turn(tribe)
    assert "won a wrestling match" in request["prompt"]


def test_chief_victory_clears_alongside_philosophy_on_chief_death():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.chief_philosophy = "expansion"
    tribe.chief_victory = "won a wrestling match"
    tribe.population = 5

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._lose_population(tribe, 1, cause="wolf_attack")

    assert tribe.chief_name == ""
    assert tribe.chief_victory == ""


def test_water_decree_clears_once_water_is_actually_confirmed():
    """Regression test: real live runs showed tribes repeatedly scouting for water
    they'd already found -- the hardcoded water decree, once set, never expired on
    its own, so it kept getting fed into every future turn's prompt regardless of
    whether water had since been confirmed."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_decree = "prioritize dispatching scouts to find reliable water"
    tribe.confirmed_water_sites = [(52, 50)]

    sim._clear_resolved_water_decree(tribe)

    assert tribe.chief_decree == ""


def test_water_decree_persists_while_water_is_still_unconfirmed():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_decree = "prioritize dispatching scouts to find reliable water"
    tribe.confirmed_water_sites = []

    sim._clear_resolved_water_decree(tribe)

    assert tribe.chief_decree == "prioritize dispatching scouts to find reliable water"


def test_clearing_resolved_decree_does_not_touch_an_unrelated_decree():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_decree = "expand our territory aggressively"
    tribe.confirmed_water_sites = [(52, 50)]

    sim._clear_resolved_water_decree(tribe)

    assert tribe.chief_decree == "expand our territory aggressively"


@run_async
async def test_install_chief_survives_a_non_dict_water_decision():
    """Regression test: a real live run against llama3.2:1b crashed the whole
    simulation with 'bool' object has no attribute 'get' -- the model returned
    {"water_decision": true} instead of a nested object. Valid JSON, wrong shape;
    a weak model doing this to a nested field is a different failure mode than
    returning a non-dict top-level response (see test_ollama_client.py)."""
    sim = Simulation([{"name": "A", "model": "llama3.2:1b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.x, tribe.y = 10, 10  # mountains spawn, not on water
    fake_result = {"chief_name": "Ashgar", "guiding_philosophy": "expansion", "water_decision": True}

    with mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=fake_result)):
        await sim._install_chief(tribe)

    assert tribe.chief_name == "Ashgar"
    assert tribe.chief_decree == ""  # a bare `true` isn't a real decree to honor


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

    async def fake_elect(client, model, name, water_needed=False, context=""):
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

    async def fake_elect(client, model, name, water_needed=False, context=""):
        captured["water_needed"] = water_needed
        return {"chief_name": "Ashgar", "guiding_philosophy": "x", "water_decision": {"decreed": False, "reason": ""}}

    with mock.patch("backend.simulation.elect_chief", fake_elect):
        await sim._install_chief(tribe)

    assert captured["water_needed"] is True


@run_async
async def test_install_chief_skips_water_fact_when_already_on_a_lake():
    from backend.world import LAKE_CENTER

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.x, tribe.y = LAKE_CENTER
    captured = {}

    async def fake_elect(client, model, name, water_needed=False, context=""):
        captured["water_needed"] = water_needed
        return {"chief_name": "Ashgar", "guiding_philosophy": "x", "water_decision": {"decreed": False, "reason": ""}}

    with mock.patch("backend.simulation.elect_chief", fake_elect):
        await sim._install_chief(tribe)

    assert captured["water_needed"] is False
