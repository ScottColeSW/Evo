from backend.actions import ACTION_REGISTRY, _execute_trade
from backend.ancestral_matrix import AncestralTraumaMatrix
from backend.simulation import Simulation, Tribe
from backend.world import Landscape

# Most actions here ignore target entirely (only RELOCATE and SCOUT use it) -- (0, 0) is
# just an unused placeholder for those tests.
_NO_TARGET = (0, 0)


def _bare_simulation():
    sim = Simulation.__new__(Simulation)
    sim.world = Landscape(100)
    sim.trauma = AncestralTraumaMatrix(100)
    sim.cycle = 1
    sim.immortality_cycles = 0
    sim.recent_encounters = []
    sim.minor_settlements = []
    return sim


def _settle(sim, tribe):
    """Establishes a real territory + first wall ring, matching what
    Simulation._found_territory does when a tribe first settles -- most build
    actions now place a real footprint via backend/architect.py, which needs a
    territory_center to place anything into."""
    sim._found_territory(tribe)


def _unlock_all_ring0_sections(sim, tribe):
    """Directly unlocks every section of the first wall ring, bypassing the real
    EXPAND_TERRITORY action loop -- EXPAND_TERRITORY's own unlock-one-section-per-
    call behavior is covered directly by its own tests; most other tests just need
    "every section is available to build," not to re-exercise that sequence."""
    for sec in tribe.wall_rings[0]["sections"]:
        sec["unlocked"] = True


def _complete_ring0(sim, tribe, tier=0):
    """Directly marks every section of the first wall ring as unlocked and fully
    built, optionally reinforced to `tier`. Bypasses the real CONSTRUCT_WALL action
    loop -- CONSTRUCT_WALL's own progress/reinforcement math is covered directly by
    its own tests; most other tests just need "the wall already stands.\""""
    for sec in tribe.wall_rings[0]["sections"]:
        sec["unlocked"] = True
        sec["progress"] = 100
        sec["tier"] = tier


def test_first_harvest_at_a_tile_yields_full_amount():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == 10


def test_repeated_harvest_at_the_same_tile_yields_less_each_time():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)
    first_gain = tribe.wood
    tribe.wood = 0
    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)
    second_gain = tribe.wood

    assert second_gain < first_gain


def test_harvesting_elsewhere_is_unaffected_by_a_depleted_tile():
    sim = _bare_simulation()
    depleted_tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(5):
        depleted_tribe.wood = 0
        ACTION_REGISTRY["GATHER_WOOD"](sim, depleted_tribe, "forest", _NO_TARGET)

    fresh_tribe = Tribe("tribe_1", "Mountain Tribe", "qwen2.5:3b", 10, 45, "#fb923c")
    fresh_tribe.wood = 0
    ACTION_REGISTRY["GATHER_WOOD"](sim, fresh_tribe, "forest", _NO_TARGET)

    assert fresh_tribe.wood == 10  # full yield at an untouched tile


def test_wood_yield_is_scaled_down_outside_forest():
    """Regression test: GATHER_WOOD/GATHER_STONE/HUNT_DEER used to pay the exact same
    flat yield in every biome (only local depletion scaled them) -- a tribe standing on
    a bare mountain peak could "hunt deer" as effectively as one deep in a forest.
    Resources should actually depend on what's around the tribe."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 20, 20, "#fb923c")
    tribe.wood = 0

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "mountains", _NO_TARGET)

    assert 0 < tribe.wood < 10  # some wood, but far less than a forest tile


def test_stone_yield_is_reduced_outside_mountains():
    """Raised from a 0.1x multiplier to 0.25x (2026-09-02 tuning pass, see
    BIOME_YIELD_MULTIPLIER's comment) -- still clearly worse than mountains' 1.0x, but
    no longer an effective lockout on ever bootstrapping a Quarry off-mountain."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.stone = 0

    ACTION_REGISTRY["GATHER_STONE"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.stone == 2  # round(10 * 0.25)


def test_storage_cap_is_the_base_amount_with_no_warehouses():
    from backend.actions import _storage_cap
    from backend import config

    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    assert _storage_cap(tribe) == config.STORAGE_CAP_BASE


def test_storage_cap_rises_with_each_warehouse_built():
    from backend.actions import _storage_cap
    from backend import config

    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.warehouses_built = 2

    assert _storage_cap(tribe) == config.STORAGE_CAP_BASE + 2 * config.WAREHOUSE_STORAGE_BONUS_PER_BUILDING


def test_gather_wood_is_wasted_once_storage_is_already_full():
    """Explicit design goal: the tribe should be *told*, as the real result of the
    turn it just took, not have the overflow silently vanish. Explicit follow-up:
    "these guys need punishment for choosing the wrong thing... for waste when
    they overfill the storage" -- real waste now radiates real negative trauma."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = config.STORAGE_CAP_BASE

    result = ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == config.STORAGE_CAP_BASE  # unchanged, nothing fit
    assert "wood stores are already full" in result
    assert "DREAD" in sim.trauma.bias_string(50, 50)


def test_gather_wood_partially_fits_right_at_the_edge_of_the_cap():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = config.STORAGE_CAP_BASE - 3  # first harvest at this tile yields 10

    result = ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == config.STORAGE_CAP_BASE
    assert "wood stores are nearly full" in result
    assert "only 3 of 10 fits" in result
    assert "DREAD" in sim.trauma.bias_string(50, 50)


def test_gather_wood_below_the_cap_is_unaffected():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    result = ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == 10
    assert result is None
    assert "DREAD" not in sim.trauma.bias_string(50, 50)  # no waste, no punishment


def test_build_warehouse_is_repeatable_and_raises_the_storage_cap():
    from backend.actions import _storage_cap
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.WAREHOUSE_WOOD_COST * 3
    tribe.stone = config.WAREHOUSE_STONE_COST * 3
    cap_before = _storage_cap(tribe)

    result = ACTION_REGISTRY["BUILD_WAREHOUSE"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.warehouses_built == 1
    assert _storage_cap(tribe) == cap_before + config.WAREHOUSE_STORAGE_BONUS_PER_BUILDING
    assert "warehouse rises" in result

    ACTION_REGISTRY["BUILD_WAREHOUSE"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.warehouses_built == 2
    assert _storage_cap(tribe) == cap_before + 2 * config.WAREHOUSE_STORAGE_BONUS_PER_BUILDING


def test_build_warehouse_no_op_when_cannot_afford_it():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.WAREHOUSE_WOOD_COST - 1
    tribe.stone = config.WAREHOUSE_STONE_COST

    assert ACTION_REGISTRY["BUILD_WAREHOUSE"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.warehouses_built == 0


def test_second_fire_at_the_same_tile_costs_nothing_and_gains_no_pride():
    """Regression test: a real 8-cycle live run showed a model spamming BUILD_FIRE at
    the same tile every cycle, each one radiating more ancestral pride at zero
    additional benefit -- a self-reinforcing loop that made staying in one place look
    increasingly attractive. A second fire where one already burns should do nothing."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50

    ACTION_REGISTRY["BUILD_FIRE"](sim, tribe, "plains", _NO_TARGET)
    wood_after_first = tribe.wood
    pride_after_first = float(sim.trauma.ghost_tensor[50, 50])

    ACTION_REGISTRY["BUILD_FIRE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.wood == wood_after_first  # no wood spent the second time
    assert float(sim.trauma.ghost_tensor[50, 50]) == pride_after_first  # no extra pride


def test_construct_wall_adds_progress_and_costs_proportional_resources():
    """Explicit request: a wall is built in stages, like a crop, not finished in one
    action -- one CONSTRUCT_WALL call should leave a real, partial construction, not
    an instantly-complete wall or nothing at all."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _unlock_all_ring0_sections(sim, tribe)
    tribe.wood = 50
    tribe.stone = 50

    from backend import city_layout

    ring_i, sec_i = city_layout.next_wall_work_section(tribe)  # first real (non-natural) section in compass order
    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    section = tribe.wall_rings[ring_i]["sections"][sec_i]
    assert 0 < section["progress"] < 100
    assert tribe.wood < 50
    assert tribe.stone < 50


def test_construct_wall_no_op_when_cannot_afford_the_proportional_cost():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _unlock_all_ring0_sections(sim, tribe)
    tribe.wood = 0
    tribe.stone = 0

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.wall_rings[0]["sections"][0]["progress"] == 0


def test_construct_wall_no_op_when_nothing_is_unlocked():
    """A freshly founded territory starts with every non-natural section locked --
    EXPAND_TERRITORY must unlock at least one before CONSTRUCT_WALL has anything to
    do (see actions._expand_territory's own tests)."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = 100
    tribe.stone = 100

    result = ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    assert result is not None and "unlocked" in result
    assert (tribe.wood, tribe.stone) == (100, 100)


def test_construct_wall_is_a_no_op_once_complete():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe, tier=config.WALL_MAX_LAYERS)  # every ring-0 section fully built and maxed
    tribe.wood = 100
    tribe.stone = 100
    wood_before, stone_before = tribe.wood, tribe.stone

    result = ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    assert "EXPAND_TERRITORY" in result  # every ring-0 section is done -- nudge toward the next one
    assert (tribe.wood, tribe.stone) == (wood_before, stone_before)


def test_larger_population_builds_wall_progress_faster():
    """Reuses _labor_multiplier -- the same 'more hands get more done' concept
    _harvest already uses -- rather than a separate team-size notion."""
    sim = _bare_simulation()
    small = Tribe("tribe_0", "Small Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, small)
    _unlock_all_ring0_sections(sim, small)
    small.wood, small.stone, small.population = 100, 100, 8
    big = Tribe("tribe_1", "Big Tribe", "gemma2:2b", 60, 60, "#f97316")
    _settle(sim, big)
    _unlock_all_ring0_sections(sim, big)
    big.wood, big.stone, big.population = 100, 100, 40

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, small, "plains", _NO_TARGET)
    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, big, "plains", _NO_TARGET)

    small_progress = small.wall_rings[0]["sections"][0]["progress"]
    big_progress = big.wall_rings[0]["sections"][0]["progress"]
    assert big_progress > small_progress


def test_build_long_house_requires_the_wall_to_be_complete_first():
    """Explicit request: BUILD_LONG_HOUSE is gated on the first wall ring already
    being complete -- defense before shelter."""
    from backend import city_layout

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _unlock_all_ring0_sections(sim, tribe)
    tribe.wood = 100
    tribe.stone = 100
    ring_i, sec_i = city_layout.next_wall_work_section(tribe)
    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)  # one section, partial only
    assert 0 < tribe.wall_rings[ring_i]["sections"][sec_i]["progress"] < 100

    result = ACTION_REGISTRY["BUILD_LONG_HOUSE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.long_houses_built == 0
    assert "wall ring must be finished" in result


def test_build_long_house_succeeds_once_wall_is_complete():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe)
    tribe.wood = 200
    tribe.stone = 200
    wood_before, stone_before = tribe.wood, tribe.stone

    result = ACTION_REGISTRY["BUILD_LONG_HOUSE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.long_houses_built == 1
    assert tribe.wood < wood_before
    assert tribe.stone < stone_before
    assert any(t["name"] == "Master Builder" for t in tribe.trophies)
    assert "long house rises" in result
    assert any(b["type"] == "long_house" for b in tribe.buildings)


def test_build_long_house_no_op_when_cannot_afford_it():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe)
    tribe.wood = config.LONG_HOUSE_WOOD_COST - 1
    tribe.stone = config.LONG_HOUSE_STONE_COST

    ACTION_REGISTRY["BUILD_LONG_HOUSE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.long_houses_built == 0


def test_build_long_house_repeats_as_population_grows():
    """Explicit correction: "most structures they only need 1 of. but house
    builds are dependant on population needs.\""""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe)
    tribe.wood = 200
    tribe.stone = 200

    ACTION_REGISTRY["BUILD_LONG_HOUSE"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.long_houses_built == 1
    # No more housing need yet at the same population -- a no-op, not a second house.
    assert ACTION_REGISTRY["BUILD_LONG_HOUSE"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.long_houses_built == 1

    tribe.population = config.HOUSING_POPULATION_PER_LONG_HOUSE * 2
    result = ACTION_REGISTRY["BUILD_LONG_HOUSE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.long_houses_built == 2
    assert "another long house rises" in result


def test_build_long_house_is_a_no_op_once_already_built():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe)
    tribe.wood = 200
    tribe.stone = 200
    ACTION_REGISTRY["BUILD_LONG_HOUSE"](sim, tribe, "plains", _NO_TARGET)
    wood_after, stone_after = tribe.wood, tribe.stone
    trophies_after = list(tribe.trophies)

    result = ACTION_REGISTRY["BUILD_LONG_HOUSE"](sim, tribe, "plains", _NO_TARGET)

    assert result is None
    assert (tribe.wood, tribe.stone) == (wood_after, stone_after)
    assert tribe.trophies == trophies_after


def test_build_keep_requires_ten_long_houses():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = 200
    tribe.stone = 200

    result = ACTION_REGISTRY["BUILD_KEEP"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.keep_built is False
    assert "10 long houses" in result

    tribe.long_houses_built = config.KEEP_LONG_HOUSES_REQUIRED
    wood_before, stone_before = tribe.wood, tribe.stone
    result = ACTION_REGISTRY["BUILD_KEEP"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.keep_built is True
    assert tribe.wood < wood_before
    assert tribe.stone < stone_before
    assert any(t["name"] == "Keep Warden" for t in tribe.trophies)
    assert "keep rises" in result
    assert any(b["type"] == "keep" for b in tribe.buildings)


def test_build_fortress_requires_keep_and_forty_long_houses():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = 200
    tribe.stone = 200
    tribe.long_houses_built = config.FORTRESS_LONG_HOUSES_REQUIRED

    result = ACTION_REGISTRY["BUILD_FORTRESS"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.fortress_built is False
    assert "keep must be built" in result

    tribe.keep_built = True
    result = ACTION_REGISTRY["BUILD_FORTRESS"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.fortress_built is True
    assert any(t["name"] == "Fortress Warden" for t in tribe.trophies)
    assert "fortress rises" in result


def test_build_castle_requires_the_fortress_first():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = 200
    tribe.stone = 200

    result = ACTION_REGISTRY["BUILD_CASTLE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.castle_built is False
    assert "fortress must be built" in result


def test_build_castle_succeeds_once_fortress_and_seventy_long_houses_exist():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    # A Castle's 8x8 footprint doesn't fit inside a freshly-founded (radius-12)
    # territory alongside the wall ring sitting right at that same boundary -- by the
    # time CASTLE_LONG_HOUSES_REQUIRED long houses and a Fortress are real, EXPAND_
    # TERRITORY would have grown real territory well past this in actual play.
    tribe.territory_radius = 40
    tribe.wood = 300
    tribe.stone = 300
    tribe.fortress_built = True
    tribe.long_houses_built = config.CASTLE_LONG_HOUSES_REQUIRED
    wood_before, stone_before = tribe.wood, tribe.stone

    result = ACTION_REGISTRY["BUILD_CASTLE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.castle_built is True
    assert tribe.wood < wood_before
    assert tribe.stone < stone_before
    assert any(t["name"] == "Castle Builder" for t in tribe.trophies)
    assert "castle rises" in result


def test_wall_can_be_reinforced_with_a_second_layer_once_complete():
    from backend import city_layout, config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe, tier=1)  # every section already built, one reinforcement tier already applied
    tribe.wood = 300
    tribe.stone = 300
    ring_i, sec_i = city_layout.next_wall_work_section(tribe)

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.wall_rings[ring_i]["sections"][sec_i]["tier"] == config.WALL_MAX_LAYERS


def test_wall_reinforcement_stops_at_the_layer_cap():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe, tier=config.WALL_MAX_LAYERS)
    tribe.wood = 1000
    tribe.stone = 1000
    wood_before, stone_before = tribe.wood, tribe.stone

    assert "EXPAND_TERRITORY" in ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)
    assert (tribe.wood, tribe.stone) == (wood_before, stone_before)


def test_build_moat_requires_wall_fully_reinforced():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.MOAT_WOOD_COST
    tribe.stone = config.MOAT_STONE_COST

    assert ACTION_REGISTRY["BUILD_MOAT"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.moat_built is False

    _complete_ring0(sim, tribe, tier=config.WALL_MAX_LAYERS)
    result = ACTION_REGISTRY["BUILD_MOAT"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.moat_built is True
    assert any(t["name"] == "Moat Digger" for t in tribe.trophies)
    assert "moat is dug" in result


def test_build_road_grants_a_flat_expedition_speed_bonus():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 100
    tribe.stone = 100

    result = ACTION_REGISTRY["BUILD_ROAD"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.road_built is True
    assert "road is built" in result
    assert config.ROAD_SPEED_BONUS > 0


def test_build_road_is_a_no_op_once_already_built():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 100
    tribe.stone = 100
    tribe.road_built = True
    wood_before = tribe.wood

    result = ACTION_REGISTRY["BUILD_ROAD"](sim, tribe, "plains", _NO_TARGET)

    assert result is None
    assert tribe.wood == wood_before


def test_expand_territory_no_op_without_any_territory_yet():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 200
    tribe.stone = 200

    assert ACTION_REGISTRY["EXPAND_TERRITORY"](sim, tribe, "plains", _NO_TARGET) is None


def test_expand_territory_unlocks_one_section_without_moving_the_radius_yet():
    """Live-run correction: "Wall Sections are being rendered on screen as a
    box around the settlement instead of portions of Wall being placed just
    inside the Territory dotted outline." territory_radius (what drawTerritory
    actually draws) used to grow by its own separately-scaled increment on
    every call, independent of the wall ring's real geometry -- now it only
    ever moves when a whole new ring is actually created (see the test below),
    not when a call merely unlocks one more section within the current ring."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = 200
    tribe.stone = 200
    unlocked_before = sum(1 for s in tribe.wall_rings[0]["sections"] if s["unlocked"])

    result = ACTION_REGISTRY["EXPAND_TERRITORY"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.territory_radius == config.WALL_RING_RADIUS_STEP  # unchanged -- still ring 0
    unlocked_after = sum(1 for s in tribe.wall_rings[0]["sections"] if s["unlocked"])
    assert unlocked_after == unlocked_before + 1
    assert any(t["name"] == "Territory Expander" for t in tribe.trophies)
    assert "territory expands" in result


def test_expand_territory_no_op_when_unaffordable():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.TERRITORY_EXPANSION_WOOD_COST - 1
    tribe.stone = config.TERRITORY_EXPANSION_STONE_COST

    assert ACTION_REGISTRY["EXPAND_TERRITORY"](sim, tribe, "plains", _NO_TARGET) is None


def test_expand_territory_requires_current_ring_fully_reinforced_before_opening_next():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _unlock_all_ring0_sections(sim, tribe)  # every section unlocked, none built yet
    tribe.wood = 200
    tribe.stone = 200

    result = ACTION_REGISTRY["EXPAND_TERRITORY"](sim, tribe, "plains", _NO_TARGET)

    assert result is not None and "fully reinforced" in result
    assert len(tribe.wall_rings) == 1


def test_expand_territory_opens_a_new_ring_once_the_current_one_is_maxed():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe, tier=config.WALL_MAX_LAYERS)
    tribe.wood = 200
    tribe.stone = 200

    result = ACTION_REGISTRY["EXPAND_TERRITORY"](sim, tribe, "plains", _NO_TARGET)

    assert len(tribe.wall_rings) == 2
    assert tribe.wall_rings[1]["radius"] > tribe.wall_rings[0]["radius"]
    # See "wall sections rendered as a box inside the territory outline" fix --
    # territory_radius now always matches the real outermost ring's own radius.
    assert tribe.territory_radius == tribe.wall_rings[1]["radius"] == config.WALL_RING_RADIUS_STEP * 2
    assert "territory expands" in result


def test_build_dock_is_a_no_op_before_fishing_is_learned():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50

    result = ACTION_REGISTRY["BUILD_DOCK"](sim, tribe, "plains", _NO_TARGET)

    assert result is None
    assert tribe.dock_built is False
    assert tribe.wood == 50  # no wood spent on a no-op


def test_build_dock_is_a_real_reward_even_though_it_cant_boost_a_manual_catch():
    """Explicit correction: "they don't need to CATCH_FISH once they know how"
    -- CATCH_FISH retires from available_actions the instant fishing_learned is
    set (see Simulation._prepare_turn), and BUILD_DOCK requires fishing_learned
    already, so a dock can never coexist with a reachable manual catch. Its own
    bonus (config.DOCK_FISH_CATCH_BONUS_FRACTION) now applies to the passive
    daily supply instead (see test_advance_fish_supply_dock_bonus_stacks_with_
    fishery in test_simulation.py) -- this just confirms building it still
    works and no longer touches CATCH_FISH's own yield at all."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50
    tribe.fishing_learned = True

    dock_result = ACTION_REGISTRY["BUILD_DOCK"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.dock_built is True
    assert "dock rises" in dock_result


def test_sawmill_triples_every_future_wood_harvest():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    with mock.patch("backend.actions._harvest", return_value=10):
        ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)
        assert tribe.wood == 10

        tribe.sawmill_built = True
        ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)
        assert tribe.wood == 40  # +10 unboosted, then +30 (10 * SAWMILL_WOOD_MULTIPLIER)


def test_quarry_triples_every_future_stone_harvest():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.stone = 0

    with mock.patch("backend.actions._harvest", return_value=10):
        ACTION_REGISTRY["GATHER_STONE"](sim, tribe, "mountains", _NO_TARGET)
        assert tribe.stone == 10

        tribe.quarry_built = True
        ACTION_REGISTRY["GATHER_STONE"](sim, tribe, "mountains", _NO_TARGET)
        assert tribe.stone == 40  # +10 unboosted, then +30 (10 * QUARRY_STONE_MULTIPLIER)


def test_food_multiplier_stacks_cooking_and_kitchen():
    """Redesigned 2026-09-02 ("that's a mess, let's build it back up properly"):
    cooking now multiplies food production at the harvest point, the same shape
    SAWMILL_WOOD_MULTIPLIER/QUARRY_STONE_MULTIPLIER already use, instead of dividing
    upkeep consumption."""
    from backend import config
    from backend.actions import _food_multiplier

    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    assert _food_multiplier(tribe) == 1.0

    tribe.cooking_learned = True
    assert _food_multiplier(tribe) == config.COOKING_FOOD_MULTIPLIER

    tribe.kitchen_built = True
    assert _food_multiplier(tribe) == config.COOKING_FOOD_MULTIPLIER * config.KITCHEN_FOOD_MULTIPLIER


def test_cooking_triples_every_future_gather_food_harvest():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0

    with mock.patch("backend.actions._harvest", return_value=10):
        ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "plains", _NO_TARGET)
        assert tribe.food == 10

        tribe.cooking_learned = True
        ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "plains", _NO_TARGET)
        assert tribe.food == 40  # +10 unboosted, then +30 (10 * COOKING_FOOD_MULTIPLIER)


def test_cooking_triples_every_future_hunt_deer_catch():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0
    tribe.cooking_learned = True

    with mock.patch("backend.actions.random.random", return_value=1.0), \
         mock.patch("backend.actions._harvest", return_value=15):
        ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.food == 45  # 15 * COOKING_FOOD_MULTIPLIER


def test_build_sawmill_requires_a_real_successful_wood_gather():
    """Explicit correction: "the Sawmill is... online easily if they Gather Wood
    successfully" -- replaced the old Long House/fishing/scouted-site gate."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.SAWMILL_WOOD_COST
    tribe.stone = config.SAWMILL_STONE_COST

    assert ACTION_REGISTRY["BUILD_SAWMILL"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.sawmill_built is False

    tribe.wood_ever_gathered = True
    result = ACTION_REGISTRY["BUILD_SAWMILL"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.sawmill_built is True
    assert tribe.lumber_site is None  # no site required, none was ever scouted
    assert "sawmill rises" in result


def test_build_sawmill_uses_a_scouted_site_opportunistically():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.SAWMILL_WOOD_COST
    tribe.stone = config.SAWMILL_STONE_COST
    tribe.wood_ever_gathered = True
    tribe.lumber_sites.append((7, 7))

    ACTION_REGISTRY["BUILD_SAWMILL"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.lumber_site == (7, 7)


def test_build_quarry_requires_a_real_successful_stone_gather():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.QUARRY_WOOD_COST
    tribe.stone = config.QUARRY_STONE_COST

    assert ACTION_REGISTRY["BUILD_QUARRY"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.quarry_built is False

    tribe.stone_ever_gathered = True
    result = ACTION_REGISTRY["BUILD_QUARRY"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.quarry_built is True
    assert tribe.quarry_site is None  # no site required, none was ever scouted
    assert "quarry opens" in result


def test_build_mine_requires_quarry_and_a_discovered_site():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.MINE_WOOD_COST
    tribe.stone = config.MINE_STONE_COST
    tribe.quarry_built = True

    # No discovered site yet -- quarry_built alone isn't enough.
    assert ACTION_REGISTRY["BUILD_MINE"](sim, tribe, "mountains", _NO_TARGET) is None
    assert tribe.mine_built is False

    tribe.mine_sites.append({"x": 10, "y": 10, "biome": "mountains", "resource": "Orosite Ore"})
    result = ACTION_REGISTRY["BUILD_MINE"](sim, tribe, "mountains", _NO_TARGET)

    assert tribe.mine_built is True
    assert tribe.mine_resource_name == "Orosite Ore"
    assert "Orosite Ore" in result


def test_build_mine_locks_in_the_most_recently_discovered_site():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.MINE_WOOD_COST
    tribe.stone = config.MINE_STONE_COST
    tribe.quarry_built = True
    tribe.mine_sites.append({"x": 10, "y": 10, "biome": "mountains", "resource": "Orosite Ore"})
    tribe.mine_sites.append({"x": 20, "y": 60, "biome": "forest", "resource": "Whisperwood Amber"})

    ACTION_REGISTRY["BUILD_MINE"](sim, tribe, "mountains", _NO_TARGET)

    assert tribe.mine_resource_name == "Whisperwood Amber"


def test_build_tannery_requires_a_real_successful_hunt():
    """Explicit correction: "the Tannery should come online easily, as they only
    need to have hunted" -- replaced the old Long House/fishing/scouted-Rabbit-
    Warren gate."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.TANNERY_WOOD_COST
    tribe.stone = config.TANNERY_STONE_COST

    assert ACTION_REGISTRY["BUILD_TANNERY"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.tannery_built is False

    tribe.hunt_ever_succeeded = True
    result = ACTION_REGISTRY["BUILD_TANNERY"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.tannery_built is True
    assert tribe.tannery_site is None  # no site required, none was ever scouted
    assert "tannery is built" in result


def test_build_tannery_uses_a_scouted_rabbit_warren_opportunistically():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.TANNERY_WOOD_COST
    tribe.stone = config.TANNERY_STONE_COST
    tribe.hunt_ever_succeeded = True
    # A Deer Stand alone doesn't count -- only a Rabbit Warren produces fur.
    tribe.wildlife_sites.append({"x": 5, "y": 5, "type": "Deer Stand"})
    tribe.wildlife_sites.append({"x": 10, "y": 12, "type": "Rabbit Warren"})

    ACTION_REGISTRY["BUILD_TANNERY"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.tannery_site == (10, 12)


def test_hunt_deer_yields_a_meat_bonus_once_tannery_is_built():
    """Explicit request: "it also gives the meat to the kitchen (2 meat per
    catch) which cooks it (multiplier)"."""
    from unittest import mock

    from backend import config

    # Two separate sims/tribes, not one reused across both calls -- _harvest
    # depletes world state on every call, so reusing one would understate the
    # second hunt's real yield and mask the bonus.
    sim_a = _bare_simulation()
    tribe_a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    with mock.patch("backend.actions.random.random", return_value=1.0):
        ACTION_REGISTRY["HUNT_DEER"](sim_a, tribe_a, "forest", _NO_TARGET)
    without_tannery = tribe_a.food

    sim_b = _bare_simulation()
    tribe_b = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe_b.tannery_built = True
    with mock.patch("backend.actions.random.random", return_value=1.0):
        ACTION_REGISTRY["HUNT_DEER"](sim_b, tribe_b, "forest", _NO_TARGET)

    assert tribe_b.food == without_tannery + config.TANNERY_MEAT_BONUS_PER_HUNT


def test_gather_eggs_success_sets_the_hatchery_prerequisite():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.eggs_ever_gathered is True


def test_build_hatchery_requires_a_real_wild_egg_find():
    """Explicit follow-up: "the Flock and the Eggs self generate. So, maybe
    after they GATHER_EGGS in the wild, they can have a Hatchery."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.HATCHERY_WOOD_COST
    tribe.stone = config.HATCHERY_STONE_COST

    assert ACTION_REGISTRY["BUILD_HATCHERY"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.hatchery_built is False

    tribe.eggs_ever_gathered = True
    result = ACTION_REGISTRY["BUILD_HATCHERY"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.hatchery_built is True
    assert "hatchery is built" in result


def test_build_bath_house_has_no_prerequisite_beyond_being_settled_and_affordable():
    """Explicit request: "bath house bolsters Well-Being upkeep once built" --
    no proven-success gate the way Sawmill/Quarry/Tannery/Hatchery need, the
    same "infrastructure from the moment it's unlocked" shape Warehouse uses."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    assert ACTION_REGISTRY["BUILD_BATH_HOUSE"](sim, tribe, "plains", _NO_TARGET) is None  # not settled yet
    assert tribe.bath_house_built is False

    _settle(sim, tribe)
    tribe.wood = config.BATH_HOUSE_WOOD_COST
    tribe.stone = config.BATH_HOUSE_STONE_COST
    result = ACTION_REGISTRY["BUILD_BATH_HOUSE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.bath_house_built is True
    assert "bath house is built" in result


def test_build_bath_house_is_a_no_op_once_already_built():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.bath_house_built = True
    tribe.wood = config.BATH_HOUSE_WOOD_COST
    tribe.stone = config.BATH_HOUSE_STONE_COST

    assert ACTION_REGISTRY["BUILD_BATH_HOUSE"](sim, tribe, "plains", _NO_TARGET) is None


def test_build_library_requires_a_long_house_first():
    """Explicit request: a Library condenses the tribe's own remembered history and
    unlocks RESEARCH -- gated on long_houses_built, the same "building homes" real
    prerequisite Kitchen/Sawmill/Quarry already use."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.LIBRARY_WOOD_COST
    tribe.stone = config.LIBRARY_STONE_COST

    assert ACTION_REGISTRY["BUILD_LIBRARY"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.library_built is False

    tribe.long_houses_built = 1
    result = ACTION_REGISTRY["BUILD_LIBRARY"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.library_built is True
    assert "library is built" in result


def test_build_library_is_a_no_op_once_already_built():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.long_houses_built = 1
    tribe.library_built = True
    tribe.wood = config.LIBRARY_WOOD_COST
    tribe.stone = config.LIBRARY_STONE_COST

    assert ACTION_REGISTRY["BUILD_LIBRARY"](sim, tribe, "plains", _NO_TARGET) is None


def test_research_requires_a_library_first():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 100

    assert ACTION_REGISTRY["RESEARCH"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.research_completed == 0


def test_research_distills_the_tribes_own_memory_into_a_library_entry():
    """The Library's real payoff: RESEARCH pulls the tribe's own highest-weight
    remembered episodes (same ranking TribeMemory.consolidate uses for its own
    taboo cut) into one permanent Library entry, spending a little wood and
    permanently counting toward the next era's discount (Simulation.
    _advance_era_if_ready)."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.library_built = True
    tribe.wood = 100
    tribe.memory.remember("a wolf attacked near the river", cycle=1, weight=0.9)
    tribe.memory.remember("gathered some extra wood", cycle=2, weight=0.1)

    result = ACTION_REGISTRY["RESEARCH"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.research_completed == 1
    assert len(tribe.library_entries) == 1
    assert "wolf attacked near the river" in tribe.library_entries[0]["summary"]
    assert tribe.wood == 100 - config.RESEARCH_WOOD_COST
    assert "insight" in result


def test_research_with_no_memory_yet_is_a_real_no_op_not_a_silent_one():
    """Deliberately left reachable even with nothing to study yet -- same as
    CONSTRUCT_WALL's own "let the action's own message surface instead" case,
    not a guaranteed no-op the affordability table should hide."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.library_built = True
    tribe.wood = 100

    result = ACTION_REGISTRY["RESEARCH"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.research_completed == 0
    assert tribe.library_entries == []
    assert tribe.wood == 100  # nothing spent on a no-op
    assert "worth recording" in result


def test_build_well_has_no_prerequisite_beyond_being_settled_and_affordable():
    """Explicit request: a Well boosts water's passive income, no proven-success
    gate needed, the same "infrastructure from the moment it's unlocked" shape
    Bath House/Warehouse already use."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    assert ACTION_REGISTRY["BUILD_WELL"](sim, tribe, "plains", _NO_TARGET) is None  # not settled yet
    assert tribe.well_built is False

    _settle(sim, tribe)
    tribe.wood = config.WELL_WOOD_COST
    tribe.stone = config.WELL_STONE_COST
    result = ACTION_REGISTRY["BUILD_WELL"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.well_built is True
    assert "well is dug" in result


def test_build_well_is_a_no_op_once_already_built():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.well_built = True
    tribe.wood = config.WELL_WOOD_COST
    tribe.stone = config.WELL_STONE_COST

    assert ACTION_REGISTRY["BUILD_WELL"](sim, tribe, "plains", _NO_TARGET) is None


def test_gather_ore_requires_a_real_mine():
    """Explicit correction: "GATHER_ORE only comes in if they Discover a Mine.
    They do not harvest on a Discovery, so they have to fetch it once."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")

    assert ACTION_REGISTRY["GATHER_ORE"](sim, tribe, "mountains", _NO_TARGET) is None
    assert tribe.ore_ever_gathered is False

    tribe.mine_built = True
    tribe.mine_resource_name = "Orosite Ore"
    result = ACTION_REGISTRY["GATHER_ORE"](sim, tribe, "mountains", _NO_TARGET)

    assert tribe.ore_ever_gathered is True
    assert tribe.unique_resources["Orosite Ore"] > 0
    assert "Orosite Ore" in result


def test_build_forge_requires_mine_and_at_least_one_ore():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.FORGE_WOOD_COST
    tribe.stone = config.FORGE_STONE_COST

    # No mine at all yet.
    assert ACTION_REGISTRY["BUILD_FORGE"](sim, tribe, "plains", _NO_TARGET) is None

    tribe.mine_built = True
    tribe.mine_resource_name = "Orosite Ore"
    tribe.unique_resources["Orosite Ore"] = 0

    # A mine exists, but nothing's actually been produced yet.
    result = ACTION_REGISTRY["BUILD_FORGE"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.forge_built is False
    assert "not enough ore" in result

    tribe.unique_resources["Orosite Ore"] = 1
    result = ACTION_REGISTRY["BUILD_FORGE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.forge_built is True
    assert tribe.wood == 0
    assert tribe.stone == 0
    assert any(t["name"] == "Blacksmith" for t in tribe.trophies)
    assert "forge is built" in result


def test_forge_item_requires_forge_built_and_consumes_ore_and_wood():
    from unittest import mock

    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.mine_resource_name = "Orosite Ore"
    tribe.unique_resources["Orosite Ore"] = 5
    tribe.wood = config.FORGE_ITEM_WOOD_COST

    # No forge yet -- ore/wood in stock isn't enough on its own.
    assert ACTION_REGISTRY["FORGE_ITEM"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.items == []

    tribe.forge_built = True
    with mock.patch("backend.actions.random.choice", side_effect=["tool", "Whetstone"]):
        result = ACTION_REGISTRY["FORGE_ITEM"](sim, tribe, "plains", _NO_TARGET)

    assert len(tribe.items) == 1
    item = tribe.items[0]
    assert item == {"name": "Whetstone", "type": "tool", "value": config.ITEM_VALUE_BY_TYPE["tool"], "cycle_made": sim.cycle}
    assert tribe.unique_resources["Orosite Ore"] == 5 - config.FORGE_ITEM_ORE_COST
    assert tribe.wood == 0
    assert "Whetstone" in result
    assert any(t["name"] == "Artisan" for t in tribe.trophies)


def test_forge_item_is_a_no_op_without_enough_ore_or_wood():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.forge_built = True
    tribe.mine_resource_name = "Orosite Ore"
    tribe.unique_resources["Orosite Ore"] = 0
    tribe.wood = config.FORGE_ITEM_WOOD_COST

    assert ACTION_REGISTRY["FORGE_ITEM"](sim, tribe, "plains", _NO_TARGET) is None

    tribe.unique_resources["Orosite Ore"] = config.FORGE_ITEM_ORE_COST
    tribe.wood = 0

    assert ACTION_REGISTRY["FORGE_ITEM"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.items == []


def test_forge_item_is_capped_and_the_cap_rises_with_warehouses():
    """Explicit follow-up to the passive-income storage-cap fix: tribe.items had
    no ceiling at all, the same unbounded-hoarding shape that fix closed for bulk
    resources."""
    from unittest import mock

    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.forge_built = True
    tribe.mine_resource_name = "Orosite Ore"
    tribe.unique_resources["Orosite Ore"] = 100
    tribe.wood = 1000
    tribe.items = [{"name": "Whetstone", "type": "tool", "value": 8, "cycle_made": 0}] * config.ITEM_STORAGE_CAP_BASE

    result = ACTION_REGISTRY["FORGE_ITEM"](sim, tribe, "plains", _NO_TARGET)

    assert len(tribe.items) == config.ITEM_STORAGE_CAP_BASE  # nothing added
    assert "already full" in result
    assert tribe.wood == 1000  # no cost spent on a no-op
    assert tribe.unique_resources["Orosite Ore"] == 100

    tribe.warehouses_built = 1
    with mock.patch("backend.actions.random.choice", side_effect=["tool", "Whetstone"]):
        result = ACTION_REGISTRY["FORGE_ITEM"](sim, tribe, "plains", _NO_TARGET)

    assert len(tribe.items) == config.ITEM_STORAGE_CAP_BASE + 1
    assert "already full" not in result


def test_use_item_converts_value_to_wood_and_stone():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.items = [{"name": "War Hammer", "type": "weapon", "value": config.ITEM_VALUE_BY_TYPE["weapon"], "cycle_made": 1}]
    wood_before, stone_before = tribe.wood, tribe.stone

    result = ACTION_REGISTRY["USE_ITEM"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.items == []
    stone_gain = round(config.ITEM_VALUE_BY_TYPE["weapon"] * config.USE_ITEM_STONE_SHARE)
    wood_gain = config.ITEM_VALUE_BY_TYPE["weapon"] - stone_gain
    assert tribe.wood == wood_before + wood_gain
    assert tribe.stone == stone_before + stone_gain
    assert "War Hammer" in result


def test_use_item_is_a_no_op_with_no_items():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 50, 50, "#c084fc")
    wood_before, stone_before = tribe.wood, tribe.stone

    assert ACTION_REGISTRY["USE_ITEM"](sim, tribe, "plains", _NO_TARGET) is None
    assert (tribe.wood, tribe.stone) == (wood_before, stone_before)


def test_trade_exchanges_unique_resources_too():
    """Explicit request confirms the original "Mine & unique resource" design
    gap: "maybe some hunters want a Tannery and they can trade furs too." Trade
    used to only ever swap the same four generic resources."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    partner = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 60, 60, "#fb923c")
    tribe.unique_resources = {"Fur": 100}
    partner.unique_resources = {"Orosite Ore": 50}

    _execute_trade(sim, tribe, partner)

    assert tribe.unique_resources["Fur"] == 100 - round(100 * config.TRADE_GIFT_FRACTION)
    assert tribe.unique_resources["Orosite Ore"] == round(50 * config.TRADE_GIFT_FRACTION)
    assert partner.unique_resources["Orosite Ore"] == 50 - round(50 * config.TRADE_GIFT_FRACTION)
    assert partner.unique_resources["Fur"] == round(100 * config.TRADE_GIFT_FRACTION)


def test_trade_exchanges_one_crafted_item_each_way_when_either_side_has_any():
    """A forged item is discrete, unlike the fractional resource gifts above -- each
    side that has any items gives up its oldest one. Also guards against a tribe
    with zero items immediately receiving back the very item it was just given."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    partner = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 60, 60, "#fb923c")
    tribe_item = {"name": "Whetstone", "type": "tool", "value": 8, "cycle_made": 1}
    tribe.items = [tribe_item]
    partner.items = []  # partner starts with nothing to give back

    _execute_trade(sim, tribe, partner)

    assert tribe.items == []
    assert partner.items == [tribe_item]


def test_trade_swaps_one_item_each_way_when_both_sides_have_some():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    partner = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 60, 60, "#fb923c")
    tribe_item = {"name": "Whetstone", "type": "tool", "value": 8, "cycle_made": 1}
    partner_item = {"name": "War Hammer", "type": "weapon", "value": 12, "cycle_made": 2}
    tribe.items = [tribe_item]
    partner.items = [partner_item]

    _execute_trade(sim, tribe, partner)

    assert tribe.items == [partner_item]
    assert partner.items == [tribe_item]


def test_build_kitchen_requires_cooking_and_long_house_first():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = config.KITCHEN_WOOD_COST
    tribe.stone = config.KITCHEN_STONE_COST

    assert ACTION_REGISTRY["BUILD_KITCHEN"](sim, tribe, "plains", _NO_TARGET) is None
    assert tribe.kitchen_built is False

    tribe.cooking_learned = True
    tribe.long_houses_built = 1
    result = ACTION_REGISTRY["BUILD_KITCHEN"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.kitchen_built is True
    assert "kitchen is built" in result


def test_plant_crop_spends_wood_and_adds_a_plot():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    tribe.wood = 50

    from backend import config
    ACTION_REGISTRY["PLANT_CROP"](sim, tribe, "river", _NO_TARGET)

    assert tribe.farm_plots == 1
    assert tribe.wood == 50 - config.PLANT_CROP_WOOD_COST


def test_plant_crop_does_nothing_without_enough_wood():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    ACTION_REGISTRY["PLANT_CROP"](sim, tribe, "river", _NO_TARGET)

    assert tribe.farm_plots == 0


def test_plant_crop_is_capped_at_max_farm_plots():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 1000
    tribe.farm_plots = config.MAX_FARM_PLOTS

    ACTION_REGISTRY["PLANT_CROP"](sim, tribe, "river", _NO_TARGET)

    assert tribe.farm_plots == config.MAX_FARM_PLOTS  # no further growth past the cap
    assert tribe.wood == 1000  # and no wood spent trying


def test_gather_eggs_sets_pending_hatch_on_a_successful_roll():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.0):  # below any chance
        ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "river", _NO_TARGET)

    assert tribe.pending_hatch == {"parents": None}  # founding egg -- nothing to cross yet


def test_gather_eggs_does_nothing_on_a_failed_roll():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.999):  # above any chance
        ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "river", _NO_TARGET)

    assert tribe.pending_hatch is None


def test_gather_eggs_refuses_a_second_egg_while_one_is_already_pending():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.pending_hatch = {"parents": None}

    with mock.patch("backend.actions.random.random", return_value=0.0):
        result = ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "river", _NO_TARGET)

    assert tribe.pending_hatch == {"parents": None}  # unchanged, not overwritten
    assert result == "an egg is already being tended -- one thing at a time"


def test_gather_eggs_crosses_the_two_most_recent_flock_members_once_available():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.flock_lineage = [
        {"trait": "first", "parents": [], "cycle": 1, "note": ""},
        {"trait": "second", "parents": [], "cycle": 2, "note": ""},
        {"trait": "third", "parents": [], "cycle": 3, "note": ""},
    ]

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "river", _NO_TARGET)

    assert [p["trait"] for p in tribe.pending_hatch["parents"]] == ["second", "third"]


def test_catch_fish_catches_food_on_a_successful_roll():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["CATCH_FISH"](sim, tribe, "river", _NO_TARGET)

    assert tribe.food > 0


def test_catch_fish_does_nothing_on_a_failed_roll():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0

    with mock.patch("backend.actions.random.random", return_value=0.999):
        ACTION_REGISTRY["CATCH_FISH"](sim, tribe, "river", _NO_TARGET)

    assert tribe.food == 0
    assert tribe.fishing_learned is False


def test_first_successful_catch_learns_fishing_and_celebrates():
    """Explicit request: "learning to fish" isn't a separate knowledge system --
    the first catch just flips fishing_learned, awards a trophy, and throws the same
    kind of party water discovery/settling/game discovery already get."""
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 100

    with mock.patch("backend.actions.random.random", return_value=0.0):
        result = ACTION_REGISTRY["CATCH_FISH"](sim, tribe, "river", _NO_TARGET)

    assert tribe.fishing_learned is True
    assert any(t["name"] == "Angler" for t in tribe.trophies)
    assert any("celebrates learning to fish" in entry for entry in tribe.history)
    assert "first catch" in result


def test_cook_food_learns_cooking_unconditionally_once_chosen():
    """Explicit correction: COOK_FOOD no longer checks for a fire currently standing
    at this tile -- its real prerequisites (a proven hunt and a proven fire, ever)
    are gated at the availability layer (Simulation._prepare_turn), not here. Once
    the action is actually chosen, it always succeeds."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    result = ACTION_REGISTRY["COOK_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.cooking_learned is True
    assert any(t["name"] == "Master Chef" for t in tribe.trophies)
    assert "learns to cook" in result


def test_cook_food_is_a_no_op_once_already_learned():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.cooking_learned = True
    trophies_before = list(tribe.trophies)

    result = ACTION_REGISTRY["COOK_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert result is None
    assert tribe.trophies == trophies_before


def test_build_fire_marks_fire_ever_built():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50

    ACTION_REGISTRY["BUILD_FIRE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.fire_ever_built is True


def test_hunt_deer_success_marks_hunt_ever_succeeded():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.99):  # miss the wolf hazard
        ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.hunt_ever_succeeded is True


def test_hunt_deer_wolf_attack_does_not_mark_hunt_ever_succeeded():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.0):  # trigger the wolf hazard
        ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.hunt_ever_succeeded is False


def test_hunt_deer_wolf_attack_marks_a_map_encounter():
    """Explicit request: 'I do want to see the Wolves encountered marked for
    them' -- every other hazard/conflict already gets a momentary map marker via
    recent_encounters; the wolf-pack hazard never did."""
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.0):  # trigger the wolf hazard
        ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "forest", _NO_TARGET)

    assert sim.recent_encounters == [{"x": 50, "y": 50, "kind": "wolf_attack", "label": "Wolf pack!", "outcome": "struck"}]


def test_later_catches_do_not_re_learn_or_re_celebrate():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.fishing_learned = True
    tribe.food = 100

    with mock.patch("backend.actions.random.random", return_value=0.0):
        result = ACTION_REGISTRY["CATCH_FISH"](sim, tribe, "river", _NO_TARGET)

    assert len(tribe.trophies) == 0
    assert not any("celebrates learning to fish" in entry for entry in tribe.history)
    assert "first catch" not in result


def test_hunting_success_also_depletes_local_game():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.food = 0

    ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "plains", _NO_TARGET)  # plains has no wolf hazard

    assert tribe.food == 9  # round(15 * 0.6 plains game multiplier)
    assert sim.world.scarcity("game", 65, 85) > 0


def test_forage_yields_the_most_on_plains():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.food = 0

    ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.food == 10  # round(10 * 1.0 plains forage multiplier)


def test_forage_yield_is_lower_in_forest_than_plains():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0

    ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.food == 6  # round(10 * 0.6 forest forage multiplier)


def test_forage_carries_no_hazard_unlike_hunting():
    import random
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0
    tribe.population = 10

    with mock.patch("backend.actions.random.random", return_value=0.0):  # would trigger a wolf hazard if this were HUNT_DEER
        ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.population == 10  # no population lost
    assert tribe.food > 0


def test_foraging_also_depletes_the_local_forage_scarcity():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.food = 0

    ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert sim.world.scarcity("forage", 65, 85) > 0


def test_harvest_yield_is_unchanged_at_or_below_starting_population():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0
    tribe.population = 4  # below the 8-person baseline

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == 10  # no penalty for a smaller-than-baseline tribe


def test_harvest_yield_scales_up_for_a_larger_tribe():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0
    tribe.population = 16  # double the 8-person baseline

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == 20  # round(10 * (16/8) labor multiplier)


def test_expedition_capacity_matches_the_floor_at_baseline_population():
    from backend.actions import expedition_capacity
    from backend import config

    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    assert expedition_capacity(tribe) == config.MAX_CONCURRENT_EXPEDITIONS


def test_expedition_capacity_grows_with_population():
    from backend.actions import expedition_capacity

    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 25

    assert expedition_capacity(tribe) == 5  # 25 // EXPEDITION_SLOT_POPULATION_DIVISOR (5)


def test_a_third_scout_succeeds_once_population_growth_raises_capacity():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 25  # capacity 5, well past the old fixed cap of 2
    tribe.expeditions = [
        {"lead_scout": "A", "day": 1, "max_days": 3, "phase": "outbound"},
        {"lead_scout": "B", "day": 1, "max_days": 3, "phase": "outbound"},
    ]

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (60, 60))

    assert "no one left to send" not in note
    assert len(tribe.expeditions) == 3


def test_relocate_moves_the_tribe_toward_target():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["RELOCATE"](sim, tribe, "plains", (80, 50))

    assert (tribe.x, tribe.y) != (50, 50)
    assert tribe.x > 50  # moved toward the target, not away or nowhere


def test_relocate_moves_five_times_as_fast_from_an_evolved_toll_road():
    """Explicit request: "travel speed is 5x on toll roads.\""""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    # Evolving the road itself also wears the tile a little (unavoidable, same
    # wear_trail call) -- computed against the exact bonus this leaves behind
    # rather than assuming a bare config.MOVEMENT_SPEED baseline.
    for _ in range(config.ROAD_EVOLVE_CROSSINGS + 1):
        sim.world.wear_trail(50, 50, 0.01, tribe_id=tribe.id)
    trail_bonus = sim.world.trail_speed_bonus(50, 50, config.MAX_TRAIL_BONUS_SPEED)
    expected_speed = round((config.MOVEMENT_SPEED + trail_bonus) * config.TOLL_ROAD_SPEED_MULTIPLIER)

    ACTION_REGISTRY["RELOCATE"](sim, tribe, "plains", (80, 50))

    assert tribe.x - 50 == expected_speed
    assert expected_speed > config.MOVEMENT_SPEED * config.TOLL_ROAD_SPEED_MULTIPLIER - 1  # genuinely ~5x, not 1x


def test_relocate_costs_stamina():
    """Without a cost here, RELOCATE would be strictly free compared to every gathering
    action, which all cost time and risk -- marching should be tiring."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 40
    tribe.water = 30

    ACTION_REGISTRY["RELOCATE"](sim, tribe, "plains", (80, 50))

    assert tribe.food == 39
    assert tribe.water == 29


def test_relocate_stamina_cost_never_goes_negative():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0
    tribe.water = 1

    ACTION_REGISTRY["RELOCATE"](sim, tribe, "plains", (80, 50))

    assert tribe.food == 0
    assert tribe.water == 0


def test_other_actions_never_move_the_tribe():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}

    for action in ("GATHER_WOOD", "GATHER_STONE", "HUNT_DEER", "BUILD_FIRE", "CONSTRUCT_WALL", "RAID", "TRADE"):
        ACTION_REGISTRY[action](sim, tribe, "plains", (80, 80))
        assert (tribe.x, tribe.y) == (50, 50), f"{action} should not move the tribe"


def test_scout_does_not_move_the_tribe_but_launches_an_expedition():
    """Explicit request: "scout directions rotate on a 20 degree angle
    starting with the South East" -- target_vector (here, (10,10)) is
    deliberately ignored for SCOUT specifically; the real heading comes from
    tribe.scout_rotation_index instead (see test_scout_rotation_* below)."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))

    assert (tribe.x, tribe.y) == (50, 50)  # scouting doesn't relocate the tribe
    assert len(tribe.expeditions) == 1
    assert tribe.expeditions[0]["target"] != [10, 10]
    assert tribe.expeditions[0]["day"] == 0
    assert tribe.expeditions[0]["phase"] == "outbound"
    assert "depart" in note


def test_reflect_into_grid_leaves_in_bounds_values_alone():
    from backend.actions import _reflect_into_grid

    assert _reflect_into_grid(50, 100) == 50
    assert _reflect_into_grid(0, 100) == 0
    assert _reflect_into_grid(99, 100) == 99


def test_reflect_into_grid_bounces_an_overshoot_back_inward():
    """Bug report: "at least 5 quarries on the map edge... this random distribution
    of stuff to find is wrong." Reflecting (not clamping) means different overshoot
    amounts land at different points, instead of every overshoot collapsing onto the
    same boundary value."""
    from backend.actions import _reflect_into_grid

    assert _reflect_into_grid(-9, 100) == 9
    assert _reflect_into_grid(-7, 100) == 7
    assert _reflect_into_grid(108, 100) == 90  # 2*99 - 108


def test_scout_targets_near_an_edge_no_longer_all_collapse_onto_the_same_coordinate():
    """Live bug: a tribe near the map's edge scouting mountains kept landing every
    westward-ish dispatch on the exact same x=0 -- hard-clamping an overshooting
    target throws away the heading, collapsing a whole arc of distinct angles onto
    one identical coordinate."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 16, 27, "#fb923c")

    targets = set()
    for _ in range(6):
        tribe.expeditions.clear()
        ACTION_REGISTRY["SCOUT"](sim, tribe, "mountains", (0, 0))
        targets.add(tuple(tribe.expeditions[0]["target"]))

    edge_x_count = sum(1 for (x, _y) in targets if x == 0)
    assert edge_x_count < len(targets)  # not every heading collapses onto x=0


def test_scout_rotation_starts_southwest_and_advances_by_a_fixed_step():
    """Explicit request: "scout directions rotate on a 20 degree angle" (starting
    direction changed from southeast to southwest 2026-09-02 -- see config.
    SCOUT_ROTATION_START_ANGLE_DEGREES's own comment -- to steer every tribe's first
    scout away from a specific cramped map corner that caused other live bugs)."""
    from backend.simulation import _compass_direction

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (0, 0))
    first_target = tribe.expeditions[0]["target"]
    assert _compass_direction(first_target[0] - 50, first_target[1] - 50) == "southwest"
    assert tribe.scout_rotation_index == 1

    tribe.expeditions.clear()  # room for a second party
    ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (0, 0))
    second_target = tribe.expeditions[0]["target"]

    assert tribe.scout_rotation_index == 2
    assert second_target != first_target


def test_second_tribes_opening_scout_heads_a_different_way_than_the_first():
    """Live report: "the scouts went exact the same way" -- confirmed against a
    fresh run's own snapshots: two tribes' opening SCOUT (both starting at
    scout_rotation_index=0 before this fix) computed the identical (dx,dy)
    offset off their own position, so every tribe's very first scout walked
    the same heading regardless of where it started. Tribe.__init__ now seeds
    scout_rotation_index from the tribe's own spawn index (config.
    SCOUT_ROTATION_TRIBE_STAGGER_STEPS) so this can't happen."""
    from backend import config
    from backend.simulation import _compass_direction

    sim = _bare_simulation()
    tribe_0 = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe_1 = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 50, 50, "#fb923c")

    ACTION_REGISTRY["SCOUT"](sim, tribe_0, "plains", (0, 0))
    ACTION_REGISTRY["SCOUT"](sim, tribe_1, "plains", (0, 0))

    target_0 = tribe_0.expeditions[0]["target"]
    target_1 = tribe_1.expeditions[0]["target"]
    heading_0 = _compass_direction(target_0[0] - 50, target_0[1] - 50)
    heading_1 = _compass_direction(target_1[0] - 50, target_1[1] - 50)
    assert heading_0 != heading_1
    # +1 each: _scout advances the index by one step after every real dispatch.
    assert tribe_1.scout_rotation_index == config.SCOUT_ROTATION_TRIBE_STAGGER_STEPS + 1


def test_scout_rotation_ignores_target_vector_entirely():
    sim = _bare_simulation()
    same_target = (77, 3)
    headings = []
    for _ in range(3):
        tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
        ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", same_target)
        headings.append(tuple(tribe.expeditions[0]["target"]))

    assert len(set(headings)) == 1  # same tribe.scout_rotation_index (0) each time -> same heading
    assert same_target not in headings


def test_exploration_party_dispatches_with_its_own_rotation_and_a_longer_patience():
    """Explicit request: "a smart Chief will send one Scout and one
    Exploration Party" -- a real, distinct expedition kind, not SCOUT renamed."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    result = ACTION_REGISTRY["EXPLORATION_PARTY"](sim, tribe, "plains", (0, 0))

    assert len(tribe.expeditions) == 1
    exp = tribe.expeditions[0]
    assert exp["kind"] == "explore"
    assert exp["wood_gathered"] == 0 and exp["stone_gathered"] == 0
    assert tribe.explore_rotation_index == 1
    assert tribe.scout_rotation_index == 0  # its own separate counter, untouched
    assert "exploration party" in result


def test_exploration_party_shares_expedition_capacity_with_scout():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(config.MAX_CONCURRENT_EXPEDITIONS):
        ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (0, 0))

    result = ACTION_REGISTRY["EXPLORATION_PARTY"](sim, tribe, "plains", (0, 0))

    assert "no one left to send" in result


def test_advance_exploration_party_outbound_gathers_real_wood_and_stone():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    exp = {
        "pos": [50, 50], "day": 1, "max_days": 6, "wood_gathered": 0, "stone_gathered": 0,
        "food_gathered": 0, "water_gathered": 0, "phase": "outbound", "lead_scout": "Rivenna",
    }

    ended = sim._advance_exploration_party_outbound(tribe, exp, "plains", "Rivenna")

    assert ended is False
    assert exp["wood_gathered"] > 0
    assert exp["stone_gathered"] > 0
    assert exp["phase"] == "outbound"


def test_advance_exploration_party_outbound_turns_back_at_carry_capacity():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    exp = {
        "pos": [50, 50], "day": 1, "max_days": 6,
        "wood_gathered": config.EXPLORATION_PARTY_CARRY_CAPACITY, "stone_gathered": 0,
        "food_gathered": 0, "water_gathered": 0, "phase": "outbound", "lead_scout": "Rivenna",
    }

    ended = sim._advance_exploration_party_outbound(tribe, exp, "plains", "Rivenna")

    assert ended is True
    assert exp["phase"] == "returning"
    assert any("laden with all they can carry" in e for e in tribe.history)


def test_advance_exploration_party_outbound_turns_back_at_the_day_limit():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    exp = {
        "pos": [50, 50], "day": 6, "max_days": 6, "wood_gathered": 0, "stone_gathered": 0,
        "food_gathered": 0, "water_gathered": 0, "phase": "outbound", "lead_scout": "Rivenna",
    }

    ended = sim._advance_exploration_party_outbound(tribe, exp, "plains", "Rivenna")

    assert ended is True
    assert exp["phase"] == "returning"
    assert any("after 6 days out" in e for e in tribe.history)


def test_advance_exploration_party_outbound_spots_a_nearby_rival_settlement():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    rival = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 55, 50, "#fb923c")
    sim.tribes = {"tribe_0": tribe, "tribe_1": rival}
    exp = {
        "pos": [50, 50], "day": 1, "max_days": 6, "wood_gathered": 0, "stone_gathered": 0,
        "food_gathered": 0, "water_gathered": 0, "phase": "outbound", "lead_scout": "Rivenna",
    }

    sim._advance_exploration_party_outbound(tribe, exp, "plains", "Rivenna")

    assert any("spots Mountain Tribe's settlement" in e for e in tribe.history)


def test_advance_exploration_party_outbound_discovers_a_landmark():
    """Explicit request: "leave Landmarks (with a reason to go there - maybe a
    fun unique resource but not ore) as they find them.\""""
    from unittest import mock

    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    exp = {
        "pos": [50, 50], "day": 1, "max_days": 6, "wood_gathered": 0, "stone_gathered": 0,
        "food_gathered": 0, "water_gathered": 0, "phase": "outbound", "lead_scout": "Rivenna",
    }

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_exploration_party_outbound(tribe, exp, "plains", "Rivenna")

    assert len(tribe.landmarks) == 1
    landmark = tribe.landmarks[0]
    assert landmark["x"] == 50 and landmark["y"] == 50
    assert landmark["resource"] in config.LANDMARK_NAMES
    assert sum(tribe.unique_resources.values()) > 0
    assert any("discovers" in e for e in tribe.history)


def test_advance_exploration_party_outbound_does_not_rediscover_the_same_spot():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    tribe.landmarks = [{"x": 50, "y": 50, "resource": "Ancient Grove"}]
    exp = {
        "pos": [50, 50], "day": 1, "max_days": 6, "wood_gathered": 0, "stone_gathered": 0,
        "food_gathered": 0, "water_gathered": 0, "phase": "outbound", "lead_scout": "Rivenna",
    }

    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._advance_exploration_party_outbound(tribe, exp, "plains", "Rivenna")

    assert len(tribe.landmarks) == 1  # unchanged -- already known at this exact spot


def test_exploration_party_credits_real_wood_and_stone_home():
    """Only EXPLORATION_PARTY should ever bring real wood/stone home -- SCOUT/
    HUNTING_PARTY/SEND_TRADE_EMISSARY never populate these fields at all."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b", "x": 50, "y": 50}])
    tribe = sim.tribes["tribe_0"]
    tribe.wood = 0
    tribe.stone = 0
    exp = {
        "kind": "explore", "pos": [50, 50], "origin": [50, 50], "target": [50, 50],
        "day": 3, "phase": "returning", "found": None, "terrain_report": None,
        "food_gathered": 5, "water_gathered": 5, "wood_gathered": 12, "stone_gathered": 9,
        "lead_scout": "Rivenna", "determination": 0.5, "max_days": 6, "path": [[50, 50]],
    }
    tribe.expeditions = [exp]

    sim._advance_one_expedition(tribe, exp)

    assert tribe.wood == 12
    assert tribe.stone == 9


def test_scout_launch_gives_the_expedition_a_named_lead_and_determination_trait():
    """The exploration party isn't a second LLM agent making its own choices (that
    would double Ollama calls per tribe per cycle for a party the tribe already
    decided to send) -- but it does get its own procedurally-generated character, not
    every expedition behaving identically."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))

    exp = tribe.expeditions[0]
    assert exp["lead_scout"]  # a real, non-empty name
    assert 0.0 <= exp["determination"] <= 1.0
    span = config.EXPEDITION_DETERMINATION_DAY_VARIANCE
    assert config.EXPEDITION_MAX_DAYS - span <= exp["max_days"] <= config.EXPEDITION_MAX_DAYS + span


def test_scout_launch_is_deterministic_per_tribe_and_cycle():
    """Same tribe, same cycle should always produce the same scout -- re-reading an
    expedition's state (e.g. across a websocket reconnect) shouldn't change who's
    leading it."""
    sim = _bare_simulation()
    tribe_a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe_b = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["SCOUT"](sim, tribe_a, "plains", (10, 10))
    ACTION_REGISTRY["SCOUT"](sim, tribe_b, "plains", (10, 10))

    assert tribe_a.expeditions[0]["lead_scout"] == tribe_b.expeditions[0]["lead_scout"]
    assert tribe_a.expeditions[0]["determination"] == tribe_b.expeditions[0]["determination"]


def _minor_settlement(x=10, y=10, raids_remaining=None):
    from backend import config

    return {
        "x": x, "y": y, "wood": 100, "stone": 100, "food": 100, "water": 100,
        "raids_remaining": config.MINOR_SETTLEMENT_MAX_RAIDS if raids_remaining is None else raids_remaining,
        "depleted_at_cycle": None,
    }


def test_raid_always_succeeds_against_a_nearby_minor_settlement():
    """Explicit request: 'no llm or advanced logic like battle... stealing only.'
    No population risk either way, unlike a real rival tribe -- a minor settlement
    has no people to fight back with."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    settlement = _minor_settlement(x=10, y=10)
    sim.minor_settlements = [settlement]
    pop_before = tribe.population
    wood_before = tribe.wood

    result = ACTION_REGISTRY["RAID"](sim, tribe, "plains", (10, 10))

    stolen = round(100 * config.MINOR_SETTLEMENT_RAID_STEAL_FRACTION)
    assert tribe.wood == wood_before + stolen
    assert settlement["wood"] == 100 - stolen
    assert settlement["raids_remaining"] == config.MINOR_SETTLEMENT_MAX_RAIDS - 1
    assert tribe.population == pop_before  # no risk to the raiding tribe
    assert "raided an outlying settlement" in result


def test_minor_settlement_is_depleted_after_max_raids_and_stops_being_a_target():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    settlement = _minor_settlement(x=10, y=10, raids_remaining=1)
    sim.minor_settlements = [settlement]

    ACTION_REGISTRY["RAID"](sim, tribe, "plains", (10, 10))

    assert settlement["raids_remaining"] == 0
    assert settlement["depleted_at_cycle"] == sim.cycle

    # Depleted -- no longer found as a raid target until it respawns.
    note = ACTION_REGISTRY["RAID"](sim, tribe, "plains", (10, 10))
    assert "no rival" in note


def test_trade_with_a_minor_settlement_is_smaller_and_does_not_deplete_it():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    settlement = _minor_settlement(x=10, y=10)
    sim.minor_settlements = [settlement]
    wood_before = tribe.wood

    result = ACTION_REGISTRY["TRADE"](sim, tribe, "plains", (10, 10))

    gained = round(100 * config.MINOR_SETTLEMENT_TRADE_FRACTION)
    assert tribe.wood == wood_before + gained
    assert gained < round(100 * config.MINOR_SETTLEMENT_RAID_STEAL_FRACTION)  # smaller than a raid's take
    assert settlement["raids_remaining"] == config.MINOR_SETTLEMENT_MAX_RAIDS  # untouched by trading
    assert tribe.trades_completed == 1
    assert "traded peacefully with an outlying settlement" in result


def test_raid_with_no_rival_nearby_does_nothing():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}

    note = ACTION_REGISTRY["RAID"](sim, tribe, "plains", (10, 10))

    assert "no rival" in note
    assert tribe.population == 8


def test_raid_win_steals_resources_and_absorbs_some_of_the_defenders_population():
    """Population moves rather than just vanishing on a raid win -- captured or
    defecting survivors, not a pointless flat loss with no benefit to the winner."""
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.wood = 20
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.0):  # any positive win chance wins
        note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert "raided Mountain Tribe" in note
    assert attacker.wood == 56  # 50 starting + round(20 * 0.3) stolen
    assert defender.wood == 14  # 20 - 6
    assert defender.population == 6  # 8 - round(8 * RAID_POPULATION_ABSORB_FRACTION (0.2)) = 8 - 2
    assert attacker.population == 9  # 8 + 2 absorbed - 1 (RAID_ATTACKER_POPULATION_LOSS_ON_WIN)


def test_raid_win_awards_first_conquest_trophy_only_on_the_first_win():
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.population = 20  # plenty of survivors, won't trigger a merge
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))
        ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert len([t for t in attacker.trophies if t["name"] == "First Conquest"]) == 1


def test_raid_win_grants_a_chief_proposed_raiding_award():
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.population = 20  # plenty of survivors, won't trigger a merge
    attacker.custom_awards = [{"name": "Blooded Spear", "category": "raiding", "cycle": 1}]
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert any(t["name"] == "Blooded Spear" for t in attacker.trophies)


def test_raid_that_reduces_defender_to_zero_population_merges_into_the_attacker():
    from unittest import mock

    sim = _bare_simulation()
    sim.cycle = 5
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.population = 1  # a single raid at 20% (rounded up to a minimum of 1) finishes them
    defender.era = "tribal_synapse"
    defender.wood, defender.stone, defender.food, defender.water = 5, 5, 5, 5
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.0):
        note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert note == "raided Mountain Tribe, fully absorbing its survivors -- Forest Tribe becomes Forest Tribe (Advanced)!"
    assert attacker.name == "Forest Tribe (Advanced)"
    assert attacker.era == "tribal_synapse"  # inherits the higher of the two eras
    assert attacker.chief_name == ""  # chief-less, awaiting the next cycle's succession
    assert "tribe_1" not in sim.tribes
    assert defender.extinct is True


def test_repelled_raid_that_reduces_the_attacker_to_zero_population_merges_into_the_defender():
    """Mirror of the win-side merge test, roles reversed: a repelled raid can fully
    absorb the attacker into the defender."""
    from unittest import mock

    sim = _bare_simulation()
    sim.cycle = 5
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    attacker.population = 3  # survives the flat attrition (2), then absorption finishes them
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.era = "tribal_synapse"
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.999):  # forces a loss
        note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert "fully repelled and absorbed" in note
    assert "tribe_0" not in sim.tribes
    assert attacker.extinct is True
    assert defender.era == "tribal_synapse"  # inherits the higher of the two eras


def test_raid_loss_gives_the_defender_loot_and_captives():
    """Explicit request: "Raids that fail give the winning Tribe people and
    inventory" -- a repelled defense now mirrors a successful raid, roles
    reversed: the defender loots the attacker and absorbs some of its population,
    not just avoids loss."""
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.wood = 20
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.999):  # forces a loss
        note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert "repelled" in note
    # attacker: 8 pop - 2 (RAID_ATTACKER_POPULATION_LOSS_ON_LOSS) = 6, then -1 absorbed (20% of 6) = 5
    assert attacker.population == 5
    assert defender.population == 9  # +1 absorbed
    # attacker: 50 wood - 15 (30% stolen) = 35; defender: 20 + 15 = 35
    assert attacker.wood == 35
    assert defender.wood == 35
    assert any(t["name"] == "Raid Breaker" for t in defender.trophies)


def test_raid_loss_that_wipes_the_attacker_through_attrition_alone_does_not_also_absorb():
    """Ordering regression guard: if the pre-existing small attrition cost alone
    already kills the attacker, the new absorption logic must not also run on an
    already-extinct tribe."""
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    attacker.population = 2  # exactly RAID_ATTACKER_POPULATION_LOSS_ON_LOSS
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.999):
        note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert attacker.extinct is True
    assert attacker.population == 0
    assert "wiped out" in note


def test_strike_raider_camp_fails_with_no_known_sighting_at_that_location():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.raider_sightings = []

    note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, tribe, "plains", (60, 60))

    assert "no known raider camp" in note


def test_strike_raider_camp_success_removes_the_sighting_and_loots_food():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.raider_sightings = [(60, 60)]
    tribe.food = 100

    with mock.patch("backend.actions.random.random", return_value=0.0):  # forces a win
        note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, tribe, "plains", (60, 60))

    assert "destroyed" in note
    assert tribe.raider_sightings == []
    assert tribe.food > 100
    assert "PRIDE" in sim.trauma.bias_string(60, 60)


def test_strike_raider_camp_failure_costs_population_and_leaves_sighting_intact():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.raider_sightings = [(60, 60)]
    tribe.population = 10

    with mock.patch("backend.actions.random.random", return_value=0.999):  # forces a loss
        note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, tribe, "plains", (60, 60))

    assert "escaped" in note
    assert tribe.raider_sightings == [(60, 60)]
    assert tribe.population < 10
    assert "DREAD" in sim.trauma.bias_string(50, 50)


def test_strike_raider_camp_win_chance_scales_with_population():
    from unittest import mock

    from backend import config

    sim = _bare_simulation()
    low = Tribe("tribe_0", "Small Tribe", "gemma2:2b", 50, 50, "#c084fc")
    low.raider_sightings = [(60, 60)]
    low.population = 1
    high = Tribe("tribe_1", "Big Tribe", "gemma2:2b", 50, 50, "#f97316")
    high.raider_sightings = [(60, 60)]
    high.population = 200  # capped at STRIKE_RAIDER_CAMP_MAX_WIN_CHANCE

    # A roll that clears the capped high-population chance but not the low one.
    roll = config.STRIKE_RAIDER_CAMP_BASE_WIN_CHANCE + 0.001
    with mock.patch("backend.actions.random.random", return_value=roll):
        low_note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, low, "plains", (60, 60))
        high_note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, high, "plains", (60, 60))

    assert "escaped" in low_note
    assert "destroyed" in high_note


def test_strike_raider_camp_unlocked_only_from_tribal_synapse():
    from backend.eras import unlocked_actions_through

    assert "STRIKE_RAIDER_CAMP" not in unlocked_actions_through("primitive_dawn")
    assert "STRIKE_RAIDER_CAMP" in unlocked_actions_through("tribal_synapse")


def test_cook_food_unlocked_from_primitive_dawn():
    """Explicit request: "this can happen early." COOK_FOOD is no longer gated to a
    later era at all -- its real prerequisites (a proven hunt and a proven fire) are
    what gate it now (see Simulation._prepare_turn), and both are only reachable
    post-settling anyway, so this can't fire before a tribe has a camp regardless."""
    from backend.eras import unlocked_actions_through

    assert "COOK_FOOD" in unlocked_actions_through("primitive_dawn")


def test_declare_alliance_sets_symmetric_stance():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    result = ACTION_REGISTRY["DECLARE_ALLIANCE"](sim, a, "plains", (51, 51))

    assert a.stance_toward["tribe_1"] == "ALLIED"
    assert b.stance_toward["tribe_0"] == "ALLIED"
    assert "declares an alliance" in result
    assert a.pending_cultural_crossover == "tribe_1"


def test_declare_alliance_does_not_requeue_a_crossover_while_already_allied():
    """See Simulation._resolve_cultural_crossover -- only a genuinely new
    alliance should trigger it, not a redundant re-declaration."""
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.stance_toward["tribe_1"] = "ALLIED"
    b.stance_toward["tribe_0"] = "ALLIED"
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["DECLARE_ALLIANCE"](sim, a, "plains", (51, 51))

    assert a.pending_cultural_crossover is None


def test_declare_war_sets_symmetric_stance():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    result = ACTION_REGISTRY["DECLARE_WAR"](sim, a, "plains", (51, 51))

    assert a.stance_toward["tribe_1"] == "WAR"
    assert b.stance_toward["tribe_0"] == "WAR"
    assert "declares war" in result


def test_declare_war_requires_real_contact_range():
    from backend import config

    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    far = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 50 + config.DIPLOMACY_CONTACT_RADIUS + 5, 50, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": far}

    result = ACTION_REGISTRY["DECLARE_WAR"](sim, a, "plains", (far.x, far.y))

    assert "no rival tribe has been encountered" in result
    assert a.stance_toward == {}


def test_declare_war_is_a_no_op_if_already_at_war():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.stance_toward["tribe_1"] = "WAR"
    b.stance_toward["tribe_0"] = "WAR"
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    result = ACTION_REGISTRY["DECLARE_WAR"](sim, a, "plains", (51, 51))

    assert "already at war" in result


def test_declare_alliance_ends_a_previously_declared_war():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.stance_toward["tribe_1"] = "WAR"
    b.stance_toward["tribe_0"] = "WAR"
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    result = ACTION_REGISTRY["DECLARE_ALLIANCE"](sim, a, "plains", (51, 51))

    assert a.stance_toward["tribe_1"] == "ALLIED"
    assert b.stance_toward["tribe_0"] == "ALLIED"
    assert "sues for peace" in result
    assert a.pending_cultural_crossover == "tribe_1"  # a war ending into alliance is also new


def test_declare_alliance_with_no_rival_tribe():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": a}

    result = ACTION_REGISTRY["DECLARE_ALLIANCE"](sim, a, "plains", (51, 51))

    assert "no rival tribe has been encountered" in result
    assert a.stance_toward == {}


def test_declare_alliance_requires_real_contact_range():
    """Explicit correction: "they can't make an ALLIANCE if they have not made
    contact with another Tribe or Settlement" -- a rival that exists but is far
    outside config.DIPLOMACY_CONTACT_RADIUS shouldn't be a valid target."""
    from backend import config

    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    far = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 50 + config.DIPLOMACY_CONTACT_RADIUS + 5, 50, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": far}

    result = ACTION_REGISTRY["DECLARE_ALLIANCE"](sim, a, "plains", (far.x, far.y))

    assert "no rival tribe has been encountered" in result
    assert a.stance_toward == {}


def test_declare_stance_picks_the_nearest_rival_not_the_first():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    far = Tribe("tribe_1", "Far Tribe", "gemma2:2b", 90, 90, "#fb923c")
    near = Tribe("tribe_2", "Near Tribe", "gemma2:2b", 52, 52, "#34d399")
    sim.tribes = {"tribe_0": a, "tribe_1": far, "tribe_2": near}

    ACTION_REGISTRY["DECLARE_ALLIANCE"](sim, a, "plains", (52, 52))

    assert "tribe_2" in a.stance_toward
    assert "tribe_1" not in a.stance_toward


def test_raid_ignores_extinct_tribes_and_self():
    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    dead_rival = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    dead_rival.extinct = True
    sim.tribes = {"tribe_0": attacker, "tribe_1": dead_rival}

    note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert "no rival" in note


def test_trade_with_no_rival_nearby_does_nothing():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}

    note = ACTION_REGISTRY["TRADE"](sim, tribe, "plains", (10, 10))

    assert "no rival" in note


def test_trade_exchanges_resources_both_ways_with_no_loss_of_population():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.wood, a.stone, a.food, a.water = 100, 0, 0, 0
    b.wood, b.stone, b.food, b.water = 0, 100, 0, 0
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    note = ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert "opened trade with Mountain Tribe" in note
    # 15% of A's wood (100) moves to B, 15% of B's stone (100) moves to A
    assert a.wood == 85
    assert a.stone == 15
    assert b.wood == 15
    assert b.stone == 85
    assert a.population == 8 and b.population == 8  # no cost, unlike a raid


def test_trade_increments_both_sides_trade_counter():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert a.trades_completed == 1
    assert b.trades_completed == 1


def test_trade_awards_first_contact_trophy_to_both_sides_on_first_trade():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert any(t["name"] == "First Contact" for t in a.trophies)
    assert any(t["name"] == "First Contact" for t in b.trophies)


def test_trade_does_not_re_award_first_contact_on_a_later_trade():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))
    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert len([t for t in a.trophies if t["name"] == "First Contact"]) == 1


def test_trade_grants_a_chief_proposed_trading_award_to_both_sides():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.custom_awards = [{"name": "Merchant's Mark", "category": "trading", "cycle": 1}]
    b.custom_awards = [{"name": "Silver Tongue", "category": "trading", "cycle": 1}]
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert any(t["name"] == "Merchant's Mark" for t in a.trophies)
    assert any(t["name"] == "Silver Tongue" for t in b.trophies)


def test_trade_does_not_grant_a_proposed_award_from_a_different_category():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.custom_awards = [{"name": "Keeper of the Trails", "category": "scouting", "cycle": 1}]
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert not any(t["name"] == "Keeper of the Trails" for t in a.trophies)


def test_trade_ignores_extinct_tribes_and_self():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    dead_rival = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    dead_rival.extinct = True
    sim.tribes = {"tribe_0": tribe, "tribe_1": dead_rival}

    note = ACTION_REGISTRY["TRADE"](sim, tribe, "plains", (51, 51))

    assert "no rival" in note


def test_breed_does_nothing_with_only_a_chief_and_no_trophy_holder():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert "no one with enough standing" in note
    assert tribe.pending_birth is None
    assert tribe.food == 40  # unchanged, nothing was spent


def test_breed_pairs_the_chief_with_the_most_recent_trophy_holder():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [
        {"name": "Water Bringer", "chief": "Ashgar", "cycle": 1},
        {"name": "Master Pathfinder", "chief": "BriMir", "cycle": 10},
    ]

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert tribe.pending_birth == {"parent_a": "Ashgar", "parent_b": "BriMir"}
    assert "Ashgar and BriMir decide to start a family" in note


def test_breed_deducts_its_solo_cost():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]

    ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert tribe.food == 40 - config.BREED_FOOD_COST
    assert tribe.water == config.STARTING_WATER - config.BREED_WATER_COST


def test_breed_now_costs_real_food_and_water():
    """Live-run correction (2026-09-02): BREED used to be free (config.
    BREED_FOOD_COST/WATER_COST = 0) -- a weaker model latched onto it as a
    reflexive default with nothing weighing against it (63.8% of turns in one
    run). Costs roughly one gathering action's worth of each resource now."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]
    tribe.food = 100
    tribe.water = 100

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert "Ashgar and BriMir decide to start a family" in note
    assert tribe.pending_birth == {"parent_a": "Ashgar", "parent_b": "BriMir"}
    assert tribe.food == 100 - config.BREED_FOOD_COST
    assert tribe.water == 100 - config.BREED_WATER_COST


def test_breed_fails_during_a_real_starvation_crisis_now_that_it_has_a_cost():
    """The reversal of the test above: a positive cost means BREED can genuinely be
    blocked during a 0-food/0-water crisis now -- the exact scenario the original
    zero-cost design was built to avoid, deliberately accepted as the tradeoff for
    stopping reflexive BREED spam outside a crisis."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]
    tribe.food = 0
    tribe.water = 0

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert tribe.pending_birth is None
    assert "too little food and water" in note


def test_breed_refuses_at_the_population_cap():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]
    tribe.population = config.POPULATION_GROWTH_CAP

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert "no room" in note
    assert tribe.pending_birth is None


def test_scout_launches_a_second_party_while_capacity_remains():
    """A tribe can run up to config.MAX_CONCURRENT_EXPEDITIONS parties at once -- a
    chief with people to spare shouldn't be capped at just one no matter what."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))
    first_expedition = tribe.expeditions[0]

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (80, 80))

    assert len(tribe.expeditions) == 2
    assert tribe.expeditions[0] is first_expedition  # the first party is untouched
    assert "depart" in note


def test_scout_refuses_once_at_capacity():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(config.MAX_CONCURRENT_EXPEDITIONS):
        ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))
    parties = list(tribe.expeditions)

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (80, 80))

    assert tribe.expeditions == parties  # unchanged, no third party squeezed in
    assert "no one left to send" in note


def test_hunting_party_does_not_move_the_tribe_but_launches_an_expedition():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    note = ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))

    assert (tribe.x, tribe.y) == (50, 50)
    assert len(tribe.expeditions) == 1
    assert tribe.expeditions[0]["kind"] == "hunt"
    assert tribe.expeditions[0]["target"] == [10, 10]
    assert tribe.expeditions[0]["day"] == 0
    assert tribe.expeditions[0]["phase"] == "outbound"
    assert tribe.expeditions[0]["food_caught"] == 0
    assert "depart" in note


def test_hunting_party_launch_uses_its_own_max_days_baseline():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))

    span = config.EXPEDITION_DETERMINATION_DAY_VARIANCE
    max_days = tribe.expeditions[0]["max_days"]
    assert config.HUNTING_PARTY_MAX_DAYS - span <= max_days <= config.HUNTING_PARTY_MAX_DAYS + span


def test_hunting_party_launches_a_second_party_while_capacity_remains():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))
    first_expedition = tribe.expeditions[0]

    note = ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (80, 80))

    assert len(tribe.expeditions) == 2
    assert tribe.expeditions[0] is first_expedition
    assert "depart" in note


def test_hunting_party_refuses_once_at_capacity():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(config.MAX_CONCURRENT_EXPEDITIONS):
        ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))
    parties = list(tribe.expeditions)

    note = ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (80, 80))

    assert tribe.expeditions == parties
    assert "no one left to send" in note


def test_scout_and_hunting_party_can_both_be_out_at_once():
    """Hunting and scouting share the same expedition list and capacity -- a tribe can
    mix party types, not just repeat the same one."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))
    hunting_expedition = tribe.expeditions[0]

    ACTION_REGISTRY["SCOUT"](sim, tribe, "forest", (80, 80))

    assert len(tribe.expeditions) == 2


def test_send_trade_emissary_requires_the_wall_to_be_finished_first():
    """Explicit request: "it's unwise to Trade before we have a full Wall.\""""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    result = ACTION_REGISTRY["SEND_TRADE_EMISSARY"](sim, tribe, "plains", (10, 10))

    assert "wall ring must be finished" in result
    assert tribe.expeditions == []


def test_send_trade_emissary_dispatches_an_expedition_but_does_not_move_the_tribe():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe)

    note = ACTION_REGISTRY["SEND_TRADE_EMISSARY"](sim, tribe, "plains", (10, 10))

    assert (tribe.x, tribe.y) == (50, 50)
    assert len(tribe.expeditions) == 1
    assert tribe.expeditions[0]["kind"] == "trade"
    assert tribe.expeditions[0]["target"] == [10, 10]
    assert tribe.expeditions[0]["phase"] == "outbound"
    assert "depart" in note


def test_send_trade_emissary_shares_expedition_capacity_with_scout_and_hunt():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    _complete_ring0(sim, tribe)
    for _ in range(config.MAX_CONCURRENT_EXPEDITIONS):
        ACTION_REGISTRY["SEND_TRADE_EMISSARY"](sim, tribe, "plains", (10, 10))
    parties = list(tribe.expeditions)

    note = ACTION_REGISTRY["SEND_TRADE_EMISSARY"](sim, tribe, "plains", (80, 80))

    assert tribe.expeditions == parties
    assert "no one left to send" in note


def test_execute_trade_is_reused_identically_by_instant_trade_and_the_emissary():
    """Both TRADE and SEND_TRADE_EMISSARY resolve a found partner through the same
    _execute_trade helper -- confirms the refactor didn't change instant TRADE's
    own behavior."""
    from backend.actions import _execute_trade

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    partner = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    tribe.wood, partner.wood = 100, 200
    sim.tribes = {"tribe_0": tribe, "tribe_1": partner}

    note = ACTION_REGISTRY["TRADE"](sim, tribe, "plains", (51, 51))

    assert "opened trade" in note
    assert tribe.wood != 100 and partner.wood != 200  # goods actually moved
