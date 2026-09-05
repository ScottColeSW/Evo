from backend import config
from backend.simulation import Tribe
from backend.world import Landscape, biome_at, find_nearby_site, mark_visited_sector, sector_of, site_seed_points


def test_sector_of_buckets_by_the_configured_size():
    size = config.TRIBE_MAP_SECTOR_SIZE
    assert sector_of(0, 0) == (0, 0)
    assert sector_of(size - 1, size - 1) == (0, 0)
    assert sector_of(size, size) == (1, 1)
    assert sector_of(2 * size + 3, 5) == (2, 0)


def test_mark_visited_sector_records_the_tiles_bucket():
    """Explicit request (Tribe Map): a coarse "ground we've actually walked"
    record, distinct from the positive-find lists (lumber_sites etc.)."""
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    assert tribe.visited_sectors == set()

    mark_visited_sector(tribe, 55, 62)

    assert sector_of(55, 62) in tribe.visited_sectors


def test_site_seed_points_are_deterministic():
    """Same "pure function of coordinates" philosophy biome_at itself already
    follows -- no persisted state, safe to call repeatedly/from anywhere."""
    assert site_seed_points("quarry", 100) == site_seed_points("quarry", 100)


def test_site_seed_points_differ_by_type():
    """Each site type gets its own independent seed set -- explicit request: 'a
    twisted sparse matrix assignment,' not one shared layout every type reuses."""
    assert site_seed_points("lumber", 100) != site_seed_points("quarry", 100)


def test_site_seed_points_are_sparse_not_every_cell_filled():
    """SITE_SEED_FILL_PROBABILITY < 1 -- most cells of the underlying grid should
    end up empty, not one seed per cell."""
    from backend.world import SITE_SEED_GRID_CELL_SIZE

    points = site_seed_points("quarry", 100)
    max_possible_cells = (100 // SITE_SEED_GRID_CELL_SIZE + 1) ** 2
    assert 0 < len(points) < max_possible_cells


def test_site_seed_points_never_land_on_an_unbuildable_biome():
    for seed_type in ("lumber", "wildlife", "quarry", "mine"):
        for x, y in site_seed_points(seed_type, 100):
            assert biome_at(x, y) not in config.UNBUILDABLE_BIOMES


def test_find_nearby_site_returns_none_when_nothing_is_within_radius():
    # (1, 1) is far from every real seed point's own grid cell given the coarse
    # SITE_SEED_GRID_CELL_SIZE -- radius 1 is far tighter than any cell.
    assert find_nearby_site("quarry", 1, 1, 100, set(), radius=1) is None


def test_find_nearby_site_finds_a_real_seeded_point_within_radius():
    seed_x, seed_y = site_seed_points("quarry", 100)[0]

    found = find_nearby_site("quarry", seed_x, seed_y, 100, set(), radius=0)

    assert found == (seed_x, seed_y)


def test_find_nearby_site_excludes_already_known_points():
    seed_x, seed_y = site_seed_points("quarry", 100)[0]

    found = find_nearby_site("quarry", seed_x, seed_y, 100, {(seed_x, seed_y)}, radius=0)

    assert found is None


def test_wear_trail_accumulates_and_caps_at_full_wear():
    land = Landscape(100)
    land.wear_trail(10, 10, 0.3)
    land.wear_trail(10, 10, 0.3)
    assert land.trails[(10, 10)]["wear"] == 0.6

    land.wear_trail(10, 10, 0.9)
    assert land.trails[(10, 10)]["wear"] == 1.0  # capped, never exceeds full wear


def test_wear_trail_records_the_most_recent_walkers_color():
    """Regression: a trail used to have no notion of who wore it, so every tribe's
    path rendered in the same shared amber-to-gold gradient regardless of whose it
    was. The most recent walker's color wins on a shared tile."""
    land = Landscape(100)
    land.wear_trail(10, 10, 0.3, color="#c084fc")
    assert land.trails[(10, 10)]["color"] == "#c084fc"

    land.wear_trail(10, 10, 0.3, color="#fb923c")
    assert land.trails[(10, 10)]["color"] == "#fb923c"  # overwritten by the latest walker


def test_wear_trail_keeps_the_existing_color_when_none_given():
    land = Landscape(100)
    land.wear_trail(10, 10, 0.3, color="#c084fc")
    land.wear_trail(10, 10, 0.3)  # no color passed this time
    assert land.trails[(10, 10)]["color"] == "#c084fc"


def test_wear_trail_affects_only_that_tile():
    land = Landscape(100)
    land.wear_trail(10, 10, 0.5)
    assert land.trail_speed_bonus(10, 10, max_bonus=3) == 1.5
    assert land.trail_speed_bonus(11, 10, max_bonus=3) == 0.0  # untouched tile


def test_trail_speed_bonus_scales_linearly_with_wear():
    land = Landscape(100)
    land.wear_trail(10, 10, 0.25)
    assert land.trail_speed_bonus(10, 10, max_bonus=4) == 1.0


def test_decay_trails_reduces_wear_but_not_below_zero():
    land = Landscape(100)
    land.wear_trail(10, 10, 0.05)
    land.decay_trails(0.03)
    assert round(land.trails[(10, 10)]["wear"], 6) == 0.02

    land.decay_trails(0.03)
    assert (10, 10) not in land.trails  # fully decayed, removed rather than negative


def test_wear_trail_tracks_crossings_and_first_owner_separately_from_wear():
    """Explicit request: "trails that have been traversed more than 5 times by
    anyone will automatically evolve into visible and owned roads... The first
    trailblazer gets the ownership." Crossings never decay (unlike wear) and
    ownership is set once, from whoever wore the tile first, even if a
    different tribe wears it far more since."""
    from backend import config

    land = Landscape(100)
    land.wear_trail(10, 10, 0.1, tribe_id="tribe_a")
    assert land.trails[(10, 10)]["crossings"] == 1
    assert land.trails[(10, 10)]["owner"] == "tribe_a"

    for _ in range(config.ROAD_EVOLVE_CROSSINGS - 1):
        land.wear_trail(10, 10, 0.1, tribe_id="tribe_b")

    assert land.trails[(10, 10)]["crossings"] == config.ROAD_EVOLVE_CROSSINGS
    assert land.trails[(10, 10)]["owner"] == "tribe_a"  # unchanged despite tribe_b's heavier use


def test_is_toll_road_only_once_crossings_exceed_the_threshold():
    from backend import config

    land = Landscape(100)
    for _ in range(config.ROAD_EVOLVE_CROSSINGS):
        land.wear_trail(10, 10, 0.1, tribe_id="tribe_a")
    assert land.is_toll_road(10, 10) is False  # exactly at the threshold, not yet over it

    land.wear_trail(10, 10, 0.1, tribe_id="tribe_a")
    assert land.is_toll_road(10, 10) is True


def test_road_owner_is_none_for_an_untouched_tile():
    land = Landscape(100)
    assert land.road_owner(10, 10) is None


def test_decay_trails_never_deletes_an_evolved_road():
    """A road that's crossed enough to have evolved is a real, permanent
    structure now -- it shouldn't revert to open ground just from disuse the
    way an ordinary trail's cosmetic wear does."""
    from backend import config

    land = Landscape(100)
    for _ in range(config.ROAD_EVOLVE_CROSSINGS + 1):
        land.wear_trail(10, 10, 0.01, tribe_id="tribe_a")

    for _ in range(200):  # far more than enough to fully decay ordinary wear
        land.decay_trails(0.05)

    assert (10, 10) in land.trails
    assert land.is_toll_road(10, 10) is True


def test_fresh_tile_has_no_scarcity():
    land = Landscape(100)
    assert land.scarcity("wood", 10, 10) == 0.0


def test_harvesting_raises_scarcity_at_that_tile_only():
    land = Landscape(100)
    land.deplete("wood", 10, 10, amount=0.15, max_scarcity=0.8)
    assert land.scarcity("wood", 10, 10) == 0.15
    assert land.scarcity("wood", 11, 10) == 0.0  # a different tile is untouched
    assert land.scarcity("stone", 10, 10) == 0.0  # a different resource is untouched


def test_scarcity_is_capped_below_total_depletion():
    land = Landscape(100)
    for _ in range(20):
        land.deplete("wood", 10, 10, amount=0.15, max_scarcity=0.8)
    assert land.scarcity("wood", 10, 10) == 0.8


def test_regeneration_reduces_scarcity_over_time():
    land = Landscape(100)
    land.deplete("wood", 10, 10, amount=0.5, max_scarcity=0.8)
    land.regenerate(0.1)
    assert round(land.scarcity("wood", 10, 10), 2) == 0.4


def test_regeneration_fully_clears_a_tile_eventually():
    land = Landscape(100)
    land.deplete("wood", 10, 10, amount=0.1, max_scarcity=0.8)
    for _ in range(20):
        land.regenerate(0.1)
    assert land.scarcity("wood", 10, 10) == 0.0
    assert ("wood", 10, 10) not in land.depletion  # cleaned up, not just floored at 0


def test_biome_at_covers_all_eight_regions():
    # (10, 10) moved from "mountains" to "volcano" once VOLCANO_CENTER=(10, 12)/
    # VOLCANO_RADIUS=4 was added (map dream, phase 1) -- (10, 40) was still real
    # mountains at that point, comfortably outside the volcano's radius. (50, 90)
    # moved from "plains" to "desert" once DESERT_NORTH_BOUNDARY_BASE=78 carved
    # out the southern band -- (50, 70) is still real plains, north of that
    # boundary.
    #
    # Map dream, phase 2: turning the map into a real island (west/north/south
    # ocean insets) swallowed several of the points above outright -- (80, 10),
    # (10, 40), (50, 90), and the original volcano's own (10, 10) all now land in
    # new ocean. Each was replaced with a fresh point of the same biome, found by
    # sampling biome_at directly rather than estimated, and the volcano itself
    # was relocated to (32, 30) (see VOLCANO_CENTER's own comment) since its old
    # spot became unreachable ocean.
    assert biome_at(65, 51) == "forest"
    assert biome_at(19, 40) == "mountains"
    assert biome_at(50, 70) == "plains"
    assert biome_at(40, 37) == "river"
    assert biome_at(95, 50) == "ocean"
    assert biome_at(80, 77) == "desert"
    assert biome_at(32, 30) == "volcano"
    from backend.world import LAKE_CENTER
    assert biome_at(*LAKE_CENTER) == "lake"


def test_desert_zone_does_not_swallow_ground_north_of_its_boundary():
    """Map dream, phase 1: Desert claims the southern band that would otherwise
    fall through to forest/plains -- a point clearly north of
    DESERT_NORTH_BOUNDARY_BASE (78) at the same x must stay whatever it already
    was, not flip to desert.

    Map dream, phase 2: the new south coast ocean inset squeezed the desert band
    down to as little as ~0.3-6 tiles wide (checked by sampling both boundary
    functions across the grid), too thin for the original x=50/offset-5 check --
    at x=50 an offset of 5 now overshoots straight into the new south ocean. x=80
    with a smaller offset of 2 is confirmed (by direct sampling, not estimated)
    to land cleanly on each side of the boundary without also crossing the
    south coast."""
    from backend.world import _desert_north_boundary

    x = 80
    boundary = _desert_north_boundary(x)
    assert biome_at(x, round(boundary) + 2) == "desert"
    assert biome_at(x, round(boundary) - 2) != "desert"


def test_volcano_zone_is_a_clean_circle_around_its_center():
    """Map dream, phase 1: "the volcano is a Hazard they will die if they go
    there" -- a small, fixed circle (not a wavy boundary, this is a one-off
    feature, not an organic terrain type)."""
    from backend.world import VOLCANO_CENTER, VOLCANO_RADIUS

    vx, vy = VOLCANO_CENTER
    assert biome_at(vx, vy) == "volcano"
    assert biome_at(vx + VOLCANO_RADIUS, vy) == "volcano"  # right at the edge
    assert biome_at(vx + VOLCANO_RADIUS + 3, vy) != "volcano"  # clearly outside


def test_desert_and_volcano_movement_and_yields_are_harsh():
    """Explicit request: Desert is a real, harsh biome (slower, low-yield), not
    just a recolor. Volcano keeps a real, non-zero movement multiplier -- a 0.0
    (like ocean's) would make physics.terrain_aware_step treat it as impassable
    and deflect around it, making the hazard (config.VOLCANO_HAZARD_CHANCE)
    unreachable and moot."""
    from backend.actions import BIOME_YIELD_MULTIPLIER

    assert config.TERRAIN_MOVEMENT_MULTIPLIER["desert"] < config.TERRAIN_MOVEMENT_MULTIPLIER["plains"]
    assert 0.0 < config.TERRAIN_MOVEMENT_MULTIPLIER["volcano"] < config.TERRAIN_MOVEMENT_MULTIPLIER["plains"]
    for resource in ("wood", "stone", "game", "forage"):
        assert "desert" in BIOME_YIELD_MULTIPLIER[resource]
        assert "volcano" in BIOME_YIELD_MULTIPLIER[resource]
    assert "desert" not in config.FARMABLE_BIOMES
    assert "desert" not in config.UNBUILDABLE_BIOMES  # harsh, but buildable -- mirrors mountains
    assert "volcano" in config.UNBUILDABLE_BIOMES  # lethal ground, unlike mountains


def test_lake_center_and_its_tributary_are_lake_biome():
    from backend.world import LAKE_CENTER, LAKE_TRIBUTARY_BRANCH_X, _river_center_y

    assert biome_at(*LAKE_CENTER) == "lake"
    by = round(_river_center_y(LAKE_TRIBUTARY_BRANCH_X))
    assert biome_at(LAKE_TRIBUTARY_BRANCH_X, by) == "river"  # the fork point itself


def test_lake_tributary_actually_connects_the_river_to_the_lake():
    """A real fork, not two disconnected features -- the midpoint between the branch
    point and the lake center should read as lake (the connecting stream), not plains."""
    from backend.world import LAKE_CENTER, LAKE_TRIBUTARY_BRANCH_X, _river_center_y

    bx, by = LAKE_TRIBUTARY_BRANCH_X, _river_center_y(LAKE_TRIBUTARY_BRANCH_X)
    lx, ly = LAKE_CENTER
    mid_x, mid_y = round((bx + lx) / 2), round((by + ly) / 2)
    assert biome_at(mid_x, mid_y) == "lake"


def test_lake_does_not_extend_beyond_its_radius():
    from backend.world import LAKE_CENTER, LAKE_RADIUS

    lx, ly = LAKE_CENTER
    assert biome_at(lx, ly + LAKE_RADIUS + 1) != "lake"


def test_ocean_occupies_the_entire_east_edge():
    for y in range(0, 100, 10):
        assert biome_at(99, y) == "ocean"


def test_coastline_is_wavy_not_a_straight_line():
    """The whole point: OCEAN_X_START is a reference point, not literally where every
    row's coastline sits anymore."""
    from backend.world import _coast_boundary_x

    boundaries = {round(_coast_boundary_x(y)) for y in range(20, 100)}
    assert len(boundaries) > 1


def test_coast_band_is_cliffs_on_a_headland_and_shoals_in_a_bay():
    from backend.world import _coast_boundary_x, _coast_is_headland

    # Scan for a real headland and a real bay rather than assuming specific
    # coordinates -- the wave's exact shape is an implementation detail.
    headland_y = next(y for y in range(20, 100) if _coast_is_headland(y))
    bay_y = next(y for y in range(20, 100) if not _coast_is_headland(y))

    headland_x = round(_coast_boundary_x(headland_y)) - 1  # just inland of the boundary
    bay_x = round(_coast_boundary_x(bay_y)) - 1
    assert biome_at(headland_x, headland_y) == "cliffs"
    assert biome_at(bay_x, bay_y) == "shoals"


def test_river_mouth_does_not_extend_past_a_receded_coastline():
    """Regression test: _is_river's mouth used to always cut off at the flat
    OCEAN_X_START regardless of the wavy coast -- wherever the coast recedes into a
    bay (its own boundary_x drops below OCEAN_X_START), the river kept extending to
    the old fixed line anyway, sticking several tiles out into open ocean. The
    river's own course does pass through exactly such a bay near its mouth (the
    coastline's and river's sine waves aren't related), which is what originally
    surfaced this."""
    from backend.world import OCEAN_X_START, _coast_boundary_x, _river_center_y

    found_a_receded_stretch = False
    for x in range(OCEAN_X_START - 15, OCEAN_X_START):
        y = round(_river_center_y(x))
        boundary = _coast_boundary_x(y)
        if boundary >= OCEAN_X_START:
            continue  # not a receded stretch at this point on the river's path
        found_a_receded_stretch = True
        if x >= boundary:
            assert biome_at(x, y) == "ocean"
    assert found_a_receded_stretch  # confirms this test actually exercised the bug


def test_river_originates_near_the_mountains():
    """Map dream, phase 2: _is_river's source end now clips to
    max(RIVER_SOURCE_X, _west_coast_boundary(y)), the same way its mouth already
    clips to the real east coastline -- RIVER_SOURCE_X=15 itself now sits inside
    the new west ocean at every y along the river's early course, so the river's
    real origin is wherever real land actually begins, not the old fixed
    constant. x=21 is the first point (scanning from RIVER_SOURCE_X outward)
    where the river's true course, using its own _river_center_y(x) rather than
    a value sampled at a different x, clears the west coast."""
    from backend.world import _river_center_y

    assert biome_at(21, round(_river_center_y(21))) == "river"


def test_river_reaches_the_coast():
    """The river must actually connect to the ocean, not fade out into forest or
    plains before reaching it -- an Earth-like river runs from source to sea. Scans
    the river's own course for its last river tile and checks the very next step is
    ocean (not forest/plains) -- robust to exactly where the wavy coastline sits,
    unlike asserting a single hardcoded coordinate."""
    from backend.world import RIVER_SOURCE_X, OCEAN_X_START, _river_center_y

    last_river_x = None
    for x in range(RIVER_SOURCE_X, OCEAN_X_START):
        if biome_at(x, round(_river_center_y(x))) == "river":
            last_river_x = x
    assert last_river_x is not None
    y = round(_river_center_y(last_river_x))
    assert biome_at(last_river_x + 1, y) == "ocean"


def test_river_crosses_more_than_one_biome_on_its_way_to_the_sea():
    from backend.world import OCEAN_X_START, RIVER_SOURCE_X, _river_center_y

    biomes_crossed = set()
    for x in range(RIVER_SOURCE_X, OCEAN_X_START):
        y = round(_river_center_y(x))
        for dy in (-4, 0, 4):  # sample just off the river's own centerline
            biomes_crossed.add(biome_at(x, y + dy))
    assert {"mountains", "plains", "forest"}.issubset(biomes_crossed)


def test_west_north_south_coasts_are_wavy_not_straight_lines():
    """Map dream, phase 2: same "the whole point" check as
    test_coastline_is_wavy_not_a_straight_line, extended to the three new
    coastlines that turned the map into a real island."""
    from backend.world import _north_coast_boundary, _south_coast_boundary, _west_coast_boundary

    assert len({round(_west_coast_boundary(y)) for y in range(0, 100)}) > 1
    assert len({round(_north_coast_boundary(x)) for x in range(0, 100)}) > 1
    assert len({round(_south_coast_boundary(x)) for x in range(0, 100)}) > 1


def test_west_north_south_oceans_actually_appear_in_biome_at():
    """Confirms the three new coastlines are actually wired into biome_at's
    chain, not just computed and ignored -- a point safely inside each new
    inset (base minus the sine waves' own amplitude, so it's ocean regardless
    of where in the wave this particular column/row falls) must read as ocean."""
    from backend.world import _south_coast_boundary

    assert biome_at(3, 50) == "ocean"  # west: WEST_COAST_INSET_BASE=18, deep inside any wave trough
    assert biome_at(50, 3) == "ocean"  # north: NORTH_COAST_INSET_BASE=18, same margin
    south_y = round(_south_coast_boundary(50)) + 8  # well past the south boundary at x=50
    assert biome_at(50, south_y) == "ocean"


def test_west_north_south_coast_bands_are_cliffs_or_shoals():
    """Same cliffs-on-a-headland/shoals-in-a-bay texture as the existing east
    coast (test_coast_band_is_cliffs_on_a_headland_and_shoals_in_a_bay),
    confirmed for each of the three new coastlines -- scans for a real
    headland and a real bay on each rather than assuming specific coordinates."""
    from backend.world import (
        _is_headland_like, _north_coast_boundary, _south_coast_boundary, _west_coast_boundary,
    )

    # Sampled from the interior (30-70), away from the map's four corners --
    # near a corner, two coastlines' insets overlap and biome_at's chain order
    # lets whichever is checked first (west, then north, then south) claim the
    # tile outright, which isn't the headland/bay texture this test is after.
    for boundary_fn, sample_range, ocean_on_increasing_side, land_of in (
        (_west_coast_boundary, range(30, 70), False, lambda c, b: (round(b) + 1, c)),
        (_north_coast_boundary, range(30, 70), False, lambda c, b: (c, round(b) + 1)),
        (_south_coast_boundary, range(30, 70), True, lambda c, b: (c, round(b) - 1)),
    ):
        headland_c = next(c for c in sample_range if _is_headland_like(boundary_fn, c, ocean_on_increasing_side))
        bay_c = next(c for c in sample_range if not _is_headland_like(boundary_fn, c, ocean_on_increasing_side))
        hx, hy = land_of(headland_c, boundary_fn(headland_c))
        bx, by = land_of(bay_c, boundary_fn(bay_c))
        assert biome_at(hx, hy) == "cliffs"
        assert biome_at(bx, by) == "shoals"


def test_volcano_clears_every_coastline_with_real_margin():
    """Map dream, phase 2: the volcano's original (10, 12) center fell inside
    the new west+north ocean bands outright -- its relocated (32, 30) must
    clear every one of the four coastlines by more than VOLCANO_RADIUS, not
    just barely poke over the line."""
    from backend.world import VOLCANO_CENTER, VOLCANO_RADIUS, _north_coast_boundary, _west_coast_boundary

    vx, vy = VOLCANO_CENTER
    assert vx - _west_coast_boundary(vy) > VOLCANO_RADIUS
    assert vy - _north_coast_boundary(vx) > VOLCANO_RADIUS
    assert biome_at(vx, vy) == "volcano"


def test_river_still_connects_real_land_to_real_land_end_to_end():
    """Map dream, phase 2: the river's west (source) end now clips to the real
    west coastline instead of the old fixed RIVER_SOURCE_X (see
    test_river_originates_near_the_mountains) -- confirms the river, sampled
    along its own course, never starts or ends in the new ocean at a handful
    of representative x values, the same connectivity test_river_reaches_the_
    coast already runs for the east end."""
    from backend.world import OCEAN_X_START, RIVER_SOURCE_X, _river_center_y

    river_xs = [x for x in range(RIVER_SOURCE_X, OCEAN_X_START) if biome_at(x, round(_river_center_y(x))) == "river"]
    assert river_xs  # the river exists somewhere along this stretch
    first_x, last_x = min(river_xs), max(river_xs)
    assert biome_at(first_x - 1, round(_river_center_y(first_x - 1))) != "river"
    assert biome_at(last_x + 1, round(_river_center_y(last_x + 1))) == "ocean"


def test_nearest_water_returns_own_tile_when_already_on_water():
    land = Landscape(100)
    assert land.nearest_water(40, 37) == (40, 37)  # on the river
    assert land.nearest_water(95, 50) == (95, 50)  # in the ocean


def test_nearest_water_matches_brute_force_reference():
    """Regression test: an earlier ring-by-ring search stopped at the first ring
    containing any match, which can be farther in true Euclidean distance than a match
    in a nominally "later" ring along a shallower angle. Caught by comparing against
    this same brute-force scan before trusting the faster version -- replaced with the
    brute-force approach directly rather than debugging the ring search further, since
    it only runs once per tribe and costs nothing that matters."""
    land = Landscape(100)

    def brute_force(x, y):
        best, best_dist = None, None
        for cx in range(100):
            for cy in range(100):
                if biome_at(cx, cy) in ("river", "ocean"):
                    dist = (cx - x) ** 2 + (cy - y) ** 2
                    if best_dist is None or dist < best_dist:
                        best, best_dist = (cx, cy), dist
        return best, best_dist

    for spawn in [(10, 10), (80, 10), (50, 90), (0, 0), (99, 0), (0, 99)]:
        result = land.nearest_water(*spawn)
        expected, expected_dist = brute_force(*spawn)
        result_dist = (result[0] - spawn[0]) ** 2 + (result[1] - spawn[1]) ** 2
        assert result_dist == expected_dist, f"{spawn}: got {result} ({result_dist}), expected {expected} ({expected_dist})"
