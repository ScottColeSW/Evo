from backend.world import Landscape, biome_at


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


def test_biome_at_covers_all_six_regions():
    assert biome_at(80, 10) == "forest"
    assert biome_at(10, 10) == "mountains"
    assert biome_at(50, 90) == "plains"
    assert biome_at(40, 37) == "river"
    assert biome_at(95, 50) == "ocean"
    from backend.world import LAKE_CENTER
    assert biome_at(*LAKE_CENTER) == "lake"


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
    from backend.world import RIVER_SOURCE_X, _river_center_y

    assert biome_at(RIVER_SOURCE_X, round(_river_center_y(RIVER_SOURCE_X))) == "river"


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
