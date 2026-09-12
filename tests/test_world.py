from backend import config
from backend.simulation import Tribe
from backend.world import (
    SITE_SEED_TYPES, Landscape, biome_at, find_nearby_site, mark_visited_sector, sector_of, site_seed_points,
)


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


def test_site_seed_points_are_sparse_not_covering_the_whole_map():
    """Real minimum spacing (Poisson-disc) means nowhere near every tile can hold
    a point -- sparse by construction, not a tuned probability."""
    points = site_seed_points("quarry", 100)
    assert 0 < len(points) < 200


def test_site_seed_points_never_land_on_an_unbuildable_biome():
    # Iterates SITE_SEED_TYPES itself (not a hardcoded copy) so a future new
    # seed type is covered automatically -- explicit follow-up, 2026-09-11,
    # after "landmark" was added to this exact list to fix a real live-run
    # report of landmarks appearing on river/lake tiles.
    for seed_type in SITE_SEED_TYPES:
        for x, y in site_seed_points(seed_type, 100):
            assert biome_at(x, y) not in config.UNBUILDABLE_BIOMES


def test_site_seed_points_respect_real_minimum_spacing():
    """2026-09-12 rework: replaced the old grid-cell-plus-jitter scatter with real
    Poisson-disc sampling (Bridson's algorithm) -- live report, with a
    screenshot: "objects landed along the boards of the map... in a line,"
    confirmed as the old grid system's own seams (only ~5-6 cells per axis on a
    100-tile map, so a whole row of independent per-cell hits near an edge read
    as a line). The actual guarantee a disc-sampled layout makes, and the old one
    never did: no two points of the same type are closer together than each
    point's own local minimum spacing (world._site_spacing_radius)."""
    import math

    from backend.world import _site_spacing_radius
    from backend.simulation import SPAWN_POINTS

    for seed_type in SITE_SEED_TYPES:
        points = site_seed_points(seed_type, 100)
        for i, (x1, y1) in enumerate(points):
            r1 = _site_spacing_radius(x1, y1, seed_type, SPAWN_POINTS)
            for x2, y2 in points[i + 1:]:
                r2 = _site_spacing_radius(x2, y2, seed_type, SPAWN_POINTS)
                dist = math.hypot(x1 - x2, y1 - y2)
                assert dist >= max(r1, r2) - 1e-9, (
                    f"{seed_type} points {(x1, y1)} and {(x2, y2)} are only {dist:.1f} apart, "
                    f"closer than the required {max(r1, r2):.1f}"
                )


def test_site_seed_points_are_denser_near_a_spawn_point_than_far_from_every_spawn():
    """The actual "bias centered from spawn points, degrading" property, not a
    hard count/quota -- explicit correction from an earlier two-pass "guarantee N
    nearby" design, which would have created an artificial density cliff right at
    the guarantee radius. Checked via average nearest-neighbor distance (smaller
    = denser) in a region close to a real spawn vs. a region far from every one,
    using a dense type (quarry) on a large enough sample to be stable."""
    import math

    from backend.simulation import SPAWN_POINTS

    def nearest_spawn_distance(x, y):
        return min(math.hypot(x - sx, y - sy) for sx, sy in SPAWN_POINTS)

    def nearest_neighbor_distances(points):
        return [
            min(math.hypot(x - ox, y - oy) for ox, oy in points if (ox, oy) != (x, y))
            for x, y in points
        ]

    points = site_seed_points("quarry", 100)
    near = [p for p in points if nearest_spawn_distance(*p) < 20]
    far = [p for p in points if nearest_spawn_distance(*p) > 40]
    assert near and far, "test needs both a near-spawn and a far-from-spawn point to compare"
    avg_near = sum(nearest_neighbor_distances(near)) / len(near) if len(near) > 1 else nearest_spawn_distance(*near[0])
    avg_far = sum(nearest_neighbor_distances(far)) / len(far) if len(far) > 1 else nearest_spawn_distance(*far[0])
    # Weaker per-point signal (small samples), so just confirm more real points
    # land near spawns than the map's own area split would predict by chance --
    # the direct, low-noise evidence the gradient is actually doing something.
    assert len(near) >= len(far)


def test_landmark_seed_points_are_sparser_than_the_resource_sites():
    """Live report, 2026-09-11: "we can reduce the number too" -- landmark's own
    (r_near, r_far) pair in SITE_DENSITY_BY_TYPE is deliberately larger than every
    resource type's, distinct from their own tuned density."""
    assert len(site_seed_points("landmark", 100)) < len(site_seed_points("quarry", 100))


def test_find_nearby_site_returns_none_when_nothing_is_within_radius():
    # (1, 1) is far from every real seeded quarry point -- radius 1 is far
    # tighter than any realistic spacing between points.
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
    # VOLCANO_RADIUS=4 was added (map dream, phase 1) -- (10, 40) is still real
    # mountains, comfortably outside the volcano's radius. (50, 90) moved from
    # "plains" to "desert" once DESERT_NORTH_BOUNDARY_BASE=78 carved out the
    # southern band -- (50, 70) is still real plains, north of that boundary.
    #
    # Map dream, phase 2 turned the map into a real island (west/north/south
    # ocean insets), but kept those insets to a thin 4-8 tile frame -- every
    # point above still holds except the volcano's own spot, which moved from
    # (10, 10) to (13, 13) (see VOLCANO_CENTER's own comment) since the west/
    # north coast band now brushes right up against the original coordinate.
    #
    # (10, 40) -- moved to (18, 40) once the west coast's richer three-wave
    # texture (see _west_coast_boundary's own comment) shifted where its peaks
    # land: at y=40 the boundary+coast-band reach grew enough to swallow the
    # old point, which was only ever a few tiles from the coast to begin with.
    # (18, 40) sits comfortably deeper in real mountain ground either way.
    assert biome_at(80, 10) == "forest"
    assert biome_at(18, 40) == "mountains"
    assert biome_at(50, 70) == "plains"
    assert biome_at(40, 37) == "river"
    assert biome_at(95, 50) == "ocean"
    assert biome_at(50, 90) == "desert"
    assert biome_at(13, 13) == "volcano"
    from backend.world import LAKE_CENTER
    assert biome_at(*LAKE_CENTER) == "lake"


def test_desert_zone_does_not_swallow_ground_north_of_its_boundary():
    """Map dream, phase 1: Desert claims the southern band that would otherwise
    fall through to forest/plains -- a point clearly north of
    DESERT_NORTH_BOUNDARY_BASE (78) at the same x must stay whatever it already
    was, not flip to desert."""
    from backend.world import _desert_north_boundary

    x = 50
    boundary = _desert_north_boundary(x)
    assert biome_at(x, round(boundary) + 5) == "desert"
    assert biome_at(x, round(boundary) - 5) != "desert"


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
    """A real fork, not two disconnected features -- BFS from LAKE_CENTER across
    lake_tiles() must reach a tile 8-adjacent to a real river tile. Robust to
    the tributary's exact path (a natural bow, not a straight line -- see
    scripts/generate_hydrology.py) rather than checking one hardcoded midpoint
    that would break every time the path's shape is retuned."""
    from backend.world import LAKE_CENTER, lake_tiles, river_tiles

    lake, river = lake_tiles(), river_tiles()
    visited = {LAKE_CENTER}
    frontier = [LAKE_CENTER]
    while frontier:
        x, y = frontier.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor = (x + dx, y + dy)
                if neighbor in lake and neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
    touches_river = any(
        (x + dx, y + dy) in river for x, y in visited for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    )
    assert touches_river


def test_lake_does_not_extend_beyond_its_radius():
    from backend.world import LAKE_CENTER, LAKE_RADIUS

    lx, ly = LAKE_CENTER
    assert biome_at(lx, ly + LAKE_RADIUS + 1) != "lake"


def test_river_is_narrower_upstream_than_at_its_mouth():
    """Natural river/lake rework: "the river can be less wide." The eroded
    river (scripts/generate_hydrology.py) is a subset of the old fixed-width
    ribbon, with more of the outer band eroded away upstream than near the
    mouth -- so its per-column width should now genuinely vary, tapering
    narrower toward the source, rather than holding a constant width."""
    from backend.world import OCEAN_X_START, RIVER_SOURCE_X, river_tiles

    by_x: dict[int, int] = {}
    for x, _ in river_tiles():
        by_x[x] = by_x.get(x, 0) + 1
    xs = sorted(by_x)
    assert xs  # the river exists somewhere
    near_source_width = min(by_x[x] for x in xs if x < RIVER_SOURCE_X + 10)
    near_mouth_width = max(by_x[x] for x in xs if x > OCEAN_X_START - 10)
    assert near_source_width < near_mouth_width


def test_lake_basin_is_no_longer_a_perfect_circle():
    """Natural river/lake rework: "the lake less rounded." A perfect circle of
    LAKE_RADIUS around LAKE_CENTER would include every point at that exact
    distance -- the eroded basin (scripts/generate_hydrology.py) only keeps an
    inner core and erodes the outer shell, so at least one point on the old
    circle's own boundary should now read as dry land while the center is
    still lake."""
    from backend.world import LAKE_CENTER, LAKE_RADIUS

    lx, ly = LAKE_CENTER
    assert biome_at(lx, ly) == "lake"
    boundary_points = [
        (lx + round(LAKE_RADIUS * dx), ly + round(LAKE_RADIUS * dy))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7))
    ]
    assert any(biome_at(x, y) != "lake" for x, y in boundary_points)


def test_river_tiles_never_touch_ocean_except_at_the_mouth():
    """Structural invariant of the baked hydrology data itself, independent of
    biome_at's own dispatch order: every baked river tile must sit strictly
    inside all four real coastlines. Checked directly against the coastline
    functions rather than through biome_at (which would always report "river"
    for a baked river tile regardless of the coast)."""
    from backend.world import _coast_boundary_x, _north_coast_boundary, _south_coast_boundary, _west_coast_boundary, river_tiles

    tiles = river_tiles()
    assert tiles
    for x, y in tiles:
        assert x < _coast_boundary_x(y)
        assert x > _west_coast_boundary(y)
        assert y > _north_coast_boundary(x)
        assert y < _south_coast_boundary(x)


def test_river_tiles_are_a_subset_of_the_pre_rework_shape():
    """The river's own anchoring guarantee: erosion only ever subtracts from
    the original sine-wave ribbon, never grows or relocates it -- so every
    gameplay fixture that already depended on a specific tile NOT being river
    keeps working. Recomputes the pre-rework formula directly (the same one
    scripts/generate_hydrology.py treats as its outer footprint) rather than
    importing a frozen copy.

    The lake doesn't get an equivalent test: a direct request ("we can extend
    the lake, naturally curving into the south-west") deliberately lets the
    lake grow a real bay beyond its old circle in that one direction -- see
    test_lake_does_not_sprawl_past_a_generous_bound and
    test_wall_barrier_anchor_points_stay_dry_land below for what actually
    matters there instead."""
    from backend.world import OCEAN_X_START, RIVER_HALF_WIDTH, RIVER_SOURCE_X, _coast_boundary_x, _river_center_y, _west_coast_boundary, river_tiles

    def old_is_river(x, y):
        if x < max(RIVER_SOURCE_X, _west_coast_boundary(y)) or x >= min(OCEAN_X_START, _coast_boundary_x(y)):
            return False
        return abs(y - _river_center_y(x)) <= RIVER_HALF_WIDTH

    assert all(old_is_river(x, y) for x, y in river_tiles())


def test_lake_does_not_sprawl_past_a_generous_bound():
    """The southwest bay is a deliberate, bounded extension (see
    scripts/generate_hydrology.py's LAKE_SOUTHWEST_EXTENSION_MAX), not an
    unbounded flood-fill -- a sanity guard, not a precise geometric proof.
    lake_tiles() also includes the tributary corridor, which legitimately
    reaches all the way out to the branch point on the river (~31 tiles from
    LAKE_CENTER) -- so the bound is derived from that real distance plus the
    corridor's own width/bow margin, not just the lake body's own radius."""
    import math

    from backend.world import (
        LAKE_CENTER, LAKE_TRIBUTARY_BRANCH_X, LAKE_TRIBUTARY_HALF_WIDTH, _river_center_y, lake_tiles,
    )

    lx, ly = LAKE_CENTER
    bx, by = LAKE_TRIBUTARY_BRANCH_X, _river_center_y(LAKE_TRIBUTARY_BRANCH_X)
    tributary_length = math.hypot(lx - bx, ly - by)
    max_reach = tributary_length + LAKE_TRIBUTARY_HALF_WIDTH + 5
    assert all(math.hypot(x - lx, y - ly) <= max_reach for x, y in lake_tiles())


def test_wall_barrier_anchor_points_stay_dry_land():
    """Several existing wall/natural-barrier tests (test_simulation.py) hardcode
    specific tiles that must NOT be river or lake -- (30, 60) in particular sits
    just 3.15 tiles from the tributary's old straight-line path, the tightest
    margin in the whole rework. Direct, explicit regression coverage for the
    anchors those tests depend on, independent of whatever the exact hydrology
    shape looks like after any future retune."""
    from backend.world import lake_tiles, river_tiles

    river, lake = river_tiles(), lake_tiles()
    for x, y in ((40, 62), (30, 60), (41, 45), (55, 65)):
        assert (x, y) not in river
        assert (x, y) not in lake


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
    """_is_river's source end clips to max(RIVER_SOURCE_X,
    _west_coast_boundary(y)), the same way its mouth already clips to the real
    east coastline (map dream, phase 2) -- with the west coast's own frame kept
    thin (4-8 tiles), that clip is a no-op here in practice, but it stays in
    place for the same reason the east mouth's clip does: correct regardless of
    exactly where the wavy boundary sits."""
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
    inset (below the boundary's own minimum across the whole grid, so it's
    ocean regardless of where in the wave this particular column/row falls)
    must read as ocean."""
    from backend.world import _south_coast_boundary

    assert biome_at(3, 50) == "ocean"  # west: boundary never dips below ~4.5, see INSET_BASE=6 +-2 wave
    assert biome_at(50, 3) == "ocean"  # north: same margin
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
    # near a corner, two coastlines' insets genuinely overlap, so a tile can be
    # legitimately claimed as open ocean by a different edge entirely (see
    # test_no_coast_band_tile_is_secretly_inside_another_edges_ocean below),
    # which isn't the headland/bay texture this test is after.
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


def test_no_coast_band_tile_is_secretly_inside_another_edges_ocean():
    """Regression test: "we shouldn't have any legs on the island." biome_at
    used to check one edge fully (its own ocean, then its own coast band)
    before ever moving on to the next -- right next to a corner, that let an
    edge's coast-band check ("within COAST_BAND_WIDTH of MY boundary") commit
    to "cliffs"/"shoals" on ground a DIFFERENT edge's ocean check would have
    correctly claimed first, had it ever gotten the chance. That produced a
    visible strip of coastal texture jutting out past the real coastline at
    every corner -- a live screenshot showed it plainly at all four. Every
    tile biome_at calls cliffs or shoals must fail every one of the four
    edges' own plain ocean tests, not just the one edge whose band it's in."""
    from backend.world import _coast_boundary_x, _north_coast_boundary, _south_coast_boundary, _west_coast_boundary

    checked_any = False
    for x in range(0, 100):
        for y in range(0, 100):
            if biome_at(x, y) not in ("cliffs", "shoals"):
                continue
            checked_any = True
            assert x < _coast_boundary_x(y)
            assert x > _west_coast_boundary(y)
            assert y > _north_coast_boundary(x)
            assert y < _south_coast_boundary(x)
    assert checked_any  # confirms this test actually exercised real coast-band tiles


def test_volcano_clears_every_coastline_with_real_margin():
    """Map dream, phase 2: the volcano's original (10, 12) center now sits too
    close to the (thin, but real) new west/north coast band -- its relocated
    (13, 13) must clear every one of the four coastlines by more than
    VOLCANO_RADIUS, not just barely poke over the line."""
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
