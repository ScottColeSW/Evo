from backend.world import Landscape, biome_at


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


def test_biome_at_covers_all_five_regions():
    assert biome_at(80, 10) == "forest"
    assert biome_at(10, 10) == "mountains"
    assert biome_at(50, 90) == "plains"
    assert biome_at(40, 37) == "river"
    assert biome_at(95, 50) == "ocean"


def test_ocean_occupies_the_entire_east_edge():
    for y in range(0, 100, 10):
        assert biome_at(99, y) == "ocean"


def test_river_originates_near_the_mountains():
    from backend.world import RIVER_SOURCE_X, _river_center_y

    assert biome_at(RIVER_SOURCE_X, round(_river_center_y(RIVER_SOURCE_X))) == "river"


def test_river_reaches_the_coast():
    """The river must actually connect to the ocean, not fade out into forest or
    plains before reaching it -- an Earth-like river runs from source to sea."""
    from backend.world import OCEAN_X_START, _river_center_y

    mouth_y = round(_river_center_y(OCEAN_X_START - 1))
    assert biome_at(OCEAN_X_START - 1, mouth_y) == "river"
    assert biome_at(OCEAN_X_START, mouth_y) == "ocean"


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
