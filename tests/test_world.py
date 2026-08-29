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


def test_biome_at_covers_all_four_regions():
    assert biome_at(85, 85) == "forest"
    assert biome_at(10, 45) == "mountains"
    assert biome_at(50, 50) == "river"
    assert biome_at(65, 85) == "plains"
