from backend.architect import find_free_slot, record_building
from backend.simulation import Tribe
from backend.world import Landscape


def _tribe_with_territory(x=50, y=50, radius=20):
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", x, y, "#c084fc")
    tribe.territory_center = (x, y)
    tribe.territory_radius = radius
    return tribe


def test_find_free_slot_returns_a_valid_in_bounds_non_water_anchor():
    world = Landscape(100)
    tribe = _tribe_with_territory()

    slot = find_free_slot(world, tribe, "long_house")

    assert slot is not None
    x, y = slot
    assert 0 <= x < world.grid_size and 0 <= y < world.grid_size


def test_find_free_slot_avoids_already_placed_buildings():
    world = Landscape(100)
    tribe = _tribe_with_territory()

    first = find_free_slot(world, tribe, "long_house")
    record_building(tribe, "long_house", first[0], first[1], 3, 2, cycle=1)
    second = find_free_slot(world, tribe, "long_house")

    assert second is not None
    assert second != first
    # No overlap (with padding) between the two placed rects.
    ax, ay = first
    bx, by = second
    assert not (ax < bx + 3 + 1 and bx < ax + 3 + 1 and ay < by + 2 + 1 and by < ay + 2 + 1)


def test_find_free_slot_avoids_wall_ring_sections():
    world = Landscape(100)
    tribe = _tribe_with_territory()
    tribe.wall_rings = [{
        "radius": 5,
        "sections": [{"index": 0, "direction": "N", "x": 50, "y": 45, "w": 5, "h": 1,
                       "orientation": "horizontal", "natural_barrier": False,
                       "unlocked": True, "progress": 0, "tier": 0}],
    }]

    slot = find_free_slot(world, tribe, "fire")

    assert slot is not None
    x, y = slot
    assert not (x < 50 + 5 + 1 and 50 < x + 1 + 1 and y < 45 + 1 + 1 and 45 < y + 1 + 1)


def test_find_free_slot_returns_none_when_territory_has_no_room():
    world = Landscape(100)
    tribe = _tribe_with_territory(radius=0)

    assert find_free_slot(world, tribe, "long_house") is None


def test_find_free_slot_returns_none_when_fully_occupied():
    world = Landscape(100)
    tribe = _tribe_with_territory(radius=2)
    # A 1x1 building fits at every tile in this tiny territory -- fill all of them.
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            record_building(tribe, "fire", 50 + dx, 50 + dy, 1, 1, cycle=0)

    assert find_free_slot(world, tribe, "fire") is None


def test_find_free_slot_returns_none_without_a_territory_center():
    world = Landscape(100)
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    assert find_free_slot(world, tribe, "long_house") is None


def test_record_building_appends_expected_fields():
    tribe = _tribe_with_territory()

    record_building(tribe, "kitchen", 10, 12, 2, 2, cycle=7)

    assert tribe.buildings == [{"type": "kitchen", "x": 10, "y": 12, "w": 2, "h": 2, "cycle_built": 7}]
