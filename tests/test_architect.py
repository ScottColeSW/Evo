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


def test_find_free_slot_spills_just_outside_a_full_territory():
    """Explicit request (2026-09-06, alongside config.MAX_WALL_RINGS): "they
    will have to build outside the walls once they hit that point." A fully
    occupied territory no longer means no more room -- a slot just past
    territory_radius, within the config.WALL_RING_RADIUS_STEP buffer, is
    found instead of returning None outright."""
    from backend import config

    world = Landscape(100)
    tribe = _tribe_with_territory(radius=5)
    # Fill the entire walled territory with 1x1s -- nothing fits inside it.
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            record_building(tribe, "fire", 50 + dx, 50 + dy, 1, 1, cycle=0)

    slot = find_free_slot(world, tribe, "fire")

    assert slot is not None
    x, y = slot
    dist = max(abs(x - 50), abs(y - 50))  # this spiral moves in square rings
    assert dist > tribe.territory_radius  # genuinely outside the walls
    assert dist <= tribe.territory_radius + config.WALL_RING_RADIUS_STEP  # but still bounded, not anywhere on the map


def test_find_free_slot_returns_none_when_territory_has_no_room():
    """Explicit request (2026-09-06): find_free_slot now searches a bounded
    buffer (config.WALL_RING_RADIUS_STEP) past territory_radius too, so a
    tribe whose walls are complete can still build just outside them -- "no
    room anywhere" now means the whole search disc, not just the walled area.
    One big occupied rectangle spanning the entire expanded search zone
    reproduces that cheaply, without placing hundreds of individual 1x1s."""
    from backend import config

    world = Landscape(100)
    tribe = _tribe_with_territory(radius=0)
    span = tribe.territory_radius + config.WALL_RING_RADIUS_STEP + 5  # padding slack
    record_building(tribe, "fire", 50 - span, 50 - span, span * 2, span * 2, cycle=0)

    assert find_free_slot(world, tribe, "long_house") is None


def test_find_free_slot_returns_none_when_fully_occupied():
    """See the test above -- the fully-occupied area now has to cover the
    expanded search zone (territory_radius + config.WALL_RING_RADIUS_STEP),
    not just the bare territory_radius."""
    from backend import config

    world = Landscape(100)
    tribe = _tribe_with_territory(radius=2)
    span = tribe.territory_radius + config.WALL_RING_RADIUS_STEP + 5  # padding slack
    record_building(tribe, "fire", 50 - span, 50 - span, span * 2, span * 2, cycle=0)

    assert find_free_slot(world, tribe, "fire") is None


def test_find_free_slot_returns_none_without_a_territory_center():
    world = Landscape(100)
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    assert find_free_slot(world, tribe, "long_house") is None


def test_record_building_appends_expected_fields():
    tribe = _tribe_with_territory()

    record_building(tribe, "kitchen", 10, 12, 2, 2, cycle=7)

    assert tribe.buildings == [{"type": "kitchen", "x": 10, "y": 12, "w": 2, "h": 2, "cycle_built": 7}]


def test_a_remembered_answer_is_the_same_as_a_fresh_scan_and_skips_the_scan():
    """2026-10-10: _prepare_turn asks for every BUILD_* each turn and a failed scan of a packed territory was 65% of a mid-game tick."""
    from unittest import mock
    from backend import architect

    world = Landscape(100)
    tribe = _tribe_with_territory()
    fresh = architect._scan_for_free_slot(world, 3, 2, 50, 50, [], 1, tribe.territory_radius + 8)
    first = find_free_slot(world, tribe, "long_house")
    with mock.patch.object(architect, "_scan_for_free_slot", side_effect=AssertionError("scanned again")):
        again = find_free_slot(world, tribe, "long_house")
    assert first == again


def test_a_new_building_or_wall_section_or_different_radius_is_not_answered_from_memory():
    world = Landscape(100)
    tribe = _tribe_with_territory()
    first = find_free_slot(world, tribe, "long_house")
    record_building(tribe, "long_house", first[0], first[1], 3, 2, cycle=1)
    second = find_free_slot(world, tribe, "long_house")
    assert second != first  # the building moved the key; a stale answer would have placed it on top of the first

    tribe.territory_radius = 2
    packed = _tribe_with_territory(radius=2)
    for dx in range(-14, 15, 2):
        for dy in range(-14, 15, 2):
            record_building(packed, "long_house", 50 + dx, 50 + dy, 3, 2, cycle=1)
    assert find_free_slot(world, packed, "castle") is None
    packed.territory_radius = 40
    assert find_free_slot(world, packed, "castle") is not None  # a wider territory is a different question


def test_two_worlds_do_not_share_remembered_answers():
    tribe_a, tribe_b = _tribe_with_territory(), _tribe_with_territory()
    world_a, world_b = Landscape(100), Landscape(100)
    find_free_slot(world_a, tribe_a, "long_house")
    assert not getattr(world_b, "_free_slot_cache", None)
