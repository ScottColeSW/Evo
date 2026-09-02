from backend import config
from backend.city_layout import (
    _is_natural_barrier,
    breach_outer_ring,
    build_ring,
    inner_ring_defense_bonus,
    next_unlockable_section,
    next_wall_work_section,
    ring_fully_built,
    ring_fully_reinforced,
    wall_defense_fraction,
)
from backend.simulation import Tribe
from backend.world import Landscape


def _tribe_at(x=50, y=50):
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", x, y, "#c084fc")
    tribe.territory_center = (x, y)
    tribe.territory_radius = config.WALL_RING_RADIUS_STEP
    return tribe


def test_build_ring_produces_the_expected_section_count_and_radius():
    world = Landscape(100)
    tribe = _tribe_at()

    ring = build_ring(world, tribe, ring_index=0)

    assert ring["radius"] == config.WALL_RING_RADIUS_STEP
    assert len(ring["sections"]) == config.WALL_RING_SECTION_COUNT
    directions = {s["direction"] for s in ring["sections"]}
    assert directions == {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


def test_build_ring_second_ring_sits_further_out():
    world = Landscape(100)
    tribe = _tribe_at()

    ring0 = build_ring(world, tribe, ring_index=0)
    ring1 = build_ring(world, tribe, ring_index=1)

    assert ring1["radius"] > ring0["radius"]


def test_is_natural_barrier_true_when_footprint_crosses_water():
    world = Landscape(100)
    # (40, 37) is real river/lake terrain used elsewhere in this test suite.
    assert _is_natural_barrier(world, 40, 37, 5, 1) is True


def test_is_natural_barrier_false_on_plain_land():
    world = Landscape(100)
    assert _is_natural_barrier(world, 50, 50, 5, 1) is False


def test_natural_barrier_sections_start_unlocked_others_do_not():
    world = Landscape(100)
    tribe = _tribe_at()

    ring = build_ring(world, tribe, ring_index=0)

    for sec in ring["sections"]:
        assert sec["unlocked"] == sec["natural_barrier"]


def _plain_ring():
    """A ring with every section forced non-natural, for tests that don't care
    about real map geometry."""
    return {"radius": 40, "sections": [
        {"index": i, "direction": d, "x": 0, "y": 0, "w": 5, "h": 1, "orientation": "horizontal",
         "natural_barrier": False, "unlocked": False, "progress": 0, "tier": 0}
        for i, d in enumerate(("N", "NE", "E", "SE", "S", "SW", "W", "NW"))
    ]}


def test_next_wall_work_section_picks_unfinished_construction_first():
    tribe = _tribe_at()
    ring = _plain_ring()
    ring["sections"][0]["unlocked"] = True
    ring["sections"][3]["unlocked"] = True
    ring["sections"][3]["progress"] = 100
    tribe.wall_rings = [ring]

    assert next_wall_work_section(tribe) == (0, 0)


def test_next_wall_work_section_picks_reinforcement_once_construction_done():
    tribe = _tribe_at()
    ring = _plain_ring()
    ring["sections"][0]["unlocked"] = True
    ring["sections"][0]["progress"] = 100
    tribe.wall_rings = [ring]

    assert next_wall_work_section(tribe) == (0, 0)


def test_next_wall_work_section_none_once_everything_maxed():
    tribe = _tribe_at()
    ring = _plain_ring()
    for sec in ring["sections"]:
        sec["unlocked"] = True
        sec["progress"] = 100
        sec["tier"] = config.WALL_MAX_LAYERS
    tribe.wall_rings = [ring]

    assert next_wall_work_section(tribe) is None


def test_next_wall_work_section_none_when_nothing_unlocked():
    tribe = _tribe_at()
    tribe.wall_rings = [_plain_ring()]

    assert next_wall_work_section(tribe) is None


def test_next_unlockable_section_skips_natural_barriers_and_already_unlocked():
    tribe = _tribe_at()
    ring = _plain_ring()
    ring["sections"][0]["natural_barrier"] = True
    ring["sections"][0]["unlocked"] = True
    ring["sections"][1]["unlocked"] = True
    tribe.wall_rings = [ring]

    assert next_unlockable_section(tribe) == (0, 2)


def test_next_unlockable_section_none_when_ring_fully_unlocked():
    tribe = _tribe_at()
    ring = _plain_ring()
    for sec in ring["sections"]:
        sec["unlocked"] = True
    tribe.wall_rings = [ring]

    assert next_unlockable_section(tribe) is None


def test_ring_fully_built_requires_every_section_at_100_or_natural():
    ring = _plain_ring()
    for sec in ring["sections"]:
        sec["progress"] = 100
    assert ring_fully_built(ring) is True

    ring["sections"][0]["progress"] = 50
    assert ring_fully_built(ring) is False


def test_ring_fully_reinforced_requires_every_section_at_max_tier_or_natural():
    ring = _plain_ring()
    for sec in ring["sections"]:
        sec["tier"] = config.WALL_MAX_LAYERS
    assert ring_fully_reinforced(ring) is True

    ring["sections"][0]["tier"] = 0
    assert ring_fully_reinforced(ring) is False


def test_wall_defense_fraction_zero_with_no_rings():
    tribe = _tribe_at()
    assert wall_defense_fraction(tribe) == 0.0


def test_wall_defense_fraction_full_when_outer_ring_fully_reinforced():
    tribe = _tribe_at()
    ring = _plain_ring()
    for sec in ring["sections"]:
        sec["unlocked"] = True
        sec["progress"] = 100
        sec["tier"] = config.WALL_MAX_LAYERS
    tribe.wall_rings = [ring]

    assert wall_defense_fraction(tribe) == 1.0


def test_wall_defense_fraction_natural_barrier_contributes_weaker_value():
    tribe = _tribe_at()
    ring = _plain_ring()
    ring["sections"][0]["natural_barrier"] = True
    ring["sections"][0]["unlocked"] = True
    tribe.wall_rings = [ring]

    # 1 natural section (0.5) + 7 unbuilt sections (0.0), averaged.
    expected = config.NATURAL_BARRIER_DEFENSE_FRACTION / config.WALL_RING_SECTION_COUNT
    assert wall_defense_fraction(tribe) == expected


def test_inner_ring_defense_bonus_zero_with_only_one_ring():
    tribe = _tribe_at()
    tribe.wall_rings = [_plain_ring()]
    assert inner_ring_defense_bonus(tribe) == 0.0


def test_inner_ring_defense_bonus_counts_only_fully_reinforced_inner_rings():
    tribe = _tribe_at()
    maxed_ring = _plain_ring()
    for sec in maxed_ring["sections"]:
        sec["tier"] = config.WALL_MAX_LAYERS
    partial_ring = _plain_ring()
    outer_ring = _plain_ring()
    tribe.wall_rings = [maxed_ring, partial_ring, outer_ring]

    assert inner_ring_defense_bonus(tribe) == config.RAIDER_DEFENSE_PER_INNER_RING_BONUS


def test_breach_outer_ring_resets_only_the_outermost_non_natural_sections():
    tribe = _tribe_at()
    inner_ring = _plain_ring()
    for sec in inner_ring["sections"]:
        sec["progress"] = 100
        sec["tier"] = config.WALL_MAX_LAYERS
    outer_ring = _plain_ring()
    outer_ring["sections"][0]["natural_barrier"] = True
    outer_ring["sections"][0]["progress"] = 100  # should never be reset -- terrain
    for sec in outer_ring["sections"][1:]:
        sec["progress"] = 100
        sec["tier"] = config.WALL_MAX_LAYERS
    tribe.wall_rings = [inner_ring, outer_ring]

    breach_outer_ring(tribe)

    assert all(sec["progress"] == 100 and sec["tier"] == config.WALL_MAX_LAYERS for sec in inner_ring["sections"])
    assert outer_ring["sections"][0]["progress"] == 100  # natural barrier untouched
    assert all(sec["progress"] == 0 and sec["tier"] == 0 for sec in outer_ring["sections"][1:])


def test_breach_outer_ring_does_nothing_without_any_rings():
    tribe = _tribe_at()
    breach_outer_ring(tribe)  # must not raise
