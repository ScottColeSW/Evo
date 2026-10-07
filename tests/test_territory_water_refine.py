"""2026-10-07 (the owner, about a live run's Tribe 1): the territory circle kept most of a river's course (15% of its tiles). After the barrier rule picks a center, one
within TERRITORY_WATER_REFINE_MAX_SHIFT tiles of the tribe with clearly less open water is preferred; a tribe on dry ground does not move."""
import math

from backend import config
from backend.simulation import Simulation


def _sim_with_tribe_at(x, y):
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": x, "y": y}, {"name": "B", "model": "gemma2:2b", "x": 5, "y": 95}])
    return sim, sim.tribes["tribe_0"]


def test_the_live_runs_river_site_gets_a_much_drier_center_the_tribe_still_stands_inside():
    sim, tribe = _sim_with_tribe_at(57, 40)  # where the live run's Tribe 1 settled, on the river
    old = (57, 43)  # what the barrier rule alone chose there
    assert sim._unbuildable_share(*old) > 0.14
    center = sim._choose_territory_center(tribe)
    assert center != old
    assert sim._unbuildable_share(*center) < 0.05
    assert math.hypot(center[0] - tribe.x, center[1] - tribe.y) <= config.TERRITORY_WATER_REFINE_MAX_SHIFT
    assert math.hypot(center[0] - tribe.x, center[1] - tribe.y) < config.WALL_RING_RADIUS_STEP  # the tribe is inside its own ring


def test_a_tribe_on_dry_ground_does_not_move():
    sim, tribe = _sim_with_tribe_at(44, 52)  # plains with no water anywhere inside the ring
    assert sim._unbuildable_share(44, 52) < config.TERRITORY_WATER_REFINE_MIN_GAIN
    assert sim._choose_territory_center(tribe) == (44, 52)


def test_the_new_center_keeps_the_natural_barrier_cap():
    from backend import city_layout

    sim, tribe = _sim_with_tribe_at(57, 40)
    center = sim._choose_territory_center(tribe)
    ring = city_layout.build_ring(sim.world, center, ring_index=0)
    assert sum(1 for s in ring["sections"] if s["natural_barrier"]) <= config.TERRITORY_MAX_ACCEPTABLE_NATURAL_BARRIERS
