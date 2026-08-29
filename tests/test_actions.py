from backend.actions import ACTION_REGISTRY
from backend.ancestral_matrix import AncestralTraumaMatrix
from backend.simulation import Simulation, Tribe
from backend.world import Landscape


def _bare_simulation():
    sim = Simulation.__new__(Simulation)
    sim.world = Landscape(100)
    sim.trauma = AncestralTraumaMatrix(100)
    sim.cycle = 1
    return sim


def test_first_harvest_at_a_tile_yields_full_amount():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "plains")

    assert tribe.wood == 10


def test_repeated_harvest_at_the_same_tile_yields_less_each_time():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "plains")
    first_gain = tribe.wood
    tribe.wood = 0
    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "plains")
    second_gain = tribe.wood

    assert second_gain < first_gain


def test_harvesting_elsewhere_is_unaffected_by_a_depleted_tile():
    sim = _bare_simulation()
    depleted_tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(5):
        depleted_tribe.wood = 0
        ACTION_REGISTRY["GATHER_WOOD"](sim, depleted_tribe, "plains")

    fresh_tribe = Tribe("tribe_1", "Mountain Tribe", "qwen2.5:3b", 10, 45, "#fb923c")
    fresh_tribe.wood = 0
    ACTION_REGISTRY["GATHER_WOOD"](sim, fresh_tribe, "mountains")

    assert fresh_tribe.wood == 10  # full yield at an untouched tile


def test_second_fire_at_the_same_tile_costs_nothing_and_gains_no_pride():
    """Regression test: a real 8-cycle live run showed a model spamming BUILD_FIRE at
    the same tile every cycle, each one radiating more ancestral pride at zero
    additional benefit -- a self-reinforcing loop that made staying in one place look
    increasingly attractive. A second fire where one already burns should do nothing."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50

    ACTION_REGISTRY["BUILD_FIRE"](sim, tribe, "plains")
    wood_after_first = tribe.wood
    pride_after_first = float(sim.trauma.ghost_tensor[50, 50])

    ACTION_REGISTRY["BUILD_FIRE"](sim, tribe, "plains")

    assert tribe.wood == wood_after_first  # no wood spent the second time
    assert float(sim.trauma.ghost_tensor[50, 50]) == pride_after_first  # no extra pride


def test_second_wall_at_the_same_tile_costs_nothing():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50
    tribe.stone = 50

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains")
    wood_after_first, stone_after_first = tribe.wood, tribe.stone

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains")

    assert (tribe.wood, tribe.stone) == (wood_after_first, stone_after_first)


def test_hunting_success_also_depletes_local_game():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.food = 0

    ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "plains")  # plains has no wolf hazard

    assert tribe.food == 15
    assert sim.world.scarcity("game", 65, 85) > 0
