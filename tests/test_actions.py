from backend.actions import ACTION_REGISTRY
from backend.ancestral_matrix import AncestralTraumaMatrix
from backend.simulation import Simulation, Tribe
from backend.world import Landscape

# Most actions here ignore target entirely (only RELOCATE and SCOUT use it) -- (0, 0) is
# just an unused placeholder for those tests.
_NO_TARGET = (0, 0)


def _bare_simulation():
    sim = Simulation.__new__(Simulation)
    sim.world = Landscape(100)
    sim.trauma = AncestralTraumaMatrix(100)
    sim.cycle = 1
    sim.immortality_cycles = 0
    sim.recent_encounters = []
    return sim


def test_first_harvest_at_a_tile_yields_full_amount():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == 10


def test_repeated_harvest_at_the_same_tile_yields_less_each_time():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)
    first_gain = tribe.wood
    tribe.wood = 0
    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)
    second_gain = tribe.wood

    assert second_gain < first_gain


def test_harvesting_elsewhere_is_unaffected_by_a_depleted_tile():
    sim = _bare_simulation()
    depleted_tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(5):
        depleted_tribe.wood = 0
        ACTION_REGISTRY["GATHER_WOOD"](sim, depleted_tribe, "forest", _NO_TARGET)

    fresh_tribe = Tribe("tribe_1", "Mountain Tribe", "qwen2.5:3b", 10, 45, "#fb923c")
    fresh_tribe.wood = 0
    ACTION_REGISTRY["GATHER_WOOD"](sim, fresh_tribe, "forest", _NO_TARGET)

    assert fresh_tribe.wood == 10  # full yield at an untouched tile


def test_wood_yield_is_scaled_down_outside_forest():
    """Regression test: GATHER_WOOD/GATHER_STONE/HUNT_DEER used to pay the exact same
    flat yield in every biome (only local depletion scaled them) -- a tribe standing on
    a bare mountain peak could "hunt deer" as effectively as one deep in a forest.
    Resources should actually depend on what's around the tribe."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Mountain Tribe", "gemma2:2b", 20, 20, "#fb923c")
    tribe.wood = 0

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "mountains", _NO_TARGET)

    assert 0 < tribe.wood < 10  # some wood, but far less than a forest tile


def test_stone_yield_is_negligible_outside_mountains():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.stone = 0

    ACTION_REGISTRY["GATHER_STONE"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.stone == 1  # round(10 * 0.1)


def test_second_fire_at_the_same_tile_costs_nothing_and_gains_no_pride():
    """Regression test: a real 8-cycle live run showed a model spamming BUILD_FIRE at
    the same tile every cycle, each one radiating more ancestral pride at zero
    additional benefit -- a self-reinforcing loop that made staying in one place look
    increasingly attractive. A second fire where one already burns should do nothing."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50

    ACTION_REGISTRY["BUILD_FIRE"](sim, tribe, "plains", _NO_TARGET)
    wood_after_first = tribe.wood
    pride_after_first = float(sim.trauma.ghost_tensor[50, 50])

    ACTION_REGISTRY["BUILD_FIRE"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.wood == wood_after_first  # no wood spent the second time
    assert float(sim.trauma.ghost_tensor[50, 50]) == pride_after_first  # no extra pride


def test_construct_wall_adds_progress_and_costs_proportional_resources():
    """Explicit request: a wall is built in stages, like a crop, not finished in one
    action -- one CONSTRUCT_WALL call should leave a real, partial construction, not
    an instantly-complete wall or nothing at all."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50
    tribe.stone = 50

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    wall = sim.world.constructions[(50, 50)]
    assert wall["type"] == "wall"
    assert 0 < wall["progress"] < 100
    assert tribe.wood < 50
    assert tribe.stone < 50


def test_construct_wall_no_op_when_cannot_afford_the_proportional_cost():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0
    tribe.stone = 0

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    assert (50, 50) not in sim.world.constructions


def test_construct_wall_is_a_no_op_once_complete():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 100
    tribe.stone = 100

    for _ in range(6):  # comfortably more than enough calls to reach 100%
        ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    assert sim.world.constructions[(50, 50)]["progress"] == 100
    wood_at_completion, stone_at_completion = tribe.wood, tribe.stone

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    assert (tribe.wood, tribe.stone) == (wood_at_completion, stone_at_completion)


def test_larger_population_builds_wall_progress_faster():
    """Reuses _labor_multiplier -- the same 'more hands get more done' concept
    _harvest already uses -- rather than a separate team-size notion."""
    sim = _bare_simulation()
    small = Tribe("tribe_0", "Small Tribe", "gemma2:2b", 50, 50, "#c084fc")
    small.wood, small.stone, small.population = 100, 100, 8
    big = Tribe("tribe_1", "Big Tribe", "gemma2:2b", 60, 60, "#f97316")
    big.wood, big.stone, big.population = 100, 100, 40

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, small, "plains", _NO_TARGET)
    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, big, "plains", _NO_TARGET)

    assert sim.world.constructions[(60, 60)]["progress"] > sim.world.constructions[(50, 50)]["progress"]


def test_plant_crop_spends_wood_and_adds_a_plot():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50

    from backend import config
    ACTION_REGISTRY["PLANT_CROP"](sim, tribe, "river", _NO_TARGET)

    assert tribe.farm_plots == 1
    assert tribe.wood == 50 - config.PLANT_CROP_WOOD_COST


def test_plant_crop_does_nothing_without_enough_wood():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0

    ACTION_REGISTRY["PLANT_CROP"](sim, tribe, "river", _NO_TARGET)

    assert tribe.farm_plots == 0


def test_plant_crop_is_capped_at_max_farm_plots():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 1000
    tribe.farm_plots = config.MAX_FARM_PLOTS

    ACTION_REGISTRY["PLANT_CROP"](sim, tribe, "river", _NO_TARGET)

    assert tribe.farm_plots == config.MAX_FARM_PLOTS  # no further growth past the cap
    assert tribe.wood == 1000  # and no wood spent trying


def test_gather_eggs_sets_pending_hatch_on_a_successful_roll():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.0):  # below any chance
        ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "river", _NO_TARGET)

    assert tribe.pending_hatch == {"parents": None}  # founding egg -- nothing to cross yet


def test_gather_eggs_does_nothing_on_a_failed_roll():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.999):  # above any chance
        ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "river", _NO_TARGET)

    assert tribe.pending_hatch is None


def test_gather_eggs_refuses_a_second_egg_while_one_is_already_pending():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.pending_hatch = {"parents": None}

    with mock.patch("backend.actions.random.random", return_value=0.0):
        result = ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "river", _NO_TARGET)

    assert tribe.pending_hatch == {"parents": None}  # unchanged, not overwritten
    assert result == "an egg is already being tended -- one thing at a time"


def test_gather_eggs_crosses_the_two_most_recent_flock_members_once_available():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.flock_lineage = [
        {"trait": "first", "parents": [], "cycle": 1, "note": ""},
        {"trait": "second", "parents": [], "cycle": 2, "note": ""},
        {"trait": "third", "parents": [], "cycle": 3, "note": ""},
    ]

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["GATHER_EGGS"](sim, tribe, "river", _NO_TARGET)

    assert [p["trait"] for p in tribe.pending_hatch["parents"]] == ["second", "third"]


def test_gather_fish_catches_food_on_a_successful_roll():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["GATHER_FISH"](sim, tribe, "river", _NO_TARGET)

    assert tribe.food > 0


def test_gather_fish_does_nothing_on_a_failed_roll():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0

    with mock.patch("backend.actions.random.random", return_value=0.999):
        ACTION_REGISTRY["GATHER_FISH"](sim, tribe, "river", _NO_TARGET)

    assert tribe.food == 0
    assert tribe.fishing_learned is False


def test_first_successful_catch_learns_fishing_and_celebrates():
    """Explicit request: "learning to fish" isn't a separate knowledge system --
    the first catch just flips fishing_learned, awards a trophy, and throws the same
    kind of party water discovery/settling/game discovery already get."""
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 100

    with mock.patch("backend.actions.random.random", return_value=0.0):
        result = ACTION_REGISTRY["GATHER_FISH"](sim, tribe, "river", _NO_TARGET)

    assert tribe.fishing_learned is True
    assert any(t["name"] == "Angler" for t in tribe.trophies)
    assert any("celebrates learning to fish" in entry for entry in tribe.history)
    assert "first catch" in result


def test_cook_food_requires_an_existing_fire():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    result = ACTION_REGISTRY["COOK_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.cooking_learned is False
    assert "no fire" in result


def test_cook_food_learns_cooking_once_a_fire_exists():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.world.add_construction(50, 50, "fire", sim.cycle)

    result = ACTION_REGISTRY["COOK_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.cooking_learned is True
    assert any(t["name"] == "Master Chef" for t in tribe.trophies)
    assert "learns to cook" in result


def test_cook_food_is_a_no_op_once_already_learned():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.world.add_construction(50, 50, "fire", sim.cycle)
    tribe.cooking_learned = True
    trophies_before = list(tribe.trophies)

    result = ACTION_REGISTRY["COOK_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert result is None
    assert tribe.trophies == trophies_before


def test_later_catches_do_not_re_learn_or_re_celebrate():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.fishing_learned = True
    tribe.food = 100

    with mock.patch("backend.actions.random.random", return_value=0.0):
        result = ACTION_REGISTRY["GATHER_FISH"](sim, tribe, "river", _NO_TARGET)

    assert len(tribe.trophies) == 0
    assert not any("celebrates learning to fish" in entry for entry in tribe.history)
    assert "first catch" not in result


def test_hunting_success_also_depletes_local_game():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.food = 0

    ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "plains", _NO_TARGET)  # plains has no wolf hazard

    assert tribe.food == 9  # round(15 * 0.6 plains game multiplier)
    assert sim.world.scarcity("game", 65, 85) > 0


def test_forage_yields_the_most_on_plains():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.food = 0

    ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.food == 10  # round(10 * 1.0 plains forage multiplier)


def test_forage_yield_is_lower_in_forest_than_plains():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0

    ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.food == 6  # round(10 * 0.6 forest forage multiplier)


def test_forage_carries_no_hazard_unlike_hunting():
    import random
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0
    tribe.population = 10

    with mock.patch("backend.actions.random.random", return_value=0.0):  # would trigger a wolf hazard if this were HUNT_DEER
        ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.population == 10  # no population lost
    assert tribe.food > 0


def test_foraging_also_depletes_the_local_forage_scarcity():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.food = 0

    ACTION_REGISTRY["GATHER_FOOD"](sim, tribe, "plains", _NO_TARGET)

    assert sim.world.scarcity("forage", 65, 85) > 0


def test_harvest_yield_is_unchanged_at_or_below_starting_population():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0
    tribe.population = 4  # below the 8-person baseline

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == 10  # no penalty for a smaller-than-baseline tribe


def test_harvest_yield_scales_up_for_a_larger_tribe():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 0
    tribe.population = 16  # double the 8-person baseline

    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)

    assert tribe.wood == 20  # round(10 * (16/8) labor multiplier)


def test_expedition_capacity_matches_the_floor_at_baseline_population():
    from backend.actions import expedition_capacity
    from backend import config

    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    assert expedition_capacity(tribe) == config.MAX_CONCURRENT_EXPEDITIONS


def test_expedition_capacity_grows_with_population():
    from backend.actions import expedition_capacity

    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 25

    assert expedition_capacity(tribe) == 5  # 25 // EXPEDITION_SLOT_POPULATION_DIVISOR (5)


def test_a_third_scout_succeeds_once_population_growth_raises_capacity():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 25  # capacity 5, well past the old fixed cap of 2
    tribe.expeditions = [
        {"lead_scout": "A", "day": 1, "max_days": 3, "phase": "outbound"},
        {"lead_scout": "B", "day": 1, "max_days": 3, "phase": "outbound"},
    ]

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (60, 60))

    assert "no one left to send" not in note
    assert len(tribe.expeditions) == 3


def test_relocate_moves_the_tribe_toward_target():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["RELOCATE"](sim, tribe, "plains", (80, 50))

    assert (tribe.x, tribe.y) != (50, 50)
    assert tribe.x > 50  # moved toward the target, not away or nowhere


def test_relocate_costs_stamina():
    """Without a cost here, RELOCATE would be strictly free compared to every gathering
    action, which all cost time and risk -- marching should be tiring."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 40
    tribe.water = 30

    ACTION_REGISTRY["RELOCATE"](sim, tribe, "plains", (80, 50))

    assert tribe.food == 39
    assert tribe.water == 29


def test_relocate_stamina_cost_never_goes_negative():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.food = 0
    tribe.water = 1

    ACTION_REGISTRY["RELOCATE"](sim, tribe, "plains", (80, 50))

    assert tribe.food == 0
    assert tribe.water == 0


def test_other_actions_never_move_the_tribe():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}

    for action in ("GATHER_WOOD", "GATHER_STONE", "HUNT_DEER", "BUILD_FIRE", "CONSTRUCT_WALL", "RAID", "TRADE", "IDLE"):
        ACTION_REGISTRY[action](sim, tribe, "plains", (80, 80))
        assert (tribe.x, tribe.y) == (50, 50), f"{action} should not move the tribe"


def test_scout_does_not_move_the_tribe_but_launches_an_expedition():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))

    assert (tribe.x, tribe.y) == (50, 50)  # scouting doesn't relocate the tribe
    assert len(tribe.expeditions) == 1
    assert tribe.expeditions[0]["target"] == [10, 10]
    assert tribe.expeditions[0]["day"] == 0
    assert tribe.expeditions[0]["phase"] == "outbound"
    assert "depart" in note


def test_scout_launch_gives_the_expedition_a_named_lead_and_determination_trait():
    """The exploration party isn't a second LLM agent making its own choices (that
    would double Ollama calls per tribe per cycle for a party the tribe already
    decided to send) -- but it does get its own procedurally-generated character, not
    every expedition behaving identically."""
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))

    exp = tribe.expeditions[0]
    assert exp["lead_scout"]  # a real, non-empty name
    assert 0.0 <= exp["determination"] <= 1.0
    span = config.EXPEDITION_DETERMINATION_DAY_VARIANCE
    assert config.EXPEDITION_MAX_DAYS - span <= exp["max_days"] <= config.EXPEDITION_MAX_DAYS + span


def test_scout_launch_is_deterministic_per_tribe_and_cycle():
    """Same tribe, same cycle should always produce the same scout -- re-reading an
    expedition's state (e.g. across a websocket reconnect) shouldn't change who's
    leading it."""
    sim = _bare_simulation()
    tribe_a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe_b = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["SCOUT"](sim, tribe_a, "plains", (10, 10))
    ACTION_REGISTRY["SCOUT"](sim, tribe_b, "plains", (10, 10))

    assert tribe_a.expeditions[0]["lead_scout"] == tribe_b.expeditions[0]["lead_scout"]
    assert tribe_a.expeditions[0]["determination"] == tribe_b.expeditions[0]["determination"]


def test_raid_with_no_rival_nearby_does_nothing():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}

    note = ACTION_REGISTRY["RAID"](sim, tribe, "plains", (10, 10))

    assert "no rival" in note
    assert tribe.population == 8


def test_raid_win_steals_resources_and_absorbs_some_of_the_defenders_population():
    """Population moves rather than just vanishing on a raid win -- captured or
    defecting survivors, not a pointless flat loss with no benefit to the winner."""
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.wood = 20
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.0):  # any positive win chance wins
        note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert "raided Mountain Tribe" in note
    assert attacker.wood == 56  # 50 starting + round(20 * 0.3) stolen
    assert defender.wood == 14  # 20 - 6
    assert defender.population == 6  # 8 - round(8 * RAID_POPULATION_ABSORB_FRACTION (0.2)) = 8 - 2
    assert attacker.population == 9  # 8 + 2 absorbed - 1 (RAID_ATTACKER_POPULATION_LOSS_ON_WIN)


def test_raid_win_awards_first_conquest_trophy_only_on_the_first_win():
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.population = 20  # plenty of survivors, won't trigger a merge
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))
        ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert len([t for t in attacker.trophies if t["name"] == "First Conquest"]) == 1


def test_raid_win_grants_a_chief_proposed_raiding_award():
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.population = 20  # plenty of survivors, won't trigger a merge
    attacker.custom_awards = [{"name": "Blooded Spear", "category": "raiding", "cycle": 1}]
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert any(t["name"] == "Blooded Spear" for t in attacker.trophies)


def test_raid_that_reduces_defender_to_zero_population_merges_into_the_attacker():
    from unittest import mock

    sim = _bare_simulation()
    sim.cycle = 5
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.population = 1  # a single raid at 20% (rounded up to a minimum of 1) finishes them
    defender.era = "bronze_age"
    defender.wood, defender.stone, defender.food, defender.water = 5, 5, 5, 5
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.0):
        note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert note == "raided Mountain Tribe, fully absorbing its survivors -- Forest Tribe becomes Forest Tribe (Advanced)!"
    assert attacker.name == "Forest Tribe (Advanced)"
    assert attacker.era == "bronze_age"  # inherits the higher of the two eras
    assert attacker.chief_name == ""  # chief-less, awaiting the next cycle's succession
    assert "tribe_1" not in sim.tribes
    assert defender.extinct is True


def test_raid_loss_costs_the_attacker_without_stealing_anything():
    from unittest import mock

    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    defender.wood = 20
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}

    with mock.patch("backend.actions.random.random", return_value=0.999):  # forces a loss
        note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert "repelled" in note
    assert defender.wood == 20  # untouched
    assert attacker.population == 6  # 8 - RAID_ATTACKER_POPULATION_LOSS_ON_LOSS (2)
    assert defender.population == 8  # untouched


def test_strike_raider_camp_fails_with_no_known_sighting_at_that_location():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.raider_sightings = []

    note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, tribe, "plains", (60, 60))

    assert "no known raider camp" in note


def test_strike_raider_camp_success_removes_the_sighting_and_loots_food():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.raider_sightings = [(60, 60)]
    tribe.food = 100

    with mock.patch("backend.actions.random.random", return_value=0.0):  # forces a win
        note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, tribe, "plains", (60, 60))

    assert "destroyed" in note
    assert tribe.raider_sightings == []
    assert tribe.food > 100
    assert "PRIDE" in sim.trauma.bias_string(60, 60)


def test_strike_raider_camp_failure_costs_population_and_leaves_sighting_intact():
    from unittest import mock

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.raider_sightings = [(60, 60)]
    tribe.population = 10

    with mock.patch("backend.actions.random.random", return_value=0.999):  # forces a loss
        note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, tribe, "plains", (60, 60))

    assert "escaped" in note
    assert tribe.raider_sightings == [(60, 60)]
    assert tribe.population < 10
    assert "DREAD" in sim.trauma.bias_string(50, 50)


def test_strike_raider_camp_win_chance_scales_with_population():
    from unittest import mock

    from backend import config

    sim = _bare_simulation()
    low = Tribe("tribe_0", "Small Tribe", "gemma2:2b", 50, 50, "#c084fc")
    low.raider_sightings = [(60, 60)]
    low.population = 1
    high = Tribe("tribe_1", "Big Tribe", "gemma2:2b", 50, 50, "#f97316")
    high.raider_sightings = [(60, 60)]
    high.population = 200  # capped at STRIKE_RAIDER_CAMP_MAX_WIN_CHANCE

    # A roll that clears the capped high-population chance but not the low one.
    roll = config.STRIKE_RAIDER_CAMP_BASE_WIN_CHANCE + 0.001
    with mock.patch("backend.actions.random.random", return_value=roll):
        low_note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, low, "plains", (60, 60))
        high_note = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, high, "plains", (60, 60))

    assert "escaped" in low_note
    assert "destroyed" in high_note


def test_strike_raider_camp_unlocked_only_from_bronze_age():
    from backend.eras import unlocked_actions_through

    assert "STRIKE_RAIDER_CAMP" not in unlocked_actions_through("stone_age")
    assert "STRIKE_RAIDER_CAMP" in unlocked_actions_through("bronze_age")


def test_cook_food_unlocked_only_from_bronze_age():
    from backend.eras import unlocked_actions_through

    assert "COOK_FOOD" not in unlocked_actions_through("stone_age")
    assert "COOK_FOOD" in unlocked_actions_through("bronze_age")


def test_raid_ignores_extinct_tribes_and_self():
    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    dead_rival = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    dead_rival.extinct = True
    sim.tribes = {"tribe_0": attacker, "tribe_1": dead_rival}

    note = ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))

    assert "no rival" in note


def test_trade_with_no_rival_nearby_does_nothing():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}

    note = ACTION_REGISTRY["TRADE"](sim, tribe, "plains", (10, 10))

    assert "no rival" in note


def test_trade_exchanges_resources_both_ways_with_no_loss_of_population():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.wood, a.stone, a.food, a.water = 100, 0, 0, 0
    b.wood, b.stone, b.food, b.water = 0, 100, 0, 0
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    note = ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert "opened trade with Mountain Tribe" in note
    # 15% of A's wood (100) moves to B, 15% of B's stone (100) moves to A
    assert a.wood == 85
    assert a.stone == 15
    assert b.wood == 15
    assert b.stone == 85
    assert a.population == 8 and b.population == 8  # no cost, unlike a raid


def test_trade_increments_both_sides_trade_counter():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert a.trades_completed == 1
    assert b.trades_completed == 1


def test_trade_awards_first_contact_trophy_to_both_sides_on_first_trade():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert any(t["name"] == "First Contact" for t in a.trophies)
    assert any(t["name"] == "First Contact" for t in b.trophies)


def test_trade_does_not_re_award_first_contact_on_a_later_trade():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))
    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert len([t for t in a.trophies if t["name"] == "First Contact"]) == 1


def test_trade_grants_a_chief_proposed_trading_award_to_both_sides():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.custom_awards = [{"name": "Merchant's Mark", "category": "trading", "cycle": 1}]
    b.custom_awards = [{"name": "Silver Tongue", "category": "trading", "cycle": 1}]
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert any(t["name"] == "Merchant's Mark" for t in a.trophies)
    assert any(t["name"] == "Silver Tongue" for t in b.trophies)


def test_trade_does_not_grant_a_proposed_award_from_a_different_category():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    a.custom_awards = [{"name": "Keeper of the Trails", "category": "scouting", "cycle": 1}]
    sim.tribes = {"tribe_0": a, "tribe_1": b}

    ACTION_REGISTRY["TRADE"](sim, a, "plains", (51, 51))

    assert not any(t["name"] == "Keeper of the Trails" for t in a.trophies)


def test_trade_ignores_extinct_tribes_and_self():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    dead_rival = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    dead_rival.extinct = True
    sim.tribes = {"tribe_0": tribe, "tribe_1": dead_rival}

    note = ACTION_REGISTRY["TRADE"](sim, tribe, "plains", (51, 51))

    assert "no rival" in note


def test_breed_does_nothing_with_only_a_chief_and_no_trophy_holder():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert "no one with enough standing" in note
    assert tribe.pending_birth is None
    assert tribe.food == 40  # unchanged, nothing was spent


def test_breed_pairs_the_chief_with_the_most_recent_trophy_holder():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [
        {"name": "Water Bringer", "chief": "Ashgar", "cycle": 1},
        {"name": "Master Pathfinder", "chief": "BriMir", "cycle": 10},
    ]

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert tribe.pending_birth == {"parent_a": "Ashgar", "parent_b": "BriMir"}
    assert "Ashgar and BriMir decide to start a family" in note


def test_breed_deducts_its_solo_cost():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]

    ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert tribe.food == 40 - config.BREED_FOOD_COST
    assert tribe.water == config.STARTING_WATER - config.BREED_WATER_COST


def test_breed_is_free_and_succeeds_even_with_almost_no_food_or_water():
    """BREED costs nothing (see config.BREED_FOOD_COST/WATER_COST) -- the two real
    eligible windows watched live were both inside a full starvation spiral, so
    affordability can't be a second gate stacked on top of the real constraint
    (two distinct named individuals)."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]
    tribe.food = 0
    tribe.water = 0

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert "Ashgar and BriMir decide to start a family" in note
    assert tribe.pending_birth == {"parent_a": "Ashgar", "parent_b": "BriMir"}
    assert tribe.food == 0
    assert tribe.water == 0


def test_breed_refuses_at_the_population_cap():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.chief_name = "Ashgar"
    tribe.trophies = [{"name": "Water Bringer", "chief": "BriMir", "cycle": 1}]
    tribe.population = config.POPULATION_GROWTH_CAP

    note = ACTION_REGISTRY["BREED"](sim, tribe, "plains", (0, 0))

    assert "no room" in note
    assert tribe.pending_birth is None


def test_scout_launches_a_second_party_while_capacity_remains():
    """A tribe can run up to config.MAX_CONCURRENT_EXPEDITIONS parties at once -- a
    chief with people to spare shouldn't be capped at just one no matter what."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))
    first_expedition = tribe.expeditions[0]

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (80, 80))

    assert len(tribe.expeditions) == 2
    assert tribe.expeditions[0] is first_expedition  # the first party is untouched
    assert "depart" in note


def test_scout_refuses_once_at_capacity():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(config.MAX_CONCURRENT_EXPEDITIONS):
        ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))
    parties = list(tribe.expeditions)

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (80, 80))

    assert tribe.expeditions == parties  # unchanged, no third party squeezed in
    assert "no one left to send" in note


def test_hunting_party_does_not_move_the_tribe_but_launches_an_expedition():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    note = ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))

    assert (tribe.x, tribe.y) == (50, 50)
    assert len(tribe.expeditions) == 1
    assert tribe.expeditions[0]["kind"] == "hunt"
    assert tribe.expeditions[0]["target"] == [10, 10]
    assert tribe.expeditions[0]["day"] == 0
    assert tribe.expeditions[0]["phase"] == "outbound"
    assert tribe.expeditions[0]["food_caught"] == 0
    assert "depart" in note


def test_hunting_party_launch_uses_its_own_max_days_baseline():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))

    span = config.EXPEDITION_DETERMINATION_DAY_VARIANCE
    max_days = tribe.expeditions[0]["max_days"]
    assert config.HUNTING_PARTY_MAX_DAYS - span <= max_days <= config.HUNTING_PARTY_MAX_DAYS + span


def test_hunting_party_launches_a_second_party_while_capacity_remains():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))
    first_expedition = tribe.expeditions[0]

    note = ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (80, 80))

    assert len(tribe.expeditions) == 2
    assert tribe.expeditions[0] is first_expedition
    assert "depart" in note


def test_hunting_party_refuses_once_at_capacity():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for _ in range(config.MAX_CONCURRENT_EXPEDITIONS):
        ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))
    parties = list(tribe.expeditions)

    note = ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (80, 80))

    assert tribe.expeditions == parties
    assert "no one left to send" in note


def test_scout_and_hunting_party_can_both_be_out_at_once():
    """Hunting and scouting share the same expedition list and capacity -- a tribe can
    mix party types, not just repeat the same one."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))
    hunting_expedition = tribe.expeditions[0]

    ACTION_REGISTRY["SCOUT"](sim, tribe, "forest", (80, 80))

    assert len(tribe.expeditions) == 2
    assert tribe.expeditions[0] is hunting_expedition
    assert tribe.expeditions[1]["kind"] == "scout"
