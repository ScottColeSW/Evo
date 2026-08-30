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


def test_second_wall_at_the_same_tile_costs_nothing():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.wood = 50
    tribe.stone = 50

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)
    wood_after_first, stone_after_first = tribe.wood, tribe.stone

    ACTION_REGISTRY["CONSTRUCT_WALL"](sim, tribe, "plains", _NO_TARGET)

    assert (tribe.wood, tribe.stone) == (wood_after_first, stone_after_first)


def test_hunting_success_also_depletes_local_game():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Plains Tribe", "gemma2:2b", 65, 85, "#34d399")
    tribe.food = 0

    ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "plains", _NO_TARGET)  # plains has no wolf hazard

    assert tribe.food == 9  # round(15 * 0.6 plains game multiplier)
    assert sim.world.scarcity("game", 65, 85) > 0


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
    assert tribe.expedition is not None
    assert tribe.expedition["target"] == [10, 10]
    assert tribe.expedition["day"] == 0
    assert tribe.expedition["phase"] == "outbound"
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

    assert tribe.expedition["lead_scout"]  # a real, non-empty name
    assert 0.0 <= tribe.expedition["determination"] <= 1.0
    span = config.EXPEDITION_DETERMINATION_DAY_VARIANCE
    assert config.EXPEDITION_MAX_DAYS - span <= tribe.expedition["max_days"] <= config.EXPEDITION_MAX_DAYS + span


def test_scout_launch_is_deterministic_per_tribe_and_cycle():
    """Same tribe, same cycle should always produce the same scout -- re-reading an
    expedition's state (e.g. across a websocket reconnect) shouldn't change who's
    leading it."""
    sim = _bare_simulation()
    tribe_a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe_b = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["SCOUT"](sim, tribe_a, "plains", (10, 10))
    ACTION_REGISTRY["SCOUT"](sim, tribe_b, "plains", (10, 10))

    assert tribe_a.expedition["lead_scout"] == tribe_b.expedition["lead_scout"]
    assert tribe_a.expedition["determination"] == tribe_b.expedition["determination"]


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


def test_trade_ignores_extinct_tribes_and_self():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    dead_rival = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    dead_rival.extinct = True
    sim.tribes = {"tribe_0": tribe, "tribe_1": dead_rival}

    note = ACTION_REGISTRY["TRADE"](sim, tribe, "plains", (51, 51))

    assert "no rival" in note


def test_scout_while_already_out_does_not_launch_a_second_expedition():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (10, 10))
    first_expedition = tribe.expedition

    note = ACTION_REGISTRY["SCOUT"](sim, tribe, "plains", (80, 80))

    assert tribe.expedition is first_expedition  # unchanged, not replaced
    assert "field" in note


def test_hunting_party_does_not_move_the_tribe_but_launches_an_expedition():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    note = ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))

    assert (tribe.x, tribe.y) == (50, 50)
    assert tribe.expedition is not None
    assert tribe.expedition["kind"] == "hunt"
    assert tribe.expedition["target"] == [10, 10]
    assert tribe.expedition["day"] == 0
    assert tribe.expedition["phase"] == "outbound"
    assert tribe.expedition["food_caught"] == 0
    assert "depart" in note


def test_hunting_party_launch_uses_its_own_max_days_baseline():
    from backend import config

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")

    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))

    span = config.EXPEDITION_DETERMINATION_DAY_VARIANCE
    assert config.HUNTING_PARTY_MAX_DAYS - span <= tribe.expedition["max_days"] <= config.HUNTING_PARTY_MAX_DAYS + span


def test_hunting_party_while_already_out_does_not_launch_a_second_expedition():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))
    first_expedition = tribe.expedition

    note = ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (80, 80))

    assert tribe.expedition is first_expedition
    assert "field" in note


def test_scout_while_a_hunting_party_is_out_does_not_launch_over_it():
    """Hunting and scouting share the same single expedition slot -- a tribe can only
    have one party in the field at a time, whichever kind it is."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    ACTION_REGISTRY["HUNTING_PARTY"](sim, tribe, "forest", (10, 10))
    hunting_expedition = tribe.expedition

    ACTION_REGISTRY["SCOUT"](sim, tribe, "forest", (80, 80))

    assert tribe.expedition is hunting_expedition
