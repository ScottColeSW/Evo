"""2026-10-11 (the owner): the relationship ladder, ENEMY < UNTRUSTWORTHY < UNTRUSTED < NEUTRAL < TRUSTED_TRADE_PARTNER (backend/standing.py), what moves it and what it opens."""
from unittest import mock

from backend import config, standing
from backend.actions import ACTION_REGISTRY, _execute_trade, at_war, war_remaining
from backend.simulation import Simulation


def _two():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}, {"name": "B", "model": "qwen2.5:3b", "x": 60, "y": 60}])
    a, b = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    a.discovered_rivals.add(b.id)
    b.discovered_rivals.add(a.id)
    a.barracks_built = b.barracks_built = 1
    sim.cycle = 100
    return sim, a, b


def test_the_bands_are_in_order_and_a_rival_just_met_starts_untrusted():
    sim, a, b = _two()
    assert standing.ORDER == [standing.ENEMY, standing.UNTRUSTWORTHY, standing.UNTRUSTED, standing.NEUTRAL, standing.TRUSTED]
    assert standing.band(a, b.id) == standing.UNTRUSTED
    for value, expected in ((-4, standing.UNTRUSTWORTHY), (-3, standing.UNTRUSTWORTHY), (-2.9, standing.UNTRUSTED), (1.9, standing.UNTRUSTED),
                            (2, standing.NEUTRAL), (5.9, standing.NEUTRAL), (6, standing.TRUSTED)):
        a.standing[b.id] = value
        assert standing.band(a, b.id) == expected, value


def test_scores_are_clamped_and_decay_a_quarter_point_a_day_toward_zero_both_ways_but_never_for_an_enemy():
    sim, a, b = _two()
    standing.adjust(a, b.id, 50)
    assert a.standing[b.id] == config.STANDING_MAX
    a.standing[b.id] = 1.0
    a.standing["tribe_9"] = -0.1
    standing.decay_day(a)
    assert a.standing[b.id] == 1.0 - config.STANDING_DECAY_PER_DAY and "tribe_9" not in a.standing      # a small grudge is gone, not overshot to positive
    a.standing_locked.add(b.id)
    a.standing[b.id] = config.STANDING_MIN
    standing.decay_day(a)
    standing.adjust(a, b.id, 5)
    assert a.standing[b.id] == config.STANDING_MIN and standing.band(a, b.id) == standing.ENEMY


def test_trade_needs_neutral_or_better_on_both_sides_and_the_envoy_is_how_trust_is_built():
    sim, a, b = _two()
    a.wood = b.wood = 100
    assert "do not trust each other enough" in _execute_trade(sim, a, b, require_neutral=True)
    a.standing[b.id] = 2.0
    assert "do not trust each other enough" in _execute_trade(sim, a, b, require_neutral=True)         # the other side's view counts too
    b.standing[a.id] = 2.0
    assert "opened trade" in _execute_trade(sim, a, b, require_neutral=True)
    sim, a, b = _two()
    a.wood = b.wood = 100
    assert "opened trade" in _execute_trade(sim, a, b)                                                 # the envoy needs no trust
    assert a.standing[b.id] == config.TRADE_STANDING_GAIN and b.standing[a.id] == config.TRADE_STANDING_GAIN
    b.standing[a.id] = config.STANDING_UNTRUSTWORTHY_AT
    assert "no trust left" in _execute_trade(sim, a, b)                                                # an untrustworthy partner refuses even the envoy


def test_a_caught_spy_costs_two_points_and_the_third_makes_an_enemy_that_declares_war_at_once():
    sim, a, b = _two()
    a.population = b.population = 1000
    with mock.patch("backend.actions.random.random", return_value=0.0):                                # caught every time
        ACTION_REGISTRY["SPY"](sim, b, "plains", (a.x, a.y))
        assert b.standing == {} and a.standing[b.id] == -config.SPY_CAUGHT_STANDING_DROP and a.spies_caught_of[b.id] == 1
        ACTION_REGISTRY["SPY"](sim, b, "plains", (a.x, a.y))
        assert standing.band(a, b.id) == standing.UNTRUSTWORTHY
        assert not at_war(a, b)
        note = ACTION_REGISTRY["SPY"](sim, b, "plains", (a.x, a.y))
    assert standing.band(a, b.id) == standing.ENEMY and "will never trust" in note
    assert a.stance_toward[b.id] == "WAR" and b.stance_toward.get(a.id) is None                          # the catcher declares; the spying tribe has not
    assert at_war(b, a) and war_remaining(sim, a, b) >= 10**5
    assert [e["kind"] for e in sim.recent_encounters if e["kind"] == "war_declared"] == ["war_declared"]   # the war card


def test_a_spy_that_gets_through_lowers_the_spying_chiefs_own_view_whichever_tribe_is_stronger():
    sim, a, b = _two()
    a.population, b.population = 1000, 10
    with mock.patch("backend.actions.random.random", return_value=0.99):
        weaker = ACTION_REGISTRY["SPY"](sim, a, "plains", (b.x, b.y))
        stronger = ACTION_REGISTRY["SPY"](sim, b, "plains", (a.x, a.y))
    assert a.standing[b.id] == -config.SPY_SUCCESS_STANDING_DROP and b.standing[a.id] == -config.SPY_SUCCESS_STANDING_DROP
    assert "not worth respecting" in weaker and "it stings" in stronger


def test_a_raid_that_lands_costs_the_victim_three_and_the_raider_one():
    sim, a, b = _two()
    a.population = b.population = 500
    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["RAID"](sim, a, "plains", (b.x, b.y))
    assert b.standing[a.id] == -config.RAID_VICTIM_STANDING_DROP and a.standing[b.id] == -config.RAID_ATTACKER_STANDING_DROP


def test_a_war_declared_by_one_tribe_strikes_at_once_and_closes_the_peaceful_channels_for_both():
    sim, a, b = _two()
    a.population = b.population = 1000
    with mock.patch("backend.actions.random.random", return_value=0.0):
        note = ACTION_REGISTRY["DECLARE_WAR"](sim, a, "plains", (b.x, b.y))
    assert "strikes at once" in note and a.stance_toward[b.id] == "WAR" and b.id in a.war_since and a.id not in b.stance_toward
    assert at_war(a, b) and at_war(b, a)
    _req, ctx = sim._prepare_turn(b)
    for closed in ("SCOUT", "TRADE", "SEND_TRADE_EMISSARY", "DECLARE_ALLIANCE", "DECLARE_WAR"):
        assert closed not in ctx["available_actions"], closed
    request, _ = sim._prepare_turn(b)
    assert "A has declared war on you." in request["prompt"]


def test_the_chief_is_told_where_each_rival_stands_and_TRADE_leaves_the_menu_until_someone_is_neutral():
    sim, a, b = _two()
    a.has_ever_settled = True
    sim._found_territory(a)
    a.era = "tribal_synapse"
    a.wood = a.stone = 1000
    sim.minor_settlements = []
    request, ctx = sim._prepare_turn(a)
    assert "Your people's standing with B: Untrusted." in request["prompt"]
    assert "TRADE" not in ctx["available_actions"] and "TRADE is not offered" in request["prompt"]
    a.standing[b.id] = b.standing[a.id] = 2.0
    request, ctx = sim._prepare_turn(a)
    assert "Your people's standing with B: Neutral." in request["prompt"] and "TRADE" in ctx["available_actions"]


def test_the_warband_standing_falls_with_each_attack_and_a_tribes_to_dict_carries_the_bands():
    sim, a, b = _two()
    a.population = 1000
    a.warband = {"name": "Grim", "size": 100, "wins": 0, "attacks": 0}
    sim._warband_after_battle(a, raiders_won=True, force=100, raiders_lost=5, p_before=1000, tribe_lost=50, established=True)
    sim._warband_after_battle(a, raiders_won=False, force=100, raiders_lost=5, p_before=1000, tribe_lost=0, established=True)
    assert a.standing[standing.RAIDERS] == -(config.RAIDERS_LANDED_DROP + config.RAIDERS_REPELLED_DROP)
    sent = a.to_dict()["standing"]
    assert sent["raiders"]["band"] == standing.UNTRUSTWORTHY and sent["raiders"]["score"] == -4.0


def test_a_war_declared_out_of_the_blue_costs_the_target_four_points_but_a_spy_catchers_war_does_not():
    """2026-10-11 (the owner: "if I got attacked out of the blue, I would feel both hurt and angry")."""
    sim, a, b = _two()
    a.population = b.population = 1000
    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["DECLARE_WAR"](sim, a, "plains", (b.x, b.y))
    assert b.standing[a.id] == -config.WAR_DECLARED_STANDING_DROP and standing.band(b, a.id) == standing.UNTRUSTWORTHY
    sim, a, b = _two()
    a.population = b.population = 1000
    with mock.patch("backend.actions.random.random", return_value=0.0):
        for _ in range(3):
            ACTION_REGISTRY["SPY"](sim, b, "plains", (a.x, a.y))
    assert b.standing.get(a.id, 0.0) == 0.0                                  # the spying tribe was caught at it: no grievance of its own
