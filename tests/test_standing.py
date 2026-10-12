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
    assert [e["battle"]["outcome"] for e in sim.recent_encounters if e["kind"] == "tribe_conquest_battle"] in (["strike_won"], ["strike_repelled"])   # the full-screen war splash


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


def test_a_war_declared_out_of_the_blue_makes_the_target_an_enemy_of_the_declarer_but_a_spy_catchers_war_does_not_lock_the_spier():
    """2026-10-11 (the owner: "if they declare war, obviously the relationship falls all the way to enemy")."""
    sim, a, b = _two()
    a.population = b.population = 1000
    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["DECLARE_WAR"](sim, a, "plains", (b.x, b.y))
    assert standing.band(b, a.id) == standing.ENEMY and war_remaining(sim, a, b) >= 10**5
    assert standing.band(a, b.id) == standing.UNTRUSTED                       # the declarer's own view is not changed by declaring
    battle = [e for e in sim.recent_encounters if e["kind"] == "tribe_conquest_battle"][0]["battle"]
    assert battle["attacker_name"] == "A" and battle["defender_name"] == "B" and len(battle["rounds"]) == 1
    sim, a, b = _two()
    a.population = b.population = 1000
    with mock.patch("backend.actions.random.random", return_value=0.0):
        for _ in range(3):
            ACTION_REGISTRY["SPY"](sim, b, "plains", (a.x, a.y))
    assert not standing.is_enemy(b, a.id)                                    # the spying tribe was caught at it: no grievance of its own


def test_a_spy_reports_how_the_target_sees_us_and_a_mismatch_moves_our_view_toward_theirs():
    """2026-10-11 (the owner: "a mismatch can cause conflict or incite more partnership")."""
    sim, a, b = _two()
    a.population = b.population = 1000
    b.standing[a.id] = 7.0                                                  # B trusts A as a Trusted Trade Partner; A has not warmed to B (Untrusted)
    with mock.patch("backend.actions.random.random", return_value=0.99):
        note = ACTION_REGISTRY["SPY"](sim, a, "plains", (b.x, b.y))
    assert "they see A as trusted trade partner" in note and "the chief warms to them" in note
    assert a.rival_intel[b.id]["views"][a.id]["band"] == standing.TRUSTED
    gap = standing.rank(standing.TRUSTED) - standing.rank(standing.UNTRUSTED)
    assert a.standing[b.id] == -config.SPY_SUCCESS_STANDING_DROP + gap * config.SPY_MISMATCH_STANDING_PER_RANK     # partnership invited
    sim, a, b = _two()
    a.standing[b.id] = 7.0                                                  # A trusts B; B thinks little of A
    b.standing[a.id] = -4.0
    with mock.patch("backend.actions.random.random", return_value=0.99):
        note = ACTION_REGISTRY["SPY"](sim, a, "plains", (b.x, b.y))
    assert "the chief hardens" in note and a.standing[b.id] == 7.0 - config.SPY_SUCCESS_STANDING_DROP - 3 * config.SPY_MISMATCH_STANDING_PER_RANK   # conflict: three bands apart (Trusted vs Untrustworthy)


def test_spy_is_not_offered_for_a_while_after_a_mission_and_longer_after_a_caught_one():
    """2026-10-11 (the owner: "cooldown spy"): run_20261011_161130 sent a spy every 4 cycles at war."""
    sim, a, b = _two()
    a.has_ever_settled = True
    sim._found_territory(a)
    a.era = "tribal_synapse"
    _req, ctx = sim._prepare_turn(a)
    assert "SPY" in ctx["available_actions"]
    with mock.patch("backend.actions.random.random", return_value=0.99):
        ACTION_REGISTRY["SPY"](sim, a, "plains", (b.x, b.y))
    assert a.spy_cooldown_until_cycle == sim.cycle + config.SPY_COOLDOWN_CYCLES
    request, ctx = sim._prepare_turn(a)
    assert "SPY" not in ctx["available_actions"] and "last spy mission is still fresh" in request["prompt"]
    sim.cycle = a.spy_cooldown_until_cycle
    _req, ctx = sim._prepare_turn(a)
    assert "SPY" in ctx["available_actions"]
    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["SPY"](sim, a, "plains", (b.x, b.y))
    assert a.spy_cooldown_until_cycle == sim.cycle + config.SPY_CAUGHT_COOLDOWN_CYCLES


def test_an_enemy_lock_lets_go_after_five_hard_exchanges_once_the_war_cooldown_is_over_and_both_still_stand():
    """2026-10-11 (the owner: "reset after 5 actions of war and war in cooldown, if they still stand")."""
    sim, a, b = _two()
    a.population = b.population = 1000
    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["DECLARE_WAR"](sim, a, "plains", (b.x, b.y))                     # one exchange, and b now holds a as an Enemy
    assert standing.is_enemy(b, a.id) and a.war_actions[b.id] == 1
    sim.cycle += config.WAR_MIN_CYCLES
    sim._release_cooled_wars()
    assert standing.is_enemy(b, a.id)                                                    # only one exchange so far
    for _ in range(config.WAR_ACTIONS_FOR_RESET - 1):
        with mock.patch("backend.actions.random.random", return_value=0.99):
            ACTION_REGISTRY["SPY"](sim, a, "plains", (b.x, b.y))
    assert a.war_actions[b.id] == config.WAR_ACTIONS_FOR_RESET
    sim.cycle = a.war_since[b.id] + config.WAR_MIN_CYCLES - 1                            # the war's own cooldown is not over yet
    sim._release_cooled_wars()
    assert standing.is_enemy(b, a.id)
    sim.cycle += 1
    sim._release_cooled_wars()
    assert not standing.is_enemy(b, a.id) and b.id not in a.spies_caught_of and a.war_actions == {} and standing.band(b, a.id) == standing.UNTRUSTED
    assert at_war(a, b) and war_remaining(sim, a, b) == 0                                # still at war, but an envoy may now be heard


def test_every_model_call_asks_for_the_same_context_so_a_model_has_one_runner():
    import inspect

    from backend.ollama_client import OllamaClient
    for method in (OllamaClient.generate_json, OllamaClient.generate_json_with_raw):
        assert inspect.signature(method).parameters["num_ctx"].default == 8192
    assert config.REFLECTION_CONTEXT_OWN_MODEL == 8192


def test_failed_turns_apply_no_action_and_the_third_one_sinks_the_island_with_no_winner():
    """2026-10-11 (the owner: "the Island sinks into the Ocean as opposed to a real forfeit", and "allow for 3 failures over time")."""
    import asyncio

    from backend import config

    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}, {"name": "B", "model": "qwen2.5:3b", "x": 60, "y": 60}])
    a, b = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    a.chief_name, b.chief_name = "Ashgar", "Brisa"

    async def fake_run_batch(requests):
        return {
            "tribe_0": {"intent": {}, "latency_ms": 120400.0, "raw_response": "", "error": "ReadTimeout"},
            "tribe_1": {"intent": {"visual_action": "GATHER_WOOD", "metacognitive_rationale": "wood"}, "latency_ms": 1500.0, "raw_response": "{}"},
        }

    def step():
        with (
            mock.patch.object(sim.scheduler, "run_batch", fake_run_batch),
            mock.patch.object(sim.client, "unload_model", mock.AsyncMock()),
            mock.patch.object(sim.client, "list_models", mock.AsyncMock(return_value=[])),
            mock.patch("backend.simulation.generate_endgame_narrative", mock.AsyncMock(side_effect=AssertionError("no model call after the island sinks"))),
        ):
            asyncio.run(sim.step())

    for failures in range(1, config.MODEL_FAILURES_BEFORE_ISLAND_SINKS):
        step()
        assert not sim.game_over and len(sim.model_failures) == failures
        assert any("falls silent this cycle: no order is given" in line for line in a.history)
        assert not any(line.startswith("[") and "ms]" in line and "GATHER" not in line for line in a.history)   # no action was fuzzy-matched for A
    step()
    assert sim.game_over and sim.game_over_reason == "island_sinks"
    assert "The island sinks into the ocean" in sim.game_over_summary and "No tribe wins" in sim.game_over_summary
    assert sim.game_over_narrative == "" and sim.game_over_report["headline"]["title"] == "The island sinks into the ocean"
    assert len(sim.game_over_report["island_sunk"]["failures"]) == config.MODEL_FAILURES_BEFORE_ISLAND_SINKS
