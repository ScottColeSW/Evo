"""2026-10-06 ("make sure proximity battle is working", "the visual battle of might is best"): a tribe going to a node a non-allied rival drew from within
NODE_CONTEST_WINDOW_CYCLES has to win it first; allies share freely; the fight rides the skirmish card."""
from unittest import mock

from backend import config
from backend.actions import ACTION_REGISTRY
from backend.simulation import Tribe
from tests.test_actions import _NO_TARGET
from tests.test_resource_nodes import _sim_with


def _two_tribes_at_one_grove(stance=None):
    sim, tribe, sites = _sim_with("lumber", [(3, 0)])
    rival = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 60, 60, "#fb923c")
    sim.tribes["tribe_1"] = rival
    sim.world.site_last_draw[("lumber", *sites[0])] = ("tribe_1", sim.cycle)  # the rival drew this very cycle
    if stance:
        tribe.stance_toward["tribe_1"] = stance
    return sim, tribe, rival, sites


def _gather(sim, tribe, roll):
    with mock.patch.object(sim.world, "biome", return_value="forest"), mock.patch("backend.actions.random.random", return_value=roll):
        return ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "plains", _NO_TARGET)


def test_winning_the_contest_lets_the_gather_go_on_and_shows_the_fight():
    sim, tribe, rival, sites = _two_tribes_at_one_grove()
    result = _gather(sim, tribe, 0.0)
    assert tribe.wood > 50 and "timber grove" in result
    fight = [e for e in sim.recent_encounters if e["kind"] == "node_contest"]
    assert len(fight) == 1 and fight[0]["skirmish"]["attacker_won"] is True
    assert fight[0]["skirmish"]["attacker"] == "Forest Tribe" and fight[0]["skirmish"]["defender"] == "Mountain Tribe"
    assert sim.world.site_last_draw[("lumber", *sites[0])][0] == "tribe_0"  # the winner is now the one working it


def test_losing_the_contest_draws_nothing_spends_no_use_and_costs_people():
    sim, tribe, rival, sites = _two_tribes_at_one_grove()
    before = tribe.population
    result = _gather(sim, tribe, 0.999)
    assert tribe.wood == 50 and "drove the party off" in result
    assert tribe.population == before - config.NODE_CONTEST_LOSS_POPULATION
    assert sim.world.site_uses.get(("lumber", *sites[0]), 0) == 0
    assert any(e["kind"] == "node_contest" and e["skirmish"]["attacker_won"] is False and e["skirmish"]["attacker_lost"] == 1 for e in sim.recent_encounters)
    assert any("drove" in str(h) for h in rival.history)


def test_allies_share_a_node_without_a_fight():
    sim, tribe, rival, sites = _two_tribes_at_one_grove(stance="ALLIED")
    _gather(sim, tribe, 0.999)
    assert tribe.wood > 50 and not [e for e in sim.recent_encounters if e["kind"] == "node_contest"]


def test_no_contest_once_the_window_has_passed_or_against_yourself():
    sim, tribe, rival, sites = _two_tribes_at_one_grove()
    sim.cycle += config.NODE_CONTEST_WINDOW_CYCLES + 1
    _gather(sim, tribe, 0.999)
    assert tribe.wood > 50 and not sim.recent_encounters
    sim2, tribe2, rival2, sites2 = _two_tribes_at_one_grove()
    sim2.world.site_last_draw[("lumber", *sites2[0])] = ("tribe_0", sim2.cycle)
    _gather(sim2, tribe2, 0.999)
    assert tribe2.wood > 50 and not sim2.recent_encounters
