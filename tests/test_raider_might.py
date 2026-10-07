"""2026-10-07 (the owner: "raiders strength/odds are not scaled to the average of both tribes Might value"): raiders have no Battalion of their own, so a tribe's odds
against them shift by how its Might compares with the average Might of the living tribes."""
from unittest import mock

from backend import config
from backend.actions import ACTION_REGISTRY, _raider_might_modifier
from backend.simulation import Tribe
from tests.test_actions import _bare_simulation


def _world():
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 70, 70, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}
    return sim, a, b


def _with_might(values):
    return mock.patch("backend.actions.compute_might", side_effect=lambda t: values[t.id])


def test_no_modifier_while_nobody_has_a_battalion():
    sim, a, b = _world()
    with _with_might({"tribe_0": 0, "tribe_1": 0}):
        assert _raider_might_modifier(sim, a) == 0.0


def test_a_tribe_at_the_average_is_unchanged_above_it_gains_and_below_it_loses():
    sim, a, b = _world()
    with _with_might({"tribe_0": 100, "tribe_1": 100}):
        assert _raider_might_modifier(sim, a) == 0.0
    with _with_might({"tribe_0": 300, "tribe_1": 100}):  # average 200
        assert _raider_might_modifier(sim, a) > 0 > _raider_might_modifier(sim, b)
    with _with_might({"tribe_0": 0, "tribe_1": 2000}):   # no army while the other is strong: the average is 1000
        assert _raider_might_modifier(sim, a) == config.MIGHT_MODIFIER_MIN


def test_a_strike_on_a_raider_camp_is_easier_for_the_stronger_tribe():
    sim, a, b = _world()
    a.era = b.era = "bronze_age"
    odds = {}
    for label, values in (("strong", {"tribe_0": 900, "tribe_1": 100}), ("weak", {"tribe_0": 100, "tribe_1": 900})):
        a.raider_sightings = [(55, 55)]
        sim.recent_encounters = []
        with _with_might(values), mock.patch("backend.actions.random.random", return_value=0.0):
            ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, a, "plains", (55, 55))
        odds[label] = sim.recent_encounters[-1]["skirmish"]["attacker_chance"]
    assert odds["strong"] > odds["weak"]


def test_passive_defense_odds_follow_the_tribes_might_against_the_average():
    sim, a, b = _world()
    a.population = 10
    chances = {}
    for label, values in (("strong", {"tribe_0": 900, "tribe_1": 100}), ("weak", {"tribe_0": 100, "tribe_1": 900})):
        sim.recent_encounters = []
        with _with_might(values), mock.patch("backend.simulation.random.random", return_value=0.0):
            sim._resolve_raider_attack(a)
        chances[label] = sim.recent_encounters[-1]["skirmish"]["attacker_chance"]  # the raiders' odds
    assert chances["strong"] < chances["weak"]
