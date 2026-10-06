"""2026-10-06 (the owner: "I'd like to see the battle scene when raiders strike or any battle really"): every fight's recent_encounters entry carries a `skirmish` record
for the board's card, built only from what the simulation computed."""
from unittest import mock

from backend.actions import ACTION_REGISTRY
from backend.simulation import Tribe
from backend.skirmish import skirmish
from tests.test_actions import _NO_TARGET, _bare_simulation


def _card(sim):
    entries = [e for e in sim.recent_encounters if "skirmish" in e]
    assert entries, sim.recent_encounters
    return entries[-1]["skirmish"]


def test_the_record_clamps_the_odds_and_keeps_the_facts():
    s = skirmish("t", "A", "B", True, attacker_chance=1.7, attacker_force=5, defender_lost=2, outcome="x")
    assert s["attacker_chance"] == 1.0 and s["attacker_won"] is True and s["defender_lost"] == 2 and s["defender_force"] is None
    assert skirmish("t", "A", "B", False)["attacker_chance"] is None  # no roll, no odds


def test_a_raider_strike_that_breaks_the_defense_is_a_raider_win_with_the_loss():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    tribe.food = tribe.water = tribe.wood = tribe.stone = 100
    with mock.patch("backend.simulation.random.random", return_value=0.99):
        sim._resolve_raider_attack(tribe)
    card = _card(sim)
    assert card["attacker"] == "Raiders" and card["defender"] == "Forest Tribe"
    assert card["attacker_won"] is True and card["defender_lost"] > 0 and 0 < card["attacker_chance"] < 1


def test_a_repelled_raider_strike_is_a_defender_win():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.population = 10
    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._resolve_raider_attack(tribe)
    card = _card(sim)
    assert card["attacker_won"] is False and card["defender_lost"] == 0


def test_a_tribe_raid_card_names_both_tribes_with_their_forces_and_the_real_odds():
    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}
    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))
    card = _card(sim)
    assert (card["attacker"], card["defender"]) == ("Forest Tribe", "Mountain Tribe")
    assert card["attacker_force"] == 8 and card["defender_force"] == 8  # populations before the fight, not after
    assert card["attacker_won"] is True and card["defender_lost"] == 2 and 0 < card["attacker_chance"] < 1


def test_a_failed_raid_card_shows_the_attacker_beaten():
    sim = _bare_simulation()
    attacker = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    defender = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 51, 51, "#fb923c")
    sim.tribes = {"tribe_0": attacker, "tribe_1": defender}
    with mock.patch("backend.actions.random.random", return_value=0.999):
        ACTION_REGISTRY["RAID"](sim, attacker, "plains", (51, 51))
    card = _card(sim)
    assert card["attacker_won"] is False and card["attacker_lost"] > 0


def test_a_strike_on_a_raider_camp_has_a_card_either_way():
    for roll, won in ((0.0, True), (0.999, False)):
        sim = _bare_simulation()
        tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
        tribe.era = "bronze_age"
        tribe.raider_sightings = [(55, 55)]
        with mock.patch("backend.actions.random.random", return_value=roll):
            ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, tribe, "plains", (55, 55))
        card = _card(sim)
        assert card["attacker"] == "Forest Tribe" and card["defender"] == "Raiders" and card["attacker_won"] is won


def test_a_wolf_pack_has_a_card_without_odds():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    with mock.patch("backend.actions.random.random", return_value=0.0):
        ACTION_REGISTRY["HUNT_DEER"](sim, tribe, "forest", _NO_TARGET)
    card = _card(sim)
    assert card["attacker"] == "Wolf pack" and card["attacker_chance"] is None and card["defender_lost"] >= 1
