"""2026-10-10 (the owner: "if the Raiders keep winning, the Tribe becomes them! That guy is the new chief"): a persistent warband per tribe, raids with a real force that scales with the target and
compounds with wins, casualties shown on both sides and one way (no swap), the 10% turn-over rule for an established tribe, young tribes protected, and the leader as the new chief (config.WARBAND_*)."""
from unittest import mock

from backend import config
from backend.simulation import Simulation
from tests.conftest import run_async


def _tribe(population=500):
    sim = Simulation([{"name": "Home", "model": "gemma2:2b", "x": 50, "y": 50}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    tribe.chief_name = "Old Chief"
    tribe.chief_philosophy = "peace"
    tribe.population = population
    sim.cycle = 100  # past the early-game immunity from losses (Simulation.immortality_cycles is 0 and cycle 0 counts as inside it)
    return sim, tribe


def _raiders_win():
    return mock.patch("backend.simulation.random.random", return_value=0.999)  # the defense roll fails and no chief dies


def _tribe_holds():
    return mock.patch("backend.simulation.random.random", return_value=0.0)  # the defense roll succeeds


def _card(sim):
    return [e for e in sim.recent_encounters if e.get("kind") == "raider_attack"][-1]["skirmish"]


def test_the_first_attack_founds_the_warband_and_every_later_attack_is_the_same_raiders_with_a_scaled_force():
    sim, tribe = _tribe(1000)
    with mock.patch("backend.simulation.random.random", return_value=0.0):  # attacks, and the force roll at its low end
        sim.cycle = 100
        sim._check_raider_attack(tribe)
    assert tribe.warband and tribe.warband["wins"] == 0
    assert tribe.raiders_approaching["force"] == tribe.warband["size"] == round(1000 * config.WARBAND_FORCE_RATIO * config.WARBAND_FORCE_JITTER_LOW)
    name = tribe.warband["name"]
    assert tribe.raiders_approaching["name"] == name
    tribe.raiders_approaching = None
    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim.cycle = 300
        sim._check_raider_attack(tribe)
    assert tribe.raiders_approaching["name"] == name and tribe.warband["name"] == name


def test_the_raiders_force_scales_with_the_target_and_compounds_with_wins():
    sim, tribe = _tribe(1000)
    middle = (config.WARBAND_FORCE_JITTER_LOW + config.WARBAND_FORCE_JITTER_HIGH) / 2
    with mock.patch("backend.simulation.random.random", return_value=0.5):
        plain = sim._raider_force(tribe)
        tribe.population = 4000
        bigger_target = sim._raider_force(tribe)
        tribe.population = 1000
        tribe.warband = {"name": "Grim", "size": 0, "wins": 3, "attacks": 3}
        bolder = sim._raider_force(tribe)
    assert plain == round(1000 * config.WARBAND_FORCE_RATIO * middle)
    assert bigger_target == 4 * plain
    assert bolder == round(1000 * config.WARBAND_FORCE_RATIO * config.WARBAND_FORCE_GROWTH_PER_WIN ** 3 * middle)


def test_an_established_tribe_that_loses_a_raid_loses_a_share_of_its_people_and_the_card_shows_both_sides():
    sim, tribe = _tribe(1000)
    tribe.warband = {"name": "Grim Raiders", "size": 300, "wins": 0, "attacks": 0}
    with _raiders_win():
        sim._resolve_raider_attack(tribe, force=300)
    lost = round(1000 * config.WARBAND_CASUALTY_FACTOR * 300 / 1000)         # no wall: a quarter of the force/population ratio
    card = _card(sim)
    assert card["attacker_won"] is True and card["attacker_force"] == 300 and card["defender_force"] == 1000
    assert card["defender_lost"] == lost and card["attacker_lost"] == round(300 * config.WARBAND_WINNER_LOSS_FRACTION)
    assert tribe.population == 1000 - lost and tribe.warband["wins"] == 1 and tribe.warband["size"] == 300 - card["attacker_lost"]
    assert tribe.chief_name == "Old Chief"  # a loss that leaves it standing changes nothing about who rules


def test_a_wall_cuts_the_casualties_of_a_lost_raid():
    sim, bare = _tribe(1000)
    sim2, walled = _tribe(1000)
    for sim_, tribe_ in ((sim, bare), (sim2, walled)):
        tribe_.warband = {"name": "Grim Raiders", "size": 600, "wins": 0, "attacks": 0}
    with mock.patch("backend.simulation.city_layout.wall_defense_fraction", return_value=1.0), _raiders_win():
        sim2._resolve_raider_attack(walled, force=600)
    with _raiders_win():
        sim._resolve_raider_attack(bare, force=600)
    assert _card(sim2)["defender_lost"] < _card(sim)["defender_lost"]


def test_a_young_tribe_is_protected_a_raid_costs_what_it_always_did_and_no_takeover_is_possible():
    sim, tribe = _tribe(config.WARBAND_PROTECTED_POPULATION - 1)
    tribe.warband = {"name": "Grim Raiders", "size": 5000, "wins": 5, "attacks": 5}
    with _raiders_win():
        sim._resolve_raider_attack(tribe, force=5000)                       # an enormous force, but the tribe is young
    assert tribe.population >= config.WARBAND_PROTECTED_POPULATION - 1 - config.RAIDER_ATTACK_POPULATION_LOSS_UNDEFENDED
    assert tribe.warband is not None and tribe.chief_name == "Old Chief" and tribe.forced_chief_name == ""


def test_an_established_tribe_that_ends_the_battle_under_ten_percent_is_taken_over_and_the_leader_is_chief():
    sim, tribe = _tribe(200)
    tribe.warband = {"name": "Ruthless Gnashdrex", "size": 2000, "wins": 4, "attacks": 4}
    tribe.settlement_name = "Old Town"
    tribe.pending_settlement_naming = False
    with _raiders_win():
        sim._resolve_raider_attack(tribe, force=2000)
    survivors_of_the_raiders = 2000 - round(2000 * config.WARBAND_WINNER_LOSS_FRACTION)
    assert tribe.warband is None
    assert tribe.chief_name == "" and tribe.forced_chief_name == "Ruthless Gnashdrex"
    assert tribe.population == 1 + survivors_of_the_raiders                    # the one who was left standing and the raiders who survived
    assert tribe.name == "Home" and tribe.settlement_name == "Old Town" and tribe.pending_settlement_naming is False  # nothing is renamed
    assert "Ruthless Gnashdrex" in tribe.pending_chief_context and "Old Chief" in tribe.pending_chief_context
    assert any("becomes them" in line for line in tribe.history)
    assert _card(sim)["defender_lost"] == 200 and _card(sim)["attacker_force"] == 2000


@run_async
async def test_the_election_names_the_leader_chief_whatever_name_it_offers_but_keeps_the_philosophy():
    sim, tribe = _tribe(200)
    tribe.warband = {"name": "Ruthless Gnashdrex", "size": 2000, "wins": 4, "attacks": 4}
    with _raiders_win():
        sim._resolve_raider_attack(tribe, force=2000)

    async def fake_election(client, model, name, water_needed, context):
        assert "Ruthless Gnashdrex" in context
        return {"chief_name": "Somebody Else", "guiding_philosophy": "We take what we need.", "victory_method": "strength"}

    with mock.patch("backend.simulation.elect_chief", fake_election):
        await sim._install_chief(tribe)
    assert tribe.chief_name == "Ruthless Gnashdrex" and tribe.chief_philosophy == "We take what we need." and tribe.forced_chief_name == ""


def test_when_the_tribe_holds_the_raiders_lose_people_one_way_and_nothing_is_swapped():
    sim, tribe = _tribe(1000)
    tribe.warband = {"name": "Grim Raiders", "size": 600, "wins": 2, "attacks": 2}
    with _tribe_holds():
        sim._resolve_raider_attack(tribe, force=600)
    card = _card(sim)
    expected_lost = min(600, round(600 * min(1.0, config.WARBAND_CASUALTY_FACTOR * 1000 / 600)))
    assert card["attacker_won"] is False and card["attacker_lost"] == expected_lost and card["defender_lost"] == 0
    assert tribe.population >= 1000                                         # no one joined it: the raiders' losses are just losses (a held raid also loots supplies, never people)
    assert tribe.warband["wins"] == 1 and tribe.warband["size"] == 600 - expected_lost


def test_raiders_that_lose_nearly_everything_are_broken_and_the_few_who_survive_join_the_tribe():
    sim, tribe = _tribe(380)
    tribe.warband = {"name": "Grim Raiders", "size": 100, "wins": 0, "attacks": 0}
    with _tribe_holds():
        sim._resolve_raider_attack(tribe, force=100)            # 0.25 x 380 / 100 = 95% of the raiders fall, 5 are left (under 10%)
    assert tribe.warband is None and tribe.population == 380 + 5
    assert any("are broken" in line and "5 who survive join the tribe" in line for line in tribe.history)


def test_expelled_raiders_set_the_warband_back_one_win_and_nothing_else_changes_hands():
    sim, tribe = _tribe(1000)
    tribe.warband = {"name": "Grim Raiders", "size": 50, "wins": 2, "attacks": 2}
    sim._warband_defeated(tribe)
    assert tribe.warband["wins"] == 1 and tribe.warband["size"] == 50 and tribe.population == 1000


def test_wins_make_the_warband_bolder_a_shorter_wait():
    sim, tribe = _tribe(2000)
    tribe.last_raider_attack_cycle = 0
    sim.cycle = 30          # inside the plain cooldown (35 cycles)
    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._check_raider_attack(tribe)
    assert tribe.raiders_approaching is None
    tribe.warband = {"name": "Grim Raiders", "size": 50, "wins": 2, "attacks": 2}
    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._check_raider_attack(tribe)                                                       # two wins shorten the wait
    assert tribe.raiders_approaching is not None


def test_the_chief_is_told_who_the_raiders_are_and_how_they_stand():
    sim, tribe = _tribe(1000)
    tribe.warband = {"name": "Grim Raiders", "size": 120, "wins": 2, "attacks": 3}
    request, _ctx = sim._prepare_turn(tribe)
    assert "The Grim Raiders Raiders have attacked you 3 time(s) and are 2 win(s) ahead; they number about 120 against your 1,000 people" in request["prompt"]
    tribe.warband = {"name": "Grim Raiders", "size": 50, "wins": 0, "attacks": 0}
    request, _ctx = sim._prepare_turn(tribe)
    assert "have attacked you" not in request["prompt"]


def test_a_takeover_is_recorded_on_the_tribe_and_sent_to_the_page():
    sim, tribe = _tribe(200)
    tribe.warband = {"name": "Ruthless Gnashdrex", "size": 2000, "wins": 4, "attacks": 4}
    sim.cycle = 210
    with _raiders_win():
        sim._resolve_raider_attack(tribe, force=2000)
    assert tribe.takeover == {"leader": "Ruthless Gnashdrex", "cycle": 210, "force": 2000, "tribe_before": 200, "survivors": 1, "joined": 2000 - round(2000 * config.WARBAND_WINNER_LOSS_FRACTION)}
    sent = tribe.to_dict()
    assert sent["takeover"]["leader"] == "Ruthless Gnashdrex" and sent["warband"] is None


def _expel(sim, tribe, roll):
    from backend.actions import ACTION_REGISTRY
    with mock.patch("backend.actions.random.random", return_value=roll):
        return ACTION_REGISTRY["EXPEL_RAIDERS_FROM_TERRITORY"](sim, tribe, "plains", (0, 0))


def _riding_in(tribe, force):
    tribe.raiders_approaching = {"start_x": 60, "start_y": 60, "x": 60, "y": 60, "name": "Grim", "force": force, "cycles_left": 5, "total_cycles": 10}
    tribe.warband = {"name": "Grim", "size": force, "wins": 0, "attacks": 0}


def test_a_big_warband_is_harder_to_expel_than_a_small_one_at_the_same_tribe_size():
    """2026-10-11 (the owner: incoming raiders "did not scale high enough"): every expulsion in a real run won at 0.90-0.95 whatever the raiders' force."""
    sim, tribe = _tribe(1000)
    _riding_in(tribe, 30)
    _expel(sim, tribe, 0.0)
    small = [e for e in sim.recent_encounters if e.get("label") == "Raiders expelled"][-1]["skirmish"]["attacker_chance"]
    sim2, tribe2 = _tribe(1000)
    _riding_in(tribe2, 1000)
    _expel(sim2, tribe2, 0.0)
    big = [e for e in sim2.recent_encounters if e.get("label") == "Raiders expelled"][-1]["skirmish"]["attacker_chance"]
    assert small - big >= config.EXPEL_RAIDERS_FORCE_PENALTY * (1.0 - 0.03) - 0.01


def test_a_failed_expulsion_wave_costs_people_in_proportion_to_the_raiders_force():
    sim, tribe = _tribe(1000)
    _riding_in(tribe, 800)
    _expel(sim, tribe, 0.999)                    # every wave fails
    assert tribe.population <= 1000 - config.EXPEL_RAIDERS_MAX_WAVES * round(800 * config.EXPEL_RAIDERS_FAILED_WAVE_CASUALTY_FACTOR)


def test_raiders_who_join_after_an_expulsion_are_never_more_than_the_warband_and_leave_it():
    sim, tribe = _tribe(1000)
    _riding_in(tribe, 5)                          # only five raiders came; 3% of 1000 would be 30
    _expel(sim, tribe, 0.0)
    assert tribe.population == 1005 and tribe.warband["size"] == 0
    sim2, tribe2 = _tribe(1000)
    _riding_in(tribe2, 400)
    _expel(sim2, tribe2, 0.0)
    assert tribe2.population == 1030 and tribe2.warband["size"] == 370
