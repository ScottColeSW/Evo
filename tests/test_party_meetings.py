"""2026-10-06 (the owner: "what about when their paths cross in the wild?", "scouts just get chased off", "all parties involved ... will file a report with the
respective Chief"): parties of non-allied tribes within PARTY_MEETING_RADIUS meet once; armed parties fight, a scout is only chased off, allies never fight, and every
party files a report with its own Chief when it gets home."""
from unittest import mock

from backend import config
from backend.simulation import Tribe
from tests.test_actions import _bare_simulation


def _party(kind, pos, lead, determination=0.5, food=10):
    return {"kind": kind, "pos": list(pos), "phase": "outbound", "lead_scout": lead, "determination": determination, "food_gathered": food, "path": [list(pos)]}


def _world(kind_a="hunt", kind_b="hunt", gap=2, stance=None):
    sim = _bare_simulation()
    a = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 20, 20, "#c084fc")
    b = Tribe("tribe_1", "Mountain Tribe", "gemma2:2b", 70, 70, "#fb923c")
    sim.tribes = {"tribe_0": a, "tribe_1": b}
    ea, eb = _party(kind_a, (40, 40), "Karval"), _party(kind_b, (40 + gap, 40), "Tikmen")
    a.expeditions, b.expeditions = [ea], [eb]
    if stance:
        a.stance_toward["tribe_1"] = stance
    return sim, a, b, ea, eb


def test_two_armed_parties_fight_and_the_loser_turns_back_without_its_food():
    sim, a, b, ea, eb = _world()
    with mock.patch("backend.simulation.random.random", return_value=0.0):  # tribe_0's party wins
        sim._resolve_party_meetings()
    assert ea["phase"] == "outbound" and eb["phase"] == "returning"
    assert eb["food_gathered"] == 0 and ea["food_gathered"] == 20
    assert b.population == 8 - config.FIELD_PARTY_LOSS_POPULATION and a.population == 8
    card = [e for e in sim.recent_encounters if e["kind"] == "party_meeting"][0]["skirmish"]
    assert card["attacker_won"] is True and card["attacker_force"] == config.FIELD_PARTY_SIZE and card["defender_lost"] == config.FIELD_PARTY_LOSS_POPULATION


def test_a_scout_meeting_an_armed_party_is_chased_off_and_loses_nothing():
    sim, a, b, ea, eb = _world(kind_a="scout", kind_b="hunt")
    sim._resolve_party_meetings()
    assert ea["phase"] == "returning" and eb["phase"] == "outbound"
    assert a.population == 8 and ea["food_gathered"] == 10
    assert "chased" in ea["meetings"][0]["outcome"] and "chased" in eb["meetings"][0]["outcome"]


def test_two_scouts_pass_each_other():
    sim, a, b, ea, eb = _world(kind_a="scout", kind_b="scout")
    sim._resolve_party_meetings()
    assert ea["phase"] == eb["phase"] == "outbound" and not sim.recent_encounters
    assert ea["meetings"] and eb["meetings"]  # they still saw each other, and will report it


def test_allied_parties_do_not_fight():
    sim, a, b, ea, eb = _world(stance="ALLIED")
    sim._resolve_party_meetings()
    assert not sim.recent_encounters and "meetings" not in ea


def test_parties_too_far_apart_do_not_meet():
    sim, a, b, ea, eb = _world(gap=config.PARTY_MEETING_RADIUS + 1)
    sim._resolve_party_meetings()
    assert not sim.recent_encounters and "meetings" not in ea


def test_a_pair_of_parties_meets_only_once():
    sim, a, b, ea, eb = _world()
    sim._resolve_party_meetings()
    first = len(sim.recent_encounters)
    sim._resolve_party_meetings()
    assert first == 1 and len(sim.recent_encounters) == 1 and len(ea["meetings"]) == 1


def test_each_party_files_its_own_report_with_its_own_chief_when_home():
    sim, a, b, ea, eb = _world()
    with mock.patch("backend.simulation.random.random", return_value=0.0):
        sim._resolve_party_meetings()
    sim.event_log = mock.MagicMock()
    sim._party_report_meetings(a, ea)
    sim._party_report_meetings(b, eb)
    ra, rb = a.party_meeting_reports[0], b.party_meeting_reports[0]
    assert ra["with"] == "Mountain Tribe" and ra["their_lead"] == "Tikmen" and "won" in ra["outcome"]
    assert rb["with"] == "Forest Tribe" and rb["their_lead"] == "Karval" and "lost" in rb["outcome"]
    assert "tribe_1" in a.discovered_rivals and "tribe_0" in b.discovered_rivals  # knowledge is power: each now knows who it met
    assert any("reports meeting Mountain Tribe" in str(h) for h in a.history)
    assert any("reports meeting Forest Tribe" in str(h) for h in b.history)
