"""2026-10-08: a scout's report says what came home, and the Chief's journal shows what a scouting trip gained. The report said "nothing new found" for any trip without water
or a landmark, even one that brought sites, and the journal (written when the party leaves) saw no change in anything it tracks."""
from unittest import mock

from backend.simulation import Simulation, _haul_text

PATH = [[57, 40], [45, 38], [30, 36], [17, 34]]


def _sim():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 57, "y": 40}, {"name": "B", "model": "gemma2:2b", "x": 5, "y": 95}])
    return sim, sim.tribes["tribe_0"]


def _exp(path=PATH, launched=5, found=None):
    return {"kind": "scout", "launched": launched, "pos": list(path[0]), "origin": list(path[0]), "target": list(path[-1]), "day": 2, "phase": "returning",
            "found": found, "terrain_report": None, "food_gathered": 0, "water_gathered": 0, "lead_scout": "Ashgar", "determination": 0.5, "max_days": 3, "path": path}


def _home(sim, tribe, exp):
    tribe.expeditions = [exp]
    with mock.patch.object(sim, "_celebrate_game_discovery"):
        sim._advance_one_expedition(tribe, exp)
    return [str(line) for line in tribe.history if "is home and gives" in str(line)]


def _launch_entry(tribe, cycle):
    entry = {"cycle": cycle, "action": "SCOUT", "delta": {}, "built": {}, "moved": False, "stock_before": {}, "population": tribe.population, "note": None}
    tribe.decision_journal.append(entry)
    return entry


def test_a_trip_that_brought_sites_says_so_instead_of_nothing_new_found():
    sim, tribe = _sim()
    (line,) = _home(sim, tribe, _exp())
    assert "brought home" in line and "timber grove at (" in line and "nothing new found" not in line
    assert len(tribe.lumber_sites) == 1  # and the cap still holds: one of each type


def test_a_trip_over_known_ground_is_still_reported_as_empty():
    sim, tribe = _sim()
    for _ in range(40):  # map the route out, one scout at a time, until a scout brings nothing
        tribe.history.clear()
        (line,) = _home(sim, tribe, _exp())
        if "brought home" not in line:
            break
    assert "nothing new found" in line and "brought home" not in line


def test_water_and_sites_are_both_in_the_report():
    sim, tribe = _sim()
    (line,) = _home(sim, tribe, _exp(found=(55, 55)))
    assert "fresh water confirmed at (55,55)" in line and "brought home" in line


def test_the_haul_is_added_to_the_journal_entry_for_that_launch():
    sim, tribe = _sim()
    entry = _launch_entry(tribe, 5)
    other = _launch_entry(tribe, 9)  # a later choice that must not be touched
    _home(sim, tribe, _exp(launched=5))
    assert entry["delta"].get("timber groves known") == 1 and "stone sites known" in entry["delta"]
    assert other["delta"] == {}


def test_the_readback_shows_scouting_as_a_gain_not_no_change(monkeypatch):
    from backend import config

    sim, tribe = _sim()
    monkeypatch.setattr(config, "JOURNAL_REPEAT_MIN", 2)
    tribe.decision_journal = []
    for c in (5, 8, 11):
        _launch_entry(tribe, c)
    _home(sim, tribe, _exp(launched=5))
    _home(sim, tribe, _exp(launched=8))
    (line, *_rest) = sim._journal_readback_lines(tribe)
    assert "SCOUT" in line and "timber groves known +2" in line and "no change" not in line


def test_haul_text_reads_naturally():
    assert _haul_text([]) == ""
    one = _haul_text([{"type": "lumber", "x": 5, "y": 6}])
    assert one == "brought home a timber grove at (5,6)"
    three = _haul_text([{"type": "lumber", "x": 1, "y": 2}, {"type": "quarry", "x": 3, "y": 4}, {"type": "mine", "x": 5, "y": 6, "label": "Orosite Ore"}])
    assert three == "brought home a timber grove at (1,2), a stone-rich site at (3,4) and a vein of Orosite Ore at (5,6)"
