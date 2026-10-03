"""Step 2 of docs/CHIEF-EVIDENCE-MEMORY-DESIGN.md: the decision journal read back to the Chief as plain facts, off by default."""
import json

from backend import config
from backend.simulation import Simulation


def _sim(readback):
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}], journal_readback="on" if readback else "off")
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    sim._found_territory(tribe)
    return sim, tribe


def _entry(cycle, action, **delta):
    return {"cycle": cycle, "action": action, "delta": delta, "built": {}, "moved": False,
            "stock_before": {}, "population": 100, "note": None}


def test_the_readback_is_off_by_default_and_adds_nothing_to_the_prompt():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    assert sim.journal_readback is False and sim.snapshot()["journal_readback"] == "off"
    tribe = sim.tribes["tribe_0"]
    sim.cycle = 20
    tribe.decision_journal = [_entry(c, "GATHER_STONE", stone=100) for c in range(11, 21)]
    request, _ctx = sim._prepare_turn(tribe)
    assert "In your last" not in request["prompt"]


def test_a_repeated_choice_is_read_back_as_numbers_only():
    sim, tribe = _sim(True)
    sim.cycle = 20
    tribe.decision_journal = [_entry(c, "GATHER_STONE", stone=100) for c in range(11, 17)] + [_entry(c, "SCOUT") for c in range(17, 21)]
    (line,) = sim._journal_readback_lines(tribe)
    assert line == "In your last 10 choices you picked GATHER_STONE 6 times: stone +600; nothing was built."
    assert not any(word in line.lower() for word in ("should", "must", "try", "consider", "instead"))
    request, _ctx = sim._prepare_turn(tribe)
    assert line in request["prompt"]
    assert sim.snapshot()["journal_readback"] == "on"


def test_a_repeated_choice_reports_what_was_built_over_those_choices_when_something_was():
    sim, tribe = _sim(True)
    sim.cycle = 12
    entries = [_entry(c, "GATHER_WOOD", wood=40) for c in range(3, 12)] + [_entry(12, "BUILD_LONG_HOUSE", wood=-30)]
    entries[-1]["built"] = {"long_houses_built": [0, 1]}
    tribe.decision_journal = entries
    (line,) = sim._journal_readback_lines(tribe)
    assert "GATHER_WOOD 9 times" in line and "changed over those choices: long houses built" in line


def test_no_line_when_nothing_repeats_and_the_last_high_stakes_choice_is_read_back_only_while_recent():
    sim, tribe = _sim(True)
    sim.cycle = 40
    tribe.decision_journal = [_entry(c, a) for c, a in zip(range(31, 41), ["SCOUT", "RESEARCH", "GATHER_FOOD", "HUNT_DEER", "SCOUT", "RESEARCH",
                                                                         "GATHER_WOOD", "HUNT_DEER", "GATHER_FOOD", "RESEARCH"])]
    assert sim._journal_readback_lines(tribe) == []
    tribe.decision_journal.insert(0, _entry(5, "RAID", population=-8, food=40))
    assert sim._journal_readback_lines(tribe) == []                         # cycle 5 is outside the 30-cycle lookback
    tribe.decision_journal.insert(0, _entry(25, "RAID", population=-8, food=40))
    assert sim._journal_readback_lines(tribe) == ["Your last RAID (cycle 25): population -8, food +40."]


def test_the_journal_no_longer_counts_the_action_streak_as_a_structure():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    assert not any(name.endswith("_count") for name in Simulation._journal_snapshot(tribe)["built"])
