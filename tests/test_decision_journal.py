"""Step 1 of docs/CHIEF-EVIDENCE-MEMORY-DESIGN.md: a decision journal. Recording only; nothing reads it into a prompt."""
import json

from backend import config
from backend.actions import note_conflict
from backend.simulation import Simulation


def _records(sim, kind):
    if not sim.event_log.path.exists():
        return []
    return [json.loads(line) for line in sim.event_log.path.read_text(encoding="utf-8").splitlines() if f'"kind": "{kind}"' in line]


def _turn(sim, tribe, action="GATHER_FOOD"):
    _request, ctx = sim._prepare_turn(tribe)
    ctx["available_actions"] = [action]
    sim._apply_turn(tribe, {"visual_action": action, "target_vector": [tribe.x, tribe.y]}, 10.0, ctx)


def test_a_turn_is_journaled_with_what_it_changed_and_logged():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.food = 0
    sim.cycle = 4
    _turn(sim, tribe)
    (entry,) = tribe.decision_journal
    assert entry["cycle"] == 4 and entry["action"] == "GATHER_FOOD"
    assert entry["delta"].get("food", 0) == tribe.food - 0 and entry["stock_before"]["food"] == 0
    assert entry["moved"] is False and entry["built"] == {}
    (logged,) = _records(sim, "decision")
    assert logged["data"]["action"] == "GATHER_FOOD" and logged["message"].startswith("[journal] GATHER_FOOD")


def test_the_journal_is_capped_and_can_be_switched_off(monkeypatch):
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    monkeypatch.setattr(config, "DECISION_JOURNAL_LENGTH", 3)
    for cycle in range(1, 7):
        sim.cycle = cycle
        _turn(sim, tribe)
    assert [e["cycle"] for e in tribe.decision_journal] == [4, 5, 6]

    monkeypatch.setattr(config, "DECISION_JOURNAL", "off")
    before = len(tribe.decision_journal)
    sim.cycle = 7
    _turn(sim, tribe)
    assert len(tribe.decision_journal) == before and len(_records(sim, "decision")) == 6   # nothing recorded when off


def test_a_conflict_event_names_the_recent_decisions_it_followed():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    for cycle in (1, 2, 8, 9):
        sim.cycle = cycle
        _turn(sim, tribe)
    sim.event_log.current_cycle = 10
    note_conflict(tribe, "Home Defense", "lost")
    (event,) = _records(sim, "conflict_event")
    assert [d["cycle"] for d in event["data"]["followed_decisions"]] == [8, 9]     # within the 3-cycle lookback only
