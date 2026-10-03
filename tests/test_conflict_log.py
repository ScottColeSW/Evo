"""Phase 0 of docs/CONFLICT-MODE-DESIGN.md: conflict events open a logging window, and offered and chosen actions inside it are
logged. Logging only: nothing here changes an outcome."""
import json

from backend import config
from backend.actions import note_conflict, _record_combat
from backend.simulation import Simulation


def _records(sim, kind):
    if not sim.event_log.path.exists():
        return []
    return [json.loads(line) for line in sim.event_log.path.read_text(encoding="utf-8").splitlines() if f'"kind": "{kind}"' in line]


def test_a_combat_outcome_logs_an_event_and_opens_a_window_that_a_new_event_does_not_extend():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    sim.event_log.current_cycle = 5
    _record_combat(tribe, "Raid Defense", "lost")
    assert tribe.combat_record["Raid Defense"] == {"won": 0, "lost": 1}      # the existing bookkeeping is unchanged
    assert tribe.conflict_watch["start"] == 5 and tribe.conflict_watch["until"] == 5 + config.CONFLICT_LOG_WINDOW_CYCLES
    sim.event_log.current_cycle = 8
    _record_combat(tribe, "Raiding", "won")                                  # inside the open window: logged, not extended
    assert tribe.conflict_watch["start"] == 5
    first, second = _records(sim, "conflict_event")
    assert first["data"]["starts_window"] is True and second["data"]["starts_window"] is False
    sim.event_log.current_cycle = 5 + config.CONFLICT_LOG_WINDOW_CYCLES + 1  # after it closes, a new event opens a new window
    note_conflict(tribe, "Conquest", "lost")
    assert tribe.conflict_watch["start"] == 5 + config.CONFLICT_LOG_WINDOW_CYCLES + 1


def test_the_menu_is_logged_while_a_window_is_open_and_not_otherwise():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    sim._found_territory(tribe)
    sim._prepare_turn(tribe)
    assert _records(sim, "conflict_turn") == []                              # no window, nothing logged
    sim.cycle = 3
    sim.event_log.current_cycle = 3
    note_conflict(tribe, "Home Defense", "lost")
    sim._prepare_turn(tribe)
    (turn,) = _records(sim, "conflict_turn")
    data = turn["data"]
    assert data["event"] == "Home Defense" and data["cycles_since"] == 0
    assert data["offered_count"] == len(data["offered"]) and 0 <= data["answering_share"] <= 1
    assert set(data["answering"]) <= set(data["offered"])


def test_declaring_war_logs_an_event_for_both_tribes():
    from backend import actions
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    a, b = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    a.barracks_built = 1
    a.discovered_rivals.add(b.id)
    sim.event_log.current_cycle = 7
    actions._declare_war(sim, a, "plains", (b.x, b.y))
    kinds = {(r["tribe"], r["data"]["kind"]) for r in _records(sim, "conflict_event")}
    assert ("A", "War Declared") in kinds and ("B", "War Declared Against") in kinds
    assert a.stance_toward[b.id] == "WAR"
