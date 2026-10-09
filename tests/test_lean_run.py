"""2026-10-09 (the owner: "turn off all the logging and double checking. I'm trying to measure the app instead of the models"): LEAN_RUN switches off everything that exists to
record or to double-check, and changes nothing else. Off by default."""
import asyncio

from backend import board_history, config
from backend.event_log import RunEventLog
from backend.simulation import Simulation


def _sim():
    return Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])


def _turn(sim, tribe, action="GATHER_FOOD"):
    _request, ctx = sim._prepare_turn(tribe)
    ctx["available_actions"] = [action]
    sim._apply_turn(tribe, {"visual_action": action, "target_vector": [tribe.x, tribe.y]}, 10.0, ctx)


def test_lean_run_is_off_by_default_and_the_environment_variable_turns_it_on(monkeypatch):
    monkeypatch.delenv("LEAN_RUN", raising=False)
    assert config.LEAN_RUN == "off" and config.lean_run() is False
    monkeypatch.setenv("LEAN_RUN", "on")
    assert config.lean_run() is True


def test_the_event_log_writes_nothing_in_a_lean_run(tmp_path, monkeypatch):
    log = RunEventLog(str(tmp_path))
    log.record("Tribe", "a chronicle line")
    log.record_data("Tribe", "decision", {"action": "GATHER_FOOD"})
    assert len(log.path.read_text(encoding="utf-8").splitlines()) == 2  # by default both are written

    other = RunEventLog(str(tmp_path / "lean"))
    monkeypatch.setenv("LEAN_RUN", "on")
    other.record("Tribe", "a chronicle line")
    other.record_data("Tribe", "decision", {"action": "GATHER_FOOD"})
    assert not other.path.exists()


def test_board_history_writes_nothing_in_a_lean_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAN_RUN", "on")
    path = tmp_path / "board_history.db"
    board_history.record_board_state("run_1", 0, {"cycle": 0}, path=str(path))
    assert not path.exists()
    monkeypatch.delenv("LEAN_RUN")
    board_history.record_board_state("run_1", 0, {"cycle": 0}, path=str(path))
    assert path.exists()


def test_a_lean_run_keeps_no_decision_journal_and_no_read_back(monkeypatch):
    monkeypatch.setenv("LEAN_RUN", "on")
    monkeypatch.setenv("JOURNAL_READBACK", "on")
    sim = _sim()
    tribe = sim.tribes["tribe_0"]
    assert sim.journal_readback is False  # the environment asked for it; the lean run wins
    sim.cycle = 4
    _turn(sim, tribe)
    assert list(tribe.decision_journal) == []
    assert not sim.event_log.path.exists()


def test_the_game_itself_is_the_same_in_a_lean_run(monkeypatch):
    """The same turn gives the same result with and without it; only the records differ."""
    monkeypatch.delenv("LEAN_RUN", raising=False)
    plain = _sim()
    plain_tribe = plain.tribes["tribe_0"]
    plain.cycle = 4
    _turn(plain, plain_tribe)
    monkeypatch.setenv("LEAN_RUN", "on")
    lean = _sim()
    lean_tribe = lean.tribes["tribe_0"]
    lean.cycle = 4
    _turn(lean, lean_tribe)
    assert (lean_tribe.food, lean_tribe.wood, lean_tribe.stone, lean_tribe.water, lean_tribe.population) == (
        plain_tribe.food, plain_tribe.wood, plain_tribe.stone, plain_tribe.water, plain_tribe.population)
    assert len(plain_tribe.decision_journal) == 1  # and by default the journal still records


def test_a_lean_run_builds_no_reflection_judge_even_when_asked(monkeypatch):
    monkeypatch.setenv("LEAN_RUN", "on")
    monkeypatch.setenv("REFLECTION_JUDGE", "nli")
    sim = _sim()
    assert asyncio.run(sim._reflection_judge()) is None
    assert sim.reflection_judge_status == "off"
