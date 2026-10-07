import asyncio
import functools

import pytest


@pytest.fixture(autouse=True)
def isolate_event_log(tmp_path, monkeypatch):
    """Every test that constructs a Simulation also constructs a RunEventLog
    (backend/event_log.py), which otherwise writes a real, timestamped file under the
    project's logs/ directory -- redirect it to pytest's per-test tmp_path so running
    the suite doesn't litter the repo with hundreds of test-run log files."""
    from backend import event_log
    monkeypatch.setattr(event_log, "DEFAULT_LOG_DIR", str(tmp_path))


@pytest.fixture(autouse=True)
def nudges_on_for_the_suite(monkeypatch):
    """The shipped default is NUDGES off (2026-10-07). The prompt tests that assert a nudge's text exist to check that text, so the suite runs with the switch on;
    tests of the default itself delenv it first (tests/test_nudge_switch.py)."""
    monkeypatch.setenv("NUDGES", "on")


@pytest.fixture(autouse=True)
def isolate_scoreboard(tmp_path, monkeypatch):
    """Any test that drives a tribe to extinction appends a real record to the
    project's logs/scoreboard.jsonl (backend/scoreboard.py) -- redirect it the same way
    as the event log."""
    from backend import scoreboard
    monkeypatch.setattr(scoreboard, "DEFAULT_SCOREBOARD_PATH", str(tmp_path / "scoreboard.jsonl"))


def run_async(fn):
    """Lets an async test function run under plain pytest, with no pytest-asyncio
    dependency -- just wraps it in asyncio.run()."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        asyncio.run(fn(*args, **kwargs))

    return wrapper
