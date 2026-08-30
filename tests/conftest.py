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


def run_async(fn):
    """Lets an async test function run under plain pytest, with no pytest-asyncio
    dependency -- just wraps it in asyncio.run()."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        asyncio.run(fn(*args, **kwargs))

    return wrapper
