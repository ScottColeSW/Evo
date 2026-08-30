"""Persistent, append-only record of a run's chronicle, written to disk as JSON lines.

Browser-side state -- and Tribe.history itself, capped at the last 6 entries for the
UI -- is lost on every page reload, server restart, or navigation. All three happened
repeatedly during development and lost real run history each time. This survives that,
for after-the-fact analysis of what a run actually did: major steps (chief elections,
era advances, expeditions found water) and missteps (starvation, a chief's death, a
lost raid) alike, in the order they happened.
"""

import json
import time
from pathlib import Path

# A plain module attribute (not a default argument) so tests can monkeypatch it via
# `monkeypatch.setattr(event_log, "DEFAULT_LOG_DIR", tmp_path)` -- every Simulation
# otherwise creates a real, timestamped file under the project's logs/ directory, and
# call sites all construct RunEventLog() with no arguments.
DEFAULT_LOG_DIR = "logs"


class RunEventLog:
    def __init__(self, log_dir: str | None = None):
        log_dir = log_dir if log_dir is not None else DEFAULT_LOG_DIR
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        filename = f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.path = Path(log_dir) / filename
        self.current_cycle = 0

    def record(self, tribe_name: str, message: str) -> None:
        line = json.dumps({
            "cycle": self.current_cycle,
            "tribe": tribe_name,
            "message": message,
            "ts": time.time(),
        })
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class TribeHistory(list):
    """Drop-in replacement for Tribe.history's plain list -- every append still works
    exactly as before for the live in-browser chronicle (including the [-6:] slice
    Tribe.to_dict() takes), but also mirrors the entry into the shared RunEventLog if
    one is attached. `event_log=None` (the default, and what every existing test
    constructs a bare Tribe with) makes this behave as a plain list with no side effect."""

    def __init__(self, tribe_name: str, event_log: RunEventLog | None = None):
        super().__init__()
        self.tribe_name = tribe_name
        self.event_log = event_log

    def append(self, entry: str) -> None:
        super().append(entry)
        if self.event_log is not None:
            self.event_log.record(self.tribe_name, entry)
