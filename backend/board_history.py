"""Full board-state capture, one row per simulation cycle, written to a real SQLite
database rather than JSONL -- unlike the sparse event log (backend/event_log.py, which
only records discrete narrative moments) or the scoreboard (backend/scoreboard.py, only
a tribe's lifetime summary), this captures the *entire* board snapshot every single
cycle: every tribe's full state, structures, trails, linguistic consensus -- everything
Simulation.snapshot() already returns for the frontend. A real database earns its keep
here since this is dense, structured, per-cycle data meant to be sliced and queried
later (a tribe's population over time, every board state around a given event), not
just appended-to and read back whole like the other two logs.
"""
import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = "logs/board_history.db"


def _connect(path: str | None = None) -> sqlite3.Connection:
    target = Path(path or DEFAULT_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS board_snapshots (
            run_id TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            ts REAL NOT NULL,
            snapshot_json TEXT NOT NULL,
            PRIMARY KEY (run_id, cycle)
        )
        """
    )
    return conn


def record_board_state(run_id: str, cycle: int, snapshot: dict, path: str | None = None) -> None:
    """Idempotent per (run_id, cycle) -- a re-sent snapshot for a cycle already
    recorded (e.g. a duplicate tick) overwrites rather than duplicating."""
    conn = _connect(path)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO board_snapshots (run_id, cycle, ts, snapshot_json) VALUES (?, ?, ?, ?)",
            (run_id, cycle, time.time(), json.dumps(snapshot, default=str)),
        )
    conn.close()


def list_runs(path: str | None = None) -> list[str]:
    conn = _connect(path)
    rows = conn.execute("SELECT DISTINCT run_id FROM board_snapshots ORDER BY run_id").fetchall()
    conn.close()
    return [r[0] for r in rows]


def read_run(run_id: str, path: str | None = None) -> list[dict]:
    """Every recorded cycle for one run, in order, each with its full snapshot."""
    conn = _connect(path)
    rows = conn.execute(
        "SELECT cycle, ts, snapshot_json FROM board_snapshots WHERE run_id = ? ORDER BY cycle",
        (run_id,),
    ).fetchall()
    conn.close()
    return [{"cycle": r[0], "ts": r[1], "snapshot": json.loads(r[2])} for r in rows]


def read_cycle(run_id: str, cycle: int, path: str | None = None) -> dict | None:
    conn = _connect(path)
    row = conn.execute(
        "SELECT ts, snapshot_json FROM board_snapshots WHERE run_id = ? AND cycle = ?",
        (run_id, cycle),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"cycle": cycle, "ts": row[0], "snapshot": json.loads(row[1])}
