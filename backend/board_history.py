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
import threading
import time
from pathlib import Path

DEFAULT_DB_PATH = "logs/board_history.db"


def _connect(path: str | None = None) -> sqlite3.Connection:
    target = Path(path or DEFAULT_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 2026-10-08: a live run's snapshot write failed with "database is locked" while analysis scripts were reading the history. In SQLite's default journal mode
    # a reader blocks the writer, and the default wait is short. WAL mode lets readers and the one writer work at once, and the wait is now 30 seconds.
    conn = sqlite3.connect(target, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
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


# 2026-10-10: the writer keeps one connection open per database file instead of opening, setting up and closing one every cycle. Measured (docs/APP-PERFORMANCE-2026-10-09.md): that cost about
# 120 ms a cycle, nearly all file flushing at commit and close, on the thread that serves the page. WAL with synchronous=NORMAL still never corrupts the file; the one thing it gives up is that the
# last few cycles may be lost if the machine loses power, which is acceptable for a history. Readers open their own connections as before, and WAL lets them read while this one writes.
_writers: dict[str, sqlite3.Connection] = {}
_writers_lock = threading.Lock()


def _writer(path: str | None) -> sqlite3.Connection:
    target = Path(path or DEFAULT_DB_PATH)
    key = str(target.resolve())
    with _writers_lock:
        conn = _writers.get(key)
        if conn is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(target, timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS board_snapshots (run_id TEXT NOT NULL, cycle INTEGER NOT NULL, ts REAL NOT NULL, snapshot_json TEXT NOT NULL, PRIMARY KEY (run_id, cycle))"
            )
            conn.commit()
            _writers[key] = conn
        return conn


def close_all() -> None:
    """Closes every open writer (server shutdown, and between tests so no database file stays held open)."""
    with _writers_lock:
        for conn in _writers.values():
            try:
                conn.close()
            except sqlite3.Error:
                pass
        _writers.clear()


def record_board_state(run_id: str, cycle: int, snapshot: dict, path: str | None = None, snapshot_json: str | None = None) -> None:
    """Idempotent per (run_id, cycle) -- a re-sent snapshot for a cycle already
    recorded (e.g. a duplicate tick) overwrites rather than duplicating. Does nothing in a lean run (config.LEAN_RUN): the write costs about 120 ms of file flushing
    a cycle, on the thread that serves the page."""
    from . import config
    if config.lean_run():
        return
    # snapshot_json is the already-serialized snapshot when the caller has one (the server sends the same text to the page), so it is turned into JSON once, not twice.
    payload = snapshot_json if snapshot_json is not None else json.dumps(snapshot, default=str)
    conn = _writer(path)
    with _writers_lock:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO board_snapshots (run_id, cycle, ts, snapshot_json) VALUES (?, ?, ?, ?)",
                (run_id, cycle, time.time(), payload),
            )


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
