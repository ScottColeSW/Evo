"""Derived, queryable decision table built from the full per-cycle snapshots that
board_history.py already records for every live run. Rather than duplicate that write
path with a second live instrumentation hook, this reads the existing snapshot_json
blobs and materializes one flat row per (run_id, cycle, tribe) into a `decisions`
table in the same database -- basic stats, location, discovery-site lists, and the
final action + target_vector executed that cycle, per the user's explicit scope: state
capture for spotting issues and confirming suspicions about tribe behavior, not
capturing the model's reasoning/rationale text.

Safe to re-run at any time (INSERT OR REPLACE, keyed by run_id/cycle/tribe_id) --
this is a materialization step, not a live-write path, so it can be run on demand
against whatever board_snapshots rows already exist.
"""
import json
import sqlite3

from .board_history import DEFAULT_DB_PATH, _connect

_JSON_COLUMNS = (
    "confirmed_water_sites",
    "lumber_sites",
    "wildlife_sites",
    "quarry_sites",
    "mine_sites",
    "raider_sightings",
)


def _ensure_decisions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            run_id TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            tribe_id TEXT NOT NULL,
            tribe_name TEXT,
            era TEXT,
            population INTEGER,
            wood REAL,
            stone REAL,
            food REAL,
            water REAL,
            x INTEGER,
            y INTEGER,
            confirmed_water_sites TEXT,
            lumber_sites TEXT,
            wildlife_sites TEXT,
            quarry_sites TEXT,
            mine_sites TEXT,
            raider_sightings TEXT,
            action TEXT,
            target_x INTEGER,
            target_y INTEGER,
            PRIMARY KEY (run_id, cycle, tribe_id)
        )
        """
    )


def materialize(run_id: str | None = None, path: str | None = None) -> int:
    """Reads board_snapshots (optionally scoped to one run_id) and (re)builds the
    decisions table from it. Returns the number of decision rows written."""
    conn = _connect(path or DEFAULT_DB_PATH)
    _ensure_decisions_table(conn)

    if run_id is not None:
        rows = conn.execute(
            "SELECT run_id, cycle, snapshot_json FROM board_snapshots WHERE run_id = ? ORDER BY cycle",
            (run_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT run_id, cycle, snapshot_json FROM board_snapshots ORDER BY run_id, cycle"
        ).fetchall()

    written = 0
    with conn:
        for row_run_id, cycle, snapshot_json in rows:
            snapshot = json.loads(snapshot_json)
            for tribe_id, tribe in snapshot.get("tribes", {}).items():
                target = tribe.get("last_decision_target") or [None, None]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO decisions (
                        run_id, cycle, tribe_id, tribe_name, era, population,
                        wood, stone, food, water, x, y,
                        confirmed_water_sites, lumber_sites, wildlife_sites,
                        quarry_sites, mine_sites, raider_sightings,
                        action, target_x, target_y
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_run_id, cycle, tribe_id, tribe.get("name"), tribe.get("era"),
                        tribe.get("population"), tribe.get("wood"), tribe.get("stone"),
                        tribe.get("food"), tribe.get("water"), tribe.get("x"), tribe.get("y"),
                        *(json.dumps(tribe.get(col)) for col in _JSON_COLUMNS),
                        tribe.get("last_action"), target[0], target[1],
                    ),
                )
                written += 1
    conn.close()
    return written


def query_tribe_decisions(
    run_id: str, tribe_name: str, path: str | None = None,
    start_cycle: int | None = None, end_cycle: int | None = None,
) -> list[dict]:
    """Convenience reader for one tribe's decision history within a run, ordered by
    cycle -- the shape most useful for "spot the issue" analysis over a stretch of
    time (e.g. the cycles around a suspected bug)."""
    conn = _connect(path or DEFAULT_DB_PATH)
    sql = "SELECT * FROM decisions WHERE run_id = ? AND tribe_name = ?"
    params: list = [run_id, tribe_name]
    if start_cycle is not None:
        sql += " AND cycle >= ?"
        params.append(start_cycle)
    if end_cycle is not None:
        sql += " AND cycle <= ?"
        params.append(end_cycle)
    sql += " ORDER BY cycle"
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
