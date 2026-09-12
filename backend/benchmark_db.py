"""A persistent, cross-run benchmark record for the scenario harness (backend.
benchmark_scenarios/run_benchmark.py) -- long-running by design, the same "append
forever, query later" shape backend/board_history.py and backend/scoreboard.py
already use, not a one-off report. Two real, sliceable tables rather than one wide
row with a JSON blob, for the same reason board_history.py gives for using a real
database at all: this is meant to be queried and filtered later (every trial for
one model in one scenario, everything since a given scenario_version), not just
appended-to and read back whole.

Deliberately stores only raw facts, never a computed score -- see backend.
benchmark_scoring.py's own module docstring for why baking a score in here would
silently invalidate every historical trial the moment the scoring formula is
tuned.
"""

import sqlite3
import subprocess
import time
from pathlib import Path

DEFAULT_DB_PATH = "logs/benchmark_results.db"

# Columns are simple ints/text (SQLite bool = 0/1) so an aggregate query (AVG,
# GROUP BY model) works directly in SQL without deserializing anything.
_TRIAL_TRIBE_COLUMNS = (
    "trial_id", "tribe_index", "model", "extinct", "extinction_cause",
    "era_reached", "max_population", "final_population",
    "chiefs_elected", "chief_deaths",
    "expeditions_launched", "expeditions_succeeded",
    "raids_won", "raids_lost", "raids_defended",
    "trades_completed", "spy_missions_run", "spy_missions_caught",
    "discovered_rival", "allied_with_rival", "at_war_with_rival",
    "joint_castle_completed", "settled_permanently_near_water",
)


def _connect(path: str | None = None) -> sqlite3.Connection:
    target = Path(path or DEFAULT_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trials (
            trial_id TEXT PRIMARY KEY,
            scenario_key TEXT NOT NULL,
            scenario_version INTEGER NOT NULL,
            trial_seed INTEGER NOT NULL,
            cycle_budget INTEGER NOT NULL,
            cycles_run INTEGER NOT NULL,
            ended_reason TEXT NOT NULL,
            git_commit TEXT,
            run_id TEXT,
            started_ts REAL NOT NULL,
            finished_ts REAL NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS trial_tribes (
            {", ".join(f"{c} TEXT" if c in ("model", "extinction_cause", "era_reached", "trial_id") else f"{c} INTEGER" for c in _TRIAL_TRIBE_COLUMNS)},
            PRIMARY KEY (trial_id, tribe_index),
            FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
        )
        """
    )
    return conn


def current_git_commit() -> str | None:
    """Best-effort -- a missing/unavailable git binary shouldn't break a real
    benchmark trial over a nice-to-have provenance stamp (same fail-open shape
    backend/vram_guard.py's own docstring already argues for)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def extract_tribe_facts(tribe, rival=None) -> dict:
    """The raw, per-tribe facts a completed trial records -- kept separate from
    the run loop itself so it's directly unit-testable against a hand-built Tribe.
    `rival` is the other tribe in a 2-tribe scenario (None for survival/settlement)
    -- only used to compute this tribe's own view of the relationship, never the
    rival's stats."""
    discovered_rival = rival is not None and rival.id in tribe.discovered_rivals
    stance = tribe.stance_toward.get(rival.id) if rival is not None else None
    return {
        "model": tribe.model,
        "extinct": tribe.extinct,
        "extinction_cause": tribe.extinction_cause,
        "era_reached": tribe.era,
        "max_population": tribe.max_population,
        "final_population": tribe.population,
        "chiefs_elected": tribe.chiefs_elected,
        "chief_deaths": tribe.chief_deaths,
        "expeditions_launched": tribe.expeditions_launched,
        "expeditions_succeeded": tribe.expeditions_succeeded,
        "raids_won": tribe.raids_won,
        "raids_lost": tribe.raids_lost,
        "raids_defended": tribe.raids_defended,
        "trades_completed": tribe.trades_completed,
        "spy_missions_run": tribe.spy_missions_run,
        "spy_missions_caught": tribe.spy_missions_caught,
        "discovered_rival": discovered_rival,
        "allied_with_rival": stance == "ALLIED",
        "at_war_with_rival": stance == "WAR",
        "joint_castle_completed": tribe.castle_built and stance == "ALLIED",
        "settled_permanently_near_water": tribe.settled_permanently_near_water,
    }


def record_trial(trial_facts: dict, tribe_facts: list[dict], path: str | None = None) -> None:
    """`trial_facts` needs: trial_id, scenario_key, scenario_version, trial_seed,
    cycle_budget, cycles_run, ended_reason, git_commit, run_id, started_ts,
    finished_ts. `tribe_facts` is a list of dicts shaped like extract_tribe_facts's
    return value, one per tribe in the trial, in tribe_index order."""
    conn = _connect(path)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO trials
            (trial_id, scenario_key, scenario_version, trial_seed, cycle_budget,
             cycles_run, ended_reason, git_commit, run_id, started_ts, finished_ts)
            VALUES (:trial_id, :scenario_key, :scenario_version, :trial_seed, :cycle_budget,
                    :cycles_run, :ended_reason, :git_commit, :run_id, :started_ts, :finished_ts)
            """,
            trial_facts,
        )
        for tribe_index, facts in enumerate(tribe_facts):
            row = {"trial_id": trial_facts["trial_id"], "tribe_index": tribe_index, **facts}
            placeholders = ", ".join(f":{c}" for c in _TRIAL_TRIBE_COLUMNS)
            conn.execute(
                f"INSERT OR REPLACE INTO trial_tribes ({', '.join(_TRIAL_TRIBE_COLUMNS)}) VALUES ({placeholders})",
                row,
            )
    conn.close()


def list_trials(scenario_key: str | None = None, model: str | None = None, path: str | None = None) -> list[dict]:
    """Every trial matching the given filters, newest first, each with its
    tribes' facts nested under "tribes"."""
    conn = _connect(path)
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM trials"
    clauses, params = [], []
    if scenario_key is not None:
        clauses.append("scenario_key = ?")
        params.append(scenario_key)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY started_ts DESC"
    trial_rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    results = []
    for trial in trial_rows:
        tribe_query = "SELECT * FROM trial_tribes WHERE trial_id = ? ORDER BY tribe_index"
        tribes = [dict(r) for r in conn.execute(tribe_query, (trial["trial_id"],)).fetchall()]
        if model is not None and not any(t["model"] == model for t in tribes):
            continue
        trial["tribes"] = tribes
        results.append(trial)
    conn.close()
    return results


def read_trial(trial_id: str, path: str | None = None) -> dict | None:
    conn = _connect(path)
    conn.row_factory = sqlite3.Row
    trial_row = conn.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
    if trial_row is None:
        conn.close()
        return None
    trial = dict(trial_row)
    tribes = conn.execute(
        "SELECT * FROM trial_tribes WHERE trial_id = ? ORDER BY tribe_index", (trial_id,)
    ).fetchall()
    conn.close()
    trial["tribes"] = [dict(r) for r in tribes]
    return trial


def new_trial_id(scenario_key: str, trial_seed: int) -> str:
    return f"{scenario_key}_{trial_seed}_{int(time.time())}"
