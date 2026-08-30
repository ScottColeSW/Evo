"""A persistent, cross-run benchmark record -- what an evaluator comparing local
models actually wants isn't a per-run play-by-play (that's backend/event_log.py's
job), it's a structured summary per tribe's lifetime: how long it survived, how far it
got, and how it got there (leadership turnover, whether scouting paid off, how it fared
in conflict), all attributable to the specific model that ran it. One line is appended
here every time a tribe's story ends (extinction), across every run, so a leaderboard
can be built by reading this file back and grouping by model.
"""

import json
import time
from pathlib import Path

# A plain module attribute (not a default argument) so tests can monkeypatch it, the
# same pattern as event_log.py's DEFAULT_LOG_DIR -- every extinction otherwise appends
# to a real file under the project's logs/ directory.
DEFAULT_SCOREBOARD_PATH = "logs/scoreboard.jsonl"


def record_tribe_result(tribe, cause: str, cycles_survived: int, path: str | None = None) -> None:
    path = path if path is not None else DEFAULT_SCOREBOARD_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "tribe_name": tribe.name,
        "model": tribe.model,
        "cause_of_death": cause,
        "cycles_survived": cycles_survived,
        "era_reached": tribe.era,
        "max_population": tribe.max_population,
        "chiefs_elected": tribe.chiefs_elected,
        "chief_deaths": tribe.chief_deaths,
        "expeditions_launched": tribe.expeditions_launched,
        "expeditions_succeeded": tribe.expeditions_succeeded,
        "raids_won": tribe.raids_won,
        "raids_lost": tribe.raids_lost,
        "raids_defended": tribe.raids_defended,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_all_results(path: str | None = None) -> list[dict]:
    path = path if path is not None else DEFAULT_SCOREBOARD_PATH
    file = Path(path)
    if not file.exists():
        return []
    results = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def summarize_by_model(results: list[dict]) -> list[dict]:
    """One row per model: run count, average/best survival, totals for the rest --
    the actual leaderboard shape. Sorted by average cycles survived, best first."""
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    rows = []
    for model, runs in by_model.items():
        n = len(runs)
        rows.append({
            "model": model,
            "runs": n,
            "avg_cycles_survived": round(sum(r["cycles_survived"] for r in runs) / n, 1),
            "best_cycles_survived": max(r["cycles_survived"] for r in runs),
            "best_era_reached": max(runs, key=lambda r: r["cycles_survived"])["era_reached"],
            "avg_max_population": round(sum(r["max_population"] for r in runs) / n, 1),
            "total_chiefs_elected": sum(r["chiefs_elected"] for r in runs),
            "total_chief_deaths": sum(r["chief_deaths"] for r in runs),
            "total_expeditions_launched": sum(r["expeditions_launched"] for r in runs),
            "total_expeditions_succeeded": sum(r["expeditions_succeeded"] for r in runs),
            "total_raids_won": sum(r["raids_won"] for r in runs),
            "total_raids_lost": sum(r["raids_lost"] for r in runs),
            "total_raids_defended": sum(r["raids_defended"] for r in runs),
        })
    rows.sort(key=lambda row: row["avg_cycles_survived"], reverse=True)
    return rows
