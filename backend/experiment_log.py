"""Persistent, append-only store for A/B test results (backend/scoreboard.py's sibling,
same JSONL-on-disk pattern rather than a real database -- consistent with how this project
already tracks cross-run data, no new dependency needed for the volume involved).

Every headless A/B script (scripts/ab_test_*.py) should call record_experiment_run() once
per run instead of only writing its own throwaway *_results.json, so results accumulate in
one place across sessions instead of living in disconnected per-script files that the next
experiment silently overwrites.
"""
import json
import time
from pathlib import Path

DEFAULT_EXPERIMENT_LOG_PATH = "logs/experiments.jsonl"


def record_experiment_run(
    experiment: str,
    variant: str,
    model: str,
    metrics: dict,
    run_id: int | None = None,
    notes: str = "",
    path: str | None = None,
) -> None:
    """Appends one run's result. `metrics` is deliberately free-form (different
    experiments measure different things) -- only numeric values in it get aggregated
    by summarize_experiment, non-numeric values are kept for reference but not averaged."""
    target = Path(path or DEFAULT_EXPERIMENT_LOG_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": experiment,
        "variant": variant,
        "model": model,
        "run_id": run_id,
        "metrics": metrics,
        "notes": notes,
        "ts": time.time(),
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_all_experiment_runs(path: str | None = None) -> list[dict]:
    target = Path(path or DEFAULT_EXPERIMENT_LOG_PATH)
    if not target.exists():
        return []
    runs = []
    with target.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def summarize_experiment(experiment: str, runs: list[dict] | None = None) -> dict:
    """Groups runs for one experiment by variant, summing/averaging every numeric
    metric found. Returns {variant: {"n_runs": int, "totals": {...}, "means": {...}}}."""
    if runs is None:
        runs = read_all_experiment_runs()
    by_variant: dict[str, list[dict]] = {}
    for run in runs:
        if run.get("experiment") != experiment:
            continue
        by_variant.setdefault(run["variant"], []).append(run.get("metrics", {}))

    summary = {}
    for variant, metrics_list in by_variant.items():
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                totals[key] = totals.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1
        means = {key: totals[key] / counts[key] for key in totals}
        summary[variant] = {"n_runs": len(metrics_list), "totals": totals, "means": means}
    return summary
