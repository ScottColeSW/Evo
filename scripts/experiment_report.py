"""Prints a human-readable comparison table across every A/B test logged in
logs/experiments.jsonl (backend/experiment_log.py) -- the CLI equivalent of the
/api/experiments route, for a quick look without spinning up the server.
"""
import sys

sys.path.insert(0, ".")

from backend.experiment_log import read_all_experiment_runs, summarize_experiment


def main():
    runs = read_all_experiment_runs()
    if not runs:
        print("No experiments logged yet.")
        return

    experiment_names = sorted({r["experiment"] for r in runs})
    for name in experiment_names:
        print(f"\n=== {name} ===")
        summary = summarize_experiment(name, runs)
        for variant, stats in summary.items():
            print(f"  {variant} (n={stats['n_runs']} runs)")
            for key, total in stats["totals"].items():
                mean = stats["means"][key]
                print(f"      {key}: total={total}, mean={mean:.2f}")


if __name__ == "__main__":
    main()
