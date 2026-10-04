"""Deletes the raw development data that docs/HISTORY-REPORT.md replaces (2026-10-04), after the report was written and pushed.

Dry run by default: it prints what it would delete and what it keeps, and deletes nothing. Pass --yes to delete.

Deleted:
  - logs/board_history.db, the board-snapshot database (about 5 GB). The server recreates an empty one on the next run.
  - every logs/run_*.jsonl chronicle older than KEEP_FROM (the 502 logs the report was built from, minus the newest).
Kept:
  - run logs from KEEP_FROM on (the only ones with the structured analysis records, still being read: score_run.py).
  - logs/scoreboard.jsonl, logs/experiments.jsonl, logs/benchmark_results.db (small, and still appended to by the code).

Refuses to run if the database was written in the last 10 minutes (a server is probably running).
"""
import glob
import os
import sys
import time

KEEP_FROM = "run_20261003_17"  # run names sort by date and time; this is the evening the 50-cycle floor came in
LOGS = "logs"


def size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def main(delete: bool) -> None:
    db = os.path.join(LOGS, "board_history.db")
    if os.path.exists(db) and time.time() - os.path.getmtime(db) < 600:
        sys.exit("board_history.db was written in the last 10 minutes; stop the server first.")
    runs = sorted(glob.glob(os.path.join(LOGS, "run_*.jsonl")))
    doomed = [p for p in runs if os.path.basename(p)[:-6] < KEEP_FROM]
    kept = [p for p in runs if p not in doomed]
    targets = ([db] if os.path.exists(db) else []) + doomed
    total = sum(size(p) for p in targets)
    print(f"{'DELETING' if delete else 'DRY RUN, would delete'}: {len(targets)} files, {total / 1e9:.2f} GB")
    print(f"  {db}: {size(db) / 1e9:.2f} GB")
    print(f"  {len(doomed)} chronicle logs older than {KEEP_FROM}: {sum(size(p) for p in doomed) / 1e6:.1f} MB")
    print("keeping:")
    for p in kept:
        print(f"  {p} ({size(p) / 1e3:.0f} KB)")
    for name in ("scoreboard.jsonl", "experiments.jsonl", "benchmark_results.db"):
        path = os.path.join(LOGS, name)
        if os.path.exists(path):
            print(f"  {path} ({size(path) / 1e3:.0f} KB)")
    if not delete:
        print("\nNothing deleted. Run again with --yes to delete.")
        return
    for p in targets:
        os.remove(p)
    print("done")


if __name__ == "__main__":
    main("--yes" in sys.argv)
