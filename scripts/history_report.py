"""Builds docs/HISTORY-REPORT.md from Evo's development logs (2026-10-04), so the raw logs can be deleted once the report is kept.

Sections read logs/scoreboard.jsonl and every logs/run_*.jsonl chronicle (plain message lines). Database sections (era dwell, population
curves, action mix, stance timing) live in scripts/history_report_db.py and are appended to the same report.

Usage: python scripts/history_report.py > docs/HISTORY-REPORT.md   (UTF-8)

Caveat stated in the report itself: the code changed constantly from 2026-08-30 to 2026-10-04, so rows pool runs made under different
rules. Where it matters the tables split by period.
"""
import collections
import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone

LOGS = "logs"
PERIODS = [("Aug 30 to Sep 10", "20260830", "20260910"), ("Sep 11 to Sep 18", "20260911", "20260918"),
           ("Sep 19 to Oct 4", "20260919", "20261004")]


def period_of(run_name: str) -> str:
    day = run_name.split("_")[1]
    for label, lo, hi in PERIODS:
        if lo <= day <= hi:
            return label
    return "other"


def q(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))] if values else None


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def load_chronicles():
    runs = {}
    for path in sorted(glob.glob(os.path.join(LOGS, "run_*.jsonl"))):
        name = os.path.basename(path)[:-6]
        msgs, last = [], 0
        for line in open(path, encoding="utf-8", errors="ignore"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            last = max(last, r.get("cycle") or 0)
            if not r.get("kind"):
                msgs.append((r.get("cycle") or 0, r.get("tribe"), r.get("message", "")))
        runs[name] = {"cycles": last, "msgs": msgs}
    return runs


def corpus(runs):
    out = ["## The corpus", ""]
    lengths = [r["cycles"] for r in runs.values()]
    out.append(f"{len(runs)} chronicle logs from {min(runs)[4:12]} to {max(runs)[4:12]}. Cycles per run: median {statistics.median(lengths)}, "
               f"75th percentile {q(lengths, .75)}, longest {max(lengths)}. {sum(1 for x in lengths if x >= 100)} runs reached 100 cycles, "
               f"{sum(1 for x in lengths if x >= 400)} reached 400, {sum(1 for x in lengths if x < 30)} ended under 30 (mostly test and "
               "start-up runs).")
    rows = []
    for label, lo, hi in PERIODS:
        part = [r["cycles"] for n, r in runs.items() if lo <= n.split("_")[1] <= hi]
        rows.append((label, len(part), statistics.median(part) if part else "", max(part) if part else ""))
    out += ["", md_table(["Period", "Runs", "Median cycles", "Longest"], rows), ""]
    return "\n".join(out)


def scoreboard():
    rows = [json.loads(line) for line in open(os.path.join(LOGS, "scoreboard.jsonl"), encoding="utf-8") if line.strip()]
    real = [r for r in rows if r.get("tribe_name") != "Test Tribe"]
    out = ["## How tribes ended (scoreboard.jsonl)", "",
           f"{len(real)} tribe results (test tribes excluded). Each is a tribe's whole life summary, recorded at extinction.", ""]
    causes = collections.Counter(r["cause_of_death"] for r in real)
    out += [md_table(["Cause of death", "Tribes", "Share"], [(c, n, f"{n / len(real):.0%}") for c, n in causes.most_common()]), ""]
    by_model = collections.defaultdict(list)
    for r in real:
        by_model[r["model"]].append(r)
    out += [md_table(["Model", "Tribes", "Median cycles survived", "Longest", "Median max population", "Most common cause"],
                     [(m, len(v), statistics.median(x["cycles_survived"] for x in v), max(x["cycles_survived"] for x in v),
                       statistics.median(x["max_population"] for x in v), collections.Counter(x["cause_of_death"] for x in v).most_common(1)[0][0])
                      for m, v in sorted(by_model.items(), key=lambda kv: -len(kv[1]))]), ""]
    eras = collections.Counter(r["era_reached"] for r in real)
    out += ["Era reached at extinction: " + ", ".join(f"{e} {n}" for e, n in eras.most_common()) + ".", ""]
    trades = [r["trades_completed"] for r in real]
    out += [f"Trades completed per tribe life: median {statistics.median(trades)}, {sum(1 for t in trades if t == 0)} of {len(real)} "
            f"tribes never traded. Raids won {sum(r['raids_won'] for r in real)}, lost {sum(r['raids_lost'] for r in real)}, "
            f"defended {sum(r['raids_defended'] for r in real)}.", ""]
    return "\n".join(out)


FIRSTS = [
    ("discovers fire", r"discovers fire"),
    ("first raiders sighted", r"raiders have been spotted riding in"),
    ("first raid repelled", r"raiders were spotted approaching camp and repelled"),
    ("first raid struck", r"raiders struck the camp"),
    ("celebrates finding water", r"celebrates the discovery of water"),
    ("a road takes shape", r"celebrates a real road taking shape"),
    ("forges the Tribal Synapse", r"has forged the Tribal Synapse"),
    ("crosses into the Cognitive Horizon", r"crosses into the Cognitive Horizon"),
    ("first extinction", r"has gone extinct"),
]


def timing(runs):
    out = ["## When things first happen (runs of at least 100 cycles)", "",
           "Cycle of the first matching chronicle line per tribe, so a run with two tribes contributes two. Medians hide the code changes "
           "of the period; the three columns split them.", ""]
    long_runs = {n: r for n, r in runs.items() if r["cycles"] >= 100}
    rows = []
    for label, pattern in FIRSTS:
        cells = []
        for plabel, lo, hi in PERIODS:
            firsts = []
            for n, r in long_runs.items():
                if not (lo <= n.split("_")[1] <= hi):
                    continue
                seen = {}
                for cycle, tribe, msg in r["msgs"]:
                    if tribe not in seen and re.search(pattern, msg):
                        seen[tribe] = cycle
                firsts += list(seen.values())
            cells.append(f"{statistics.median(firsts):.0f} (n={len(firsts)})" if firsts else "none")
        rows.append((label, *cells))
    out += [md_table(["First occurrence", *[p[0] for p in PERIODS]], rows), ""]
    return "\n".join(out)


COUNTS = [
    ("starvation deaths", r"^starvation claimed lives"), ("thirst deaths", r"^thirst claimed lives"),
    ("river drownings", r"current pulled someone under"), ("volcano deaths", r"volcano's toxic fumes"),
    ("rip current deaths", r"rip current off the shoals"), ("cliff deaths", r"cliffs near .* claimed a life"),
    ("raids repelled", r"raiders were spotted approaching camp and repelled"), ("raids struck", r"raiders struck the camp"),
    ("raider camps destroyed", r"patrol the raider camp .* is destroyed"), ("old per-cycle overcrowding culls", r"culled back to what the land can support"),
    ("flock lost for lack of feed", r"part of the flock is lost for lack of feed"),
]


def counts(runs):
    out = ["## What kills and what threatens (all runs)", "",
           "Chronicle line counts, per 1,000 tribe-cycles so runs of different length compare. Tribe-cycles are approximated as run cycles "
           "times two (most runs had two tribes).", ""]
    total_cycles = sum(r["cycles"] for r in runs.values()) * 2
    rows = []
    for label, pattern in COUNTS:
        n = sum(1 for r in runs.values() for _, _, m in r["msgs"] if re.search(pattern, m))
        rows.append((label, n, f"{n / total_cycles * 1000:.1f}"))
    out += [md_table(["Event", "Lines", "Per 1,000 tribe-cycles"], rows), ""]
    repelled = next(r[1] for r in rows if r[0] == "raids repelled")
    struck = next(r[1] for r in rows if r[0] == "raids struck")
    out += [f"Raid defense: {repelled} repelled against {struck} that struck, a {repelled / (repelled + struck):.0%} repel rate.", ""]
    return "\n".join(out)


def repetition_and_waste(runs):
    out = ["## Repetition and wasted choices", ""]
    guard = collections.Counter()
    for r in runs.values():
        for _, _, m in r["msgs"]:
            match = re.search(r"has repeated (\w+) too many times in a row", m)
            if match:
                guard[match.group(1)] += 1
    total_cycles = sum(r["cycles"] for r in runs.values()) * 2
    out += [f"The repetition guard (the 'Historian insists on a different choice' line) fired {sum(guard.values())} times, "
            f"{sum(guard.values()) / total_cycles * 1000:.1f} per 1,000 tribe-cycles. By action:", "",
            md_table(["Action repeated", "Times the guard fired"], guard.most_common(8)), ""]
    waste = [("item stores already full (an item action with nowhere to put it)", r"item stores are already full"),
             ("model reply the game could not read as a decision", r"unrecognized decision text"),
             ("celebration reason that is really a routine action log", r"celebration for a fresh discovery: At \(")]
    rows = [(label, sum(1 for r in runs.values() for _, _, m in r["msgs"] if re.search(p, m))) for label, p in waste]
    out += [md_table(["Waste or noise", "Lines"], rows), "",
            "The last row is a real finding: a celebration's stated reason was a per-turn action log (for example 'At (x,y) in river, chose "
            "SCOUT'), because those logs were stored at high weight as memories. They are no longer filed as Library candidates "
            "(2026-10-04), but the celebration wording still draws on them.", ""]
    return "\n".join(out)


def language(runs):
    out = ["## Language: what tribes shouted", "",
           "Celebration shouts in the chronicle (the quoted word before the exclamation mark). On 2026-09-16 the example words in the prompt "
           "were rotated per tribe to stop every tribe echoing KRA-ZUL, MEE-LO and VASH-TA. This checks whether it worked.", ""]
    seed = {"KRA-ZUL", "MEE-LO", "VASH-TA"}
    rows = []
    shouts_by_period = collections.defaultdict(collections.Counter)
    for n, r in runs.items():
        for _, _, m in r["msgs"]:
            for s in re.findall(r'"([A-Za-z\- ]+)!"', m):
                for word in s.upper().split():
                    shouts_by_period[("before the rotation" if n.split("_")[1] <= "20260915" else "after the rotation")][word] += 1
    for label in ("before the rotation", "after the rotation"):
        c = shouts_by_period[label]
        total = sum(c.values()) or 1
        rows.append((label, sum(c.values()), len(c), f"{sum(c[w] for w in seed) / total:.0%}", ", ".join(f"{w} {n}" for w, n in c.most_common(5))))
    out += [md_table(["Period", "Words shouted", "Distinct words", "Share that are the three old seed words", "Most common"], rows), ""]
    return "\n".join(out)


def main():
    runs = load_chronicles()
    print("# Evo development history, distilled\n")
    print(f"Generated {datetime.now(timezone.utc):%Y-%m-%d} from the chronicle logs and scoreboard before they were deleted. "
          "The code changed constantly from 2026-08-30 to 2026-10-04, so rows pool runs made under different rules; tables split by "
          "period where it matters. Method: `scripts/history_report.py` and `scripts/history_report_db.py`.\n")
    for section in (corpus(runs), scoreboard(), timing(runs), counts(runs), repetition_and_waste(runs), language(runs)):
        print(section)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
