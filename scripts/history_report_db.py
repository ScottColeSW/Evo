"""Database sections of the Evo history report (2026-10-04): reads every snapshot in logs/board_history.db once and prints markdown.

Usage: python scripts/history_report_db.py > part2.md   (UTF-8; takes a few minutes for a multi-GB database)

A "tribe life" is one (run, tribe name) series. Only lives of at least 100 cycles count unless a section says otherwise, because the
many short runs are start-up and test runs. Periods match scripts/history_report.py.
"""
import collections
import json
import sqlite3
import statistics
import sys

DB = "logs/board_history.db"
PERIODS = [("Aug 30 to Sep 10", "run_20260830", "run_20260911"), ("Sep 11 to Sep 18", "run_20260911", "run_20260919"),
           ("Sep 19 to Oct 3 midday", "run_20260919", "run_20261003_17"), ("Oct 3 evening on (50-cycle floor in place)", "run_20261003_17", "run_20261005")]


def in_period(run_name, period):
    return period[1] <= run_name < period[2]
MIN_CYCLES = 100
CHECKPOINTS = (25, 50, 100, 200, 300, 400, 600)


def period_of(run_id):
    for p in PERIODS:
        if in_period(run_id, p):
            return p[0]
    return "other"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def med(values):
    return f"{statistics.median(values):.0f}" if values else "n/a"


def collect():
    from backend.eras import ERAS
    era_order = [e.key for e in ERAS]
    lives = {}
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    rows = conn.execute("select run_id, cycle, snapshot_json from board_snapshots order by run_id, cycle")
    for run_id, cycle, js in rows:
        try:
            tribes = json.loads(js).get("tribes") or {}
        except ValueError:
            continue
        for t in tribes.values():
            life = lives.setdefault((run_id, t.get("name")), {
                "model": t.get("model"), "first": cycle, "last": cycle, "eras": [], "pop": {}, "max_pop": 0, "firsts": {},
                "trade_first": None, "war_first": None, "ally_first": None, "actions": collections.defaultdict(collections.Counter),
                "fame": {}})
            life["last"] = cycle
            era = t.get("era")
            if not life["eras"] or life["eras"][-1][0] != era:
                life["eras"].append((era, cycle))
            pop = t.get("population") or 0
            life["max_pop"] = max(life["max_pop"], pop)
            for cp in CHECKPOINTS:
                if cycle == cp:
                    life["pop"][cp] = pop
                    life["fame"][cp] = t.get("fame")
            for key, value in t.items():
                if key.endswith("_built") or key in ("long_houses_built", "warehouses_built"):
                    if value and key not in life["firsts"]:
                        life["firsts"][key] = cycle
            if t.get("trades_completed") and life["trade_first"] is None:
                life["trade_first"] = cycle
            stances = set((t.get("stance_toward") or {}).values())
            if "WAR" in stances and life["war_first"] is None:
                life["war_first"] = cycle
            if "ALLIED" in stances and life["ally_first"] is None:
                life["ally_first"] = cycle
            if t.get("last_action"):
                life["actions"][era][t["last_action"]] += 1
    conn.close()
    return lives, era_order


def long_lives(lives):
    return {k: v for k, v in lives.items() if v["last"] - v["first"] + 1 >= MIN_CYCLES and k[1] and "Test" not in k[1]}


def era_section(lives, era_order):
    out = ["## Eras, from the snapshots (tribe lives of at least 100 cycles)", ""]
    out.append("Exact cycle of each era change. Dwell counts only eras the tribe left (the era a run ended in is unfinished). 'Reached' is the "
               "share of tribe lives that ever entered the era.\n")
    for label, _lo, _hi in PERIODS:
        part = {k: v for k, v in lives.items() if period_of(k[0]) == label}
        if not part:
            continue
        rows = []
        for era in era_order:
            entered, dwell = [], []
            for life in part.values():
                seq = life["eras"]
                for i, (e, start) in enumerate(seq):
                    if e == era:
                        entered.append(start)
                        if i + 1 < len(seq):
                            dwell.append(seq[i + 1][1] - start)
            if entered:
                rows.append((era, f"{len(entered) / len(part):.0%}", med(entered), med(dwell), f"{min(dwell)}" if dwell else "n/a", len(dwell)))
        out += [f"**{label}** ({len(part)} tribe lives)", "",
                md_table(["Era", "Reached", "Median first cycle", "Median dwell", "Shortest dwell", "Completed stays"], rows), ""]
    return "\n".join(out)


def population_section(lives):
    out = ["## Population over time", "",
           "Median population at fixed cycles, among tribe lives that were still going at that cycle (n in brackets), and the largest "
           "population each life reached.", ""]
    rows = []
    for label, _lo, _hi in PERIODS:
        part = [v for k, v in lives.items() if period_of(k[0]) == label]
        cells = []
        for cp in CHECKPOINTS:
            vals = [v["pop"][cp] for v in part if cp in v["pop"]]
            cells.append(f"{med(vals)} ({len(vals)})" if vals else "n/a")
        peaks = sorted(v["max_pop"] for v in part)
        cells.append(f"median {med(peaks)}, 90th pct {peaks[int(.9 * (len(peaks) - 1))] if peaks else 'n/a'}, max {peaks[-1] if peaks else 'n/a'}")
        rows.append((label, *cells))
    out += [md_table(["Period", *[f"cycle {cp}" for cp in CHECKPOINTS], "Peak population"], rows), ""]
    return "\n".join(out)


def structures_section(lives):
    out = ["## When structures first appear", "",
           "Median first cycle at which each structure existed, and the share of tribe lives that ever built it (all periods pooled; "
           "split would be too thin for the rarer ones).", ""]
    names = collections.Counter(key for v in lives.values() for key in v["firsts"])
    rows = []
    for key, _ in names.most_common():
        cycles = [v["firsts"][key] for v in lives.values() if key in v["firsts"]]
        if len(cycles) / len(lives) < 0.01:
            continue
        rows.append((key.replace("_built", "").replace("_", " "), med(cycles), f"{len(cycles) / len(lives):.0%}"))
    out += [md_table(["Structure", "Median first cycle", "Share of lives that built it"], rows), ""]
    return "\n".join(out)


def diplomacy_section(lives):
    out = ["## Trade, war and alliance", ""]
    rows = []
    for label, _lo, _hi in PERIODS:
        part = [v for k, v in lives.items() if period_of(k[0]) == label]
        if not part:
            continue
        def cell(field):
            vals = [v[field] for v in part if v[field] is not None]
            return f"{med(vals)} ({len(vals) / len(part):.0%} ever)"
        rows.append((label, len(part), cell("trade_first"), cell("war_first"), cell("ally_first")))
    out += [md_table(["Period", "Tribe lives", "First trade (median cycle)", "First war stance", "First alliance"], rows), ""]
    return "\n".join(out)


def actions_section(lives, era_order):
    out = ["## What tribes chose, by era", "",
           "Share of tribe-cycles spent on each action (the action chosen that cycle), top six per era, all periods pooled.", ""]
    for era in era_order:
        total = collections.Counter()
        for v in lives.values():
            total.update(v["actions"].get(era, {}))
        n = sum(total.values())
        if n < 200:
            continue
        out.append(f"- **{era}** ({n} tribe-cycles): " + ", ".join(f"{a} {c / n:.0%}" for a, c in total.most_common(6)))
    out.append("")
    return "\n".join(out)


def slot_section(lives):
    out = ["## Early stall by slot and model", "",
           "Time spent in the first era (primitive_dawn) before the tribe left it, by tribe name and model, tribe lives that did leave it. "
           "Tribes that never left are counted separately.", ""]
    groups = collections.defaultdict(lambda: {"dwell": [], "never": 0})
    for (run, name), v in lives.items():
        key = (name if name in ("Tribe 1", "Tribe 2") else "other names", v["model"])
        seq = v["eras"]
        if seq and seq[0][0] == "primitive_dawn" and len(seq) > 1:
            groups[key]["dwell"].append(seq[1][1] - seq[0][1])
        elif seq and seq[0][0] == "primitive_dawn":
            groups[key]["never"] += 1
    rows = [(k[0], k[1], len(g["dwell"]), med(g["dwell"]), g["never"]) for k, g in sorted(groups.items(), key=lambda kv: -(len(kv[1]["dwell"]) + kv[1]["never"])) if len(g["dwell"]) + g["never"] >= 3]
    out += [md_table(["Tribe", "Model", "Left the first era", "Median cycles in it", "Never left"], rows), ""]
    return "\n".join(out)


def main():
    sys.path.insert(0, ".")
    sys.stdout.reconfigure(encoding="utf-8")
    all_lives, era_order = collect()
    lives = long_lives(all_lives)
    print(f"_Database: {len(all_lives)} tribe lives in all, {len(lives)} of at least {MIN_CYCLES} cycles used below._\n")
    for section in (era_section(lives, era_order), population_section(lives), structures_section(lives), diplomacy_section(lives),
                    actions_section(lives, era_order), slot_section(lives)):
        print(section)


if __name__ == "__main__":
    main()
