"""Scores one run log against every prediction and watch item written down so far, in one pass (2026-10-04).

Usage: python scripts/score_run.py logs/run_XXXX.jsonl

Reads only the structured records (kind and data) that RunEventLog.record_data writes, so a log made with RUN_DATA_LOG=off scores
nothing. A section with no records says so; it does not guess. Each section states the prediction it scores and its falsifier where
one was written down, so a reader does not have to remember them.
"""
import collections
import json
import statistics
import sys


def load(path):
    by_kind = collections.defaultdict(list)
    last_cycle = 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        last_cycle = max(last_cycle, r.get("cycle") or 0)
        if r.get("kind"):
            by_kind[r["kind"]].append(r)
    return by_kind, last_cycle


def head(title):
    print(f"\n== {title}")


def library(k):
    head("Library shadow (docs/LIBRARY-PALIMPSEST-SPEC.md). The old prediction (repeats above half) is retired: since 2026-10-04 RESEARCH files "
         "only patterns not already held, so repeats cannot appear. Now: how does the judge treat beliefs, and is evidence kept out of it?")
    rows = k["library_shadow"]
    if not rows:
        print("no library_shadow records (judge off, or no RESEARCH, or no Library built)")
        return
    results = [x for r in rows for x in r["data"]["results"]]
    print(f"filings {len(rows)}, {len(results)} candidates")
    for source in ("belief", "evidence", None):
        part = [x for x in results if x.get("source") == source]
        if part:
            print(f"  source {source}: " + ", ".join(f"{rel} {n}" for rel, n in collections.Counter(x['relation'] for x in part).most_common()))
    judged_evidence = [x for x in results if x.get("source") == "evidence" and x["relation"] != "keyed"]
    print(f"  evidence sent to the judge (should be 0): {len(judged_evidence)}")
    collisions = [x for x in results if x["relation"] == "collides" and x.get("source") in ("belief", None)]
    print(f"  belief collisions flagged {len(collisions)}; read by hand (weak NLI scores of about 0.65 were doubtful, 0.98 plausible):")
    for x in collisions[:5]:
        print(f"    {x['text'][:110]} | {x['reason'][:50]}")


def research_menu(k):
    head("RESEARCH on the menu (night_watch)")
    rows = [r for r in k["night_watch"] if "research_offered" in r["data"]]
    if not rows:
        print("no night_watch records with a Library")
        return
    offered = sum(1 for r in rows if r["data"]["research_offered"])
    short = sum(1 for r in rows if r["data"].get("research_wood_short"))
    print(f"{len(rows)} tribe-nights with a Library; RESEARCH offered on {offered}; short of wood on {short}")
    last = {}
    for r in rows:
        last[r["tribe"]] = r["data"]["research_completed"]
    print("  research completed by the end: " + ", ".join(f"{t} {n}" for t, n in last.items()) + " (the 50% era discount needs 13)")


def trade_gate(k):
    head("Trade gate (docs/TRADE-GATE-DESIGN.md), logging only. Question: would a gate have blocked anything, and was the cull half ever exercised?")
    acts = k["peace_gate_peace_act"]
    if not k["peace_gate"] and not acts:
        print("no peace_gate records")
        return
    for tribe in sorted({r["tribe"] for r in k["peace_gate"]}):
        nights = [r["data"] for r in k["peace_gate"] if r["tribe"] == tribe]
        opened = next((d["would_open_cycle"] for d in reversed(nights) if d.get("would_open_cycle") is not None), None)
        path = next((d["would_open"] for d in nights if d["would_open"]), None)
        costs = [r["data"] for r in k["peace_gate_cost"] if r["tribe"] == tribe]
        kinds = collections.Counter(c["kind"] for c in costs)
        print(f"{tribe}: {len(nights)} nights; first cost at cycle {costs[0]['cycle'] if costs else None} {dict(kinds)}; "
              f"would first open at cycle {opened} (first path {path}); culls {kinds.get('cull', 0)}; contact nights at end "
              f"{nights[-1].get('contacts')}")
    if acts:
        blocked = sum(1 for r in acts if r["data"]["blocked"])
        first = collections.defaultdict(lambda: None)
        for r in acts:
            first[(r["tribe"], r["data"]["act"])] = first[(r["tribe"], r["data"]["act"])] or r["data"]["cycle"]
        print(f"peace acts {len(acts)}, a gate would have blocked {blocked}; first of each: {dict(first)}")
    else:
        print("no trades or alliances at all in this run")


def pacing(k, last_cycle):
    head("Pacing and population (night_watch). Guide: no era shorter than 50 cycles; watch growth against the overcrowding line.")
    rows = k["night_watch"]
    if not rows:
        print("no night_watch records")
        return
    for tribe in sorted({r["tribe"] for r in rows}):
        mine = [r for r in rows if r["tribe"] == tribe]
        dwell = {}
        for r in mine:
            dwell[r["data"]["era"]] = max(dwell.get(r["data"]["era"], 0), r["data"]["cycles_in_era"])
        pops = [(r["cycle"], r["data"]["population"], r["data"]["headroom"]) for r in mine]
        print(f"{tribe}: longest seen in each era {dwell}; population {pops[0][1]} -> {pops[-1][1]}; "
              f"least headroom to the cull line {min(p[2] for p in pops)}")
    print(f"culls in the run: {len(k['overcrowding'])}")


def tannery(k):
    head("Tannery (tannery_day). Question: does Fur scale with the herd, and does it pile up unused?")
    rows = k["tannery_day"]
    if not rows:
        print("no tannery_day records (no Tannery built)")
        return
    for tribe in sorted({r["tribe"] for r in rows}):
        mine = [r["data"] for r in rows if r["tribe"] == tribe]
        fur = [d["fur_made"] for d in mine]
        print(f"{tribe}: {len(mine)} days; herd {mine[0]['herd']} -> {mine[-1]['herd']} (max {max(d['herd'] for d in mine)}); "
              f"Fur a day median {statistics.median(fur)}, max {max(fur)}; Fur stock {mine[0]['fur_stock']} -> {mine[-1]['fur_stock']}")


def language(k, path):
    head("Language (docs/LANGUAGE-LEXICON-DESIGN.md). Baseline question: do a tribe's words tell you anything about its actions?")
    own = k["lexicon_update"]
    if not own:
        print("no lexicon_update records")
    else:
        for tribe in sorted({r["tribe"] for r in own}):
            mine = [r["data"] for r in own if r["tribe"] == tribe]
            words = collections.Counter(w["word"] for d in mine for w in d["words"])
            last = {}
            for d in mine:
                for w in d["words"]:
                    last[w["word"]] = w
            top = sorted(last.values(), key=lambda w: -w["uses"])[:4]
            print(f"{tribe}: {len(mine)} broadcasts, {len(words)} distinct words; most used "
                  + "; ".join(f"{w['word']} x{w['uses']} ({w['dominant']} {int(w['share'] * 100)}%)" for w in top))
        print("  run `python scripts/token_signal.py <log>` for the mutual-information check against shuffles")
    for kind, label in (("witnessed_cry", "war cries witnessed"), ("witnessed_celebration", "celebrations heard"),
                        ("lexicon_heard", "heard words recorded"), ("party_overheard", "party reports")):
        rows = k[kind]
        by = collections.Counter(r["tribe"] for r in rows)
        print(f"{label}: {len(rows)} {dict(by)}")
    cries = k["witnessed_cry"]
    if cries:
        print("  cries by kind: " + ", ".join(f"{kind} {n}" for kind, n in collections.Counter(r['data']['kind'] for r in cries).most_common()))


def dud_actions(k):
    head("Actions that changed nothing (decision journal). A menu should not offer an action that cannot add anything.")
    rows = k["decision"]
    if not rows:
        print("no decision records")
        return
    duds = collections.Counter()
    notes = {}
    for r in rows:
        d = r["data"]
        if not d.get("delta") and not d.get("built") and not d.get("moved"):
            duds[d["action"]] += 1
            notes.setdefault(d["action"], d.get("note") or "")
    print(f"{len(rows)} decisions, {sum(duds.values())} with no measurable effect")
    for action, n in duds.most_common(6):
        print(f"  {action} x{n}: {notes[action][:80]}")
    full = [r for r in rows if "nothing more fits" in (r["data"].get("note") or "")]
    print(f"gathers into full storage (should be 0 after 2026-10-04): {len(full)}")


def main(path):
    k, last_cycle = load(path)
    print(f"{path}: {last_cycle} cycles, " + ", ".join(f"{kind} {len(v)}" for kind, v in sorted(k.items(), key=lambda kv: -len(kv[1]))[:8]))
    library(k)
    research_menu(k)
    trade_gate(k)
    pacing(k, last_cycle)
    tannery(k)
    language(k, path)
    dud_actions(k)


if __name__ == "__main__":
    main(sys.argv[1])
