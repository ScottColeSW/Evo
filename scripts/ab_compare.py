import collections
import json
import statistics as st
import sys

path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else None
R = [r for r in json.load(open(path, encoding="utf-8")) if (mode is None or r.get("mode") == mode) and not r.get("invalid")]
print(f"{path} mode={mode}: {len(R)} runs, per arm {dict(collections.Counter(r['variant'] for r in R))}")
by = collections.defaultdict(list)
for r in R:
    for name, t in r["tribes"].items():
        by[r["variant"]].append((r["seed"], name, t))
for v in ("full", "compact"):
    rows = by[v]
    if not rows:
        continue
    print(f"\n== {v}: {len(rows)} tribe-runs")
    print("  mean prompt chars :", round(st.mean(t["mean_prompt_chars"] for _, _, t in rows)))
    print("  mean latency ms   :", round(st.mean(t["mean_latency_ms"] for _, _, t in rows)))
    print("  final population  :", round(st.mean(t["final_population"] for _, _, t in rows)), "| min", min(t["final_population"] for _, _, t in rows), "| extinct:", sum(t["extinct"] for _, _, t in rows))
    print("  final eras        :", dict(collections.Counter(t["final_era"] for _, _, t in rows)))
    mix = collections.Counter()
    for _, _, t in rows:
        for a, n in t["action_mix"].items():
            mix[a] += n
    tot = sum(mix.values())
    print("  action mix (share):", [(a, f"{n * 100 / tot:.0f}%") for a, n in mix.most_common(9)])
    print("  built during run  :", dict(collections.Counter(b for _, _, t in rows for b in t["built_during_run"])))
    ds = [r["decision_stats"] for r in R if r["variant"] == v]
    print("  no-effect share   :", round(st.mean(d["no_effect_share"] for d in ds), 3), "| mean menu size:", round(st.mean(d["mean_menu_size"] for d in ds), 1))
    firsts = collections.defaultdict(list)
    for _, _, t in rows:
        for era, c in t["era_first_cycle"].items():
            firsts[era].append(c)
    print("  era first-entry cycle (mean, n):", {e: (round(st.mean(c)), len(c)) for e, c in firsts.items()})
print("\nper run:")
for r in sorted(R, key=lambda r: (r["seed"], r["variant"])):
    lat = [t["mean_latency_ms"] for t in r["tribes"].values()]
    ch = [t["mean_prompt_chars"] for t in r["tribes"].values()]
    pops = [t["final_population"] for t in r["tribes"].values()]
    print(f"  seed {r['seed']} {r['variant']:8s} latency {lat[0]} ms, prompt {ch} chars, pop {pops}, {r['seconds']} s")
