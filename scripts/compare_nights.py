"""Compare what a night costs and what it produces across run logs (2026-10-10).

  python scripts/compare_nights.py logs/run_A.jsonl logs/run_B.jsonl [...]

Built for the REFLECTION_MODEL comparison: one run with the default dedicated reviewer (gemma2:2b) and one with REFLECTION_MODEL=tribe (each tribe's own model reflects).
Per run it prints the effective reflection setting (run_config), the tribe models, how long a reflection call, its embedding, the whole model phase and the apply step took
(the night_timing and night_phase_timing records), how long a normal cycle and a night cycle took, and what the reflections produced: how many nights stored a
reflection, how the judge related each to what was held (new, reinforces, coexists, contradicts), and how long the reflection text was. Runs logged before the timing records
existed show "no timing records". Read-only; it never writes anything.
"""
import json
import statistics
import sys
from collections import Counter


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered), "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], "max": ordered[-1]}


def summarize(path, night_every=30):
    config = {}
    reflect, embed, phase, apply_total = [], [], [], []
    reflect_by_model = {}
    relations = Counter()
    text_lengths = []
    nights_with_memory = set()
    first_ts = {}
    for record in _read(path):
        kind = record.get("kind")
        data = record.get("data") or {}
        cycle = record.get("cycle")
        if cycle and record.get("ts") and cycle not in first_ts:
            first_ts[cycle] = record["ts"]
        if kind == "run_config":
            config = data
        elif kind == "night_timing":
            reflect.append(data.get("reflect_s", 0.0))
            embed.append(data.get("embed_s", 0.0))
            reflect_by_model.setdefault(data.get("model", "?"), []).append(data.get("reflect_s", 0.0))
        elif kind == "night_phase_timing":
            phase.append(data.get("model_phase_s", 0.0))
            apply_total.append(sum((data.get("apply_s") or {}).values()))
        elif kind == "reflection_memory":
            relations[data.get("relation", "?")] += 1
            text_lengths.append(len(data.get("text") or ""))
            nights_with_memory.add((cycle, record.get("tribe")))
    cycles = sorted(first_ts)
    durations = {a: first_ts[b] - first_ts[a] for a, b in zip(cycles, cycles[1:])}
    nights = [seconds for cycle, seconds in durations.items() if cycle % night_every == 0]
    normal = [seconds for cycle, seconds in durations.items() if cycle % night_every != 0]
    return {
        "path": path,
        "reflection_setting": config.get("reflection_model", "(not recorded)"),
        "tribe_models": sorted({t.get("model") for t in config.get("tribes", [])}),
        "code": config.get("git_commit"),
        "reflect_call_s": _stats(reflect), "embed_s": _stats(embed), "model_phase_s": _stats(phase), "apply_s": _stats(apply_total),
        "reflect_by_model": {model: _stats(values) for model, values in reflect_by_model.items()},
        "normal_cycle_s": _stats(normal), "night_cycle_s": _stats(nights),
        "reflections_stored": len(nights_with_memory), "relations": dict(relations),
        "reflection_text_chars": _stats(text_lengths),
    }


def _line(label, stats):
    if not stats:
        return f"  {label:24s} no timing records"
    return f"  {label:24s} n {stats['n']:3d}   median {stats['median']:6.1f}   p90 {stats['p90']:6.1f}   max {stats['max']:6.1f}"


def main(paths):
    for path in paths:
        s = summarize(path)
        print(f"\n== {s['path']}")
        print(f"  reflection setting: {s['reflection_setting']}   tribe models: {', '.join(m or '?' for m in s['tribe_models'])}   code: {s['code']}")
        print(_line("reflection call (s)", s["reflect_call_s"]))
        for model, stats in s["reflect_by_model"].items():
            print(_line(f"  on {model}", stats))
        print(_line("embedding (s)", s["embed_s"]))
        print(_line("whole model phase (s)", s["model_phase_s"]))
        print(_line("apply incl. judge (s)", s["apply_s"]))
        print(_line("normal cycle (s)", s["normal_cycle_s"]))
        print(_line("night cycle (s)", s["night_cycle_s"]))
        print(f"  reflections stored: {s['reflections_stored']}   judge relations: {s['relations']}")
        print(_line("reflection text (chars)", s["reflection_text_chars"]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
