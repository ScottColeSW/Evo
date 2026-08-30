"""A/B test #4: does a proactive wildlife-sighting fact (backend/simulation.py's
Simulation._build_visible_entities, new this session) actually move HUNT_DEER pick
rate, or was the real problem never "the tribe doesn't know game exists here" in the
first place?

Previously the only fact touching game was resource-depletion scarcity, which only
ever reports *past* harvesting -- a tribe standing on a pristine forest tile got zero
signal that hunting was even an option worth considering before it had already tried.
The new sighting fact (config.GAME_SIGHTING_CHANCE_BASE, config.GAME_SIGHTING_RADIUS)
gives an occasional, honest, named encounter ("wildlife sighting: signs of deer
nearby") scaled by real nearby game yield. HUNT_DEER's name and description are left
completely unchanged in both variants, isolating this one variable.

Same model, spawn point, and cycle/run counts as the previous three tests. Throwaway
analysis script.
"""
import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")

from backend import config
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "hunt_deer_sighting_fact"
MODEL = "qwen2.5:3b"
CYCLES_PER_RUN = 100
RUNS_PER_VARIANT = 3

REAL_SIGHTING_CHANCE = config.GAME_SIGHTING_CHANCE_BASE  # 0.3 as shipped


async def run_once(variant_label: str, run_id: int) -> dict:
    sim = await Simulation.create([{"name": "Test Tribe", "model": MODEL}])
    tribe = next(iter(sim.tribes.values()))
    for _ in range(CYCLES_PER_RUN):
        await sim.step()
        if tribe.extinct:
            break

    action_counts = collections.Counter()
    sightings_seen = 0
    for entry in tribe.history:
        if entry.startswith("["):
            try:
                action = entry.split("]", 1)[1].strip().split(":")[0]
                action_counts[action] += 1
            except Exception:
                pass

    return {
        "variant": variant_label,
        "run_id": run_id,
        "cycles_run": sim.cycle,
        "extinct": tribe.extinct,
        "final_population": tribe.population,
        "max_population": tribe.max_population,
        "hunt_deer_count": action_counts.get("HUNT_DEER", 0),
        "total_actions": sum(action_counts.values()),
        "action_counts": dict(action_counts),
    }


def _log_run(label, notes, result):
    record_experiment_run(
        EXPERIMENT_NAME, label, MODEL,
        metrics={
            "hunt_deer_count": result["hunt_deer_count"],
            "total_actions": result["total_actions"],
            "cycles_run": result["cycles_run"],
            "extinct": result["extinct"],
            "final_population": result["final_population"],
            "max_population": result["max_population"],
        },
        run_id=result["run_id"], notes=notes,
    )


async def main():
    all_results = []

    config.GAME_SIGHTING_CHANCE_BASE = 0.0
    print("=== VARIANT: no_sighting (baseline, feature disabled) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("no_sighting", i)
        all_results.append(result)
        _log_run("no_sighting", "GAME_SIGHTING_CHANCE_BASE=0.0 (feature off)", result)
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"pop={result['final_population']} (peak {result['max_population']}) "
              f"HUNT_DEER={result['hunt_deer_count']}/{result['total_actions']} "
              f"actions={result['action_counts']}", flush=True)

    config.GAME_SIGHTING_CHANCE_BASE = REAL_SIGHTING_CHANCE
    print(f"\n=== VARIANT: sighting_on (GAME_SIGHTING_CHANCE_BASE={REAL_SIGHTING_CHANCE}) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("sighting_on", i)
        all_results.append(result)
        _log_run("sighting_on", f"GAME_SIGHTING_CHANCE_BASE={REAL_SIGHTING_CHANCE} (shipped default)", result)
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"pop={result['final_population']} (peak {result['max_population']}) "
              f"HUNT_DEER={result['hunt_deer_count']}/{result['total_actions']} "
              f"actions={result['action_counts']}", flush=True)

    print("\n=== SUMMARY ===")
    for label in ("no_sighting", "sighting_on"):
        rows = [r for r in all_results if r["variant"] == label]
        total_hunt = sum(r["hunt_deer_count"] for r in rows)
        total_actions = sum(r["total_actions"] for r in rows)
        survived = sum(1 for r in rows if not r["extinct"])
        pct = (total_hunt / total_actions * 100) if total_actions else 0.0
        print(f"  {label}: HUNT_DEER={total_hunt}/{total_actions} ({pct:.1f}%), "
              f"survived {survived}/{len(rows)} runs")

    with open("scripts/ab_test_sighting_fact_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved full detail to scripts/ab_test_sighting_fact_results.json")


if __name__ == "__main__":
    asyncio.run(main())
