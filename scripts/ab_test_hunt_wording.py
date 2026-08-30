"""A/B test: does HUNT_DEER's "Attempt to harvest" framing (vs. the confident
"Harvest" wording every other gather action uses) suppress how often models
choose it?

Single-tribe headless runs (no rival, no RAID/TRADE noise), same model across
both variants, only the ACTION_DESCRIPTIONS["HUNT_DEER"] string changes. Counts
real HUNT_DEER picks from the persisted event log wording, plus survival outcome,
per variant. Throwaway analysis script -- see backend/actions.py:321 for the
live description this is testing against.
"""
import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")

from backend import actions
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "hunt_deer_wording"


MODEL = "qwen2.5:3b"
CYCLES_PER_RUN = 100
RUNS_PER_VARIANT = 3

BASELINE = actions.ACTION_DESCRIPTIONS["HUNT_DEER"]
CONFIDENT = BASELINE.replace("Attempt to harvest", "Harvest")

VARIANTS = {"baseline": BASELINE, "confident": CONFIDENT}


async def run_once(variant_label: str, run_id: int) -> dict:
    sim = await Simulation.create([{"name": "Test Tribe", "model": MODEL}])
    tribe = next(iter(sim.tribes.values()))
    for _ in range(CYCLES_PER_RUN):
        await sim.step()
        if tribe.extinct:
            break

    action_counts = collections.Counter()
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


async def main():
    all_results = []
    for label, description in VARIANTS.items():
        actions.ACTION_DESCRIPTIONS["HUNT_DEER"] = description
        print(f"\n=== VARIANT: {label} ===")
        print(f"  description: {description}")
        for i in range(RUNS_PER_VARIANT):
            result = await run_once(label, i)
            all_results.append(result)
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
                run_id=i, notes=description,
            )
            print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
                  f"pop={result['final_population']} (peak {result['max_population']}) "
                  f"HUNT_DEER={result['hunt_deer_count']}/{result['total_actions']} "
                  f"actions={result['action_counts']}", flush=True)

    actions.ACTION_DESCRIPTIONS["HUNT_DEER"] = BASELINE  # restore

    print("\n=== SUMMARY ===")
    for label in VARIANTS:
        rows = [r for r in all_results if r["variant"] == label]
        total_hunt = sum(r["hunt_deer_count"] for r in rows)
        total_actions = sum(r["total_actions"] for r in rows)
        survived = sum(1 for r in rows if not r["extinct"])
        pct = (total_hunt / total_actions * 100) if total_actions else 0.0
        print(f"  {label}: HUNT_DEER={total_hunt}/{total_actions} ({pct:.1f}%), "
              f"survived {survived}/{len(rows)} runs")

    with open("scripts/ab_test_hunt_wording_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved full detail to scripts/ab_test_hunt_wording_results.json")


if __name__ == "__main__":
    asyncio.run(main())
