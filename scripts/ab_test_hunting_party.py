"""A/B test #5: does adding HUNTING_PARTY (a multi-day hunting expedition, see
backend/actions.py._hunting_party) alongside instant HUNT_DEER actually help a tribe,
or does it create the failure mode discussed while designing it -- a party still out
in the field while the tribe back home starves, since a catch isn't real food until
the party walks home (same rule as SCOUT)?

Tracks, in addition to the usual survival stats, how many cycles the tribe spent at
critical hunger while a hunting party (not a scouting one) was away -- the concrete,
measurable version of "staying out too long while food is needed immediately."

Same model, spawn point, and cycle/run counts as the previous four tests. Throwaway
analysis script.
"""
import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")

from backend import config
from backend import simulation as simulation_module
from backend.eras import unlocked_actions_through as _orig_unlocked_actions_through
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "hunting_party_availability"
MODEL = "qwen2.5:3b"
CYCLES_PER_RUN = 100
RUNS_PER_VARIANT = 3


def _with_hunting_party_unlocked(era_key):
    actions_set = set(_orig_unlocked_actions_through(era_key))
    actions_set.add("HUNTING_PARTY")
    return actions_set


async def run_once(variant_label: str, run_id: int) -> dict:
    sim = await Simulation.create([{"name": "Test Tribe", "model": MODEL}])
    tribe = next(iter(sim.tribes.values()))
    cycles_starving_while_hunting_party_out = 0

    for _ in range(CYCLES_PER_RUN):
        await sim.step()
        if tribe.extinct:
            break
        if (
            tribe.food <= config.HUNGER_CRITICAL_THRESHOLD
            and tribe.expedition is not None
            and tribe.expedition.get("kind") == "hunt"
        ):
            cycles_starving_while_hunting_party_out += 1

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
        "hunting_party_count": action_counts.get("HUNTING_PARTY", 0),
        "expeditions_launched": tribe.expeditions_launched,
        "expeditions_succeeded": tribe.expeditions_succeeded,
        "cycles_starving_while_hunting_party_out": cycles_starving_while_hunting_party_out,
        "total_actions": sum(action_counts.values()),
        "action_counts": dict(action_counts),
    }


def _log_run(label, notes, result):
    record_experiment_run(
        EXPERIMENT_NAME, label, MODEL,
        metrics={
            "hunt_deer_count": result["hunt_deer_count"],
            "hunting_party_count": result["hunting_party_count"],
            "expeditions_launched": result["expeditions_launched"],
            "expeditions_succeeded": result["expeditions_succeeded"],
            "cycles_starving_while_hunting_party_out": result["cycles_starving_while_hunting_party_out"],
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

    print("=== VARIANT: baseline (HUNT_DEER only, no HUNTING_PARTY) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("baseline", i)
        all_results.append(result)
        _log_run("baseline", "HUNTING_PARTY not unlocked -- current shipped behavior", result)
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"pop={result['final_population']} (peak {result['max_population']}) "
              f"HUNT_DEER={result['hunt_deer_count']} HUNTING_PARTY={result['hunting_party_count']} "
              f"starving_while_out={result['cycles_starving_while_hunting_party_out']} "
              f"actions={result['action_counts']}", flush=True)

    simulation_module.unlocked_actions_through = _with_hunting_party_unlocked
    print("\n=== VARIANT: with_hunting_party (HUNTING_PARTY unlocked alongside HUNT_DEER) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("with_hunting_party", i)
        all_results.append(result)
        _log_run("with_hunting_party", "HUNTING_PARTY unlocked alongside HUNT_DEER (test variant)", result)
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"pop={result['final_population']} (peak {result['max_population']}) "
              f"HUNT_DEER={result['hunt_deer_count']} HUNTING_PARTY={result['hunting_party_count']} "
              f"starving_while_out={result['cycles_starving_while_hunting_party_out']} "
              f"actions={result['action_counts']}", flush=True)
    simulation_module.unlocked_actions_through = _orig_unlocked_actions_through

    print("\n=== SUMMARY ===")
    for label in ("baseline", "with_hunting_party"):
        rows = [r for r in all_results if r["variant"] == label]
        survived = sum(1 for r in rows if not r["extinct"])
        total_starving_while_out = sum(r["cycles_starving_while_hunting_party_out"] for r in rows)
        total_hp = sum(r["hunting_party_count"] for r in rows)
        print(f"  {label}: survived {survived}/{len(rows)} runs, HUNTING_PARTY picked {total_hp} times total, "
              f"cycles-starving-while-party-out={total_starving_while_out}")

    with open("scripts/ab_test_hunting_party_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved full detail to scripts/ab_test_hunting_party_results.json")


if __name__ == "__main__":
    asyncio.run(main())
