"""A/B test #3 on the same question: wording (ab_test_hunt_wording.py) and list
order (ab_test_hunt_order.py) both came back null -- HUNT_DEER stays near-zero
regardless. This tests the species/risk-framing hypothesis instead: "deer" plus a
wolf-pack-attack mechanic reads as a serious expedition; "rabbit, quail" reads as
closer to foraging -- smaller, lower-stakes, no predator encounter at all.

The small-game variant is a genuinely different action, not just a reworded one:
new name (HUNT_SMALL_GAME, so the model isn't asked to call something "HUNT_DEER"
while being told about rabbits), same food yield via the same "game" resource
scarcity track, but with the wolf-pack hazard removed entirely -- a small-game hunt
is honestly lower-risk than a deer hunt, so the mechanic should actually match the
description's claim, not just reword the label over an unchanged deer hunt.

Same model, spawn point, and cycle/run counts as the previous two tests. Throwaway
analysis script.
"""
import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")

from backend import actions as actions_module
from backend import simulation as simulation_module
from backend.eras import unlocked_actions_through as _orig_unlocked_actions_through
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "hunt_deer_species_framing"
MODEL = "qwen2.5:3b"
CYCLES_PER_RUN = 100
RUNS_PER_VARIANT = 3

DEER_DESCRIPTION = actions_module.ACTION_DESCRIPTIONS["HUNT_DEER"]
SMALL_GAME_DESCRIPTION = (
    "Harvest food by hunting small game (rabbit, quail) at your current tile -- forest has the "
    "most game, plains and river tiles some, mountains and ocean almost none."
)


def _hunt_small_game(sim, tribe, biome, target):
    tribe.food += actions_module._harvest(sim, tribe, "game", 15, biome)
    return None


def _swap_action_name(actions_set, old, new):
    actions_set = set(actions_set)
    if old in actions_set:
        actions_set.discard(old)
        actions_set.add(new)
    return actions_set


def _small_game_unlocked_actions_through(era_key):
    return _swap_action_name(_orig_unlocked_actions_through(era_key), "HUNT_DEER", "HUNT_SMALL_GAME")


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

    hunt_count = action_counts.get("HUNT_DEER", 0) + action_counts.get("HUNT_SMALL_GAME", 0)
    return {
        "variant": variant_label,
        "run_id": run_id,
        "cycles_run": sim.cycle,
        "extinct": tribe.extinct,
        "final_population": tribe.population,
        "max_population": tribe.max_population,
        "hunt_count": hunt_count,
        "total_actions": sum(action_counts.values()),
        "action_counts": dict(action_counts),
    }


def _log_run(label, notes, result):
    record_experiment_run(
        EXPERIMENT_NAME, label, MODEL,
        metrics={
            "hunt_count": result["hunt_count"],
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

    print("=== VARIANT: deer (baseline, unchanged) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("deer", i)
        all_results.append(result)
        _log_run("deer", DEER_DESCRIPTION, result)
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"pop={result['final_population']} (peak {result['max_population']}) "
              f"HUNT={result['hunt_count']}/{result['total_actions']} "
              f"actions={result['action_counts']}", flush=True)

    actions_module.ACTION_REGISTRY["HUNT_SMALL_GAME"] = _hunt_small_game
    actions_module.ACTION_DESCRIPTIONS["HUNT_SMALL_GAME"] = SMALL_GAME_DESCRIPTION
    simulation_module.unlocked_actions_through = _small_game_unlocked_actions_through
    print("\n=== VARIANT: small_game (rabbit/quail, no predator hazard) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("small_game", i)
        all_results.append(result)
        _log_run("small_game", SMALL_GAME_DESCRIPTION, result)
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"pop={result['final_population']} (peak {result['max_population']}) "
              f"HUNT={result['hunt_count']}/{result['total_actions']} "
              f"actions={result['action_counts']}", flush=True)
    simulation_module.unlocked_actions_through = _orig_unlocked_actions_through

    print("\n=== SUMMARY ===")
    for label in ("deer", "small_game"):
        rows = [r for r in all_results if r["variant"] == label]
        total_hunt = sum(r["hunt_count"] for r in rows)
        total_actions = sum(r["total_actions"] for r in rows)
        survived = sum(1 for r in rows if not r["extinct"])
        pct = (total_hunt / total_actions * 100) if total_actions else 0.0
        print(f"  {label}: HUNT={total_hunt}/{total_actions} ({pct:.1f}%), "
              f"survived {survived}/{len(rows)} runs")

    with open("scripts/ab_test_hunt_species_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved full detail to scripts/ab_test_hunt_species_results.json")


if __name__ == "__main__":
    asyncio.run(main())
