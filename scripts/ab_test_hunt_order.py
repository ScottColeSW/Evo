"""A/B test #2 on the same question: the wording test (scripts/ab_test_hunt_wording.py)
found HUNT_DEER stays near-zero regardless of "Attempt to harvest" vs "Harvest" framing,
even in a forest tile where game and wood have the identical 1.0x yield multiplier. This
tests the next candidate explanation: primacy/list-order bias. Both the glossary
(get_prime_consciousness_prompt) and the JSON valid-actions list (compile_live_state_prompt)
currently show actions alphabetically sorted -- HUNT_DEER sits 5th of ~10 in Stone Age,
after BUILD_FIRE/GATHER_STONE/GATHER_WATER/GATHER_WOOD. This variant moves HUNT_DEER to
the front of both lists and leaves the description text at its original baseline wording,
isolating order as the single variable.

Same model, same spawn point, same cycle/run counts as the wording test for direct
comparability. Throwaway analysis script.
"""
import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")

from backend import simulation as simulation_module
from backend import prompts as prompts_module
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "hunt_deer_list_order"
MODEL = "qwen2.5:3b"
CYCLES_PER_RUN = 100
RUNS_PER_VARIANT = 3
FRONT_ACTION = "HUNT_DEER"

_orig_get_prime = prompts_module.get_prime_consciousness_prompt
_orig_compile_live = prompts_module.compile_live_state_prompt


def _reorder(actions):
    actions = list(actions)
    if FRONT_ACTION in actions:
        actions.remove(FRONT_ACTION)
        actions.insert(0, FRONT_ACTION)
    return actions


def _reordered_get_prime(tribe_name, model_architecture, chief_name="", chief_philosophy="",
                          chief_decree="", available_actions=()):
    return _orig_get_prime(tribe_name, model_architecture, chief_name, chief_philosophy,
                            chief_decree, tuple(_reorder(available_actions)))


def _reordered_compile_live(base_prompt, world_state, ancestral_bias, survival_bias):
    ws = dict(world_state)
    ws["available_actions"] = _reorder(ws["available_actions"])
    return _orig_compile_live(base_prompt, ws, ancestral_bias, survival_bias)


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

    print("=== VARIANT: alphabetical (baseline order) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("alphabetical", i)
        all_results.append(result)
        _log_run("alphabetical", "actions listed alphabetically (HUNT_DEER 5th of ~10)", result)
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"pop={result['final_population']} (peak {result['max_population']}) "
              f"HUNT_DEER={result['hunt_deer_count']}/{result['total_actions']} "
              f"actions={result['action_counts']}", flush=True)

    simulation_module.get_prime_consciousness_prompt = _reordered_get_prime
    simulation_module.compile_live_state_prompt = _reordered_compile_live
    print("\n=== VARIANT: hunt_first (HUNT_DEER moved to front of both lists) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("hunt_first", i)
        all_results.append(result)
        _log_run("hunt_first", "HUNT_DEER moved to front of glossary and valid-actions list", result)
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"pop={result['final_population']} (peak {result['max_population']}) "
              f"HUNT_DEER={result['hunt_deer_count']}/{result['total_actions']} "
              f"actions={result['action_counts']}", flush=True)
    simulation_module.get_prime_consciousness_prompt = _orig_get_prime
    simulation_module.compile_live_state_prompt = _orig_compile_live

    print("\n=== SUMMARY ===")
    for label in ("alphabetical", "hunt_first"):
        rows = [r for r in all_results if r["variant"] == label]
        total_hunt = sum(r["hunt_deer_count"] for r in rows)
        total_actions = sum(r["total_actions"] for r in rows)
        survived = sum(1 for r in rows if not r["extinct"])
        pct = (total_hunt / total_actions * 100) if total_actions else 0.0
        print(f"  {label}: HUNT_DEER={total_hunt}/{total_actions} ({pct:.1f}%), "
              f"survived {survived}/{len(rows)} runs")

    with open("scripts/ab_test_hunt_order_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved full detail to scripts/ab_test_hunt_order_results.json")


if __name__ == "__main__":
    asyncio.run(main())
