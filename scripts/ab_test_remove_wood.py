"""A/B test #8: every prior intervention this session tried to make OTHER actions more
appealing against GATHER_WOOD's dominance (facts, framing, list order, richer wording)
-- all seven failed (see logs/experiments.jsonl). This is the opposite move: take away
the dominant default and see what actually fills the vacuum.

Open question this answers: is GATHER_WOOD dominance because wood is genuinely the most
appealing single choice, or a generic single-default reflex that would just latch onto
whatever's left (GATHER_STONE, most likely, or GATHER_WATER) if wood weren't available?
Either answer is informative -- if something else just becomes the new near-100%
default, that's strong evidence this is about a generic "always pick the same one
thing" reflex, not really about wood specifically.

Zeroes out BIOME_YIELD_MULTIPLIER["wood"] for the "no_wood" variant -- GATHER_WOOD
stays choosable (so we can see if the model still reaches for a now-useless action),
it just never produces anything, same mechanism already proven honest (a real,
mechanical zero yield, not removing the action from the menu). Single-tribe, both
models (this is exactly the kind of test where model differences matter, given
gemma2:2b and qwen2.5:3b already showed very different behavior patterns this
session), 100 cycles, 3 runs each. Throwaway analysis script.
"""
import asyncio
import collections
import sys

sys.path.insert(0, ".")

from backend import actions
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "remove_wood"
MODELS = ["gemma2:2b", "qwen2.5:3b"]
CYCLES_PER_RUN = 100
RUNS_PER_VARIANT = 3

ORIGINAL_WOOD_MULTIPLIER = dict(actions.BIOME_YIELD_MULTIPLIER["wood"])
ZEROED_WOOD_MULTIPLIER = {biome: 0.0 for biome in ORIGINAL_WOOD_MULTIPLIER}


async def run_once(model: str, variant_label: str, run_id: int) -> dict:
    sim = await Simulation.create([{"name": "Test Tribe", "model": model}])
    tribe = next(iter(sim.tribes.values()))
    action_counts = collections.Counter()

    for _ in range(CYCLES_PER_RUN):
        await sim.step()
        if tribe.extinct:
            break
        action_counts[tribe.last_action] += 1

    return {
        "model": model,
        "variant": variant_label,
        "run_id": run_id,
        "cycles_run": sim.cycle,
        "extinct": tribe.extinct,
        "final_population": tribe.population,
        "max_population": tribe.max_population,
        "action_counts": dict(action_counts),
        "total_actions": sum(action_counts.values()),
    }


def _log_run(model, label, notes, result):
    metrics = {
        "cycles_run": result["cycles_run"],
        "extinct": result["extinct"],
        "final_population": result["final_population"],
        "max_population": result["max_population"],
        "total_actions": result["total_actions"],
    }
    for action, count in result["action_counts"].items():
        metrics[f"action_{action}"] = count
    record_experiment_run(EXPERIMENT_NAME, f"{model}:{label}", model, metrics=metrics,
                           run_id=result["run_id"], notes=notes)


async def main():
    all_results = []

    for model in MODELS:
        print(f"\n########## MODEL: {model} ##########")

        print("=== VARIANT: baseline (wood available) ===")
        for i in range(RUNS_PER_VARIANT):
            result = await run_once(model, "baseline", i)
            all_results.append(result)
            _log_run(model, "baseline", "wood yield unchanged", result)
            top = sorted(result["action_counts"].items(), key=lambda kv: -kv[1])[:3]
            print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
                  f"pop={result['final_population']} top_actions={top}", flush=True)

        actions.BIOME_YIELD_MULTIPLIER["wood"] = ZEROED_WOOD_MULTIPLIER
        print("\n=== VARIANT: no_wood (GATHER_WOOD yields nothing anywhere) ===")
        for i in range(RUNS_PER_VARIANT):
            result = await run_once(model, "no_wood", i)
            all_results.append(result)
            _log_run(model, "no_wood", "BIOME_YIELD_MULTIPLIER['wood'] zeroed everywhere", result)
            top = sorted(result["action_counts"].items(), key=lambda kv: -kv[1])[:3]
            print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
                  f"pop={result['final_population']} top_actions={top}", flush=True)
        actions.BIOME_YIELD_MULTIPLIER["wood"] = dict(ORIGINAL_WOOD_MULTIPLIER)

    print("\n=== SUMMARY ===")
    for model in MODELS:
        for label in ("baseline", "no_wood"):
            rows = [r for r in all_results if r["model"] == model and r["variant"] == label]
            survived = sum(1 for r in rows if not r["extinct"])
            combined = collections.Counter()
            for r in rows:
                combined.update(r["action_counts"])
            total = sum(combined.values())
            top = sorted(combined.items(), key=lambda kv: -kv[1])[:4]
            top_str = ", ".join(f"{a}={c} ({c/total*100:.0f}%)" for a, c in top) if total else "no actions"
            print(f"  {model} / {label}: survived {survived}/{len(rows)}, top actions: {top_str}")


if __name__ == "__main__":
    asyncio.run(main())
