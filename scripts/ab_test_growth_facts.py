"""A/B test: does naming BUILD_WAREHOUSE/EXPAND_TERRITORY as real, computed facts in
the GROWTH IMPERATIVE LAYER actually move a tribe off the reactive gather/hunt loop,
or was that never really an information gap in the first place?

Live trace (run_20260908_082234, both tribes traced cycle-by-cycle via
board_history.db + the run's own jsonl): a tribe that reached population 1655 spent
178 dispatched turns choosing GATHER_WOOD/GATHER_EGGS/HUNT_DEER/GATHER_STONE 72% of
the time and never once chose CONSTRUCT_WALL, EXPAND_TERRITORY, BUILD_WAREHOUSE,
BUILD_DOCK, BUILD_FISHERY, BUILD_KITCHEN, or BUILD_LONG_HOUSE -- despite never having
built a warehouse (storage capped at 150 the whole game, "225 wasted" repeating in the
chronicle every cycle or two) and despite EXPAND_TERRITORY being affordable in 32 of
174 snapshots. backend/simulation.py's _warehouse_capacity_note/_wall_expansion_note
(new this session) surface both as real, checkable facts the same way
diversification_note already proved out for fishing/farming -- this test isolates
whether that fact alone moves the needle, same question every prior ab_test_* script
in this directory has asked about a different fact/wording change.

Single-tribe, gemma2:2b (the exact model that produced the traced run). 150 cycles
(not this project's usual 100) since both new facts only start mattering once
population/wood/stone actually reach the thresholds that make them true --
CYCLES_PER_RUN=100 in the traced run didn't get Tribe 1 past population ~180.
Throwaway analysis script.
"""
import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")

from backend import simulation
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "growth_note_warehouse_wall_facts"
MODEL = "gemma2:2b"
CYCLES_PER_RUN = 150
RUNS_PER_VARIANT = 3

REAL_WAREHOUSE_NOTE = simulation._warehouse_capacity_note
REAL_WALL_NOTE = simulation._wall_expansion_note


def _count_waste_events(history: list[str]) -> int:
    return sum(1 for line in history if "stores are already full" in line or "stores are nearly full" in line)


def _sections_unlocked(tribe) -> int:
    if not tribe.wall_rings:
        return 0
    return sum(1 for sec in tribe.wall_rings[0]["sections"] if sec["unlocked"])


async def run_once(variant_label: str, run_id: int) -> dict:
    sim = await Simulation.create([{"name": "Test Tribe", "model": MODEL}])
    tribe = next(iter(sim.tribes.values()))
    action_counts = collections.Counter()

    for _ in range(CYCLES_PER_RUN):
        await sim.step()
        if tribe.extinct:
            break
        if tribe.last_action:
            action_counts[tribe.last_action] += 1

    return {
        "variant": variant_label,
        "run_id": run_id,
        "cycles_run": sim.cycle,
        "extinct": tribe.extinct,
        "final_population": tribe.population,
        "max_population": tribe.max_population,
        "warehouses_built": tribe.warehouses_built,
        "wall_sections_unlocked": _sections_unlocked(tribe),
        "waste_events": _count_waste_events(tribe.history),
        "build_warehouse_count": action_counts.get("BUILD_WAREHOUSE", 0),
        "expand_territory_count": action_counts.get("EXPAND_TERRITORY", 0),
        "construct_wall_count": action_counts.get("CONSTRUCT_WALL", 0),
        "total_actions": sum(action_counts.values()),
        "action_counts": dict(action_counts),
    }


def _log_run(label, notes, result):
    record_experiment_run(
        EXPERIMENT_NAME, label, MODEL,
        metrics={
            "cycles_run": result["cycles_run"],
            "extinct": result["extinct"],
            "final_population": result["final_population"],
            "max_population": result["max_population"],
            "warehouses_built": result["warehouses_built"],
            "wall_sections_unlocked": result["wall_sections_unlocked"],
            "waste_events": result["waste_events"],
            "build_warehouse_count": result["build_warehouse_count"],
            "expand_territory_count": result["expand_territory_count"],
            "construct_wall_count": result["construct_wall_count"],
            "total_actions": result["total_actions"],
        },
        run_id=result["run_id"], notes=notes,
    )


def _print_result(result):
    print(
        f"  run {result['run_id']}: cycles={result['cycles_run']} extinct={result['extinct']} "
        f"pop={result['final_population']} (peak {result['max_population']}) "
        f"warehouses_built={result['warehouses_built']} wall_sections={result['wall_sections_unlocked']} "
        f"waste_events={result['waste_events']} "
        f"BUILD_WAREHOUSE={result['build_warehouse_count']} EXPAND_TERRITORY={result['expand_territory_count']} "
        f"CONSTRUCT_WALL={result['construct_wall_count']}",
        flush=True,
    )


async def main():
    all_results = []

    simulation._warehouse_capacity_note = lambda tribe: ""
    simulation._wall_expansion_note = lambda tribe: ""
    print("=== VARIANT: no_growth_facts (baseline, feature disabled) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("no_growth_facts", i)
        all_results.append(result)
        _log_run("no_growth_facts", "warehouse/wall growth-note facts disabled", result)
        _print_result(result)

    simulation._warehouse_capacity_note = REAL_WAREHOUSE_NOTE
    simulation._wall_expansion_note = REAL_WALL_NOTE
    print("\n=== VARIANT: growth_facts_on (shipped default) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("growth_facts_on", i)
        all_results.append(result)
        _log_run("growth_facts_on", "warehouse/wall growth-note facts enabled (shipped)", result)
        _print_result(result)

    print("\n=== SUMMARY ===")
    for label in ("no_growth_facts", "growth_facts_on"):
        rows = [r for r in all_results if r["variant"] == label]
        n = len(rows)
        avg_warehouse = sum(r["build_warehouse_count"] for r in rows) / n
        avg_expand = sum(r["expand_territory_count"] for r in rows) / n
        avg_wall_sections = sum(r["wall_sections_unlocked"] for r in rows) / n
        avg_warehouses_built = sum(r["warehouses_built"] for r in rows) / n
        avg_waste = sum(r["waste_events"] for r in rows) / n
        survived = sum(1 for r in rows if not r["extinct"])
        print(
            f"  {label}: BUILD_WAREHOUSE avg={avg_warehouse:.1f} EXPAND_TERRITORY avg={avg_expand:.1f} "
            f"wall_sections avg={avg_wall_sections:.1f} warehouses_built avg={avg_warehouses_built:.1f} "
            f"waste_events avg={avg_waste:.1f} survived {survived}/{n} runs"
        )

    with open("scripts/ab_test_growth_facts_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved full detail to scripts/ab_test_growth_facts_results.json")


if __name__ == "__main__":
    asyncio.run(main())
