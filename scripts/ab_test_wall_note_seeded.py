"""Follow-up to scripts/ab_test_growth_facts.py: that test never resolved whether
_wall_expansion_note actually changes EXPAND_TERRITORY pick rate, because
EXPAND_TERRITORY (60 wood + 60 stone simultaneously) turned out pricier than every
other building a single test tribe could afford in 150 cycles -- CONSTRUCT_WALL/
EXPAND_TERRITORY were picked zero times across all 6 runs of that test, both
variants, so the note's own trigger condition may simply never have gone true. That
result is "untested," not "the note doesn't work."

This test removes that confound directly: each run starts with a tribe forced
through _found_territory (real wall_rings, zero sections unlocked, matching every
freshly-settled tribe) and wood/stone seeded well above the 60/60 threshold, so
_wall_expansion_note's condition is true from cycle 1 in the growth_facts_on variant
-- no waiting on organic economy growth. Tracks how many cycles the condition
actually held true (a sanity check that the seeding worked as intended) alongside
EXPAND_TERRITORY/CONSTRUCT_WALL pick counts.

Single-tribe, gemma2:2b (matches the traced run and ab_test_growth_facts.py).
Shorter than the parent test (60 cycles, not 150) since the condition being tested
is active from the first cycle instead of needing time to develop. Throwaway
analysis script.
"""
import asyncio
import collections
import json
import sys

sys.path.insert(0, ".")

from backend import config, simulation
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "wall_note_seeded"
MODEL = "gemma2:2b"
CYCLES_PER_RUN = 60
RUNS_PER_VARIANT = 4
SEED_WOOD = 200
SEED_STONE = 200

REAL_WALL_NOTE = simulation._wall_expansion_note


def _sections_unlocked(tribe) -> int:
    if not tribe.wall_rings:
        return 0
    return sum(1 for sec in tribe.wall_rings[0]["sections"] if sec["unlocked"])


def _seed_settled_tribe(sim, tribe) -> None:
    """Same shape as tests/test_actions.py's _settle() helper -- a real territory
    and first wall ring via the actual _found_territory method, not a hand-faked
    flag, so wall_rings has real section dicts _wall_expansion_note actually reads."""
    tribe.has_ever_settled = True
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    sim._found_territory(tribe)
    tribe.wood = SEED_WOOD
    tribe.stone = SEED_STONE


async def run_once(variant_label: str, run_id: int) -> dict:
    sim = await Simulation.create([{"name": "Test Tribe", "model": MODEL}])
    tribe = next(iter(sim.tribes.values()))
    _seed_settled_tribe(sim, tribe)

    action_counts = collections.Counter()
    note_active_cycles = 0

    for _ in range(CYCLES_PER_RUN):
        if simulation._wall_expansion_note(tribe):  # real condition, regardless of variant patch
            note_active_cycles += 1
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
        "note_active_cycles": note_active_cycles,
        "wall_sections_unlocked": _sections_unlocked(tribe),
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
            "note_active_cycles": result["note_active_cycles"],
            "wall_sections_unlocked": result["wall_sections_unlocked"],
            "expand_territory_count": result["expand_territory_count"],
            "construct_wall_count": result["construct_wall_count"],
            "total_actions": result["total_actions"],
        },
        run_id=result["run_id"], notes=notes,
    )


def _print_result(result):
    print(
        f"  run {result['run_id']}: cycles={result['cycles_run']} extinct={result['extinct']} "
        f"pop={result['final_population']} note_active={result['note_active_cycles']}/{result['cycles_run']} "
        f"wall_sections={result['wall_sections_unlocked']} "
        f"EXPAND_TERRITORY={result['expand_territory_count']} CONSTRUCT_WALL={result['construct_wall_count']}",
        flush=True,
    )


async def main():
    all_results = []

    simulation._wall_expansion_note = lambda tribe: ""
    print("=== VARIANT: no_wall_note (baseline, fact disabled, still seeded 200/200) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("no_wall_note", i)
        all_results.append(result)
        _log_run("no_wall_note", "wall-expansion fact disabled, tribe seeded wood=stone=200", result)
        _print_result(result)

    simulation._wall_expansion_note = REAL_WALL_NOTE
    print("\n=== VARIANT: wall_note_on (shipped default, seeded 200/200) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("wall_note_on", i)
        all_results.append(result)
        _log_run("wall_note_on", "wall-expansion fact enabled (shipped), tribe seeded wood=stone=200", result)
        _print_result(result)

    print("\n=== SUMMARY ===")
    for label in ("no_wall_note", "wall_note_on"):
        rows = [r for r in all_results if r["variant"] == label]
        n = len(rows)
        avg_expand = sum(r["expand_territory_count"] for r in rows) / n
        avg_wall_sections = sum(r["wall_sections_unlocked"] for r in rows) / n
        avg_active = sum(r["note_active_cycles"] for r in rows) / n
        avg_cycles = sum(r["cycles_run"] for r in rows) / n
        survived = sum(1 for r in rows if not r["extinct"])
        print(
            f"  {label}: EXPAND_TERRITORY avg={avg_expand:.2f} wall_sections avg={avg_wall_sections:.2f} "
            f"note_active avg={avg_active:.1f}/{avg_cycles:.0f} cycles survived {survived}/{n} runs"
        )

    with open("scripts/ab_test_wall_note_seeded_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved full detail to scripts/ab_test_wall_note_seeded_results.json")


if __name__ == "__main__":
    asyncio.run(main())
