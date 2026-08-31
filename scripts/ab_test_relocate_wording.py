"""A/B test #6: gemma2:2b's RELOCATE rate more than doubles under hunger/thirst crisis
(69.6% vs 27.1% baseline, see scripts/diagnose_relocate_under_crisis_gemma.log) while
SCOUT barely registers (~2%) -- even though RELOCATE's own description already states
it costs extra food/water and produces nothing. Same question as the HUNT_DEER
investigation: is the model just not weighting an already-stated cost heavily enough,
or is wording not the driver at all (as every HUNT_DEER wording/order/framing test this
session found)?

Tests a more forceful description: states explicitly that RELOCATE is a gamble with no
guarantee of finding food or water, on top of the existing cost language. Tracks
RELOCATE and SCOUT rate specifically during hunger/thirst-critical cycles (same
definition as the diagnostic script) since that's where the real divergence is, not the
overall rate.

Single-tribe, gemma2:2b (the model that actually shows this behavior), same cycle/run
counts as prior tests. Throwaway analysis script.
"""
import asyncio
import collections
import sys

sys.path.insert(0, ".")

from backend import actions, config
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "relocate_crisis_wording"
MODEL = "gemma2:2b"
CYCLES_PER_RUN = 100
RUNS_PER_VARIANT = 3

BASELINE_TEXT = actions.ACTION_DESCRIPTIONS["RELOCATE"]
FORCEFUL_TEXT = (
    "Move your whole tribe several tiles toward target_vector this cycle, possibly over "
    "several cycles for a far destination -- a gamble with no guarantee your destination "
    "has food or water. Produces no resources while traveling, and drains your already-"
    "limited food and water reserves for the journey itself."
)


async def run_once(variant_label: str, run_id: int) -> dict:
    sim = await Simulation.create([{"name": "Test Tribe", "model": MODEL}])
    tribe = next(iter(sim.tribes.values()))
    critical_actions = collections.Counter()
    noncritical_actions = collections.Counter()

    for _ in range(CYCLES_PER_RUN):
        critical = (
            tribe.food <= config.HUNGER_CRITICAL_THRESHOLD
            or tribe.water <= config.THIRST_CRITICAL_THRESHOLD
        )
        await sim.step()
        if tribe.extinct:
            break
        (critical_actions if critical else noncritical_actions)[tribe.last_action] += 1

    return {
        "variant": variant_label,
        "run_id": run_id,
        "cycles_run": sim.cycle,
        "extinct": tribe.extinct,
        "final_population": tribe.population,
        "max_population": tribe.max_population,
        "critical_relocate": critical_actions.get("RELOCATE", 0),
        "critical_scout": critical_actions.get("SCOUT", 0),
        "critical_total": sum(critical_actions.values()),
        "noncritical_relocate": noncritical_actions.get("RELOCATE", 0),
        "noncritical_total": sum(noncritical_actions.values()),
    }


def _log_run(label, notes, result):
    record_experiment_run(
        EXPERIMENT_NAME, label, MODEL,
        metrics={
            "critical_relocate": result["critical_relocate"],
            "critical_scout": result["critical_scout"],
            "critical_total": result["critical_total"],
            "noncritical_relocate": result["noncritical_relocate"],
            "noncritical_total": result["noncritical_total"],
            "cycles_run": result["cycles_run"],
            "extinct": result["extinct"],
            "final_population": result["final_population"],
            "max_population": result["max_population"],
        },
        run_id=result["run_id"], notes=notes,
    )


async def main():
    all_results = []

    print("=== VARIANT: baseline (unchanged) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("baseline", i)
        all_results.append(result)
        _log_run("baseline", BASELINE_TEXT, result)
        pct = (result["critical_relocate"] / result["critical_total"] * 100) if result["critical_total"] else 0.0
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"critical_relocate={result['critical_relocate']}/{result['critical_total']} ({pct:.1f}%) "
              f"critical_scout={result['critical_scout']}", flush=True)

    actions.ACTION_DESCRIPTIONS["RELOCATE"] = FORCEFUL_TEXT
    print("\n=== VARIANT: forceful (explicit gamble/no-guarantee wording) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("forceful", i)
        all_results.append(result)
        _log_run("forceful", FORCEFUL_TEXT, result)
        pct = (result["critical_relocate"] / result["critical_total"] * 100) if result["critical_total"] else 0.0
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"critical_relocate={result['critical_relocate']}/{result['critical_total']} ({pct:.1f}%) "
              f"critical_scout={result['critical_scout']}", flush=True)
    actions.ACTION_DESCRIPTIONS["RELOCATE"] = BASELINE_TEXT

    print("\n=== SUMMARY ===")
    for label in ("baseline", "forceful"):
        rows = [r for r in all_results if r["variant"] == label]
        total_relocate = sum(r["critical_relocate"] for r in rows)
        total_critical = sum(r["critical_total"] for r in rows)
        total_scout = sum(r["critical_scout"] for r in rows)
        pct = (total_relocate / total_critical * 100) if total_critical else 0.0
        print(f"  {label}: RELOCATE-while-critical={total_relocate}/{total_critical} ({pct:.1f}%), "
              f"SCOUT-while-critical={total_scout}")


if __name__ == "__main__":
    asyncio.run(main())
