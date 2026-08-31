"""A/B test #7: is SCOUT's terse, jargon-y description undervaluing what it actually
does relative to RELOCATE's plain-sounding "move to safety"? User's hypothesis: the
word "scout" reads as narrow, passive reconnaissance, when the mechanic actually covers
exploring the wilderness, documenting terrain, foraging along the way, and specifically
hunting for a reliable water source -- the single hardest survival problem in the game.

Important constraint honored here: the richer variant states ONLY what SCOUT already
mechanically does (explore, report terrain, forage a flat trickle, seek water). It does
NOT claim SCOUT hunts game encountered along the way -- that was floated as part of the
richer framing but isn't actually true (SCOUT only forages a flat trickle via config.
EXPEDITION_OUTBOUND_DAILY_FOOD/WATER, it doesn't roll a real hunt) -- adding that would
be fiction dressed as fact, not a legitimate description improvement.

Tracks RELOCATE and SCOUT specifically during hunger/thirst-critical cycles, same
definition as the other RELOCATE/SCOUT diagnostics this session. Single-tribe,
gemma2:2b (the model that actually shows the RELOCATE-under-crisis bias). Throwaway
analysis script.
"""
import asyncio
import collections
import sys

sys.path.insert(0, ".")

from backend import actions, config
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "scout_richer_wording"
MODEL = "gemma2:2b"
CYCLES_PER_RUN = 100
RUNS_PER_VARIANT = 3

BASELINE_TEXT = actions.ACTION_DESCRIPTIONS["SCOUT"]
RICHER_TEXT = (
    "Send your people out to explore beyond the village into the wilderness -- they travel "
    "and camp on their own supply, documenting the terrain they pass through and foraging "
    "food and water along the way, all while searching especially for a reliable water "
    "source, for up to a few days before turning back if they find nothing. What they learn "
    "only becomes known once they've walked all the way home. Your tribe can have a couple "
    "of parties out at once (scouting or hunting, any mix) -- choosing SCOUT again sends "
    "another one if there's room, or just reports on whoever's already out once you're at "
    "capacity."
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
        "noncritical_scout": noncritical_actions.get("SCOUT", 0),
        "noncritical_total": sum(noncritical_actions.values()),
    }


def _log_run(label, notes, result):
    record_experiment_run(
        EXPERIMENT_NAME, label, MODEL,
        metrics={
            "critical_relocate": result["critical_relocate"],
            "critical_scout": result["critical_scout"],
            "critical_total": result["critical_total"],
            "noncritical_scout": result["noncritical_scout"],
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

    print("=== VARIANT: baseline (current, capacity-accurate) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("baseline", i)
        all_results.append(result)
        _log_run("baseline", BASELINE_TEXT, result)
        pct = (result["critical_scout"] / result["critical_total"] * 100) if result["critical_total"] else 0.0
        rpct = (result["critical_relocate"] / result["critical_total"] * 100) if result["critical_total"] else 0.0
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"critical_scout={result['critical_scout']}/{result['critical_total']} ({pct:.1f}%) "
              f"critical_relocate={result['critical_relocate']} ({rpct:.1f}%)", flush=True)

    actions.ACTION_DESCRIPTIONS["SCOUT"] = RICHER_TEXT
    print("\n=== VARIANT: richer (explore/document/forage/seek-water, still all true) ===")
    for i in range(RUNS_PER_VARIANT):
        result = await run_once("richer", i)
        all_results.append(result)
        _log_run("richer", RICHER_TEXT, result)
        pct = (result["critical_scout"] / result["critical_total"] * 100) if result["critical_total"] else 0.0
        rpct = (result["critical_relocate"] / result["critical_total"] * 100) if result["critical_total"] else 0.0
        print(f"  run {i}: cycles={result['cycles_run']} extinct={result['extinct']} "
              f"critical_scout={result['critical_scout']}/{result['critical_total']} ({pct:.1f}%) "
              f"critical_relocate={result['critical_relocate']} ({rpct:.1f}%)", flush=True)
    actions.ACTION_DESCRIPTIONS["SCOUT"] = BASELINE_TEXT

    print("\n=== SUMMARY ===")
    for label in ("baseline", "richer"):
        rows = [r for r in all_results if r["variant"] == label]
        total_scout = sum(r["critical_scout"] for r in rows)
        total_relocate = sum(r["critical_relocate"] for r in rows)
        total_critical = sum(r["critical_total"] for r in rows)
        spct = (total_scout / total_critical * 100) if total_critical else 0.0
        rpct = (total_relocate / total_critical * 100) if total_critical else 0.0
        print(f"  {label}: SCOUT-while-critical={total_scout}/{total_critical} ({spct:.1f}%), "
              f"RELOCATE-while-critical={total_relocate}/{total_critical} ({rpct:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
