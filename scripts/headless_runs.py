"""Headless multi-run harness: runs the real Simulation (real Ollama calls, no
websocket/browser) for a capped number of cycles per run, and prints a compact
summary of what happened -- final era, population trajectory, cause of any
extinction, and a sample of chronicle entries. Exists to answer "why do tribes
plateau in the Stone Age" with real data instead of guessing.

Not part of the app; a throwaway analysis script.
"""
import asyncio
import sys
import time

sys.path.insert(0, ".")

from backend.simulation import Simulation
from backend.eras import era_index


MAX_CYCLES = 120
MODELS = ["gemma2:2b", "qwen2.5:3b"]


async def run_once(run_id: int) -> dict:
    sim = await Simulation.create(
        [
            {"name": "Forest Tribe", "model": MODELS[0]},
            {"name": "Mountain Tribe", "model": MODELS[1]},
        ]
    )
    pop_history = {tid: [] for tid in sim.tribes}
    era_reached = {tid: sim.tribes[tid].era for tid in sim.tribes}
    t0 = time.time()
    for cycle in range(MAX_CYCLES):
        await sim.step()
        for tid, tribe in sim.tribes.items():
            pop_history[tid].append(tribe.population)
            if era_index(tribe.era) > era_index(era_reached[tid]):
                era_reached[tid] = tribe.era
        if all(t.extinct for t in sim.tribes.values()):
            break
    elapsed = time.time() - t0

    report = {"run_id": run_id, "cycles_run": sim.cycle, "elapsed_s": round(elapsed, 1), "tribes": []}
    for tid, tribe in sim.tribes.items():
        report["tribes"].append({
            "name": tribe.name,
            "model": tribe.model,
            "extinct": tribe.extinct,
            "final_era": tribe.era,
            "max_era_reached": era_reached[tid],
            "final_population": tribe.population,
            "max_population": tribe.max_population,
            "peak_population": max(pop_history[tid]) if pop_history[tid] else 0,
            "final_resources": {"wood": tribe.wood, "stone": tribe.stone, "food": tribe.food, "water": tribe.water},
            "expeditions_launched": tribe.expeditions_launched,
            "expeditions_succeeded": tribe.expeditions_succeeded,
            "raids_won": tribe.raids_won,
            "trades_completed": tribe.trades_completed,
            "chiefs_elected": tribe.chiefs_elected,
            "chief_deaths": tribe.chief_deaths,
            "last_history": list(tribe.history)[-8:],
        })
    return report


async def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    all_reports = []
    for i in range(n_runs):
        print(f"=== RUN {i} starting ===", flush=True)
        report = await run_once(i)
        all_reports.append(report)
        print(f"=== RUN {i} done in {report['elapsed_s']}s, {report['cycles_run']} cycles ===", flush=True)
        for t in report["tribes"]:
            print(f"  {t['name']} ({t['model']}): era={t['final_era']} (max {t['max_era_reached']}) "
                  f"pop={t['final_population']} (peak {t['peak_population']}) extinct={t['extinct']} "
                  f"res={t['final_resources']} exp={t['expeditions_launched']}/{t['expeditions_succeeded']} "
                  f"raids_won={t['raids_won']} trades={t['trades_completed']}", flush=True)
            for h in t["last_history"]:
                print(f"    - {h}", flush=True)

    import json
    with open("scripts/headless_run_results.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print("\nSaved full detail to scripts/headless_run_results.json")


if __name__ == "__main__":
    asyncio.run(main())
