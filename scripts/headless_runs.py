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
# Windows' console defaults stdout to the system codepage (cp1252) even when redirected
# to a file, which can't encode the emoji chronicle entries already carry (celebrations,
# trophies, etc.) -- crashed a real run mid-batch. UTF-8 output is correct everywhere
# this runs, not just a Windows workaround.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.simulation import Simulation
from backend.eras import era_index


MAX_CYCLES = 120
MODELS = ["gemma2:2b", "qwen2.5:3b"]

# For `--sweep`: Forest Tribe stays on the fixed baseline (MODELS[0]) while Mountain
# Tribe rotates through every currently-installed candidate, one run each -- comparative
# data on which models actually survive/scout/advance under identical conditions,
# instead of only ever seeing gemma2:2b/qwen2.5:3b play against each other.
SWEEP_MOUNTAIN_MODELS = [
    "qwen2.5:3b", "llama3.2:1b", "qwen2.5:1.5b", "phi4-mini", "llama3.2:latest",
    "mistral:7b", "qwen2.5-coder:7b",
]

# For `--immortal`: every mortal-mode batch this session has shown near-universal
# extinction by cycle 40-60, well before scouting matures, trade happens, breeding
# fires, or an era advances -- there's never been enough surviving runtime to see
# whether those systems actually work, only whether tribes die before reaching them.
# IMMORTAL_CYCLES_WINDOW protects population loss (see Simulation._lose_population)
# for most of the run; the remaining cycles past that, still within IMMORTAL_MAX_CYCLES,
# are a real test of whether a tribe that had time to mature can then sustain itself
# once protection lifts, not just delayed data on the same collapse.
IMMORTAL_CYCLES_WINDOW = 300
IMMORTAL_MAX_CYCLES = 400


async def run_once(
    run_id, models: list[str] | None = None, immortality_cycles: int = 0, max_cycles: int | None = None,
) -> dict:
    models = models or MODELS
    max_cycles = max_cycles or MAX_CYCLES
    sim = await Simulation.create(
        [
            {"name": "Forest Tribe", "model": models[0]},
            {"name": "Mountain Tribe", "model": models[1]},
        ],
        immortality_cycles=immortality_cycles,
    )
    pop_history = {tid: [] for tid in sim.tribes}
    era_reached = {tid: sim.tribes[tid].era for tid in sim.tribes}
    t0 = time.time()
    for cycle in range(max_cycles):
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
            "final_position": [tribe.x, tribe.y],
            "final_biome": sim.world.biome(tribe.x, tribe.y),
            "ever_confirmed_water": tribe.scout_successes > 0,
            "final_resources": {"wood": tribe.wood, "stone": tribe.stone, "food": tribe.food, "water": tribe.water},
            "expeditions_launched": tribe.expeditions_launched,
            "expeditions_succeeded": tribe.expeditions_succeeded,
            "scout_successes": tribe.scout_successes,
            "raids_won": tribe.raids_won,
            "trades_completed": tribe.trades_completed,
            "chiefs_elected": tribe.chiefs_elected,
            "chief_deaths": tribe.chief_deaths,
            "trophies": [t["name"] for t in tribe.trophies],
            "custom_awards": [a["name"] for a in tribe.custom_awards],
            "births": len(tribe.lineage),
            "lumber_sites": len(tribe.lumber_sites),
            "wildlife_sites": len(tribe.wildlife_sites),
            "quarry_sites": len(tribe.quarry_sites),
            "last_history": list(tribe.history)[-8:],
        })
    return report


def _print_report(report: dict) -> None:
    print(f"=== RUN {report['run_id']} done in {report['elapsed_s']}s, {report['cycles_run']} cycles ===", flush=True)
    for t in report["tribes"]:
        print(f"  {t['name']} ({t['model']}): era={t['final_era']} (max {t['max_era_reached']}) "
              f"pop={t['final_population']} (peak {t['peak_population']}) extinct={t['extinct']} "
              f"final_pos={t['final_position']} final_biome={t['final_biome']} "
              f"confirmed_water={t['ever_confirmed_water']} "
              f"res={t['final_resources']} exp={t['expeditions_launched']}/{t['expeditions_succeeded']} "
              f"raids_won={t['raids_won']} trades={t['trades_completed']} births={t['births']} "
              f"trophies={t['trophies']} custom_awards={t['custom_awards']} "
              f"lumber_sites={t['lumber_sites']} wildlife_sites={t['wildlife_sites']} quarry_sites={t['quarry_sites']}",
              flush=True)
        for h in t["last_history"]:
            print(f"    - {h}", flush=True)


async def main():
    import json

    mode = sys.argv[1] if len(sys.argv) > 1 else None
    sweep = mode == "--sweep"
    immortal = mode == "--immortal"
    if sweep:
        out_path = "scripts/headless_sweep_results.json"
    elif immortal:
        out_path = "scripts/headless_immortal_results.json"
    else:
        out_path = "scripts/headless_run_results.json"
    all_reports = []

    if sweep:
        for mountain_model in SWEEP_MOUNTAIN_MODELS:
            models = [MODELS[0], mountain_model]
            print(f"=== SWEEP {mountain_model} starting (Forest fixed at {MODELS[0]}) ===", flush=True)
            report = await run_once(mountain_model, models)
            all_reports.append(report)
            with open(out_path, "w") as f:
                json.dump(all_reports, f, indent=2, default=str)
            _print_report(report)
    elif immortal:
        n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        for i in range(n_runs):
            print(f"=== IMMORTAL RUN {i} starting (protected for the first "
                  f"{IMMORTAL_CYCLES_WINDOW} of up to {IMMORTAL_MAX_CYCLES} cycles) ===", flush=True)
            report = await run_once(i, immortality_cycles=IMMORTAL_CYCLES_WINDOW, max_cycles=IMMORTAL_MAX_CYCLES)
            all_reports.append(report)
            with open(out_path, "w") as f:
                json.dump(all_reports, f, indent=2, default=str)
            _print_report(report)
    else:
        n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        for i in range(n_runs):
            print(f"=== RUN {i} starting ===", flush=True)
            report = await run_once(i)
            all_reports.append(report)
            # Written after every run, not just at the end -- a crash mid-batch (e.g. an
            # earlier real run hit a Windows console encoding error on an emoji
            # chronicle entry) used to lose every already-completed run's structured
            # detail along with it.
            with open(out_path, "w") as f:
                json.dump(all_reports, f, indent=2, default=str)
            _print_report(report)

    print(f"\nSaved full detail to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
