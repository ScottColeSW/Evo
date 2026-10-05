"""A/B test (2026-10-04): the project-wide nudge switch, NUDGES=on against NUDGES=off.

Question: when the prompt lines whose job is to steer a tribe (build hints, farm hints, the Historian's "choose something different", the
survival-warning text, the growth framing; docs/NUDGE-AUDIT.md) are removed, do tribes still survive, advance and build? Only prompt text
differs between the arms: menus, gates and mechanics are identical (tests/test_nudge_switch.py).

Design: two tribes on the same model (qwen2.5:3b, as in the last live run), 250 cycles, the same seeds in both arms, arms interleaved
(seed A on, A off, B off, B on) so drift in the machine does not line up with one arm. random.seed(seed) fixes the game's own dice; the
model's sampling is not seeded, so runs differ even with the same seed. With two runs per arm the result is suggestive, not proof.

Two modes. "early" starts both tribes fresh (the settling, farming and first-building nudges). "mid" starts both from real states at about
15,000 population (backend/fixtures/mid_game_15k_a.json and _b.json, cut from run_20261004_130133 cycle 281: five long houses, a keep, a
library, a tannery, and no forge, mine, DMM, fortress or castle yet), so the build hints for the later buildings can be tested in 100 cycles
instead of 300; the fixture path is the one run_benchmark.py already uses.

Each run records, per tribe: the cycle of each era entry, population at fixed cycles, whether it survived, the action mix, and a few
structures at the end. Results go to scripts/ab_test_nudges_off_results.json after every run, so an interrupted batch keeps what it finished.

    python scripts/ab_test_nudges_off.py --mode mid --cycles 100 --seeds 3
    python scripts/ab_test_nudges_off.py --mode early --cycles 250 --seeds 2
    python scripts/ab_test_nudges_off.py --mode mid --cycles 6 --seeds 1   (a smoke test; delete the results file afterward)
"""
import argparse
import asyncio
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from backend.simulation import Simulation
from backend.tribe_fixtures import apply_tribe_fixture, load_fixture

MODEL = "qwen2.5:3b"
RESULTS = "scripts/ab_test_nudges_off_results.json"
FIXTURES = ("mid_game_15k_a", "mid_game_15k_b")
SAMPLE_CYCLES = (50, 100, 150, 200, 250)
STRUCTURE_FLAGS = ("long_houses_built", "kitchen_built", "tannery_built", "library_built", "barracks_built", "forge_built",
                   "mine_built", "coop_built", "dock_built", "sawmill_built", "quarry_built", "well_built", "fishing_learned",
                   "keep_built", "fortress_built", "castle_built", "moat_built", "dmm_built", "deer_pen_built", "road_built",
                   "hatchery_built", "bath_house_built", "warehouses_built")


async def run_once(variant: str, seed: int, cycles: int, mode: str) -> dict:
    os.environ["NUDGES"] = variant
    random.seed(seed)
    sim = await Simulation.create([{"name": "Tribe 1", "model": MODEL}, {"name": "Tribe 2", "model": MODEL}])
    if mode == "mid":
        cycle = None
        for tribe, name in zip(sim.tribes.values(), FIXTURES):
            fixture = load_fixture(name)
            cycle = fixture["cycle"]
            apply_tribe_fixture(tribe, fixture, new_cycle=cycle)
        sim.cycle = cycle
    start_cycle = sim.cycle
    start_flags = {tid: {f: getattr(t, f, 0) for f in STRUCTURE_FLAGS} for tid, t in sim.tribes.items()}
    era_first = {tid: {t.era: 0} for tid, t in sim.tribes.items()}
    pop_at = {tid: {} for tid in sim.tribes}
    actions = {tid: collections.Counter() for tid in sim.tribes}
    started = time.time()
    for _ in range(cycles):
        await sim.step()
        for tid, t in sim.tribes.items():
            era_first[tid].setdefault(t.era, sim.cycle)
            if sim.cycle in SAMPLE_CYCLES:
                pop_at[tid][sim.cycle] = t.population
            if t.last_action:
                actions[tid][t.last_action] += 1
        if all(t.extinct for t in sim.tribes.values()):
            break
        if (sim.cycle - start_cycle) % 25 == 0:
            print(f"    [{variant} seed {seed}] cycle {sim.cycle}, {int(time.time() - started)}s: "
                  + " | ".join(f"{t.name} pop={t.population} era={t.era}" for t in sim.tribes.values()), flush=True)
    tribes = {}
    for tid, t in sim.tribes.items():
        tribes[t.name] = {
            "era_first_cycle": era_first[tid], "final_era": t.era, "population_at": pop_at[tid], "final_population": t.population,
            "max_population": t.max_population, "extinct": t.extinct, "trades": t.trades_completed,
            "structures": {f: bool(getattr(t, f, False)) for f in STRUCTURE_FLAGS},
            "built_during_run": [f for f in STRUCTURE_FLAGS if getattr(t, f, 0) and not start_flags[tid][f]],
            "action_mix": dict(actions[tid].most_common()),
        }
    return {"variant": variant, "seed": seed, "mode": mode, "start_cycle": start_cycle, "cycles": sim.cycle - start_cycle, "seconds": int(time.time() - started),
            "run_id": sim.run_id, "tribes": tribes}


def load() -> list:
    try:
        return json.load(open(RESULTS, encoding="utf-8"))
    except Exception:
        return []


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--mode", choices=("early", "mid"), default="early")
    args = parser.parse_args()
    seeds = [1000 + i for i in range(args.seeds)]
    order = []
    for i, seed in enumerate(seeds):
        order += [(seed, "on"), (seed, "off")] if i % 2 == 0 else [(seed, "off"), (seed, "on")]
    results = load()
    for seed, variant in order:
        if any(r["seed"] == seed and r["variant"] == variant and r.get("mode", "early") == args.mode for r in results):
            print(f"skip seed {seed} {variant} (already recorded)")
            continue
        print(f"=== {args.mode} seed {seed}, NUDGES={variant}, {args.cycles} cycles ===", flush=True)
        results.append(await run_once(variant, seed, args.cycles, args.mode))
        json.dump(results, open(RESULTS, "w", encoding="utf-8"), indent=1)
        r = results[-1]
        print(f"=== done: {r['cycles']} cycles in {r['seconds']}s, "
              + "; ".join(f"{n}: {t['final_era']} pop {t['final_population']}{' EXTINCT' if t['extinct'] else ''}" for n, t in r["tribes"].items()),
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
