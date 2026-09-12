"""Headless benchmark harness -- runs a fixed scenario (backend.benchmark_scenarios)
against one or two models, no websocket/frontend involved, and records raw results
into the long-running logs/benchmark_results.db (backend.benchmark_db).

Usage:
    python run_benchmark.py --scenario survival --models gemma2:2b --trials 3
    python run_benchmark.py --scenario cooperation --models gemma2:2b,qwen2.5:3b
    python run_benchmark.py --report
    python run_benchmark.py --report --scenario conflict
"""

import argparse
import asyncio
import random
import time

from backend import benchmark_db, benchmark_scoring
from backend.benchmark_scenarios import SCENARIO_VERSION, SCENARIOS
from backend.simulation import Simulation


def _apply_starting_resources(sim: Simulation, scenario) -> None:
    """Overrides every tribe's starting stock to scenario.starting_resources, when
    set -- None (the default for every scenario but survival) leaves Tribe.__init__'s
    own normal defaults untouched. Kept as its own function so the override logic is
    directly unit-testable without needing a real Simulation.create() call."""
    if scenario.starting_resources is None:
        return
    for tribe in sim.tribes.values():
        for resource, amount in scenario.starting_resources.items():
            setattr(tribe, resource, amount)


async def run_trial(scenario_key: str, models: list[str], trial_seed: int) -> dict:
    scenario = SCENARIOS[scenario_key]
    random.seed(trial_seed)  # sufficient for every gameplay roll -- see the plan's own note on scope/limits

    tribe_configs = [
        {"name": f"Trial-{i}", "model": models[i % len(models)], "x": x, "y": y}
        for i, (x, y) in enumerate(scenario.spawn_positions[: scenario.tribe_count])
    ]
    started_ts = time.time()
    sim = await Simulation.create(tribe_configs)
    _apply_starting_resources(sim, scenario)
    try:
        while sim.cycle < scenario.cycle_budget and not sim.game_over:
            await sim.step()
    finally:
        await sim.shutdown()
    finished_ts = time.time()

    tribes = list(sim.tribes.values())
    rival_by_tribe = {t.id: (tribes[1 - i] if len(tribes) == 2 else None) for i, t in enumerate(tribes)}
    tribe_facts = [
        benchmark_db.extract_tribe_facts(t, rival=rival_by_tribe[t.id]) for t in tribes
    ]

    trial_id = benchmark_db.new_trial_id(scenario_key, trial_seed)
    trial_facts = {
        "trial_id": trial_id,
        "scenario_key": scenario_key,
        "scenario_version": SCENARIO_VERSION,
        "trial_seed": trial_seed,
        "cycle_budget": scenario.cycle_budget,
        "cycles_run": sim.cycle,
        "ended_reason": sim.game_over_reason or "budget_reached",
        "git_commit": benchmark_db.current_git_commit(),
        "run_id": sim.run_id,
        "started_ts": started_ts,
        "finished_ts": finished_ts,
    }
    benchmark_db.record_trial(trial_facts, tribe_facts)
    return {**trial_facts, "tribes": tribe_facts}


async def run_scenarios(scenario_keys: list[str], models: list[str], trials: int, seed_base: int) -> None:
    for scenario_key in scenario_keys:
        for trial_index in range(trials):
            trial_seed = seed_base + trial_index
            print(f"[{scenario_key}] trial {trial_index + 1}/{trials} (seed={trial_seed}, models={models}) ...")
            trial = await run_trial(scenario_key, models, trial_seed)
            scores = benchmark_scoring.score_trial(trial)
            print(
                f"  -> cycles_run={trial['cycles_run']}/{trial['cycle_budget']} "
                f"ended_reason={trial['ended_reason']} scores={scores}"
            )


def print_report(scenario_key: str | None, model: str | None) -> None:
    trials = benchmark_db.list_trials(scenario_key=scenario_key, model=model)
    if not trials:
        print("No trials recorded yet.")
        return
    by_key: dict[tuple[str, str], list[int]] = {}
    for trial in trials:
        scores = benchmark_scoring.score_trial(trial)
        for tribe, score in zip(trial["tribes"], scores):
            by_key.setdefault((trial["scenario_key"], tribe["model"]), []).append(score)

    print(f"{'scenario':<14} {'model':<20} {'trials':>6} {'mean':>7} {'min':>5} {'max':>5}")
    for (scenario, model_name), scores in sorted(by_key.items()):
        print(
            f"{scenario:<14} {model_name:<20} {len(scores):>6} "
            f"{sum(scores) / len(scores):>7.1f} {min(scores):>5} {max(scores):>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS) + ["all"], help="repeatable; default: all")
    parser.add_argument("--models", help="comma-separated, e.g. gemma2:2b,qwen2.5:3b")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--report", action="store_true", help="print a leaderboard from recorded trials instead of running any")
    args = parser.parse_args()

    if args.report:
        scenario_filter = None
        if args.scenario and "all" not in args.scenario:
            scenario_filter = args.scenario[0] if len(args.scenario) == 1 else None
        print_report(scenario_filter, args.models)
        return

    if not args.models:
        parser.error("--models is required unless --report is given")
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    scenario_keys = list(SCENARIOS) if not args.scenario or "all" in args.scenario else args.scenario
    asyncio.run(run_scenarios(scenario_keys, models, args.trials, args.seed_base))


if __name__ == "__main__":
    main()
