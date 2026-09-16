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
import sys
import time

from backend import benchmark_db, benchmark_scoring

# Explicit request, 2026-09-16: "these runs take too long to write out; we need
# to see this faster." Real cause -- not the progress print's own cadence, but
# that stdout is fully buffered (not line-buffered) whenever it isn't a real
# terminal, e.g. every `python run_benchmark.py ... > file.log 2>&1` invocation
# used to kick off a trial in the background. Every print() below was sitting
# in that buffer, invisible in the log file, until the whole process exited and
# flushed it all at once -- confirmed live: a completed war_ready_5000 trial's
# log showed nothing at all until the final flush, despite ~10 progress lines
# having already printed internally over the ~20-minute run.
sys.stdout.reconfigure(line_buffering=True)
from backend.benchmark_scenarios import SCENARIO_VERSION, SCENARIOS
from backend.simulation import Simulation
from backend.tribe_fixtures import apply_tribe_fixture, load_fixture


def _print_progress(sim: Simulation) -> None:
    """A trial is otherwise a total black box while it runs -- explicit request,
    2026-09-16, made right after a real trial (war_ready_5000, seed=301) sat silent
    for its whole duration with nothing to check on. Printed every cycle: real
    per-cycle Ollama latency (roughly a minute each, for a real two-model trial)
    means even this isn't spammy, and the follow-up request ("we need to see this
    faster") ruled out the original once-per-day-length cadence as too sparse."""
    parts = []
    for tribe in sim.tribes.values():
        stances = ",".join(f"{rid}:{s}" for rid, s in tribe.stance_toward.items()) or "-"
        parts.append(f"{tribe.name}[{tribe.model}] pop={tribe.population} era={tribe.era} stance={stances}")
    print(f"  cycle {sim.cycle}: " + " | ".join(parts))


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


def _apply_starting_fixtures(sim: Simulation, scenario) -> int | None:
    """Overrides every tribe's state to scenario.starting_fixtures, when set (see
    Scenario.starting_fixtures' own comment). Returns the fixtures' shared source
    cycle (to fast-forward sim.cycle to) when applied, None otherwise -- kept as
    its own function, same "unit-testable without a real Simulation.create()"
    reasoning _apply_starting_resources already uses."""
    if scenario.starting_fixtures is None:
        return None
    tribes = list(sim.tribes.values())
    source_cycle = None
    for tribe, fixture_name in zip(tribes, scenario.starting_fixtures):
        fixture = load_fixture(fixture_name)
        source_cycle = fixture["cycle"]
        apply_tribe_fixture(tribe, fixture, new_cycle=source_cycle)
    return source_cycle


def _only_one_tribe_left(sim: Simulation, scenario) -> bool:
    """Explicit request, 2026-09-16: "the test should end after only 1 tribe
    remains to save time on this test." A multi-tribe scenario (cooperation,
    conflict, war_ready_5000[_no_alliance]) has nothing left to compare once
    one side is gone -- Simulation.game_over only fires once EVERY tribe is
    extinct (the live game deliberately keeps playing a lone survivor, see
    DESIGN.md's own note on that), so without this a benchmark trial would
    burn the rest of cycle_budget's real Ollama inference on a tribe with no
    rival left to interact with, producing no additional benchmark signal.
    A living count covers both ways a tribe can leave the game: extinction
    (Tribe.extinct=True, stays in sim.tribes) and a conquest merge (the loser
    is deleted from sim.tribes outright, see Simulation._merge_tribes).
    Deliberately gated on tribe_count -- a 1-tribe scenario (survival,
    settlement) starts with exactly one living tribe by design and must keep
    running it for the full budget, not stop immediately on its own first
    cycle."""
    if scenario.tribe_count < 2:
        return False
    return sum(1 for t in sim.tribes.values() if not t.extinct) <= 1


def _apply_disabled_actions(sim: Simulation, scenario) -> None:
    """Sets sim.disabled_actions from scenario.disabled_actions, when set (see
    that field's own comment) -- None (the default for every scenario except
    war_ready_5000_no_alliance) leaves Simulation.__init__'s own empty-set
    default untouched. Kept as its own function, same "unit-testable without a
    real Simulation.create() call" reasoning _apply_starting_resources/
    _apply_starting_fixtures already use."""
    if scenario.disabled_actions is None:
        return
    sim.disabled_actions = set(scenario.disabled_actions)


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
    # See Scenario.starting_fixtures' own comment: cycle_budget means "how many
    # MORE cycles from here" once a fixture is applied, not an absolute cycle
    # number -- target_cycle covers both cases (a non-fixture scenario leaves
    # sim.cycle at its real 0 start, so this is exactly today's behavior).
    fixture_cycle = _apply_starting_fixtures(sim, scenario)
    if fixture_cycle is not None:
        sim.cycle = fixture_cycle
    _apply_disabled_actions(sim, scenario)
    start_cycle = sim.cycle
    target_cycle = sim.cycle + scenario.cycle_budget
    single_tribe_remaining = False
    try:
        while sim.cycle < target_cycle and not sim.game_over:
            await sim.step()
            _print_progress(sim)
            if _only_one_tribe_left(sim, scenario):
                single_tribe_remaining = True
                break
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
        "cycles_run": sim.cycle - start_cycle,
        "ended_reason": sim.game_over_reason or ("single_tribe_remaining" if single_tribe_remaining else "budget_reached"),
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
