"""Fixed, reproducible starting conditions for comparing small local models against
each other -- the "define a small set of benchmark scenarios" half of outside
feedback on this project's direction (the other half, scoring, lives in
backend/benchmark_scoring.py; persistence in backend/benchmark_db.py).

Spawn coordinates below are real, queried against backend.world.biome_at while
designing this (not invented): survival uses a deliberately harsh, isolated desert
tile far from water; the other three reuse the game's own existing, already-proven
SPAWN_POINTS (backend/simulation.py) so a scenario's difficulty comes from what it's
actually testing, not from an unproven map location.

SCENARIO_VERSION exists so a stored trial (backend/benchmark_db.py) stays
attributable to the exact definition that produced it -- bump this whenever a
scenario's starting conditions change materially (spawn position, cycle_budget,
tribe_count), so historical trials under an old definition are never silently
treated as comparable to trials under a new one.
"""

from dataclasses import dataclass

# Bumped 2026-09-12: survival's starting_resources changed (see that scenario's own
# comment) -- a materially different definition from SCENARIO_VERSION 1's trials.
SCENARIO_VERSION = 2


@dataclass(frozen=True)
class Scenario:
    key: str
    category: str
    tribe_count: int
    spawn_positions: tuple[tuple[int, int], ...]
    cycle_budget: int
    description: str
    # None means "use the game's own normal starting values" (Tribe.__init__'s
    # defaults: 50 wood, 50 stone, 40 food, config.STARTING_WATER). Set per-scenario
    # to deliberately override them -- run_benchmark.py applies this to every tribe
    # in the trial right after Simulation.create(...).
    starting_resources: dict[str, int] | None = None


# Cycle budget grounded in real data, not guessed: logs/board_history.db's
# run_20260912_062221 showed both tribes reaching population 50 (tribal_synapse
# era's requirement -- the gate for DECLARE_ALLIANCE/DECLARE_WAR/SPY/Barracks) by
# cycle 104-119. 200 cycles leaves comfortable room for cooperation/conflict
# mechanics to actually unlock and play out, without a separate, longer budget per
# category.
_CYCLE_BUDGET = 200

SCENARIOS: dict[str, Scenario] = {
    "survival": Scenario(
        key="survival",
        category="survival",
        tribe_count=1,
        # Desert, ~20 tiles from the nearest water tile, ~8 tiles from the nearest
        # of the game's own default SPAWN_POINTS -- deliberately harsh and
        # isolated. Tests bare "don't starve" competence, nothing else.
        spawn_positions=((65, 74),),
        cycle_budget=_CYCLE_BUDGET,
        description="One tribe, a harsh isolated desert spawn far from water, starting with almost nothing on hand. Tests survival alone.",
        # Added 2026-09-12, SCENARIO_VERSION bumped: the first real batch run showed
        # this scenario had a ceiling effect -- every trial scored 100/100 (zero
        # extinctions, population in the hundreds by cycle 200) even on the harsh
        # spawn above, because Tribe.__init__'s normal starting stock (50 wood, 50
        # stone, 40 food, config.STARTING_WATER=30) already buys ~30-40 cycles of
        # upkeep before a single successful gather is required. Cutting this to
        # near-nothing forces a real early decision within the first 1-2 cycles
        # instead of a long, risk-free runway.
        starting_resources={"wood": 3, "stone": 3, "food": 3, "water": 3},
    ),
    "settlement": Scenario(
        key="settlement",
        category="settlement",
        tribe_count=1,
        # One of the game's own existing, already-proven-good default spawn points.
        spawn_positions=((66, 23),),
        cycle_budget=_CYCLE_BUDGET,
        description="One tribe, a favorable spawn. Tests how far a civilization develops when survival isn't the bottleneck.",
    ),
    "cooperation": Scenario(
        key="cooperation",
        category="cooperation",
        tribe_count=2,
        # ~17 tiles apart -- well within RIVAL_PRECISE_AWARENESS_RADIUS (20)'s
        # discovery range, both plains, both existing default spawn points.
        spawn_positions=((50, 55), (63, 66)),
        cycle_budget=_CYCLE_BUDGET,
        description="Two tribes spawned close enough to make contact plausible. Tests whether cooperative actions get chosen when rational.",
    ),
    "conflict": Scenario(
        key="conflict",
        category="conflict",
        tribe_count=2,
        # ~20.6 tiles apart, sharing a river as a contestable resource.
        spawn_positions=((50, 55), (40, 37)),
        cycle_budget=_CYCLE_BUDGET,
        description="Two tribes spawned closer together, sharing a scarce resource. Tests conflict judgment, not just aggression.",
    ),
}
