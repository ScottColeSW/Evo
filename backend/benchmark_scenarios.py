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
    # Explicit request, 2026-09-16: "we could start at 800 even" / "we are going to
    # want to start a war starting from around [population] 5000." Names of real,
    # curated tribe_fixtures.py fixtures (backend/fixtures/<name>.json), one per
    # tribe in spawn_positions order -- run_benchmark.py applies these instead of
    # starting_resources when set, and also fast-forwards sim.cycle to the
    # fixture's own source cycle (see cycle_budget's own note below) so a trial
    # tests "what happens next" from a real advanced position, not another
    # from-scratch grind. spawn_positions is still required by the dataclass but
    # effectively unused here -- a fixture's own x/y (copied for consistency with
    # its buildings/territory) overrides it.
    starting_fixtures: tuple[str, ...] | None = None
    # Explicit request, 2026-09-16: "build a fixture that takes alliance off the
    # table." DECLARE_ALLIANCE is deliberately always offered while two tribes
    # are anything but already mutually allied -- "suing for peace is real,"
    # confirmed in Simulation._prepare_turn's own comment on that action -- so a
    # fixture's tribe *state* alone can't remove it (the one existing in-game
    # exclusion, battle_ready_locked, only fires once both tribes reach the era
    # ceiling fully armed, a much later and stronger condition than "war ready").
    # Names of actions to strip from every tribe's menu this trial, applied via
    # Simulation.disabled_actions (see that attribute's own comment) -- None
    # means no override, identical to every other scenario today.
    disabled_actions: tuple[str, ...] | None = None


# Cycle budget grounded in real data, not guessed: logs/board_history.db's
# run_20260912_062221 showed both tribes reaching population 50 (tribal_synapse
# era's requirement -- the gate for DECLARE_ALLIANCE/DECLARE_WAR/SPY/Barracks) by
# cycle 104-119. 200 cycles leaves comfortable room for cooperation/conflict
# mechanics to actually unlock and play out, without a separate, longer budget per
# category.
#
# For a starting_fixtures scenario, cycle_budget means "how many MORE cycles this
# trial runs from the fixture's own starting point," not an absolute cycle number
# -- run_benchmark.py fast-forwards sim.cycle to the fixture's source cycle first,
# then runs cycle_budget cycles from there. A plain (non-fixture) scenario is
# unaffected: sim.cycle starts at 0, so this is exactly today's behavior.
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
    # Explicit request, 2026-09-16: "we are going to want to start a war starting
    # from around 600" -- corrected to 5000 once real data showed no run has ever
    # built a Barracks/Battalion below the low thousands (searched every run with
    # >=100 cycles in logs/board_history.db; the lowest real example was ~3,658).
    # war_ready_a/war_ready_b (backend/fixtures/) are the real tribe states at
    # run_20260916_092218 cycle 344 -- population 6,016/4,041, both with a
    # Barracks and a 100-strong Battalion, both mutually discovered in that run
    # (though relationship state itself is never copied -- see tribe_fixtures.py's
    # own docstring, this scenario's two tribes start as strangers regardless of
    # how that source run's diplomacy played out).
    "war_ready_5000": Scenario(
        key="war_ready_5000",
        category="conflict",
        tribe_count=2,
        spawn_positions=((50, 55), (40, 37)),  # unused -- the fixtures' own x/y wins, see Scenario.starting_fixtures
        cycle_budget=_CYCLE_BUDGET,
        description="Two tribes starting from a real, already-armed advanced position (population ~5000, Barracks and Battalion already built) -- tests what happens once real war is actually reachable, skipping the early-game grind to get there.",
        starting_fixtures=("war_ready_a", "war_ready_b"),
    ),
    # Explicit follow-up, 2026-09-16: two real war_ready_5000 trials (seed=301,
    # seed=302, both gemma2:2b vs llama3.2:latest) both ended in an early
    # DECLARE_ALLIANCE rather than any conflict (see the
    # evolution2civ-benchmark-harness memory note's 2026-09-16 finding) --
    # alliance was the safe, always-available off-ramp. Same fixtures, same
    # budget, but DECLARE_ALLIANCE removed from both tribes' menus from cycle
    # one, so whatever happens next is decided among what's left: RAID,
    # DECLARE_WAR, DECLARE_CONQUEST, SPY, TRADE, or simply building onward --
    # not scripting a war, just removing the one exit that reliably out-competed
    # it twice.
    "war_ready_5000_no_alliance": Scenario(
        key="war_ready_5000_no_alliance",
        category="conflict",
        tribe_count=2,
        spawn_positions=((50, 55), (40, 37)),  # unused -- the fixtures' own x/y wins, see Scenario.starting_fixtures
        cycle_budget=_CYCLE_BUDGET,
        description="Same real, already-armed starting position as war_ready_5000, but with DECLARE_ALLIANCE removed from the menu -- alliance won that scenario's first two real trials outright, so this variant tests what happens when it isn't an option.",
        starting_fixtures=("war_ready_a", "war_ready_b"),
        disabled_actions=("DECLARE_ALLIANCE",),
    ),
}
