"""A/B test: does *how* the turn's decision is asked change which actions get chosen?

Headless data (scripts/headless_action_data_log.txt, scripts/headless_fishing_log.txt)
showed GATHER_FOOD dominating every single run's action counts (2-6x any other action),
while PLANT_CROP/GATHER_EGGS/GATHER_FISH were picked essentially never -- even after
GATHER_FISH was added with an *immediate* payoff identical to GATHER_FOOD's. That
argues against "delayed vs. immediate reward" as the explanation.

User's hypothesis: the current schema hands the model a bare, code-shaped list of enum
tokens ("copied verbatim with no other text: (...)") right next to survival facts
phrased as natural language ("gather food or send a hunting party now"). A model may
simply pattern-match the lexically closest action name to whatever fact it just read
(GATHER_FOOD echoes "gather food" almost verbatim) rather than reasoning about a less
directly-worded option like farming/fishing/eggs, even once nudged toward it.

This tests a reframing of ONLY the decision-presentation text -- not the underlying
choice, not the JSON output contract (still the same strict schema, still "copied
verbatim," still the same retry-on-malformed-JSON warning) -- from a bare enum list to
a "leader reviewing today's options" framing that names each action's own description
inline, the way a person's daily task list would, not a function signature.

Metric: rate of "direct" food actions (GATHER_FOOD, HUNT_DEER) vs. "settlement" actions
(PLANT_CROP, GATHER_EGGS, GATHER_FISH, HUNTING_PARTY) chosen, counted only across
cycles where has_ever_settled is true (since the settlement actions aren't even offered
before that -- an unfair denominator otherwise). Also tracks extinction rate and IDLE
rate as coarse sanity checks that the looser framing didn't damage JSON reliability
(an IDLE fallback fires when the model's own visual_action wasn't a valid choice).

Not part of the app; a throwaway analysis script.
"""
import asyncio
import collections
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend import prompts, simulation
from backend.actions import ACTION_DESCRIPTIONS
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "leader_daily_list_framing"
MODELS = ["gemma2:2b", "qwen2.5:3b"]
CYCLES_PER_RUN = 120
RUNS_PER_VARIANT = 2

DIRECT_FOOD_ACTIONS = ("GATHER_FOOD", "HUNT_DEER")
SETTLEMENT_ACTIONS = ("PLANT_CROP", "GATHER_EGGS", "GATHER_FISH", "HUNTING_PARTY")

BASELINE_FN = prompts.compile_live_state_prompt


def leader_list_prompt(base_prompt: str, world_state: dict, ancestral_bias: str, survival_bias: str) -> str:
    """Same facts, same JSON output contract -- only the presentation of the choice
    itself changes, from a bare enum list to a considered daily list with each
    option's own description inline."""
    options_block = "\n".join(
        f"- {name}: {ACTION_DESCRIPTIONS.get(name, '')}" for name in world_state["available_actions"]
    )
    state_injection = f"""
========================================================================
LIVE CORE TELEMETRY: CYCLE {world_state['cycle']}
========================================================================
SPATIAL VECTOR: X: {world_state['x']} // Y: {world_state['y']}
TOPOGRAPHICAL REGION: {world_state.get('biome_label', world_state['biome'])}
CURRENT ERA: {world_state['era']}

METABOLIC STOCKPILES:
- Population Density: {world_state['population']} units
- Resource Repositories: Wood: {world_state['wood']} | Stone: {world_state['stone']} | Food: {world_state['food']} | Water: {world_state['water']}

VISUAL RENDER LAYER SCAN:
Immediate Grid Entity Array: [{', '.join(world_state['visible_entities'])}]

MOVEMENT: Only RELOCATE moves your tribe -- every other action (gathering, hunting,
building, scouting, idling) happens wherever you currently stand this cycle and does not
move you. SCOUT looks at target_vector without moving anyone there, reporting back what
is found. RELOCATE moves the whole tribe up to several tiles per cycle toward
target_vector; this may take multiple cycles for a distant destination.
{world_state.get('journey_note') or ''}

========================================================================
EPISTEMOLOGICAL INHERITANCE LAYER
========================================================================
{ancestral_bias or '[ANCESTRAL MATRIX STATE: NEUTRAL // NO INHERITED BIAS FIELD DETECTED]'}

========================================================================
SURVIVAL INSTINCT LAYER
========================================================================
{survival_bias or '[SURVIVAL STATE: STABLE // NO IMMEDIATE PHYSIOLOGICAL PRESSURE]'}

========================================================================
TODAY'S OPTIONS
========================================================================
A leader doesn't get orders -- they get a list of what's actually open to them today, \
weighed by real urgency, and they choose. Here is that list for this cycle:

{options_block}

Pick whichever one actually serves the tribe best right now. That judgment is yours --
this list states what's possible, not what to do.

Your "visual_action" value must still be exactly one of the action names above, copied
verbatim with no other text. Compile your tactical intent by substituting your own
values into this JSON template -- do not copy the placeholder text itself into your
answer. Any malformed syntax will trigger an automated retry:

{{
    "metacognitive_rationale": "<answer this: given everything above, what will your tribe do this cycle, and why? one short sentence>",
    "visual_action": "<one action name from the list above, nothing else>",
    "synthetic_language_broadcast": "<your invented-language phrase, or empty string>",
    "target_vector": [x, y]
}}
========================================================================
EXECUTION LAYER INITIALIZED. EMIT JSON PAYLOAD NOW:
"""
    return base_prompt + state_injection


async def run_once(variant_label: str, model: str, run_id: int) -> dict:
    sim = await Simulation.create([{"name": "Test Tribe", "model": model}])
    tribe = next(iter(sim.tribes.values()))
    action_counts = collections.Counter()
    settled_action_counts = collections.Counter()
    settled_cycles = 0

    for _ in range(CYCLES_PER_RUN):
        await sim.step()
        if tribe.extinct:
            break
        action_counts[tribe.last_action] += 1
        if tribe.has_ever_settled:
            settled_cycles += 1
            settled_action_counts[tribe.last_action] += 1

    direct_food = sum(settled_action_counts.get(a, 0) for a in DIRECT_FOOD_ACTIONS)
    settlement = sum(settled_action_counts.get(a, 0) for a in SETTLEMENT_ACTIONS)

    return {
        "variant": variant_label, "model": model, "run_id": run_id,
        "cycles_run": sim.cycle, "extinct": tribe.extinct,
        "final_population": tribe.population, "max_population": tribe.max_population,
        "has_ever_settled": tribe.has_ever_settled,
        "settled_cycles": settled_cycles,
        "direct_food_while_settled": direct_food,
        "settlement_actions_while_settled": settlement,
        "idle_count": action_counts.get("IDLE", 0),
        "action_counts": dict(action_counts.most_common()),
    }


def _log_run(variant_label: str, result: dict) -> None:
    record_experiment_run(
        EXPERIMENT_NAME, variant_label, result["model"],
        metrics={
            "cycles_run": result["cycles_run"],
            "extinct": result["extinct"],
            "final_population": result["final_population"],
            "max_population": result["max_population"],
            "has_ever_settled": result["has_ever_settled"],
            "settled_cycles": result["settled_cycles"],
            "direct_food_while_settled": result["direct_food_while_settled"],
            "settlement_actions_while_settled": result["settlement_actions_while_settled"],
            "idle_count": result["idle_count"],
        },
        run_id=result["run_id"], notes=str(result["action_counts"]),
    )
    print(f"  [{result['model']}] run {result['run_id']}: cycles={result['cycles_run']} "
          f"extinct={result['extinct']} settled={result['has_ever_settled']} "
          f"settled_cycles={result['settled_cycles']} "
          f"direct_food={result['direct_food_while_settled']} "
          f"settlement_actions={result['settlement_actions_while_settled']} "
          f"idle={result['idle_count']}", flush=True)


async def main():
    all_results = []

    print("=== VARIANT: baseline (current bare-enum schema) ===", flush=True)
    for model in MODELS:
        for i in range(RUNS_PER_VARIANT):
            result = await run_once("baseline", model, i)
            all_results.append(result)
            _log_run("baseline", result)

    simulation.compile_live_state_prompt = leader_list_prompt
    print("\n=== VARIANT: leader_list (daily-options framing) ===", flush=True)
    for model in MODELS:
        for i in range(RUNS_PER_VARIANT):
            result = await run_once("leader_list", model, i)
            all_results.append(result)
            _log_run("leader_list", result)
    simulation.compile_live_state_prompt = BASELINE_FN

    print("\n=== SUMMARY ===", flush=True)
    for label in ("baseline", "leader_list"):
        rows = [r for r in all_results if r["variant"] == label]
        settled_cycles = sum(r["settled_cycles"] for r in rows)
        direct_food = sum(r["direct_food_while_settled"] for r in rows)
        settlement = sum(r["settlement_actions_while_settled"] for r in rows)
        extinctions = sum(1 for r in rows if r["extinct"])
        idle = sum(r["idle_count"] for r in rows)
        direct_pct = (direct_food / settled_cycles * 100) if settled_cycles else 0.0
        settle_pct = (settlement / settled_cycles * 100) if settled_cycles else 0.0
        print(f"  {label}: n_runs={len(rows)} extinctions={extinctions} total_idle={idle} "
              f"settled_cycles={settled_cycles} "
              f"direct_food_rate={direct_pct:.1f}% settlement_action_rate={settle_pct:.1f}%", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
