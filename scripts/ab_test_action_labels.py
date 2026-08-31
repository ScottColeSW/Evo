"""A/B test: does answering in plain language instead of a code-shaped enum token
change which actions get chosen?

Two prior findings motivate this, in order:

1. scripts/ab_test_leader_framing.py showed that presenting the choice as a "leader's
   daily options list" (prose descriptions, not a bare enum tuple) produced the only
   run in that whole experiment where a tribe settled AND actually used PLANT_CROP/
   GATHER_EGGS -- a 44.7% vs 0% settlement-action rate once settled. But even that
   variant still ended with "Your visual_action value must be exactly one of the
   action names above, copied verbatim: GATHER_FOOD, PLANT_CROP, ..." -- the model
   reads a prose menu, then has to answer back in code syntax anyway, right at the
   most salient point before it generates.

2. scripts/headless_runs.py's post-fishing-rebalance batch showed raw incentive
   tuning alone does nothing: GATHER_FISH went from an 8-16 coinflip to a 14-24 catch
   at 80% odds (EV 15.2, beating both GATHER_FOOD and HUNT_DEER, zero hazard, zero
   cost) and was STILL chosen exactly once across 6 tribe-runs. The bottleneck isn't
   the payoff, it's that the model reflexively answers with whatever enum token most
   directly echoes the survival fact it just read (GATHER_FOOD echoing "gather
   food"), not a reasoned comparison of options.

This variant combines both fixes: options are still framed as a leader's daily list
(reusing the ab_test_leader_framing wording, already shown to help), but the model is
asked to answer in plain language too -- "Catch fish to bring home" instead of
GATHER_FISH -- via a static, exact-match label<->action table (backend/simulation.py's
_resolve_action is temporarily swapped for a label-aware version during this variant).
No fuzzy LLM interpretation, no added inference cost: still a deterministic string
match, just against a natural-language label instead of a code token. It also assigns
the model an explicit leader role directly ("You are Chief {name}, deciding for your
tribe this cycle") -- the leader_list variant only ever talked ABOUT leadership in the
abstract, never actually cast the model in that seat.

Metric: same as ab_test_leader_framing.py -- rate of "direct" food actions (GATHER_FOOD,
HUNT_DEER) vs. "settlement" actions (PLANT_CROP, GATHER_EGGS, GATHER_FISH,
HUNTING_PARTY) chosen, counted only across cycles where has_ever_settled is true. Also
tracks confusion_count -- cycles where Tribe.last_confusion was set, i.e. a genuine
parse failure under _resolve_action (or this variant's label-aware wrapper) -- as a
sanity check that answering in plain language didn't make the model's output *less*
reliably parseable than the enum baseline.

Not part of the app; a throwaway analysis script.
"""
import asyncio
import collections
import difflib
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend import prompts, simulation
from backend.actions import ACTION_DESCRIPTIONS
from backend.experiment_log import record_experiment_run
from backend.simulation import Simulation

EXPERIMENT_NAME = "plain_language_action_labels"
MODELS = ["gemma2:2b", "qwen2.5:3b"]
CYCLES_PER_RUN = 120
RUNS_PER_VARIANT = 2

DIRECT_FOOD_ACTIONS = ("GATHER_FOOD", "HUNT_DEER")
SETTLEMENT_ACTIONS = ("PLANT_CROP", "GATHER_EGGS", "GATHER_FISH", "HUNTING_PARTY")

BASELINE_PROMPT_FN = prompts.compile_live_state_prompt
ORIGINAL_RESOLVE_ACTION = simulation._resolve_action

# A label per action the model could plausibly be asked to attempt -- IDLE
# deliberately has none, same as ACTION_DESCRIPTIONS (it's never offered).
ACTION_LABELS = {
    "GATHER_WOOD": "Chop wood",
    "GATHER_STONE": "Quarry stone",
    "GATHER_WATER": "Fetch water",
    "GATHER_FOOD": "Forage for food",
    "HUNT_DEER": "Hunt game nearby",
    "BUILD_FIRE": "Build a fire",
    "CONSTRUCT_WALL": "Build a wall",
    "PLANT_CROP": "Plant a crop",
    "GATHER_EGGS": "Gather eggs",
    "GATHER_FISH": "Catch fish to bring home",
    "SCOUT": "Send scouts to explore",
    "HUNTING_PARTY": "Send a hunting party",
    "RELOCATE": "Move the whole tribe",
    "BREED": "Start a family",
    "RAID": "Raid a rival tribe",
    "TRADE": "Trade with a rival tribe",
}
ACTION_BY_LABEL = {label: action for action, label in ACTION_LABELS.items()}


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().upper().split())


NORMALIZED_LABEL_TO_ACTION = {_normalize(label): action for action, label in ACTION_LABELS.items()}


def label_aware_resolve_action(raw: str, available_actions: list[str]) -> tuple[str, str | None]:
    """Same exact-match-first discipline as the production resolver, just against
    labels instead of enum tokens: exact normalized label match, then a fuzzy
    close-match restricted to labels actually on offer this cycle, then falls
    through to the real production resolver (covers a model that answers in the old
    enum form anyway, or gets an out-of-era action name)."""
    normalized = _normalize(raw)
    action = NORMALIZED_LABEL_TO_ACTION.get(normalized)
    if action is not None and action in available_actions:
        return action, None

    available_labels = [ACTION_LABELS[a] for a in available_actions if a in ACTION_LABELS]
    close = difflib.get_close_matches(str(raw).strip(), available_labels, n=1, cutoff=0.6)
    if close:
        return ACTION_BY_LABEL[close[0]], None

    return ORIGINAL_RESOLVE_ACTION(raw, available_actions)


def label_list_prompt(base_prompt: str, world_state: dict, ancestral_bias: str, survival_bias: str) -> str:
    """Leader-list framing (ab_test_leader_framing's wording, already shown to help)
    plus two more changes: an explicit leader role instead of talking about
    leadership in the abstract, and plain-language option labels instead of enum
    tokens for the model to answer back with."""
    options_block = "\n".join(
        f"- {ACTION_LABELS[name]}: {ACTION_DESCRIPTIONS.get(name, '')}"
        for name in world_state["available_actions"] if name in ACTION_LABELS
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

MOVEMENT: Only "Move the whole tribe" actually relocates -- every other action
(gathering, hunting, building, scouting) happens wherever you currently stand this
cycle and does not move you. "Send scouts to explore" looks at target_vector without
moving anyone there, reporting back what is found. "Move the whole tribe" moves
everyone up to several tiles per cycle toward target_vector; this may take multiple
cycles for a distant destination.
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
YOU ARE THE LEADER. TODAY'S OPTIONS
========================================================================
You are your tribe's leader, deciding what they do this cycle. A leader doesn't get
orders -- they get a list of what's actually open to them today, weighed by real
urgency, and they choose. Here is that list for this cycle:

{options_block}

Pick whichever one actually serves the tribe best right now. That judgment is yours --
this list states what's possible, not what to do.

Compile your tactical intent by substituting your own values into this JSON template --
do not copy the placeholder text itself into your answer. Any malformed syntax will
trigger an automated retry:

{{
    "metacognitive_rationale": "<answer this: given everything above, what will your tribe do this cycle, and why? one short sentence>",
    "visual_action": "<the exact wording of one option above, nothing else -- for example, \\"Fetch water\\", not a code name>",
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
    confusion_count = 0

    for _ in range(CYCLES_PER_RUN):
        await sim.step()
        if tribe.extinct:
            break
        action_counts[tribe.last_action] += 1
        if tribe.last_confusion is not None:
            confusion_count += 1
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
        "confusion_count": confusion_count,
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
            "confusion_count": result["confusion_count"],
        },
        run_id=result["run_id"], notes=str(result["action_counts"]),
    )
    print(f"  [{result['model']}] run {result['run_id']}: cycles={result['cycles_run']} "
          f"extinct={result['extinct']} settled={result['has_ever_settled']} "
          f"settled_cycles={result['settled_cycles']} "
          f"direct_food={result['direct_food_while_settled']} "
          f"settlement_actions={result['settlement_actions_while_settled']} "
          f"confusion={result['confusion_count']}", flush=True)


async def main():
    all_results = []

    print("=== VARIANT: baseline (current shipped bare-enum schema) ===", flush=True)
    for model in MODELS:
        for i in range(RUNS_PER_VARIANT):
            result = await run_once("baseline", model, i)
            all_results.append(result)
            _log_run("baseline", result)

    simulation.compile_live_state_prompt = label_list_prompt
    simulation._resolve_action = label_aware_resolve_action
    print("\n=== VARIANT: plain_language (leader role + plain-language labels) ===", flush=True)
    for model in MODELS:
        for i in range(RUNS_PER_VARIANT):
            result = await run_once("plain_language", model, i)
            all_results.append(result)
            _log_run("plain_language", result)
    simulation.compile_live_state_prompt = BASELINE_PROMPT_FN
    simulation._resolve_action = ORIGINAL_RESOLVE_ACTION

    print("\n=== SUMMARY ===", flush=True)
    for label in ("baseline", "plain_language"):
        rows = [r for r in all_results if r["variant"] == label]
        settled_cycles = sum(r["settled_cycles"] for r in rows)
        direct_food = sum(r["direct_food_while_settled"] for r in rows)
        settlement = sum(r["settlement_actions_while_settled"] for r in rows)
        extinctions = sum(1 for r in rows if r["extinct"])
        confusion = sum(r["confusion_count"] for r in rows)
        total_cycles = sum(r["cycles_run"] for r in rows)
        direct_pct = (direct_food / settled_cycles * 100) if settled_cycles else 0.0
        settle_pct = (settlement / settled_cycles * 100) if settled_cycles else 0.0
        confusion_pct = (confusion / total_cycles * 100) if total_cycles else 0.0
        print(f"  {label}: n_runs={len(rows)} extinctions={extinctions} "
              f"settled_cycles={settled_cycles} "
              f"direct_food_rate={direct_pct:.1f}% settlement_action_rate={settle_pct:.1f}% "
              f"confusion_rate={confusion_pct:.1f}% (of {total_cycles} total cycles)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
