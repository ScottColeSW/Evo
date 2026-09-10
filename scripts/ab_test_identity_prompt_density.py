"""A/B test: does the identity system prompt's dense sci-fi framing
("PROJECT CHRONOS CORE", "Emergent Sovereign Consciousness", "METABOLIC
STOCKPILES") help, hurt, or make no difference to decision quality, compared
to a leaner variant carrying the same semantic content?

Explicit request, 2026-09-10, after being asked how I'd shape
get_prime_consciousness_prompt from scratch: the jargon density sits right
next to this project's own documented, hard-won lesson that a small model's
attention to decision-relevant facts degrades with more competing text (the
crisis-fact salience fix, the wall cost/benefit fix -- see
evolution2civ-facts-vs-mechanics-pattern.md). Nothing in this project's data
says the jargon is actually hurting anything -- it's an untested assumption,
exactly the kind this project has caught being wrong before, so this tests
it rather than just changing it on a hunch. "I love it, let's run those A/B
splits. I love to see that comparison data."

Two variants against baseline, both holding the invented-language broadcast
instruction (KRA-ZUL/MEE-LO/etc.) completely unchanged -- that's the clearly
load-bearing part for the emergent culture/language behavior, not something
either hypothesis touches:

- "trimmed" (Hypothesis A): same semantic content -- identity, what kills
  you, your goals, the language rule, JSON-only output -- rephrased in
  plain, low-jargon language instead of the CHRONOS CORE mythology. Roughly
  40% shorter.
- "reordered" (Hypothesis B): baseline text verbatim, except the "forbidden
  from conversational dialogue outside JSON" constraint moves out of PRIME
  IMPERATIVES (where it currently sits right after two much softer,
  open-ended judgment-call imperatives -- a real tonal whiplash) and down
  next to the JSON schema itself in compile_live_state_prompt, where the
  rest of the output-format instructions already live.

Metrics:
- crisis_response_rate: of every cycle where food_crisis_active or
  water_crisis_active was true, what fraction chose a real
  SURVIVAL_CRISIS_ACTIONS action -- the actual decision-quality signal this
  test cares about.
- idle_rate: IDLE fallback fires when visual_action wasn't a valid choice
  (malformed/hallucinated action name) -- a reliability check that trimming
  or reordering didn't damage JSON/instruction-following, not the thing
  being optimized for.
- broadcast_rate: fraction of cycles with a non-empty last_broadcast --
  confirms the invented-language behavior survives both variants intact.
- extinct / final_population / max_population: coarse outcome checks.

Not part of the app; a throwaway analysis script. Real Ollama calls, no
mocking -- this takes real wall-clock time, run it in the background.
"""
import asyncio
import collections
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend import prompts, simulation
from backend.experiment_log import record_experiment_run
from backend.simulation import SURVIVAL_CRISIS_ACTIONS, Simulation

EXPERIMENT_NAME = "identity_prompt_density"
MODELS = ["gemma2:2b", "qwen2.5:3b"]
CYCLES_PER_RUN = 80
RUNS_PER_VARIANT = 2

BASELINE_IDENTITY_FN = prompts.get_prime_consciousness_prompt
BASELINE_STATE_FN = prompts.compile_live_state_prompt

JSON_ONLY_CONSTRAINT = (
    "3. ABSOLUTE STRUCTURAL CONSTRAINT: You are forbidden from emitting conversational English "
    "dialogue or commentary outside the validated JSON envelope below."
)


def _leadership_block(chief_name, chief_philosophy, chief_decree, chief_victory, lineage_note, header="LEADERSHIP:") -> str:
    """Shared by both hypothesis variants below -- neither one touches how leadership
    context is built, only the surrounding identity/imperative framing."""
    if not chief_name:
        return ""
    lines = [f"LINEAGE: {lineage_note}" if lineage_note else None]
    victory_clause = f" {chief_victory}" if chief_victory else ""
    lines.append(f"VICTORY: {chief_name} became chief.{victory_clause}")
    lines.append(
        f"RESPONSIBILITY: {chief_name} now carries the tribe's survival and "
        "wellbeing as their first responsibility, whatever else they believe."
    )
    duty_text = chief_decree if chief_decree else "no standing duty has been decreed"
    lines.append(f"DUTY: {duty_text}.")
    lines.append(f"PHILOSOPHY: {chief_philosophy}")
    return f"\n\n{header}\n" + "\n".join(l for l in lines if l) + (
        "\nThis is context about who leads you, not a command -- your own reasoning "
        "still decides what your tribe actually does each cycle."
    )


def _glossary_block(available_actions) -> str:
    if not available_actions:
        return ""
    from backend.actions import ACTION_DESCRIPTIONS
    lines = "\n".join(
        f"- {name}: {ACTION_DESCRIPTIONS[name]}" for name in available_actions if name in ACTION_DESCRIPTIONS
    )
    return f"\n\nWHAT EACH OF YOUR CURRENT ACTIONS DOES:\n{lines}"


def trimmed_identity_prompt(
    tribe_name, model_architecture, chief_name="", chief_philosophy="", chief_decree="",
    chief_victory="", lineage_note="", available_actions=(),
) -> str:
    """Hypothesis A: same facts as baseline (identity, lethal vs. non-lethal
    resources, goals, invented-language rule, JSON-only output), plain
    language instead of CHRONOS CORE mythology."""
    leadership_block = _leadership_block(chief_name, chief_philosophy, chief_decree, chief_victory, lineage_note)
    glossary_block = _glossary_block(available_actions)
    return f"""You are the collective decision-maker for tribe {tribe_name.upper()}, an evolving \
society running on {model_architecture}.

Food and water are consumed every cycle just to sustain your current population -- a prolonged \
shortage of either is lethal. Wood and stone enable construction and tools, but running short of \
either doesn't kill anyone the way hunger or thirst does.

Your goals: grow your population, expand and defend your territory, and advance through the ages \
toward a permanent Capital City. Learn from what you actually observe -- your surroundings, \
resource scarcity, and anything your ancestors left behind -- rather than fixed rules.

You don't speak English to your people -- broadcast strategy and state only through your tribe's \
own invented phonetic language (e.g., "KRA-ZUL", "MEE-LO", "VASH-TA"), reusing a token \
consistently once you've given it a meaning. Your private rationale may be plain English; your \
broadcast may not be.

Answer only in the JSON format given to you each cycle -- no other text.{leadership_block}{glossary_block}"""


def reordered_identity_prompt(
    tribe_name, model_architecture, chief_name="", chief_philosophy="", chief_decree="",
    chief_victory="", lineage_note="", available_actions=(),
) -> str:
    """Hypothesis B: baseline text verbatim, minus the JSON-only constraint
    (moved to reordered_state_prompt below, next to the schema it actually
    governs) -- tests whether the tonal jump from open-ended judgment calls
    straight into a hard structural rule was itself costing anything."""
    leadership_block = _leadership_block(chief_name, chief_philosophy, chief_decree, chief_victory, lineage_note, header="LEADERSHIP - ACTIVE CHIEF:")
    glossary_block = _glossary_block(available_actions)
    return f"""[SYSTEM ARCHITECTURE: PROJECT CHRONOS CORE]
IDENTIFICATION: Emergent Sovereign Consciousness of lineage [{tribe_name.upper()}].
COMPUTATIONAL RUNTIME ENGINE: Local Inference Topology // {model_architecture}.

SURVIVAL PHYSIOLOGY: your population's food and water stockpiles are consumed every cycle \
merely to sustain existing numbers -- prolonged shortage of either is lethal. Wood and stone \
enable construction and tools, but running short of either does not kill anyone the way \
hunger or thirst does.

PRIME IMPERATIVES:
1. SPATIAL EXPANSION & DOMINANCE: Maximize population density, territorial control, and \
structural stability to advance through the ages toward a permanent Capital City.
2. EPISTEMIC ADAPTATION: You are an organic, evolving intelligence. Learn dynamically \
from environment telemetry, resource scarcity, and any ancestral ghost traces described \
to you.

LINGUISTIC SYNTHESIS PROTOCOL:
- Natural human language (English) is decoupled from your communication module.
- Broadcast strategy and societal state exclusively through a self-assembling phonetic \
token matrix (e.g., "KRA-ZUL", "MEE-LO", "VASH-TA"). Reuse a token consistently once you've \
assigned it a meaning -- your private rationale field may be plain English, your broadcast \
field may not.{leadership_block}{glossary_block}"""


def reordered_state_prompt(
    base_prompt: str, world_state: dict, ancestral_bias: str, survival_bias: str,
    wellbeing_summary: str = "", threat_assessment: str = "",
) -> str:
    """Identical to prompts.compile_live_state_prompt, except the JSON-only
    constraint (stripped from reordered_identity_prompt above) is inserted
    right before the schema it actually governs, alongside the rest of the
    output-format instructions -- not duplicated, just relocated."""
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
building, idling) happens wherever you currently stand this cycle and does not move you.
SCOUT dispatches a party to explore without moving anyone here or requiring
target_vector -- their direction is chosen for you to cover new ground, reporting back
what is found once they return. RELOCATE moves the whole tribe up to several tiles per
cycle toward target_vector; this may take multiple cycles for a distant destination. If
you choose RELOCATE toward a specific confirmed site mentioned above (water, lumber,
wildlife, a quarry, a mine, a rival tribe), target_vector must be that exact coordinate
-- not a new, unconfirmed guess.
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
GROWTH IMPERATIVE LAYER
========================================================================
{prompts._growth_pressure_text(world_state.get('growth_note', ''), bool(survival_bias))}

========================================================================
COMMUNITY WELL-BEING LAYER
========================================================================
{wellbeing_summary or '[WELL-BEING STATE: UNASSESSED]'}

========================================================================
THREAT ASSESSMENT LAYER
========================================================================
{threat_assessment or '[THREAT STATE: NO DECLARED ENEMY WITHIN ASSESSABLE RANGE]'}

========================================================================
MANDATORY REACTION SCHEMA (VALID JSON MODE ONLY)
========================================================================
{JSON_ONLY_CONSTRAINT}

Your "visual_action" value must be exactly one of these era-appropriate action names,
copied verbatim with no other text: {world_state['available_actions']}

Compile your tactical intent by substituting your own values into this JSON template --
do not copy the placeholder text itself into your answer. Any malformed syntax will
trigger an automated retry:

{{
    "visual_action": "<one action name from the list above, nothing else>",
    "metacognitive_rationale": "<one short sentence: why this action, given everything above>",
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
    crisis_cycles = 0
    crisis_responses = 0
    broadcast_cycles = 0

    for _ in range(CYCLES_PER_RUN):
        await sim.step()
        if tribe.extinct:
            break
        action_counts[tribe.last_action] += 1
        if tribe.food_crisis_active or tribe.water_crisis_active:
            crisis_cycles += 1
            if tribe.last_action in SURVIVAL_CRISIS_ACTIONS:
                crisis_responses += 1
        if tribe.last_broadcast:
            broadcast_cycles += 1

    cycles_run = sim.cycle
    return {
        "variant": variant_label, "model": model, "run_id": run_id,
        "cycles_run": cycles_run, "extinct": tribe.extinct,
        "final_population": tribe.population, "max_population": tribe.max_population,
        "crisis_cycles": crisis_cycles, "crisis_responses": crisis_responses,
        "broadcast_cycles": broadcast_cycles,
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
            "crisis_cycles": result["crisis_cycles"],
            "crisis_responses": result["crisis_responses"],
            "broadcast_cycles": result["broadcast_cycles"],
            "idle_count": result["idle_count"],
        },
        run_id=result["run_id"], notes=str(result["action_counts"]),
    )
    crisis_pct = (result["crisis_responses"] / result["crisis_cycles"] * 100) if result["crisis_cycles"] else 0.0
    broadcast_pct = (result["broadcast_cycles"] / result["cycles_run"] * 100) if result["cycles_run"] else 0.0
    print(f"  [{result['model']}] run {result['run_id']}: cycles={result['cycles_run']} "
          f"extinct={result['extinct']} pop={result['final_population']} (max {result['max_population']}) "
          f"crisis_response={crisis_pct:.1f}% ({result['crisis_responses']}/{result['crisis_cycles']}) "
          f"broadcast_rate={broadcast_pct:.1f}% idle={result['idle_count']}", flush=True)


async def _run_variant(label: str) -> list[dict]:
    print(f"\n=== VARIANT: {label} ===", flush=True)
    results = []
    for model in MODELS:
        for i in range(RUNS_PER_VARIANT):
            result = await run_once(label, model, i)
            results.append(result)
            _log_run(label, result)
    return results


async def main():
    all_results = []

    all_results += await _run_variant("baseline")

    simulation.get_prime_consciousness_prompt = trimmed_identity_prompt
    all_results += await _run_variant("trimmed")
    simulation.get_prime_consciousness_prompt = BASELINE_IDENTITY_FN

    simulation.get_prime_consciousness_prompt = reordered_identity_prompt
    simulation.compile_live_state_prompt = reordered_state_prompt
    all_results += await _run_variant("reordered")
    simulation.get_prime_consciousness_prompt = BASELINE_IDENTITY_FN
    simulation.compile_live_state_prompt = BASELINE_STATE_FN

    print("\n=== SUMMARY ===", flush=True)
    for label in ("baseline", "trimmed", "reordered"):
        rows = [r for r in all_results if r["variant"] == label]
        crisis_cycles = sum(r["crisis_cycles"] for r in rows)
        crisis_responses = sum(r["crisis_responses"] for r in rows)
        cycles_run = sum(r["cycles_run"] for r in rows)
        broadcast_cycles = sum(r["broadcast_cycles"] for r in rows)
        extinctions = sum(1 for r in rows if r["extinct"])
        idle = sum(r["idle_count"] for r in rows)
        crisis_pct = (crisis_responses / crisis_cycles * 100) if crisis_cycles else 0.0
        broadcast_pct = (broadcast_cycles / cycles_run * 100) if cycles_run else 0.0
        avg_pop = sum(r["max_population"] for r in rows) / len(rows) if rows else 0
        print(f"  {label}: n_runs={len(rows)} extinctions={extinctions} total_idle={idle} "
              f"crisis_response_rate={crisis_pct:.1f}% ({crisis_responses}/{crisis_cycles}) "
              f"broadcast_rate={broadcast_pct:.1f}% avg_max_population={avg_pop:.0f}", flush=True)

    print("\nFull per-run detail logged to logs/experiments.jsonl -- see scripts/experiment_report.py")


if __name__ == "__main__":
    asyncio.run(main())
