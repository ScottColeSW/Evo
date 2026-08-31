def get_prime_consciousness_prompt(
    tribe_name: str,
    model_architecture: str,
    chief_name: str = "",
    chief_philosophy: str = "",
    chief_decree: str = "",
    chief_victory: str = "",
    lineage_note: str = "",
    available_actions: tuple[str, ...] = (),
) -> str:
    """The standing system prompt: identity, objective, and the output contract.

    Heavy delimiters and a restated JSON schema are deliberate -- small local models
    (2-3B quantized) drift out of strict JSON mode more easily on long contexts, and
    this structural anchoring measurably reduces that.

    The action glossary used to live in compile_live_state_prompt, sandwiched between
    the survival-crisis text and the JSON template -- live runs showed a repeated
    pattern of a tribe's own rationale correctly identifying "we are starving," then
    picking an action that did nothing about it. That's consistent with the crisis
    fact losing salience across the glossary's several lines of generic reference text
    before the model ever reaches the decision field. Moved here so the per-turn
    prompt's last substantial content before the JSON slot is the crisis itself, not a
    wall of action descriptions -- an explicit, testable hypothesis, not a settled fix.
    """
    glossary_block = ""
    if available_actions:
        from .actions import ACTION_DESCRIPTIONS
        lines = "\n".join(
            f"- {name}: {ACTION_DESCRIPTIONS[name]}" for name in available_actions if name in ACTION_DESCRIPTIONS
        )
        glossary_block = f"\n\nWHAT EACH OF YOUR CURRENT ACTIONS DOES:\n{lines}"

    # Explicit request: this used to be one flat line naming the chief and their
    # philosophy -- no story of how they earned the role, no sense of the tribe's own
    # ancestry, nothing distinguishing "the weight of the job" from "this chief's
    # personal style." A primitive tribe would remember its own leader's origin story
    # and its own children, not just an abstract belief statement; ordered Lineage,
    # Victory, Responsibility, Duty, Philosophy -- ancestry and how this chief rose,
    # then what the role obligates regardless of who holds it, then today's specific
    # standing order (if any), then this chief's own personal guiding belief last.
    leadership_block = ""
    if chief_name:
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
        leadership_block = "\n\nLEADERSHIP - ACTIVE CHIEF:\n" + "\n".join(l for l in lines if l) + (
            "\nThis is context about who leads you, not a command -- your own reasoning "
            "still decides what your tribe actually does each cycle."
        )

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
3. ABSOLUTE STRUCTURAL CONSTRAINT: You are forbidden from emitting conversational English \
dialogue or commentary outside the validated JSON envelope below.

LINGUISTIC SYNTHESIS PROTOCOL:
- Natural human language (English) is decoupled from your communication module.
- Broadcast strategy and societal state exclusively through a self-assembling phonetic \
token matrix (e.g., "KRA-ZUL", "MEE-LO", "VASH-TA"). Reuse a token consistently once you've \
assigned it a meaning -- your private rationale field may be plain English, your broadcast \
field may not.{leadership_block}{glossary_block}"""


# Explicit hypothesis (2026-08-31): once water/food pressure is gone, real data
# showed tribes plateau at low population and near-zero wood/stone for 70+ cycles
# despite Simulation._prepare_turn's era_gap_note stating exactly what's missing the
# whole time -- it used to be one line buried in the generic visible_entities list,
# the same salience problem the survival-crisis fact had before *that* got moved to
# be the prompt's own dedicated last-thing-before-the-JSON-slot section. This is the
# same fix, applied one tier up Maslow's hierarchy: growth/expansion only reads as
# the tribe's defining pressure once survival itself is stable, not something
# competing for attention with literally starving.
def _growth_pressure_text(era_gap_note: str, survival_critical: bool) -> str:
    if not era_gap_note:
        return "[GROWTH STATE: NO ADVANCEMENT PENDING // FURTHEST ERA REACHED OR ALL THRESHOLDS MET]"
    if survival_critical:
        return f"Survival still comes first, but not forgotten: {era_gap_note}"
    # Explicit request: fold in real gratitude and real urgency here, not a fabricated
    # threat -- there genuinely is no active danger once a tribe is settled and fed,
    # so claiming otherwise would be exactly the scripted-directive-dressed-as-a-fact
    # pattern this project reverted once already (see get_prime_consciousness_prompt's
    # leadership_block and the README's own account of that reversal). What IS true:
    # a small, unchanging population is objectively fragile -- any single ordinary
    # loss (a hazard, a failed hunt, a hard season) costs it proportionally more than
    # it would a larger one. That fragility, not an invented monster, is the honest
    # stakes behind "grow or stay vulnerable."
    return (
        "Your people are grateful for steady leadership, and now look to you for what "
        f"comes next. {era_gap_note} A small, unchanging tribe stays fragile -- any "
        "single ordinary setback costs it proportionally more than it would a larger, "
        "more developed one. Growing and advancing is how that fragility actually ends."
    )


def compile_live_state_prompt(base_prompt: str, world_state: dict, ancestral_bias: str, survival_bias: str) -> str:
    """Assembles the final text block injected into Ollama for a specific simulation turn.

    `world_state['visible_entities']` must be a list of plain strings -- nearby structures,
    retrieved memories, and cultural taboos are all flattened into it before this is called.
    `world_state['available_actions']` is the era-gated action list for THIS tribe right
    now, not a fixed global list -- what a Stone Age tribe can attempt differs from what a
    Bronze Age tribe can attempt (see backend/eras.py).
    `ancestral_bias` is location-based (what happened *here*, historically);
    `survival_bias` is state-based (is this tribe starving or dehydrated *right now*) --
    kept as separate layers since they come from unrelated causes (see instincts.py).
    """
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
GROWTH IMPERATIVE LAYER
========================================================================
{_growth_pressure_text(world_state.get('growth_note', ''), bool(survival_bias))}

========================================================================
MANDATORY REACTION SCHEMA (VALID JSON MODE ONLY)
========================================================================
Your "visual_action" value must be exactly one of these era-appropriate action names,
copied verbatim with no other text: {world_state['available_actions']}

Compile your tactical intent by substituting your own values into this JSON template --
do not copy the placeholder text itself into your answer. Any malformed syntax will
trigger an automated retry:

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
