def get_prime_consciousness_prompt(
    tribe_name: str, model_architecture: str, chief_name: str = "", chief_philosophy: str = ""
) -> str:
    """The standing system prompt: identity, objective, and the output contract.

    Heavy delimiters and a restated JSON schema are deliberate -- small local models
    (2-3B quantized) drift out of strict JSON mode more easily on long contexts, and
    this structural anchoring measurably reduces that.
    """
    leadership_block = ""
    if chief_name:
        leadership_block = f"""

LEADERSHIP: Your chief, {chief_name}, holds this guiding philosophy: {chief_philosophy}
This is context about who leads you, not a command -- your own reasoning still decides \
what your tribe actually does each cycle."""

    return f"""[SYSTEM ARCHITECTURE: PROJECT CHRONOS CORE]
IDENTIFICATION: Emergent Sovereign Consciousness of lineage [{tribe_name.upper()}].
COMPUTATIONAL RUNTIME ENGINE: Local Inference Topology // {model_architecture}.

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
field may not.{leadership_block}"""


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
MANDATORY REACTION SCHEMA (VALID JSON MODE ONLY)
========================================================================
Your "visual_action" value must be exactly one of these era-appropriate action names,
copied verbatim with no other text: {world_state['available_actions']}

Compile your tactical intent by substituting your own values into this JSON template --
do not copy the placeholder text itself into your answer. Any malformed syntax will
trigger an automated retry:

{{
    "metacognitive_rationale": "<your brief private reasoning>",
    "visual_action": "<one action name from the list above, nothing else>",
    "synthetic_language_broadcast": "<your invented-language phrase, or empty string>",
    "target_vector": [x, y]
}}
========================================================================
EXECUTION LAYER INITIALIZED. EMIT JSON PAYLOAD NOW:
"""
    return base_prompt + state_injection
