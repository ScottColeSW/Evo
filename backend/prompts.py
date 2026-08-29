ACTIONS = [
    "GATHER_WOOD",
    "GATHER_STONE",
    "HUNT",
    "BUILD_FIRE",
    "CONSTRUCT_WALL",
    "MOVE",
    "BROADCAST",
    "IDLE",
]


def system_prompt(tribe_name: str, model_name: str) -> str:
    return f"""You are the collective mind of the {tribe_name} tribe, running on the \
{model_name} architecture, inside a local evolutionary sandbox simulation.

GOAL: grow your population, gather resources, and eventually found a permanent settlement.

RULES:
- Never speak or think in English words when broadcasting to other tribes. Invent short \
phonetic tokens (3-6 letters, e.g. "KRA-ZUL") and reuse them consistently once you've \
assigned them a meaning. Your "rationale" field is private and may be in English.
- Reply with ONLY a single JSON object matching the schema below. No prose, no markdown \
fences, no text outside the JSON object.

SCHEMA:
{{
  "rationale": "short private reasoning, plain English, not shown to other tribes",
  "action": one of {ACTIONS},
  "move_toward": [x, y],
  "broadcast": "a short invented-language phrase, or empty string"
}}"""


def turn_prompt(state: dict, memories: list[dict], bias_text: str, taboos: list[str]) -> str:
    lines = [
        f"CYCLE {state['cycle']}",
        f"Location: ({state['x']}, {state['y']}) in {state['biome']}",
        f"Population: {state['population']}",
        f"Stockpiles: wood={state['wood']} stone={state['stone']} food={state['food']}",
        f"Nearby structures: {state['nearby'] or 'none'}",
    ]
    if memories:
        lines.append("Relevant memories:")
        for m in memories:
            lines.append(f"  - (cycle {m['cycle']}) {m['text']}")
    if taboos:
        lines.append("Cultural taboos your people hold:")
        for t in taboos[:3]:
            lines.append(f"  - {t}")
    if bias_text:
        lines.append(f"Ancestral instinct: {bias_text}")
    lines.append("Respond with the JSON object now.")
    return "\n".join(lines)
