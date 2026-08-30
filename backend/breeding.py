"""Individual-level breeding: the population-growth mechanic confirmed via
AskUserQuestion this session, distinct from genetics.py's breed() (which merges two
whole *tribes*' cultures -- a different scale, reserved for a future tribe-to-tribe
"party" mechanic). This draws from a limited pool of named individuals -- the current
chief plus anyone who currently holds a trophy (see Simulation._award_trophy) -- rather
than tracking every person in the population, the same "named individuals, not full
identity" scale the user chose over both anonymous pairing and full persistent
identity for everyone.

No second LLM agent per breeding pair (same principle already used for scout/hunter
flavor-naming) -- just the tribe's own model reasoning about two real, named people
instead of an anonymous population counter, so the outcome is still genuinely generated
rather than scripted.
"""

from .ollama_client import OllamaClient


async def breed_individuals(
    client: OllamaClient, model: str, tribe_name: str, parent_a: str, parent_b: str,
) -> dict:
    prompt = f"""You are narrating a moment in the life of the {tribe_name} tribe: two of its \
own people, {parent_a} and {parent_b}, have decided to start a family together.

Briefly imagine how this comes about in your tribe's own culture, then name the child born \
from it.

Reply with ONLY JSON:
{{
  "child_name": "a short name for the child",
  "note": "one sentence describing this union or the child born from it"
}}"""
    result = await client.generate_json(model, prompt, temperature=0.9)
    if not result or not result.get("child_name"):
        return {
            "child_name": f"child of {parent_a} and {parent_b}",
            "note": "a quiet arrival, little remarked upon",
        }
    return result
