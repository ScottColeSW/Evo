"""The "night cycle": periodically, a larger reviewing model looks back at a tribe's
own recent history and decides for itself whether its guiding philosophy should
change. This is the piece from the original design transcript that was never built --
see the genetics_and_night_cycle_gap memory note -- distinct from breed()/breed_
individuals' cross-tribe/cross-individual crossover. Fast small models handle every
live turn; this is the "day reviewed at night" pass, run much less often, using a
different (larger) model than whatever the tribe uses live.

Still a real, non-scripted LLM call reasoning over real facts (the tribe's own recent
chronicle, its current philosophy) -- the simulation states what actually happened, the
model decides for itself whether and how its philosophy should change, same principle
already used for elect_chief and breed_individuals. No hardcoded rule ties any specific
event pattern to any specific philosophy change.
"""

from .ollama_client import OllamaClient


async def reflect_on_history(
    client: OllamaClient, reviewer_model: str, tribe_name: str,
    current_philosophy: str, recent_events: list[str],
) -> dict:
    events_block = "\n".join(f"- {e}" for e in recent_events) or "(nothing notable recorded)"
    prompt = f"""You are reviewing, from a distance, the recent history of the {tribe_name} \
tribe. Its current guiding philosophy is: "{current_philosophy}"

Here is what has actually happened recently, in order:
{events_block}

Consider honestly whether this philosophy is still serving the tribe well, given what \
actually happened -- not what should have happened. You may keep it unchanged, adjust it, \
or replace it entirely. That judgment is yours to make, based on the tribe's own real \
experience.

Reply with ONLY JSON:
{{
  "revised_philosophy": "the guiding philosophy going forward, whether changed or the same",
  "changed": true or false,
  "reasoning": "one sentence explaining your decision"
}}"""
    result = await client.generate_json(reviewer_model, prompt, temperature=0.7, num_ctx=8192)
    if not result or not result.get("revised_philosophy"):
        return {
            "revised_philosophy": current_philosophy,
            "changed": False,
            "reasoning": "the review produced nothing usable; philosophy stands unchanged",
        }
    return result
