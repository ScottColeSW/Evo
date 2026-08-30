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


# STUB, earmarked 2026-08-30: the chief can *propose* a custom award name/category here
# (captured on Tribe.custom_awards, see Simulation._run_night_cycle), but nothing yet
# checks whether anyone has actually met it and hands it out -- that mechanical trigger
# is NOT built. Deliberately constrained to AWARD_CATEGORIES (real, already-tracked
# stats) rather than freeform criteria: the chief invents the name and what it means to
# them, the simulation can only ever honestly judge a category it already measures,
# same split already used for scout/hunt milestone trophies. TODO before this is real:
# a periodic check (same cadence as _check_for_celebration) that looks at each
# category's real counter, detects a new personal-best or milestone within it, and
# calls _award_trophy with the chief's own proposed name instead of a hardcoded one.
AWARD_CATEGORIES = ("scouting", "hunting", "trading", "raiding")


async def reflect_on_history(
    client: OllamaClient, reviewer_model: str, tribe_name: str,
    current_philosophy: str, recent_events: list[str],
) -> dict:
    events_block = "\n".join(f"- {e}" for e in recent_events) or "(nothing notable recorded)"
    categories_list = ", ".join(AWARD_CATEGORIES)
    prompt = f"""You are reviewing, from a distance, the recent history of the {tribe_name} \
tribe. Its current guiding philosophy is: "{current_philosophy}"

Here is what has actually happened recently, in order:
{events_block}

Consider honestly whether this philosophy is still serving the tribe well, given what \
actually happened -- not what should have happened. You may keep it unchanged, adjust it, \
or replace it entirely. That judgment is yours to make, based on the tribe's own real \
experience.

Separately, if you wish, you may create a new honor of your own for your people -- a title \
you will personally bestow on whoever excels at one of: {categories_list}. This is entirely \
optional; leave it out if nothing comes to mind.

Reply with ONLY JSON:
{{
  "revised_philosophy": "the guiding philosophy going forward, whether changed or the same",
  "changed": true or false,
  "reasoning": "one sentence explaining your decision",
  "proposed_award": {{"name": "a short title of your own invention", "category": "one of: {categories_list}"}} or null
}}"""
    result = await client.generate_json(reviewer_model, prompt, temperature=0.7, num_ctx=8192)
    if not result or not result.get("revised_philosophy"):
        return {
            "revised_philosophy": current_philosophy,
            "changed": False,
            "reasoning": "the review produced nothing usable; philosophy stands unchanged",
            "proposed_award": None,
        }
    return result
