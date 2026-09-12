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


# The chief can *propose* a custom award name/category here (captured on
# Tribe.custom_awards, see Simulation._run_night_cycle); Simulation._check_custom_awards
# is the mechanical half, called from the same real-event sites as the built-in
# milestone trophies (a confirmed water scout, a hunting party's catch, a completed
# trade, a won raid) to actually hand it out. Deliberately constrained to
# AWARD_CATEGORIES (real, already-tracked stats) rather than freeform criteria: the
# chief invents the name and what it means to them, the simulation can only ever
# honestly judge a category it already measures, same split already used for scout/hunt
# milestone trophies.
AWARD_CATEGORIES = ("scouting", "hunting", "trading", "raiding")


async def reflect_on_history(
    client: OllamaClient, reviewer_model: str, tribe_name: str,
    current_philosophy: str, recent_events: list[str], inventory: str = "",
) -> dict:
    events_block = "\n".join(f"- {e}" for e in recent_events) or "(nothing notable recorded)"
    categories_list = ", ".join(AWARD_CATEGORIES)
    # `inventory` (Simulation._build_night_inventory) is the tribe's actual current
    # state -- resources, settlement status, era-progress gaps -- separate from the
    # prose chronicle below. The chronicle alone tends to just echo whatever the tribe
    # has been doing turn after turn in its own recent phrasing, which made a real
    # mismatch (surplus water, zero food; still settled but still scouting for water
    # long after it's secured) easy for a reviewer to miss entirely reading prose
    # alone. Framed as what the chief actually takes stock of before turning in for
    # the night, not a separate instruction.
    inventory_block = f"\nBefore retiring for the night, the chief takes stock: {inventory}\n" if inventory else ""
    prompt = f"""You are reviewing, from a distance, the recent history of the {tribe_name} \
tribe. Its current guiding philosophy is: "{current_philosophy}"
{inventory_block}
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
  "reasoning": "your decision in ONE short sentence, 20 words or fewer -- this is a private thought, not an essay",
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


async def generate_endgame_narrative(client: OllamaClient, model: str, summary_facts: str) -> str:
    """A one-time narrative synthesis of the whole game at Simulation.
    _trigger_game_over, distinct from the plain-templated, no-model-call
    _generate_game_over_summary (the "OVERSEER LOG" facts this reads) -- an actual
    outside voice telling the whole civilization's story once, not another data
    table. Reversed 2026-09-12 from a dedicated reviewer model resident during live
    play (see config.py's own comment above ENDGAME_SUMMARY_MODEL, and _run_night_
    cycle's switch to self-review) -- this is where that freed-up "distinct outside
    voice" actually earns its keep instead: the sim has already stopped ticking by
    the time this runs, so a bigger/different model here costs nothing in VRAM
    contention, unlike one resident during ordinary turns.

    Same "facts only, model decides what to say" shape as reflect_on_history: the
    simulation states what genuinely happened (final stats, trophies, chief
    lineage, combat record -- already assembled into summary_facts), the model
    decides how to tell it. Plain narrative text, not JSON -- there's nothing here
    for the caller to act on mechanically, only prose the frontend's end-of-run
    splash displays."""
    prompt = f"""You are a chronicler looking back on a civilization simulation that has just \
ended. Here is the factual record of what happened, exactly as observed:

{summary_facts}

Write a short narrative account of this civilization's story -- three to six sentences, \
grounded only in the facts above. Do not invent specific events, names, or numbers beyond \
what's given; you may interpret and characterize what happened, not add to it. Write it as \
a story's closing, not another log entry."""
    narrative = await client.generate_text(model, prompt, temperature=0.7)
    return narrative.strip()
