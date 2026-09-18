"""The "night cycle": periodically, a dedicated reviewer model (config.
REFLECTION_MODEL -- see that constant's own comment for why it's a separate model
again as of 2026-09-18, after a 2026-09-12 through 2026-09-17 stretch of self-
review with the tribe's own live model) looks back at a tribe's own recent history
and decides for itself whether its guiding philosophy should change. This is the
piece from the original design transcript that was never built -- see the
genetics_and_night_cycle_gap memory note -- distinct from breed()/breed_
individuals' cross-tribe/cross-individual crossover. Fast small models handle every
live turn; this is the "day reviewed at night" pass, run much less often, using a
different model than whatever the tribe uses live -- not necessarily a larger one
(REFLECTION_MODEL is deliberately one of the smaller models this project uses, to
keep the VRAM-contention risk this reintroduces as small as possible).

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
    current_decree: str = "", dmm_built: bool = False,
    departure_eligible: bool = False,
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
    # Explicit design, 2026-09-13: "the decree piece is what will make it click
    # together" -- philosophy is who the chief IS (abstract, character), a decree is
    # what the chief actually WANTS DONE (concrete, an instruction the tribe's own
    # live turns already see every cycle via prompts.py's duty_text). Before this,
    # chief_decree only ever held one hardcoded, scripted text (a water-finding duty
    # set at election) -- this is the chief's own, freely invented decree instead,
    # grounded in the same real recent history as the philosophy revision above, not
    # a second scripted rule. Optional and sticky by design: leaving it out keeps
    # whatever decree already stands rather than silently clearing it, so a chief who
    # isn't moved to change course tonight doesn't accidentally erase a good standing
    # order just by not mentioning it.
    #
    # Live bug, confirmed 2026-09-13 (run_20260913_135304): this paragraph used to
    # include two concrete illustrative examples ("build us a proper kitchen before
    # winter," "find fresh water before anything else"). qwen2.5:3b returned the
    # first one back VERBATIM as its own "freely invented" decree, and gemma2:2b
    # echoed the invented "winter" framing (a concept that appears nowhere else in
    # this simulation -- no season system exists at all) in two separate decrees of
    # its own. Confirmed via the full run log: every "winter" mention in the whole
    # 102-cycle run was inside a decree/dream field, none in ordinary turn text --
    # the example had leaked into outputs that were supposed to be grounded in real
    # events, defeating the entire point of the mechanic. Fixed by describing the
    # shape abstractly instead of giving copyable content, matching this project's
    # standing "no scripted directives" rule -- a hollow illustrative example is
    # itself a kind of scripted directive once a small model starts parroting it.
    decree_line = f'Your current standing decree is: "{current_decree}"' if current_decree else "You have no standing decree right now."
    # Explicit design, 2026-09-13: "It makes real the dreams of the Chief" -- the
    # Dream Manifestation Machine (backend/actions.py._new_created_object) reads
    # tribe.chief_dream to pick which of its 6 bounded effect categories to
    # manifest next, instead of blind round-robin. Only asked once the DMM
    # actually stands (dmm_built) -- dreaming up something with nowhere to make
    # it real is pointless, same "don't ask about a mechanic that isn't reachable
    # yet" gate cooking_learned already applies to kitchen mentions in
    # instincts.py. Explicitly required to be grounded in something from
    # recent_events, not generic wish-fulfillment -- this is the same shape as
    # the user's own original idea: "a Chief that has tasted delicious Cooked
    # food could have a dream of lots and lots of food." The DMM itself still
    # names whatever gets made (CREATED_OBJECT_NAMES) -- the chief only ever
    # supplies the wish, never the label.
    dream_block = ""
    if dmm_built:
        dream_block = f"""

Separately, the tribe's Dream Manifestation Machine stands ready to make one of your dreams \
real. If something that actually happened recently -- named above -- has left you wanting \
something practical for your people, describe that dream in your own words. It must be \
grounded in something real that just happened, not a generic wish, and it should serve a real \
need (feeding the tribe, defending it, exploring further, growing it, celebrating together, or \
fighting off a threat). Leave it out if nothing recent truly calls for one -- the Machine will \
name whatever it manifests, so just describe the need, not what to call it."""
    # Plan file amber-drifting-tern.md, 2026-09-14: Beyond the Horizon era's real
    # gate. Only offered once dmm_built, the DMM has actually been used at least
    # once (config.DMM_WARMUP_CREATIONS_REQUIRED -- passed in via
    # departure_eligible, computed by the caller, not re-derived here), and the
    # tribe has reached the era itself. Deliberately describes the SHAPE of the
    # idea only, no example sentence -- this is exactly the paragraph the
    # decree-leak bug above was found in, one paragraph earlier in this same
    # function. A copyable illustrative phrase here would repeat that mistake in
    # the one place it would matter most (this dream, unlike an ordinary one, is
    # quoted verbatim in the final game-over summary).
    departure_dream_block = ""
    if departure_eligible:
        departure_dream_block = """

Separately, if something about the tribe's long journey here has left you wondering what might \
exist beyond this island, you may say so -- not a practical need this time, just what you've \
come to want after seeing everything this place has to offer. Leave it out if this island still \
feels like enough."""
    prompt = f"""You are reviewing, from a distance, the recent history of the {tribe_name} \
tribe. Its current guiding philosophy is: "{current_philosophy}"
{inventory_block}
Here is what has actually happened recently, in order:
{events_block}

Before anything else, just think. What's actually on your mind tonight, given what's \
happened? You don't owe the tribe a decision every night -- most nights, a private thought \
with nothing else attached is a completely honest answer.

Separately, if that thinking leads somewhere concrete, you may also act on it below -- but \
only if it genuinely does; none of the following are owed just because a night has passed.

You may reconsider whether the current philosophy is still serving the tribe well, given \
what actually happened -- not what should have happened. Keep it unchanged, adjust it, or \
replace it entirely -- that judgment is yours, based on the tribe's own real experience.

You may create a new honor of your own for your people -- a title you will personally \
bestow on whoever excels at one of: {categories_list}.

You may set a standing decree -- not a philosophy, an actual concrete order the tribe can \
act on starting tomorrow: a specific task, not a value or a goal statement, worded the way \
you would actually give an order, in your own voice. {decree_line} Leave it as it is unless \
something you've just reflected on genuinely calls for a new one -- this is your own idea, \
not ours, so only propose one if it truly comes from what actually happened to THIS tribe, \
not a generic-sounding order that could apply to any tribe, any time.{dream_block}{departure_dream_block}

Reply with ONLY JSON:
{{
  "private_thoughts": "whatever is actually on your mind tonight, in your own words -- this is yours alone, no one else reads it, and it's completely fine for this to be the only thing you have to say",
  "revised_philosophy": "the guiding philosophy going forward, whether changed or the same",
  "changed": true or false,
  "reasoning": "your decision in ONE short sentence, 20 words or fewer -- a private thought, not an essay",
  "proposed_award": {{"name": "a short title of your own invention", "category": "one of: {categories_list}"}} or null,
  "proposed_decree": "a short, concrete standing duty, in your own words" or null,
  "proposed_dream": "a short description of a real, grounded dream, in your own words" or null
}}"""
    result = await client.generate_json(reviewer_model, prompt, temperature=0.7, num_ctx=8192)
    if not result or not result.get("revised_philosophy"):
        return {
            "private_thoughts": (result or {}).get("private_thoughts", ""),
            "revised_philosophy": current_philosophy,
            "changed": False,
            "reasoning": "the review produced nothing usable; philosophy stands unchanged",
            "proposed_award": None,
            "proposed_decree": None,
            "proposed_dream": None,
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
a story's closing, not another log entry.

Break it into two short paragraphs, separated by a single blank line -- roughly "how it \
lived" and "how it ended" -- rather than one unbroken block."""
    narrative = await client.generate_text(model, prompt, temperature=0.7)
    return narrative.strip()
