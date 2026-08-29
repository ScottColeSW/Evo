"""A one-time, in-fiction leadership contest, run once when a tribe is created.

This exists as an alternative to hard-coding survival directives into the prompt from
outside the simulation (tried, reverted -- see instincts.py's history). Instead of Claude
telling a tribe "hunt now," the tribe generates its own chief and governing philosophy via
the model itself, and that philosophy becomes standing context in every future turn. What
the tribe actually does with that context -- follow it, ignore it, contradict it -- is
still entirely its own reasoning each cycle. The directive, if there is one, comes from
inside the simulated world, not from the engineering layer around it.

Water used to work the same way: nearest_water's exact coordinates were handed to the
election as a fact (real map geometry, the way a game master would tell players what's
nearby). That's gone now -- water is something a tribe's own scouting expeditions have to
actually go and find (see actions.py._scout / simulation.py._advance_expedition), not a
freebie the simulation gifts on day one. All the election gets is whether water is already
confirmed nearby (water_needed=False) or not (water_needed=True); if not, the newly elected
chief can decree that finding one is a priority, which still just means "the tribe should
consider dispatching scouts" -- the tribe's own reasoning still picks SCOUT (or doesn't)
each cycle.
"""

from .ollama_client import OllamaClient


async def elect_chief(client: OllamaClient, model: str, tribe_name: str, water_needed: bool = False) -> dict:
    water_context = ""
    if water_needed:
        water_context = (
            "\n\nNo reliable fresh water has been confirmed near where this tribe stands. As "
            "part of taking power, the new chief must also decide whether making water a "
            "priority -- for instance, by dispatching scouts to search for it -- is worthwhile, "
            "or whether something else matters more right now (established shelter, defensible "
            "ground, anything else the chief judges more important). This is the chief's own "
            "call, not a requirement, and doesn't specify where water actually is -- nobody "
            "knows that yet."
        )

    prompt = f"""You are narrating the founding leadership contest for the {tribe_name} \
tribe, a small group of survivors at the dawn of their society.

Imagine two or three individuals within this tribe competing for the role of chief through \
whatever trial your culture would use -- a test of strength, wisdom, cunning, endurance, or \
oratory, your choice. Briefly imagine how the contest unfolds, then declare a winner.{water_context}

Reply with ONLY JSON:
{{
  "chief_name": "a short name for the winner",
  "victory_method": "one sentence describing how they won",
  "guiding_philosophy": "one sentence describing the governing ethos this chief will hold (e.g. territorial expansion, caution and hoarding, aggressive growth, isolationism)",
  "water_decision": {{"decreed": true or false, "reason": "one sentence, only meaningful if water was mentioned above"}}
}}"""
    result = await client.generate_json(model, prompt, temperature=0.9)
    if not result or not result.get("chief_name"):
        return {
            "chief_name": f"Elder of {tribe_name}",
            "victory_method": "no contest was recorded",
            "guiding_philosophy": "steady, unremarkable stewardship",
            "water_decision": {"decreed": False, "reason": ""},
        }
    return result
