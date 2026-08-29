"""A one-time, in-fiction leadership contest, run once when a tribe is created.

This exists as an alternative to hard-coding survival directives into the prompt from
outside the simulation (tried, reverted -- see instincts.py's history). Instead of Claude
telling a tribe "hunt now," the tribe generates its own chief and governing philosophy via
the model itself, and that philosophy becomes standing context in every future turn. What
the tribe actually does with that context -- follow it, ignore it, contradict it -- is
still entirely its own reasoning each cycle. The directive, if there is one, comes from
inside the simulated world, not from the engineering layer around it.
"""

from .ollama_client import OllamaClient


async def elect_chief(client: OllamaClient, model: str, tribe_name: str) -> dict:
    prompt = f"""You are narrating the founding leadership contest for the {tribe_name} \
tribe, a small group of survivors at the dawn of their society.

Imagine two or three individuals within this tribe competing for the role of chief through \
whatever trial your culture would use -- a test of strength, wisdom, cunning, endurance, or \
oratory, your choice. Briefly imagine how the contest unfolds, then declare a winner.

Reply with ONLY JSON:
{{
  "chief_name": "a short name for the winner",
  "victory_method": "one sentence describing how they won",
  "guiding_philosophy": "one sentence describing the governing ethos this chief will hold (e.g. territorial expansion, caution and hoarding, aggressive growth, isolationism)"
}}"""
    result = await client.generate_json(model, prompt, temperature=0.9)
    if not result or not result.get("chief_name"):
        return {
            "chief_name": f"Elder of {tribe_name}",
            "victory_method": "no contest was recorded",
            "guiding_philosophy": "steady, unremarkable stewardship",
        }
    return result
