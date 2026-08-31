import json

from .ollama_client import OllamaClient


async def breed(client: OllamaClient, model: str, tribe_a: dict, tribe_b: dict, era: str) -> dict:
    """Crosses two tribes' ideology + invented lexicon into one descendant profile."""
    prompt = f"""You are a cultural recombination engine inside an evolutionary simulation.
Merge these two tribal profiles into one descendant. Introduce one small mutation \
appropriate to the '{era}' era.

Tribe A ideology: {tribe_a['ideology']}
Tribe A words: {json.dumps(tribe_a['lexicon'])}
Tribe B ideology: {tribe_b['ideology']}
Tribe B words: {json.dumps(tribe_b['lexicon'])}

Reply with ONLY JSON:
{{"ideology": "...", "lexicon": {{"token": "meaning"}}, "note": "one sentence describing what changed"}}"""
    result = await client.generate_json(model, prompt, temperature=0.9)
    if not result:
        return {
            "ideology": tribe_a["ideology"],
            "lexicon": tribe_a["lexicon"],
            "note": "crossover failed, lineage preserved unchanged",
        }
    return result


async def hatch(client: OllamaClient, model: str, parent_a: dict, parent_b: dict, era: str) -> dict:
    """A smaller, more legible sibling of breed() above -- same crossover-with-mutation
    shape, narrowed from a tribe's ideology+lexicon to a flock member's one descriptive
    trait. Used by Simulation._resolve_hatch (backend/actions.py's GATHER_EGGS) once a
    flock has at least two members to cross; a founding egg with nothing to cross yet
    skips this entirely."""
    prompt = f"""You are a cultural recombination engine inside an evolutionary simulation.
A tribe in the '{era}' era is raising a flock. Two of its members are about to produce \
an offspring. Cross their traits into one hatchling, introducing one small mutation \
appropriate to the era.

Parent A trait: {parent_a['trait']}
Parent B trait: {parent_b['trait']}

Reply with ONLY JSON:
{{"trait": "one short descriptive trait for the hatchling", "note": "one sentence describing the hatchling"}}"""
    result = await client.generate_json(model, prompt, temperature=0.9)
    if not result:
        return {
            "trait": parent_a["trait"],
            "note": "crossover failed, the hatchling favors one parent unchanged",
        }
    return result
