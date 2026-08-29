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
