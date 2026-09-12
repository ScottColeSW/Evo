from backend.reflection import AWARD_CATEGORIES, generate_endgame_narrative, reflect_on_history
from tests.conftest import run_async


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.last_prompt = None

    async def generate_json(self, model, prompt, temperature=0.7, **kwargs):
        self.last_prompt = prompt
        return self._response


class _FakeTextClient:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_model = None
        self.last_prompt = None

    async def generate_text(self, model, prompt, temperature=0.5, **kwargs):
        self.last_model = model
        self.last_prompt = prompt
        return self._response_text


@run_async
async def test_reflect_on_history_returns_the_models_result():
    client = _FakeClient({
        "revised_philosophy": "trade over war",
        "changed": True,
        "reasoning": "raiding cost more than it ever brought back",
        "proposed_award": {"name": "Keeper of the Ledger", "category": "trading"},
    })

    result = await reflect_on_history(
        client, "llama3", "Forest Tribe", "aggressive territorial expansion", ["lost a raid", "won a trade"],
    )

    assert result["revised_philosophy"] == "trade over war"
    assert result["changed"] is True
    assert result["proposed_award"] == {"name": "Keeper of the Ledger", "category": "trading"}


@run_async
async def test_reflect_on_history_falls_back_when_model_returns_nothing_usable():
    client = _FakeClient({})

    result = await reflect_on_history(client, "llama3", "Forest Tribe", "aggressive territorial expansion", [])

    assert result["revised_philosophy"] == "aggressive territorial expansion"
    assert result["changed"] is False
    assert result["proposed_award"] is None
    assert result["reasoning"]


@run_async
async def test_reflect_on_history_falls_back_when_revised_philosophy_is_missing():
    client = _FakeClient({"changed": True, "reasoning": "something shifted"})

    result = await reflect_on_history(client, "llama3", "Mountain Tribe", "isolationism", ["a wolf attack"])

    assert result["revised_philosophy"] == "isolationism"
    assert result["changed"] is False


@run_async
async def test_recent_events_are_listed_as_facts_not_summarized():
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(
        client, "llama3", "Forest Tribe", "caution and hoarding",
        ["settled near the river", "starved for three days"],
    )

    assert "- settled near the river" in client.last_prompt
    assert "- starved for three days" in client.last_prompt


@run_async
async def test_empty_history_is_stated_honestly_not_hidden():
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(client, "llama3", "Forest Tribe", "caution and hoarding", [])

    assert "(nothing notable recorded)" in client.last_prompt


@run_async
async def test_inventory_is_omitted_when_not_supplied():
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(client, "llama3", "Forest Tribe", "caution and hoarding", ["idled"])

    assert "takes stock" not in client.last_prompt


@run_async
async def test_inventory_is_included_as_a_separate_fact_when_supplied():
    """inventory (Simulation._build_night_inventory) is the tribe's actual current state,
    kept distinct from the prose chronicle -- see reflection.py's own module docstring for
    why the chronicle alone can miss a real mismatch (e.g. water secured, food still short)."""
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(
        client, "llama3", "Forest Tribe", "caution and hoarding", ["idled"],
        inventory="surplus water, zero food, still scouting",
    )

    assert "surplus water, zero food, still scouting" in client.last_prompt


@run_async
async def test_prompt_offers_the_award_as_optional_not_required():
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(client, "llama3", "Forest Tribe", "caution and hoarding", [])

    assert "This is entirely optional" in client.last_prompt
    for category in AWARD_CATEGORIES:
        assert category in client.last_prompt


@run_async
async def test_prompt_asks_for_honest_judgment_not_a_scripted_outcome():
    """Matches the design-philosophy convention already tested in test_leadership.py's own
    water-fact tests: the model is given real facts and left to judge for itself, never told
    what to conclude."""
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(client, "llama3", "Forest Tribe", "caution and hoarding", ["a poor harvest"])

    assert "not what should have happened" in client.last_prompt
    assert "That judgment is yours to make" in client.last_prompt


@run_async
async def test_generate_endgame_narrative_returns_the_models_stripped_text():
    client = _FakeTextClient("  A civilization rose, thrived, and faded.  \n")

    narrative = await generate_endgame_narrative(client, "mistral:7b", "OVERSEER LOG: ...")

    assert narrative == "A civilization rose, thrived, and faded."


@run_async
async def test_generate_endgame_narrative_passes_the_model_and_facts_through():
    client = _FakeTextClient("a tale")

    await generate_endgame_narrative(client, "mistral:7b", "OVERSEER LOG: Tribe A reached the Bronze Age.")

    assert client.last_model == "mistral:7b"
    assert "OVERSEER LOG: Tribe A reached the Bronze Age." in client.last_prompt


@run_async
async def test_generate_endgame_narrative_does_not_invite_invented_specifics():
    """Same "facts only, model decides what to say" shape as reflect_on_history --
    the model may interpret the given facts, not add new ones."""
    client = _FakeTextClient("a tale")

    await generate_endgame_narrative(client, "mistral:7b", "OVERSEER LOG: ...")

    assert "Do not invent specific events, names, or numbers" in client.last_prompt
