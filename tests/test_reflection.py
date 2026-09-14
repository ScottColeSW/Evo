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
    assert result["proposed_decree"] is None
    assert result["proposed_dream"] is None
    assert result["reasoning"]


@run_async
async def test_prompt_states_there_is_no_standing_decree_when_none_exists():
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(client, "llama3", "Forest Tribe", "caution and hoarding", [])

    assert "You have no standing decree right now." in client.last_prompt


@run_async
async def test_prompt_states_the_current_decree_when_one_exists():
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(
        client, "llama3", "Forest Tribe", "caution and hoarding", [],
        current_decree="find fresh water before anything else",
    )

    assert 'Your current standing decree is: "find fresh water before anything else"' in client.last_prompt


@run_async
async def test_prompt_frames_the_decree_as_the_chiefs_own_idea_not_ours():
    """Design-philosophy check, same convention as the honest-judgment test above --
    this must read as an invitation to propose the chief's own idea, not an
    instruction telling the model what to decide."""
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(client, "llama3", "Forest Tribe", "caution and hoarding", [])

    assert "this is your own idea, not ours" in client.last_prompt


@run_async
async def test_decree_prompt_gives_no_copyable_example_text():
    """Live bug, confirmed 2026-09-13 (run_20260913_135304): the decree paragraph
    used to include two concrete illustrative examples. qwen2.5:3b returned one of
    them back VERBATIM as its own "freely invented" decree, and gemma2:2b echoed
    the invented "winter" framing from the other in two separate decrees of its
    own -- a concept that appears nowhere else in this simulation. The whole point
    of the mechanic is a decree grounded in what actually happened to THIS tribe;
    a copyable stock example defeats that the moment a small model parrots it. The
    prompt must describe the shape of a decree without ever giving content a model
    could copy wholesale."""
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(client, "llama3", "Forest Tribe", "caution and hoarding", [])

    assert "build us a proper kitchen before winter" not in client.last_prompt
    assert "find fresh water before anything else" not in client.last_prompt
    assert "winter" not in client.last_prompt.lower()


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
async def test_prompt_omits_the_dream_question_without_a_dmm():
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(client, "llama3", "Forest Tribe", "caution and hoarding", [])

    assert "Dream Manifestation Machine" not in client.last_prompt


@run_async
async def test_prompt_asks_for_a_grounded_dream_once_the_dmm_stands():
    """Explicit request, 2026-09-13: "It makes real the dreams of the Chief" --
    the dream must be grounded in something that actually happened, not generic
    wish-fulfillment, mirroring the original idea: a chief who tasted good food
    dreaming of Kitchen. Also confirms the DMM, not the chief, names the result."""
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(
        client, "llama3", "Forest Tribe", "caution and hoarding", ["survived a raid"], dmm_built=True,
    )

    assert "Dream Manifestation Machine" in client.last_prompt
    assert "grounded in something real that just happened" in client.last_prompt
    assert "the Machine will name whatever it manifests" in client.last_prompt
    assert "proposed_dream" in client.last_prompt


@run_async
async def test_prompt_omits_the_departure_question_when_not_eligible():
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(
        client, "llama3", "Forest Tribe", "caution and hoarding", [], dmm_built=True,
    )

    assert "beyond this island" not in client.last_prompt


@run_async
async def test_prompt_asks_about_departure_once_eligible():
    """Beyond the Horizon era's real gate -- plan file amber-drifting-tern.md.
    Still returned via the existing proposed_dream field, no new JSON key --
    classification happens downstream (Simulation._run_night_cycle)."""
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(
        client, "llama3", "Forest Tribe", "caution and hoarding", ["reached the last stage of development"],
        dmm_built=True, departure_eligible=True,
    )

    assert "beyond this island" in client.last_prompt
    # Same JSON field as the ordinary dream -- no second key added to the schema.
    assert client.last_prompt.count('"proposed_dream"') == 1


@run_async
async def test_departure_prompt_gives_no_copyable_example_text():
    """Explicit design-philosophy carry-over from the 2026-09-14 decree-leak
    fix: this dream is quoted VERBATIM in the eventual game-over summary, so a
    copyable example here would be worse than the original bug, not just a
    repeat of it. Describes the shape only."""
    client = _FakeClient({"revised_philosophy": "x"})

    await reflect_on_history(
        client, "llama3", "Forest Tribe", "caution and hoarding", [],
        dmm_built=True, departure_eligible=True,
    )

    assert "winter" not in client.last_prompt.lower()
    # No quoted illustrative sentence -- unlike the decree paragraph's old bug,
    # there should be no quotation marks wrapping an invented example at all
    # in the departure paragraph specifically.
    departure_start = client.last_prompt.index("Separately, if something about the tribe's long journey")
    departure_end = client.last_prompt.index("feels like enough.") + len("feels like enough.")
    departure_paragraph = client.last_prompt[departure_start:departure_end]
    assert '"' not in departure_paragraph


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


@run_async
async def test_generate_endgame_narrative_asks_for_two_paragraphs():
    """Explicit request, 2026-09-14: the final-summary narrative was rendering as
    one dense, unbroken block on the game-over splash. white-space: pre-wrap
    already renders a blank line as visible paragraph spacing (see frontend/
    index.html's .game-over-narrative) -- the missing piece was ever asking the
    model to write one in the first place."""
    client = _FakeTextClient("a tale")

    await generate_endgame_narrative(client, "mistral:7b", "OVERSEER LOG: ...")

    assert "two short paragraphs" in client.last_prompt
