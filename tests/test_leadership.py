from backend.leadership import elect_chief
from tests.conftest import run_async


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.last_prompt = None

    async def generate_json(self, model, prompt, temperature=0.7, **kwargs):
        self.last_prompt = prompt
        return self._response


@run_async
async def test_elect_chief_returns_the_models_result():
    client = _FakeClient({
        "chief_name": "Ashgar",
        "victory_method": "won a test of endurance",
        "guiding_philosophy": "aggressive territorial expansion",
    })

    result = await elect_chief(client, "gemma2:2b", "Forest Tribe")

    assert result["chief_name"] == "Ashgar"
    assert result["guiding_philosophy"] == "aggressive territorial expansion"


@run_async
async def test_elect_chief_falls_back_when_model_returns_nothing_usable():
    client = _FakeClient({})

    result = await elect_chief(client, "gemma2:2b", "Forest Tribe")

    assert result["chief_name"] == "Elder of Forest Tribe"
    assert result["guiding_philosophy"]
    assert result["water_decision"] == {"decreed": False, "reason": ""}


@run_async
async def test_elect_chief_falls_back_when_chief_name_is_missing():
    client = _FakeClient({"victory_method": "unclear", "guiding_philosophy": "chaos"})

    result = await elect_chief(client, "gemma2:2b", "Mountain Tribe")

    assert result["chief_name"] == "Elder of Mountain Tribe"


@run_async
async def test_water_fact_is_omitted_when_water_not_needed():
    client = _FakeClient({"chief_name": "Ashgar", "guiding_philosophy": "x"})

    await elect_chief(client, "gemma2:2b", "Forest Tribe", water_needed=False)

    assert "No reliable fresh water" not in client.last_prompt


@run_async
async def test_water_fact_is_included_as_a_fact_not_an_instruction():
    """Only whether water is confirmed nearby is supplied (information); whether to act
    on it -- and where to actually look -- is explicitly left to the chief's own
    judgment, not commanded. No coordinates are ever handed over: water is something a
    tribe's own scouts have to go find (see actions.py._scout)."""
    client = _FakeClient({"chief_name": "Ashgar", "guiding_philosophy": "x"})

    await elect_chief(client, "gemma2:2b", "Forest Tribe", water_needed=True)

    assert "No reliable fresh water has been confirmed" in client.last_prompt
    assert "chief's own call, not a requirement" in client.last_prompt
    assert "doesn't specify where water actually is" in client.last_prompt
