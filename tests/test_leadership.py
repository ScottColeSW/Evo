from backend.leadership import elect_chief
from tests.conftest import run_async


class _FakeClient:
    def __init__(self, response):
        self._response = response

    async def generate_json(self, model, prompt, temperature=0.7, **kwargs):
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


@run_async
async def test_elect_chief_falls_back_when_chief_name_is_missing():
    client = _FakeClient({"victory_method": "unclear", "guiding_philosophy": "chaos"})

    result = await elect_chief(client, "gemma2:2b", "Mountain Tribe")

    assert result["chief_name"] == "Elder of Mountain Tribe"
