from unittest import mock

import httpx

from backend.ollama_client import OllamaClient
from tests.conftest import run_async


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@run_async
async def test_generate_json_returns_the_parsed_dict_on_a_well_formed_response():
    client = OllamaClient()
    fake = _FakeResponse({"response": '{"chief_name": "Ashgar"}'})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        result = await client.generate_json("gemma2:2b", "prompt")

    assert result == {"chief_name": "Ashgar"}


@run_async
async def test_generate_json_returns_empty_dict_when_the_model_emits_a_bare_json_string():
    """Regression test: a real live run against llama3.2:1b crashed the whole
    simulation with 'str' object has no attribute 'get' -- format:"json" guarantees
    valid JSON, not a JSON *object*, and a weak/small model can emit a bare string
    (still valid JSON) instead of the expected {...}. Every caller assumes a dict."""
    client = OllamaClient()
    fake = _FakeResponse({"response": '"Elder of Forest Tribe"'})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        result = await client.generate_json("llama3.2:1b", "prompt")

    assert result == {}


@run_async
async def test_generate_json_returns_empty_dict_when_the_model_emits_a_json_list():
    client = OllamaClient()
    fake = _FakeResponse({"response": "[1, 2, 3]"})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        result = await client.generate_json("llama3.2:1b", "prompt")

    assert result == {}


@run_async
async def test_generate_json_returns_empty_dict_on_invalid_json():
    client = OllamaClient()
    fake = _FakeResponse({"response": "not json at all"})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        result = await client.generate_json("gemma2:2b", "prompt")

    assert result == {}
