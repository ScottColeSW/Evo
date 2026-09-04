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


@run_async
async def test_list_loaded_models_returns_names_from_api_ps():
    client = OllamaClient()
    fake = _FakeResponse({"models": [{"name": "gemma2:2b"}, {"name": "qwen2.5:3b"}]})

    with mock.patch.object(httpx.AsyncClient, "get", mock.AsyncMock(return_value=fake)):
        result = await client.list_loaded_models()

    assert result == ["gemma2:2b", "qwen2.5:3b"]


@run_async
async def test_list_loaded_models_returns_empty_list_when_ollama_is_unreachable():
    client = OllamaClient()

    with mock.patch.object(httpx.AsyncClient, "get", mock.AsyncMock(side_effect=httpx.ConnectError("down"))):
        result = await client.list_loaded_models()

    assert result == []


@run_async
async def test_unload_model_returns_promptly_once_ollama_confirms_eviction():
    """Explicit report: "the quit isn't cleaning up after itself and making
    sure the system stops." Ollama's own /api/generate keep_alive=0 response
    comes back done:true well before the model actually leaves VRAM -- this
    polls list_loaded_models() for the real signal instead of trusting that
    response."""
    client = OllamaClient()
    post_fake = _FakeResponse({"done": True})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=post_fake)), \
         mock.patch.object(client, "list_loaded_models", mock.AsyncMock(return_value=[])) as mock_list, \
         mock.patch("backend.ollama_client.asyncio.sleep", mock.AsyncMock()) as mock_sleep:
        await client.unload_model("qwen2.5:3b")

    mock_list.assert_awaited_once()
    mock_sleep.assert_not_awaited()  # gone on the very first check -- no need to wait at all


@run_async
async def test_unload_model_polls_past_ollamas_own_eviction_lag():
    """Live-confirmed: ~8s of real lag observed evicting two ~2-3GB models via
    direct API probing, well after Ollama's own response already said done."""
    client = OllamaClient()
    post_fake = _FakeResponse({"done": True})
    # Still resident for the first two checks, gone by the third.
    list_results = [["qwen2.5:3b"], ["qwen2.5:3b"], []]

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=post_fake)), \
         mock.patch.object(client, "list_loaded_models", mock.AsyncMock(side_effect=list_results)), \
         mock.patch("backend.ollama_client.asyncio.sleep", mock.AsyncMock()) as mock_sleep:
        await client.unload_model("qwen2.5:3b")

    assert mock_sleep.await_count == 2


@run_async
async def test_unload_model_gives_up_after_the_bounded_number_of_attempts():
    """A genuinely stuck Ollama can't hang shutdown forever -- best-effort, same
    as a failed unload request."""
    from backend.ollama_client import UNLOAD_POLL_MAX_ATTEMPTS

    client = OllamaClient()
    post_fake = _FakeResponse({"done": True})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=post_fake)), \
         mock.patch.object(client, "list_loaded_models", mock.AsyncMock(return_value=["qwen2.5:3b"])) as mock_list, \
         mock.patch("backend.ollama_client.asyncio.sleep", mock.AsyncMock()):
        await client.unload_model("qwen2.5:3b")  # must not raise

    assert mock_list.await_count == UNLOAD_POLL_MAX_ATTEMPTS


@run_async
async def test_unload_model_still_best_effort_when_the_post_itself_fails():
    client = OllamaClient()

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(side_effect=httpx.ConnectError("down"))), \
         mock.patch.object(client, "list_loaded_models", mock.AsyncMock()) as mock_list:
        await client.unload_model("qwen2.5:3b")  # must not raise

    mock_list.assert_not_awaited()  # no point polling if the unload request never even went out
