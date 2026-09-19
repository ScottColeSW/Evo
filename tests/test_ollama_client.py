from unittest import mock

import httpx

from backend.ollama_client import OllamaClient
from tests.conftest import run_async


class _FakeResponse:
    # status_code/text added 2026-09-18 alongside _raise_with_body's own real
    # 500-body-surfacing fix -- matches real httpx.Response's shape closely
    # enough for raise_for_status() to behave the same way here as it does for
    # real network failures (a >=400 status actually raises).
    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.request = httpx.Request("POST", "http://localhost:11434/api/generate")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Server error '{self.status_code}' for url", request=self.request, response=self,
            )

    def json(self):
        return self._payload


@run_async
async def test_generate_json_surfaces_the_response_body_on_a_server_error():
    """Live report, 2026-09-18: a real 500 from /api/generate crashed a tick
    with nothing but 'Server error 500' to go on -- plain raise_for_status()
    discards the body, but Ollama's own 500s carry a real reason there
    (commonly VRAM exhaustion or a model crash mid-generation). The raised
    exception's own message must include it, so a live traceback actually
    says why next time."""
    client = OllamaClient()
    fake = _FakeResponse({}, status_code=500, text='{"error": "model requires more system memory"}')

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        try:
            await client.generate_json("gemma2:2b", "prompt")
            assert False, "expected an HTTPStatusError"
        except httpx.HTTPStatusError as exc:
            assert "model requires more system memory" in str(exc)
            assert "gemma2:2b" in str(exc)


@run_async
async def test_generate_text_surfaces_the_response_body_on_a_server_error():
    client = OllamaClient()
    fake = _FakeResponse({}, status_code=500, text='{"error": "CUDA out of memory"}')

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        try:
            await client.generate_text("gemma2:2b", "prompt")
            assert False, "expected an HTTPStatusError"
        except httpx.HTTPStatusError as exc:
            assert "CUDA out of memory" in str(exc)


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
async def test_generate_json_with_raw_returns_both_the_parsed_dict_and_the_raw_text():
    """Explicit request, 2026-09-09: the live debug view needs the model's
    actual, unparsed text, not just whatever survived JSON parsing --
    generate_json (still used by every other caller in this codebase) discards
    it; this is the one method that doesn't."""
    client = OllamaClient()
    fake = _FakeResponse({"response": '{"chief_name": "Ashgar"}'})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        parsed, raw = await client.generate_json_with_raw("gemma2:2b", "prompt")

    assert parsed == {"chief_name": "Ashgar"}
    assert raw == '{"chief_name": "Ashgar"}'


@run_async
async def test_generate_json_with_raw_still_returns_the_text_on_a_degenerate_response():
    """Same "not a dict" guard generate_json's own test already covers -- the
    raw text should still come through even when parsing produces {} instead
    of a usable dict, so the debug view can show what actually happened."""
    client = OllamaClient()
    fake = _FakeResponse({"response": "not json at all"})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        parsed, raw = await client.generate_json_with_raw("gemma2:2b", "prompt")

    assert parsed == {}
    assert raw == "not json at all"


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


@run_async
async def test_embed_returns_the_real_vector_on_success():
    client = OllamaClient()
    fake = _FakeResponse({"embedding": [0.1, 0.2, 0.3]})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)) as mock_post:
        result = await client.embed("the tribe worries about defense", model="nomic-embed-text")

    assert result == [0.1, 0.2, 0.3]
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"model": "nomic-embed-text", "prompt": "the tribe worries about defense"}


@run_async
async def test_embed_fails_open_on_a_server_error():
    """Explicit request, 2026-09-19: "I like honest and upgrade." embed()
    must never raise -- a bad embedding call falls back to TribeMemory's own
    token-overlap reinforcement path (remember_reflection), the same "a
    missing nice-to-have shouldn't break the real work" pattern vram_guard.py
    already uses."""
    client = OllamaClient()
    fake = _FakeResponse({}, status_code=500, text='{"error": "model not found"}')

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        result = await client.embed("some reflection text")

    assert result is None


@run_async
async def test_embed_fails_open_when_the_connection_itself_fails():
    client = OllamaClient()

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(side_effect=httpx.ConnectError("down"))):
        result = await client.embed("some reflection text")

    assert result is None


@run_async
async def test_embed_fails_open_on_a_malformed_response():
    """format="json"-style guarantee doesn't apply to /api/embeddings -- a
    missing or wrong-shaped "embedding" key must degrade the same as a real
    HTTP failure, not crash the caller."""
    client = OllamaClient()
    fake = _FakeResponse({"embedding": "not-a-list"})

    with mock.patch.object(httpx.AsyncClient, "post", mock.AsyncMock(return_value=fake)):
        result = await client.embed("some reflection text")

    assert result is None
