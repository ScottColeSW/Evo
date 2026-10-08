"""2026-10-08: thinking models (and models that cannot generate text) are not offered to play. A live run on qwen3:4b took a median of 53 seconds a decision."""
import asyncio

import httpx

from backend import ollama_client
from backend.ollama_client import OllamaClient

TAGS = {"models": [
    {"name": "qwen2.5:3b", "digest": "a"}, {"name": "qwen3:4b", "digest": "b"}, {"name": "nomic-embed-text:latest", "digest": "c"},
    {"name": "gemma3:4b", "digest": "d"}, {"name": "deepseek-r1:7b", "digest": "e"},
]}
SHOW = {
    "qwen2.5:3b": ["completion", "tools"], "qwen3:4b": ["completion", "tools", "thinking"], "nomic-embed-text:latest": ["embedding"],
    "gemma3:4b": ["completion", "vision"], "deepseek-r1:7b": None,  # an Ollama that reports no capabilities for it: judged by name
}


def _client(show=SHOW):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        if request.url.path == "/api/show":
            import json
            name = json.loads(request.content)["model"]
            caps = show.get(name)
            return httpx.Response(200, json={} if caps is None else {"capabilities": caps})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    ollama_client.httpx.AsyncClient = lambda *a, **k: real(*a, transport=transport, **{x: y for x, y in k.items() if x != "transport"})
    return OllamaClient(), real


def test_only_models_that_can_generate_and_do_not_think_are_listed():
    ollama_client._CAPABILITY_CACHE.clear()
    client, real = _client()
    try:
        names = asyncio.run(client.list_models())
    finally:
        ollama_client.httpx.AsyncClient = real
    assert names == ["qwen2.5:3b", "gemma3:4b"]


def test_when_ollama_reports_no_capabilities_a_known_thinking_name_is_still_left_out():
    ollama_client._CAPABILITY_CACHE.clear()
    client, real = _client({**SHOW, "qwen3:4b": None})
    try:
        names = asyncio.run(client.list_models())
    finally:
        ollama_client.httpx.AsyncClient = real
    assert "qwen3:4b" not in names and "deepseek-r1:7b" not in names and "qwen2.5:3b" in names
