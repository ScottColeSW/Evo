from unittest import mock

from backend.app import _unload_stale_models
from backend.ollama_client import OllamaClient
from tests.conftest import run_async


@run_async
async def test_unload_stale_models_evicts_everything_ollama_reports_loaded():
    """Regression: force-killing the server process (rather than a graceful STOP or
    tab close, both of which reach Simulation.shutdown()) used to leave whatever
    models it had loaded resident in Ollama's VRAM until their keep_alive window
    expired on its own -- confirmed live via orphaned llama-server.exe processes
    still running well after the server that loaded them was gone."""
    with (
        mock.patch.object(OllamaClient, "list_loaded_models", mock.AsyncMock(return_value=["gemma2:2b", "qwen2.5:3b"])),
        mock.patch.object(OllamaClient, "unload_model", mock.AsyncMock()) as unload,
    ):
        await _unload_stale_models()

    assert unload.await_args_list == [mock.call("gemma2:2b"), mock.call("qwen2.5:3b")]


@run_async
async def test_unload_stale_models_is_a_no_op_when_nothing_is_loaded():
    with (
        mock.patch.object(OllamaClient, "list_loaded_models", mock.AsyncMock(return_value=[])),
        mock.patch.object(OllamaClient, "unload_model", mock.AsyncMock()) as unload,
    ):
        await _unload_stale_models()

    unload.assert_not_called()
