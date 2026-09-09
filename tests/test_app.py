import json
from unittest import mock

from backend.app import _tick_session, _unload_stale_models
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


@run_async
async def test_tick_session_returns_immediately_when_there_is_no_sim():
    ws = mock.AsyncMock()
    await _tick_session(ws, {})
    ws.send_str.assert_not_called()


@run_async
async def test_tick_session_does_nothing_while_paused():
    """Live report: "the game is paused but it sure seems to be working the
    drive... is there something hitting the disc in Pause mode?" sim.step()
    itself already no-ops while paused, but this function used to keep
    computing a snapshot and writing it to board_history.db every tick anyway,
    for a cycle number that was never actually changing."""
    sim = mock.Mock()
    sim.paused = True
    sim.step = mock.AsyncMock()

    with mock.patch("backend.app.record_board_state") as record:
        await _tick_session(mock.AsyncMock(), {"sim": sim})

    sim.step.assert_not_called()
    sim.snapshot.assert_not_called()
    record.assert_not_called()


@run_async
async def test_tick_session_sends_the_snapshot_on_a_normal_tick():
    sim = mock.Mock()
    sim.paused = False
    sim.step = mock.AsyncMock()
    sim.snapshot.return_value = {"cycle": 5}
    sim.run_id, sim.cycle = "run_x", 5
    ws = mock.AsyncMock()

    with mock.patch("backend.app.record_board_state") as record:
        await _tick_session(ws, {"sim": sim})

    sim.step.assert_awaited_once()
    record.assert_called_once_with("run_x", 5, {"cycle": 5})
    ws.send_str.assert_awaited_once_with(json.dumps({"cycle": 5}))


@run_async
async def test_tick_session_logs_and_never_sends_when_sim_step_itself_fails():
    """Regression: a bare except around the whole tick used to swallow a real
    simulation-logic crash (confirmed live: config.NIGHT_CYCLE_REVIEWER_MODEL
    pointing at a model that was never pulled crashed _run_night_cycle every
    NIGHT_CYCLE_EVERY_N_CYCLES, invisibly, for the life of this project) the
    exact same way it swallowed an ordinary dropped websocket connection. A
    sim.step() failure must never reach ws.send_str -- there's no real
    snapshot to send -- and must not propagate out of _tick_session (the
    broadcast loop calling this ticks every other session too)."""
    sim = mock.Mock()
    sim.paused = False
    sim.step = mock.AsyncMock(side_effect=RuntimeError("boom"))
    ws = mock.AsyncMock()

    with mock.patch("backend.app.record_board_state") as record, mock.patch("traceback.print_exc") as print_exc:
        await _tick_session(ws, {"sim": sim})  # must not raise

    record.assert_not_called()
    ws.send_str.assert_not_called()
    print_exc.assert_called_once()


@run_async
async def test_tick_session_sends_the_debug_snapshot_for_an_observer_and_never_steps():
    """Explicit request, 2026-09-09: the live LLM-console debug page attaches
    to a run already going in another tab (see ws_handler's OBSERVE command)
    rather than owning its own sim -- it must never call sim.step() itself,
    since the owning session's own _tick_session call already does, in the
    same broadcast_loop gather. Stepping here too would advance the shared
    sim twice per tick."""
    sim = mock.Mock()
    sim.paused = False
    sim.step = mock.AsyncMock()
    sim.debug_snapshot.return_value = {"cycle": 5, "tribes": {}}
    ws = mock.AsyncMock()

    with mock.patch("backend.app.record_board_state") as record:
        await _tick_session(ws, {"sim": sim, "observer": True})

    sim.step.assert_not_called()
    record.assert_not_called()
    ws.send_str.assert_awaited_once_with(json.dumps({"cycle": 5, "tribes": {}}))


@run_async
async def test_tick_session_observer_swallows_a_send_failure():
    sim = mock.Mock()
    sim.debug_snapshot.return_value = {"cycle": 1, "tribes": {}}
    ws = mock.AsyncMock()
    ws.send_str = mock.AsyncMock(side_effect=ConnectionResetError("gone"))

    await _tick_session(ws, {"sim": sim, "observer": True})  # must not raise


@run_async
async def test_tick_session_still_swallows_a_send_failure_after_a_successful_tick():
    """The one case this bare except is actually for -- a viewer's connection
    drops between the tick finishing and the send going out."""
    sim = mock.Mock()
    sim.paused = False
    sim.step = mock.AsyncMock()
    sim.snapshot.return_value = {"cycle": 1}
    sim.run_id, sim.cycle = "run_x", 1
    ws = mock.AsyncMock()
    ws.send_str = mock.AsyncMock(side_effect=ConnectionResetError("gone"))

    with mock.patch("backend.app.record_board_state"):
        await _tick_session(ws, {"sim": sim})  # must not raise
