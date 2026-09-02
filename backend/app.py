import asyncio
import json
from pathlib import Path

import httpx
from aiohttp import WSMsgType, web

from . import config
from .board_history import record_board_state
from .experiment_log import read_all_experiment_runs, summarize_experiment
from .ollama_client import OllamaClient
from .scoreboard import read_all_results, summarize_by_model
from .simulation import Simulation

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

routes = web.RouteTableDef()


@routes.get("/")
async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(FRONTEND_DIR / "index.html")


@routes.get("/api/models")
async def models(request: web.Request) -> web.Response:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{config.OLLAMA_URL}/api/tags")
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            names = []
    return web.json_response({"models": names})


@routes.get("/api/scoreboard")
async def scoreboard(request: web.Request) -> web.Response:
    """The cross-run benchmark (backend/scoreboard.py) -- every tribe's lifetime
    summary, across every run this machine has ever completed, plus a per-model
    leaderboard rollup. This is what an evaluator comparing local models actually
    wants, not a per-run play-by-play (that's the in-browser Chronicle's job)."""
    results = read_all_results()
    return web.json_response({"results": results, "by_model": summarize_by_model(results)})


@routes.get("/api/experiments")
async def experiments(request: web.Request) -> web.Response:
    """The A/B-test log (backend/experiment_log.py) -- every headless hypothesis test
    run against this codebase (wording, list order, framing, whatever comes next),
    grouped by experiment name with a per-variant comparison. This is the standing
    record of "did the thing we tried actually change model behavior," not a single
    run's play-by-play."""
    runs = read_all_experiment_runs()
    experiment_names = sorted({r["experiment"] for r in runs})
    by_experiment = {name: summarize_experiment(name, runs) for name in experiment_names}
    return web.json_response({"runs": runs, "by_experiment": by_experiment})


@routes.get("/ws")
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    """Each connection owns its own Simulation. Two browser tabs (or two clients
    hitting this server) get two fully independent worlds, rather than one shared
    global simulation where the second START silently replaces the first."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    session = {"sim": None}
    request.app["sessions"][ws] = session

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            command = data.get("command")
            if command == "START":
                tribe_configs = data.get("tribes", [])
                if tribe_configs:
                    immortality_cycles = int(data.get("immortality_cycles") or 0)
                    session["sim"] = await Simulation.create(
                        tribe_configs, config.OLLAMA_URL, immortality_cycles
                    )
            elif command == "TOGGLE_PAUSE" and session["sim"] is not None:
                session["sim"].toggle_pause()
            elif command == "ADD_TRIBE" and session["sim"] is not None:
                name = data.get("name") or "New Tribe"
                model = data.get("model")
                if model:
                    await session["sim"].add_tribe(name, model, data.get("x"), data.get("y"))
            elif command == "STOP" and session["sim"] is not None:
                # Explicit end of this run -- PAUSE only stops stepping, it never
                # released the models a run had loaded. Distinct from just closing
                # the tab (see the finally block below, which catches that case too).
                await session["sim"].shutdown()
                session["sim"] = None
    finally:
        # A tab closing or reloading mid-game used to leave that session's models
        # resident in Ollama's VRAM until their keep_alive window expired on its own
        # -- only reaching the all-extinct GAME OVER state ever unloaded them. This
        # is the same cleanup an explicit STOP does, just triggered by disconnection
        # instead of a command.
        ended_session = request.app["sessions"].pop(ws, None)
        if ended_session is not None and ended_session.get("sim") is not None:
            await ended_session["sim"].shutdown()
    return ws


async def _tick_session(ws: web.WebSocketResponse, session: dict) -> None:
    sim = session.get("sim")
    if sim is None:
        return
    try:
        await sim.step()
        snapshot = sim.snapshot()
        record_board_state(sim.run_id, sim.cycle, snapshot)
        await ws.send_str(json.dumps(snapshot))
    except Exception:
        pass  # connection may have dropped between the tick starting and finishing


async def broadcast_loop(app: web.Application) -> None:
    while True:
        # Snapshot the dict before iterating -- a connection can close and remove
        # itself mid-tick from another coroutine.
        sessions = list(app["sessions"].items())
        if sessions:
            await asyncio.gather(*(_tick_session(ws, session) for ws, session in sessions))
        await asyncio.sleep(config.TICK_SECONDS)


async def _unload_stale_models() -> None:
    """A previous server process that got force-killed (rather than STOP/tab-close,
    both of which reach Simulation.shutdown()) leaves whatever models it had loaded
    sitting in Ollama's VRAM until their keep_alive window expires on its own --
    confirmed live (task manager showed two orphaned llama-server.exe processes well
    after the server that loaded them was gone). This app assumes exclusive ownership
    of the local Ollama instance's model lifecycle, so every fresh startup is a
    reasonable point to guarantee a clean slate regardless of how the last process
    ended, rather than relying on every shutdown path being graceful."""
    client = OllamaClient(config.OLLAMA_URL)
    for model in await client.list_loaded_models():
        print(f"[startup] unloading stale model left resident from a previous run: {model}")
        await client.unload_model(model)


async def on_startup(app: web.Application) -> None:
    app["sessions"] = {}
    await _unload_stale_models()
    app["bg_task"] = asyncio.create_task(broadcast_loop(app))


async def on_cleanup(app: web.Application) -> None:
    app["bg_task"].cancel()


def create_app() -> web.Application:
    app = web.Application()
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
