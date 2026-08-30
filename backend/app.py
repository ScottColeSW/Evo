import asyncio
import json
from pathlib import Path

import httpx
from aiohttp import WSMsgType, web

from . import config
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
                    session["sim"] = await Simulation.create(tribe_configs, config.OLLAMA_URL)
            elif command == "TOGGLE_PAUSE" and session["sim"] is not None:
                session["sim"].toggle_pause()
            elif command == "ADD_TRIBE" and session["sim"] is not None:
                name = data.get("name") or "New Tribe"
                model = data.get("model")
                if model:
                    await session["sim"].add_tribe(name, model, data.get("x"), data.get("y"))
    finally:
        request.app["sessions"].pop(ws, None)
    return ws


async def _tick_session(ws: web.WebSocketResponse, session: dict) -> None:
    sim = session.get("sim")
    if sim is None:
        return
    try:
        await sim.step()
        await ws.send_str(json.dumps(sim.snapshot()))
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


async def on_startup(app: web.Application) -> None:
    app["sessions"] = {}
    app["bg_task"] = asyncio.create_task(broadcast_loop(app))


async def on_cleanup(app: web.Application) -> None:
    app["bg_task"].cancel()


def create_app() -> web.Application:
    app = web.Application()
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
