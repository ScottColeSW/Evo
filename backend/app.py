import asyncio
import json
from pathlib import Path

import httpx
from aiohttp import WSMsgType, web

from . import config
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


@routes.get("/ws")
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app["clients"].add(ws)
    sim_holder = request.app["sim_holder"]

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
                    sim_holder["sim"] = await Simulation.create(tribe_configs, config.OLLAMA_URL)
            elif command == "TOGGLE_PAUSE" and sim_holder["sim"] is not None:
                sim_holder["sim"].toggle_pause()
    finally:
        request.app["clients"].discard(ws)
    return ws


async def broadcast_loop(app: web.Application) -> None:
    sim_holder = app["sim_holder"]
    while True:
        sim = sim_holder["sim"]
        if sim is not None:
            await sim.step()
            payload = json.dumps(sim.snapshot())
            dead = set()
            for ws in app["clients"]:
                try:
                    await ws.send_str(payload)
                except Exception:
                    dead.add(ws)
            app["clients"] -= dead
        await asyncio.sleep(config.TICK_SECONDS)


async def on_startup(app: web.Application) -> None:
    app["clients"] = set()
    app["sim_holder"] = {"sim": None}
    app["bg_task"] = asyncio.create_task(broadcast_loop(app))


async def on_cleanup(app: web.Application) -> None:
    app["bg_task"].cancel()


def create_app() -> web.Application:
    app = web.Application()
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
