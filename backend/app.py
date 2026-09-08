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
                # Immediate feedback instead of waiting up to TICK_SECONDS for the
                # next broadcast tick -- also what makes it safe for _tick_session to
                # skip a paused sim entirely below (see its own comment) without the
                # frontend's pause indicator ever going stale.
                await ws.send_str(json.dumps(session["sim"].snapshot()))
            elif command == "ADD_TRIBE" and session["sim"] is not None:
                name = data.get("name") or "New Tribe"
                model = data.get("model")
                if model:
                    await session["sim"].add_tribe(name, model, data.get("x"), data.get("y"))
            elif command == "STOP" and session["sim"] is not None:
                # Explicit end of this run -- PAUSE only stops stepping, it never
                # released the models a run had loaded. Distinct from just closing
                # the tab (see the finally block below, which catches that case too).
                #
                # Explicit request: "since I can click Quit anytime, it should
                # come up when I quit" -- the end-of-run splash (Simulation.
                # _generate_game_over_summary) used to only ever appear on the
                # two automatic endings (extinction, era ceiling); a manual
                # QUIT just closed the socket and reloaded straight back to the
                # picker with no summary at all. _trigger_game_over already
                # does the real stop-and-unload work (same as the old bare
                # shutdown() call) -- this just also tags it and sends one
                # final snapshot so the frontend's own game-over check fires
                # before the session is torn down.
                sim = session["sim"]
                await sim._trigger_game_over("manual_quit")
                record_board_state(sim.run_id, sim.cycle, sim.snapshot())
                await ws.send_str(json.dumps(sim.snapshot()))
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
    # Live report: "the game is paused but it sure seems to be working the drive...
    # is there something hitting the disc in Pause mode?" -- confirmed: sim.step()
    # itself already no-ops while paused (returns before self.cycle even
    # increments), but this function used to keep computing a fresh snapshot and
    # writing it to board_history.db every tick regardless, forever, for a cycle
    # number that was never actually changing. TOGGLE_PAUSE above now sends one
    # immediate snapshot the moment the state actually flips, so skipping every
    # tick entirely while paused doesn't leave the frontend's pause indicator
    # stale -- there's simply nothing left to record or broadcast until unpaused.
    if sim.paused:
        return
    # Live-bug-adjacent finding: this used to be one try/except around the whole
    # tick, "connection may have dropped between the tick starting and finishing"
    # -- but that same bare except was also silently swallowing genuine
    # simulation-logic crashes (confirmed: config.NIGHT_CYCLE_REVIEWER_MODEL
    # pointing at a model that was never pulled crashed _run_night_cycle every
    # NIGHT_CYCLE_EVERY_N_CYCLES for the life of this project, invisibly -- the
    # whole tick, including record_board_state, silently never happened that
    # cycle, was never logged, and looked identical to an ordinary dropped
    # connection). Split in two: a real sim.step()/snapshot/record failure is
    # printed (server_err.log, same convention as this app's own launcher
    # already redirects to) instead of vanishing; only the actual send -- the
    # one step that can legitimately fail just because a viewer's tab closed
    # mid-tick -- stays silently swallowed.
    try:
        await sim.step()
        snapshot = sim.snapshot()
        record_board_state(sim.run_id, sim.cycle, snapshot)
    except Exception:
        import traceback
        traceback.print_exc()
        return
    try:
        await ws.send_str(json.dumps(snapshot))
    except Exception:
        pass  # connection may have dropped between the tick finishing and the send


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
