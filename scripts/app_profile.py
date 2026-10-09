"""Measure the app, not the models (2026-10-09).

The real Simulation runs with a stub in place of Ollama that answers at once (or after a fixed pretend "thinking" delay), so every millisecond measured is the app's own.
A tick is timed the way backend/app.py _tick_session does it: step, snapshot, the SQLite board-history write, json.dumps for the websocket.

  python scratch/app_profile.py --mode fresh|mid --cycles N --config default|lean [--delay SECONDS] [--profile]

--config lean turns off all optional logging and checking: run data records, the chronicle file mirror, the decision journal, the journal read-back, the reflection judge, the
Library shadow judging, and the board-history write. --delay makes each model call take that long, to measure how long the event loop is blocked while models "think".
Output is one JSON line on stdout (and, with --profile, the top of a cProfile listing). Temp files go to the TMP folder, never to logs/.
"""
import argparse
import asyncio
import cProfile
import io
import json
import os
import pstats
import random
import re
import statistics as st
import sys
import tempfile
import time

sys.path.insert(0, ".")
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("fresh", "mid"), default="fresh")
parser.add_argument("--cycles", type=int, default=100)
parser.add_argument("--config", choices=("default", "lean"), default="default")
parser.add_argument("--delay", type=float, default=0.0)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--cache-biome", action="store_true")
args = parser.parse_args()

TMP = tempfile.mkdtemp()
os.environ["NUDGES"] = "off"
os.environ["PROMPT_FORMAT"] = "full"
os.environ["JOURNAL_READBACK"] = "off"
os.environ["REFLECTION_JUDGE"] = "off"
if args.config == "lean":
    os.environ["LEAN_RUN"] = "on"  # the real switch (backend/config.py), not patches

from backend import board_history, config, event_log, ollama_client  # noqa: E402

event_log.DEFAULT_LOG_DIR = TMP
board_history.DEFAULT_DB_PATH = os.path.join(TMP, "board_history.db")
if args.cache_biome:
    import functools
    from backend import world as _world
    _world.biome_at = functools.lru_cache(maxsize=None)(_world.biome_at)
from backend import simulation as simmod  # noqa: E402
from backend.tribe_fixtures import apply_tribe_fixture, load_fixture  # noqa: E402

rng = random.Random(7)
ACTION_LIST = re.compile(r"era-appropriate action names,\s*copied verbatim with no other text: \[(.*?)\]", re.S)
calls = {"turn": 0, "other": 0}


async def fake_generate_json_with_raw(self, model, prompt, temperature=0.7, num_ctx=4096, keep_alive="5m"):
    if args.delay:
        await asyncio.sleep(args.delay)
    m = ACTION_LIST.search(prompt)
    if m:
        calls["turn"] += 1
        names = re.findall(r"'([A-Z_]+)'", m.group(1))
        action = rng.choice(names)
        reply = {"visual_action": action, "metacognitive_rationale": "stub", "synthetic_language_broadcast": "KRA-ZUL", "target_vector": None}
        return reply, json.dumps(reply)
    calls["other"] += 1
    return {}, "{}"


async def fake_generate_json(self, model, prompt, temperature=0.7, num_ctx=4096, keep_alive="5m"):
    return (await fake_generate_json_with_raw(self, model, prompt, temperature, num_ctx, keep_alive))[0]


async def fake_generate_text(self, *a, **k):
    return ""


async def fake_embed(self, *a, **k):
    return None


async def fake_list_models(self, *a, **k):
    return ["stub-model"]


async def fake_none(self, *a, **k):
    return None


ollama_client.OllamaClient.generate_json_with_raw = fake_generate_json_with_raw
ollama_client.OllamaClient.generate_json = fake_generate_json
ollama_client.OllamaClient.generate_text = fake_generate_text
ollama_client.OllamaClient.embed = fake_embed
ollama_client.OllamaClient.list_models = fake_list_models
ollama_client.OllamaClient.unload_model = fake_none
ollama_client.OllamaClient.list_loaded_models = lambda self: asyncio.sleep(0, result=[])


async def fake_vram(self, *a, **k):
    return True, ""


simmod.HardwareVRAMBoundaryGuard.verify_vram_safety_margin = fake_vram


async def lag_monitor(samples, stop):
    """How late a 5 ms sleep wakes up: the event loop could not run anything else for that long."""
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.005)
        samples.append((time.perf_counter() - t - 0.005) * 1000)


async def main():
    sim = await simmod.Simulation.create([{"name": "Tribe 1", "model": "stub-model"}, {"name": "Tribe 2", "model": "stub-model"}])
    if args.mode == "mid":
        cycle = None
        for tribe, name in zip(sim.tribes.values(), ("mid_game_15k_a", "mid_game_15k_b")):
            fixture = load_fixture(name)
            cycle = fixture["cycle"]
            apply_tribe_fixture(tribe, fixture, new_cycle=cycle)
        sim.cycle = cycle
    segments = {"step": [], "snapshot": [], "db_write": [], "json_dumps": [], "tick_total": []}
    sizes, lags, per_cycle_blocked = [], [], []
    stop = asyncio.Event()
    monitor = asyncio.create_task(lag_monitor(lags, stop))
    for _ in range(args.cycles):
        before = len(lags)
        t0 = time.perf_counter()
        await sim.step()
        t1 = time.perf_counter()
        snapshot = sim.snapshot()
        t2 = time.perf_counter()
        board_history.record_board_state(sim.run_id, sim.cycle, snapshot)  # a no-op in a lean run
        t3 = time.perf_counter()
        payload = json.dumps(snapshot)
        t4 = time.perf_counter()
        for key, v in zip(segments, ((t1 - t0), (t2 - t1), (t3 - t2), (t4 - t3), (t4 - t0))):
            segments[key].append(v * 1000)
        sizes.append(len(payload))
        per_cycle_blocked.append(sum(x for x in lags[before:] if x > 20))
        await asyncio.sleep(0)
    stop.set()
    await monitor
    return segments, sizes, lags, per_cycle_blocked


def summarize(v):
    v = sorted(v)
    return {"mean": round(st.mean(v), 1), "median": round(v[len(v) // 2], 1), "p95": round(v[int(len(v) * 0.95) - 1], 1), "max": round(v[-1], 1)}


profiler = cProfile.Profile() if args.profile else None
if profiler:
    profiler.enable()
started = time.perf_counter()
segments, sizes, lags, blocked = asyncio.run(main())
elapsed = time.perf_counter() - started
if profiler:
    profiler.disable()
result = {
    "config": args.config, "cache_biome": args.cache_biome, "mode": args.mode, "cycles": args.cycles, "delay_s": args.delay, "wall_seconds": round(elapsed, 1),
    "turn_calls": calls["turn"], "other_model_calls": calls["other"],
    "ms_per_tick": {k: summarize(v) for k, v in segments.items()},
    "snapshot_bytes": {"first": sizes[0], "last": sizes[-1], "mean": round(st.mean(sizes))},
    "loop_lag_ms": {"samples": len(lags), "over_20ms": sum(1 for x in lags if x > 20), "over_100ms": sum(1 for x in lags if x > 100), "max": round(max(lags), 1) if lags else 0},
    "loop_blocked_ms_per_cycle": summarize(blocked) if blocked else None,
}
result["files_written_to_temp"] = sorted(os.listdir(TMP))
print(json.dumps(result))
if profiler:
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(28)
    print("\n".join(line[:170] for line in s.getvalue().splitlines()[:46]))
