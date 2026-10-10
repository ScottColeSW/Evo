"""2026-10-10: every tribe's night reflection runs at the same time (Simulation._run_night_cycles) instead of one after another, a hung call cannot freeze the night, and the reflection can optionally use the
tribe's own model. What a tribe receives is unchanged. Inside two real nights the second tribe's reflection was fast and the waits were long; Ollama still runs one request per model at a time unless
OLLAMA_NUM_PARALLEL is raised, so these tests use a fake model."""
import asyncio
import random
import time
from unittest import mock

from backend import config
from backend.simulation import Simulation
from tests.conftest import run_async


def _two_tribes():
    sim = Simulation([{"name": "Alpha", "model": "qwen2.5:3b", "x": 40, "y": 37}, {"name": "Beta", "model": "qwen2.5:3b", "x": 70, "y": 70}])
    for tribe in sim.tribes.values():
        tribe.chief_name = "Chief " + tribe.name
    return sim


def _fake_reflect(delay=0.0, log=None):
    async def fake(client, reviewer_model, tribe_name, current_philosophy, recent_events, inventory="", current_decree="", dmm_built=False, departure_eligible=False, **kwargs):
        if log is not None:
            log.append(("model", tribe_name, reviewer_model, kwargs.get("num_ctx")))
        if delay:
            await asyncio.sleep(delay)
        return {"private_thoughts": f"{tribe_name} thinks about the river.", "revised_philosophy": f"{tribe_name} philosophy",
                "changed": True, "reasoning": "ok", "proposed_decree": f"{tribe_name} decree"}
    return fake


@run_async
async def test_both_tribes_reflect_at_the_same_time():
    sim = _two_tribes()
    started = time.perf_counter()
    with mock.patch("backend.simulation.reflect_on_history", _fake_reflect(delay=0.3)), mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        await sim._run_night_cycles(list(sim.tribes.values()))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5  # one after another would take at least 0.6 s
    assert [t.chief_philosophy for t in sim.tribes.values()] == ["Alpha philosophy", "Beta philosophy"]
    assert all(t.reflections_run == 1 for t in sim.tribes.values())


@run_async
async def test_what_each_tribe_receives_is_the_same_as_doing_the_nights_one_after_another():
    def outcome(sim):
        return [(t.chief_philosophy, t.chief_decree, t.reflections_run, t.last_reflection, list(t.history)[-3:], t.pending_birth) for t in sim.tribes.values()]

    with mock.patch("backend.simulation.reflect_on_history", _fake_reflect()):
        one_by_one = _two_tribes()
        random.seed(7)
        with mock.patch.object(one_by_one.client, "embed", mock.AsyncMock(return_value=None)):
            for tribe in one_by_one.tribes.values():
                await one_by_one._run_night_cycle(tribe)
        together = _two_tribes()
        random.seed(7)
        with mock.patch.object(together.client, "embed", mock.AsyncMock(return_value=None)):
            await together._run_night_cycles(list(together.tribes.values()))
    assert outcome(together) == outcome(one_by_one)


@run_async
async def test_the_quick_steps_of_every_tribe_run_before_any_model_call():
    sim = _two_tribes()
    order = []
    real_pressure = sim._advance_population_pressure
    sim._advance_population_pressure = lambda tribe, night=False: (order.append(("quick", tribe.name)), real_pressure(tribe, night=night))[1]
    with mock.patch("backend.simulation.reflect_on_history", _fake_reflect(log=order)), mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        await sim._run_night_cycles(list(sim.tribes.values()))
    kinds = [kind for kind, *_ in order]
    assert kinds[:2] == ["quick", "quick"] and kinds[2:] == ["model", "model"]


@run_async
async def test_a_tribe_that_went_extinct_during_the_night_is_not_given_a_reflection():
    sim = _two_tribes()
    beta = sim.tribes["tribe_1"]

    async def reflect_and_extinct(client, reviewer_model, tribe_name, *args, **kwargs):
        beta.extinct = True
        return {"private_thoughts": "x", "revised_philosophy": "late", "changed": True, "reasoning": ""}

    with mock.patch("backend.simulation.reflect_on_history", reflect_and_extinct), mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        await sim._run_night_cycles(list(sim.tribes.values()))
    assert beta.reflections_run == 0 and sim.tribes["tribe_0"].reflections_run == 1


@run_async
async def test_a_hung_reflection_is_skipped_after_the_time_limit_and_the_night_goes_on(monkeypatch):
    monkeypatch.setattr(config, "NIGHT_REFLECTION_TIMEOUT_SECONDS", 0.2)
    sim = _two_tribes()

    async def hang_for_alpha(client, reviewer_model, tribe_name, *args, **kwargs):
        if tribe_name == "Alpha":
            await asyncio.sleep(30)
        return {"private_thoughts": "ok", "revised_philosophy": "Beta philosophy", "changed": True, "reasoning": ""}

    started = time.perf_counter()
    with mock.patch("backend.simulation.reflect_on_history", hang_for_alpha), mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        await sim._run_night_cycles(list(sim.tribes.values()))
    assert time.perf_counter() - started < 2.0
    assert sim.tribes["tribe_1"].chief_philosophy == "Beta philosophy"          # Beta's night was not held up
    assert sim.tribes["tribe_0"].chief_philosophy != "Alpha philosophy"        # Alpha's was skipped
    assert "night reflection" in sim.event_log.path.read_text(encoding="utf-8")


def test_the_default_reflection_model_is_unchanged_and_tribe_makes_each_tribe_use_its_own(monkeypatch):
    sim = _two_tribes()
    monkeypatch.delenv("REFLECTION_MODEL", raising=False)
    assert [sim._reflection_model_for(t) for t in sim.tribes.values()] == [config.REFLECTION_MODEL] * 2
    assert not sim._reflection_uses_tribe_model()
    monkeypatch.setenv("REFLECTION_MODEL", "tribe")
    assert sim._reflection_uses_tribe_model()
    assert [sim._reflection_model_for(t) for t in sim.tribes.values()] == ["qwen2.5:3b", "qwen2.5:3b"]


@run_async
async def test_the_own_model_option_uses_the_turn_context_and_the_default_passes_no_context(monkeypatch):
    monkeypatch.delenv("REFLECTION_MODEL", raising=False)
    default_calls = []
    sim = _two_tribes()
    with mock.patch("backend.simulation.reflect_on_history", _fake_reflect(log=default_calls)), mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        await sim._run_night_cycles(list(sim.tribes.values()))
    assert all(model == config.REFLECTION_MODEL and ctx is None for _, _, model, ctx in default_calls)

    monkeypatch.setenv("REFLECTION_MODEL", "tribe")
    own_calls = []
    sim = _two_tribes()
    with mock.patch("backend.simulation.reflect_on_history", _fake_reflect(log=own_calls)), mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        await sim._run_night_cycles(list(sim.tribes.values()))
    assert all(model == "qwen2.5:3b" and ctx == config.REFLECTION_CONTEXT_OWN_MODEL for _, _, model, ctx in own_calls)


@run_async
async def test_a_tribe_whose_own_model_equals_the_reflection_model_by_name_still_gets_the_default_call():
    """The own-model option is decided by the setting, never by the names happening to match (gemma2:2b tribes with the gemma2:2b reviewer)."""
    sim = Simulation([{"name": "G", "model": config.REFLECTION_MODEL, "x": 40, "y": 37}])
    sim.tribes["tribe_0"].chief_name = "Chief G"
    calls = []
    with mock.patch("backend.simulation.reflect_on_history", _fake_reflect(log=calls)), mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        await sim._run_night_cycles(list(sim.tribes.values()))
    assert calls and calls[0][3] is None
