import dataclasses
from unittest import mock

from backend import benchmark_db
from backend.benchmark_scenarios import SCENARIOS, Scenario
from backend.simulation import Simulation
from tests.conftest import run_async

import run_benchmark

_FAKE_CHIEF = {"chief_name": "Test Chief", "victory_method": "a coin flip", "guiding_philosophy": "test philosophy"}


async def _fake_run_batch(self, requests):
    # Patched onto the class (ModelBatchScheduler.run_batch), not an instance --
    # `self` is the scheduler instance Python passes automatically on attribute
    # access, since a real Simulation (and therefore its scheduler) doesn't exist
    # yet at mock-setup time for these tests (it's constructed inside run_trial).
    return {
        r["id"]: {"intent": {"visual_action": "SCOUT", "target_vector": [55, 55]}, "latency_ms": 0.0}
        for r in requests
    }


def _small_budget_scenario(key, cycles=5):
    """A short-cycle copy of a real scenario -- small enough to stay well under
    NIGHT_CYCLE_EVERY_N_CYCLES (30), so a mocked run doesn't also need to stub
    reflect_on_history's own real LLM call."""
    return dataclasses.replace(SCENARIOS[key], cycle_budget=cycles)


def test_apply_starting_resources_overrides_every_tribe_when_set():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    scenario = Scenario(
        key="test", category="survival", tribe_count=2, spawn_positions=((0, 0), (1, 1)),
        cycle_budget=1, description="", starting_resources={"wood": 3, "stone": 3, "food": 3, "water": 3},
    )

    run_benchmark._apply_starting_resources(sim, scenario)

    for tribe in sim.tribes.values():
        assert (tribe.wood, tribe.stone, tribe.food, tribe.water) == (3, 3, 3, 3)


def test_apply_starting_resources_leaves_defaults_untouched_when_none():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    before = (tribe.wood, tribe.stone, tribe.food, tribe.water)
    scenario = Scenario(
        key="test", category="settlement", tribe_count=1, spawn_positions=((0, 0),),
        cycle_budget=1, description="", starting_resources=None,
    )

    run_benchmark._apply_starting_resources(sim, scenario)

    assert (tribe.wood, tribe.stone, tribe.food, tribe.water) == before


@run_async
async def test_run_trial_stops_at_the_cycle_budget_and_records_a_row(tmp_path, monkeypatch):
    db_path = str(tmp_path / "benchmark.db")
    monkeypatch.setattr(benchmark_db, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setitem(SCENARIOS, "survival", _small_budget_scenario("survival"))

    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)), \
         mock.patch("backend.scheduler.ModelBatchScheduler.run_batch", _fake_run_batch), \
         mock.patch("backend.ollama_client.OllamaClient.unload_model", mock.AsyncMock()):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        trial = await run_benchmark.run_trial("survival", ["gemma2:2b"], trial_seed=0)

    assert trial["cycles_run"] == 5
    assert trial["ended_reason"] == "budget_reached"
    assert len(trial["tribes"]) == 1

    stored = benchmark_db.list_trials(path=db_path)
    assert len(stored) == 1
    assert stored[0]["trial_id"] == trial["trial_id"]


@run_async
async def test_run_trial_is_deterministic_given_the_same_seed(tmp_path, monkeypatch):
    """The whole point of seeding: two trials with the same seed against the same
    fixed action sequence must land on identical facts."""
    db_path = str(tmp_path / "benchmark.db")
    monkeypatch.setattr(benchmark_db, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setitem(SCENARIOS, "survival", _small_budget_scenario("survival", cycles=10))

    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)), \
         mock.patch("backend.scheduler.ModelBatchScheduler.run_batch", _fake_run_batch), \
         mock.patch("backend.ollama_client.OllamaClient.unload_model", mock.AsyncMock()):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        first = await run_benchmark.run_trial("survival", ["gemma2:2b"], trial_seed=42)
        second = await run_benchmark.run_trial("survival", ["gemma2:2b"], trial_seed=42)

    assert first["tribes"][0]["final_population"] == second["tribes"][0]["final_population"]
    assert first["tribes"][0]["extinct"] == second["tribes"][0]["extinct"]


@run_async
async def test_run_trial_extracts_facts_for_both_tribes_in_a_two_tribe_scenario(tmp_path, monkeypatch):
    db_path = str(tmp_path / "benchmark.db")
    monkeypatch.setattr(benchmark_db, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setitem(SCENARIOS, "cooperation", _small_budget_scenario("cooperation"))

    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)), \
         mock.patch("backend.scheduler.ModelBatchScheduler.run_batch", _fake_run_batch), \
         mock.patch("backend.ollama_client.OllamaClient.unload_model", mock.AsyncMock()):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        trial = await run_benchmark.run_trial("cooperation", ["gemma2:2b", "qwen2.5:3b"], trial_seed=0)

    assert len(trial["tribes"]) == 2
    assert trial["tribes"][0]["model"] == "gemma2:2b"
    assert trial["tribes"][1]["model"] == "qwen2.5:3b"
