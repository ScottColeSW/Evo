import dataclasses
from unittest import mock

from backend import benchmark_db
from backend.benchmark_scenarios import SCENARIOS, Scenario
from backend.simulation import Simulation
from tests.conftest import run_async

import run_benchmark

_FAKE_CHIEF = {"chief_name": "Test Chief", "victory_method": "a coin flip", "guiding_philosophy": "test philosophy"}
# A real food surplus can trigger _check_for_celebration's breeding chance even within
# these short trials, hitting a real, unmocked breed_individuals() network call --
# same class of gap elect_chief above already guards against.
_FAKE_CHILD = {"child_name": "Test Child", "note": ""}


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


def test_apply_disabled_actions_sets_sim_disabled_actions_when_set():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    scenario = Scenario(
        key="test", category="conflict", tribe_count=1, spawn_positions=((0, 0),),
        cycle_budget=1, description="", disabled_actions=("DECLARE_ALLIANCE",),
    )

    run_benchmark._apply_disabled_actions(sim, scenario)

    assert sim.disabled_actions == {"DECLARE_ALLIANCE"}


def test_apply_disabled_actions_leaves_the_empty_default_untouched_when_none():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    scenario = Scenario(
        key="test", category="conflict", tribe_count=1, spawn_positions=((0, 0),),
        cycle_budget=1, description="", disabled_actions=None,
    )

    run_benchmark._apply_disabled_actions(sim, scenario)

    assert sim.disabled_actions == set()


def test_war_ready_5000_no_alliance_actually_removes_it_from_a_real_tribes_menu():
    """End-to-end check, not just the plumbing: applying war_ready_5000_no_alliance's
    real fixtures and disabled_actions together onto a real Simulation must leave
    DECLARE_ALLIANCE out of a battle-ready tribe's actual _prepare_turn menu."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    scenario = SCENARIOS["war_ready_5000_no_alliance"]

    source_cycle = run_benchmark._apply_starting_fixtures(sim, scenario)
    sim.cycle = source_cycle
    run_benchmark._apply_disabled_actions(sim, scenario)

    # Real bug, 2026-09-16: the fixed fixtures aren't symmetric -- war_ready_b
    # starts at departure_era (the top rung, no next_era), which makes it
    # eligible for _prepare_turn's own endgame_locked/battle_ready_locked
    # narrowing the instant a living rival exists (that check needs no
    # discovery at all). war_ready_a (object_creator_era) never reaches that
    # path. A first version of this test only checked tribe index 0 -- exactly
    # why the original disabled_actions filter (placed after, not before,
    # endgame_only/battle_ready_only's own narrowing) passed here in testing
    # but still let DECLARE_ALLIANCE fire in a real trial: those narrowing
    # steps each fail open to whatever survives when their own filtered
    # subset is empty, and DECLARE_ALLIANCE was exactly the one survivor at
    # the real cycle this broke. Both tribes must be checked.
    for tribe in sim.tribes.values():
        _, ctx = sim._prepare_turn(tribe)
        assert "DECLARE_ALLIANCE" not in ctx["available_actions"], tribe.name
        assert ctx["available_actions"]  # never empty


def test_apply_starting_fixtures_applies_the_real_war_ready_fixtures():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    scenario = SCENARIOS["war_ready_5000"]

    source_cycle = run_benchmark._apply_starting_fixtures(sim, scenario)

    tribes = list(sim.tribes.values())
    assert source_cycle is not None and source_cycle > 0
    assert tribes[0].population > 3000
    assert tribes[1].population > 3000
    assert tribes[0].barracks_built
    assert tribes[1].barracks_built
    # Identity stays the trial's own, never the fixture's -- see
    # tribe_fixtures.py's own docstring.
    assert tribes[0].model == "gemma2:2b"
    assert tribes[1].model == "qwen2.5:3b"


@run_async
async def test_run_trial_reports_cycles_actually_run_not_the_absolute_cycle(tmp_path, monkeypatch):
    """Real bug, 2026-09-16: a fixture-started trial (war_ready_5000's fixtures both
    start at cycle 344) recorded cycles_run as sim.cycle itself -- the absolute cycle
    number -- rather than how many cycles the trial actually ran. Harmless for
    war_ready_5000's own conflict scoring (score_conflict never reads cycles_run),
    but would silently break score_survival's cycles_run/cycle_budget fraction (which
    assumes cycles_run <= cycle_budget) for any future fixture-started survival
    scenario."""
    db_path = str(tmp_path / "benchmark.db")
    monkeypatch.setattr(benchmark_db, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setitem(SCENARIOS, "war_ready_5000", _small_budget_scenario("war_ready_5000", cycles=5))

    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)), \
         mock.patch("backend.simulation.breed_individuals", mock.AsyncMock(return_value=_FAKE_CHILD)), \
         mock.patch("backend.scheduler.ModelBatchScheduler.run_batch", _fake_run_batch), \
         mock.patch("backend.ollama_client.OllamaClient.unload_model", mock.AsyncMock()):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        trial = await run_benchmark.run_trial("war_ready_5000", ["gemma2:2b", "qwen2.5:3b"], trial_seed=0)

    assert trial["cycles_run"] == 5  # not 344 + 5 == 349, the absolute cycle number


def test_apply_starting_fixtures_returns_none_when_not_set():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    scenario = Scenario(
        key="test", category="settlement", tribe_count=1, spawn_positions=((0, 0),),
        cycle_budget=1, description="", starting_fixtures=None,
    )

    assert run_benchmark._apply_starting_fixtures(sim, scenario) is None


def _two_tribe_scenario(category="conflict", cycle_budget=1):
    return Scenario(
        key="test", category=category, tribe_count=2, spawn_positions=((0, 0), (1, 1)),
        cycle_budget=cycle_budget, description="",
    )


def test_only_one_tribe_left_is_false_for_a_single_tribe_scenario_even_if_extinct():
    """1-tribe scenarios (survival, settlement) start with exactly one living
    tribe by design -- this must never fire for them, or a trial would stop on
    its own first check before ever really running."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    sim.tribes["tribe_0"].extinct = True
    scenario = Scenario(
        key="test", category="survival", tribe_count=1, spawn_positions=((0, 0),),
        cycle_budget=1, description="",
    )

    assert run_benchmark._only_one_tribe_left(sim, scenario) is False


def test_only_one_tribe_left_is_false_while_both_tribes_are_alive():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])

    assert run_benchmark._only_one_tribe_left(sim, _two_tribe_scenario()) is False


def test_only_one_tribe_left_is_true_once_a_rival_goes_extinct():
    """Extinction (Tribe.extinct=True) leaves the tribe in sim.tribes -- see
    Simulation._lose_population."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    sim.tribes["tribe_1"].extinct = True

    assert run_benchmark._only_one_tribe_left(sim, _two_tribe_scenario()) is True


def test_only_one_tribe_left_is_true_after_a_conquest_merge_removes_the_loser():
    """A conquest merge deletes the loser from sim.tribes outright instead of
    marking it extinct -- see Simulation._merge_tribes."""
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    del sim.tribes["tribe_1"]

    assert run_benchmark._only_one_tribe_left(sim, _two_tribe_scenario()) is True


@run_async
async def test_run_trial_stops_early_once_only_one_tribe_is_left(tmp_path, monkeypatch):
    """Explicit request, 2026-09-16: "the test should end after only 1 tribe
    remains to save time on this test." Simulation.game_over only fires once
    EVERY tribe is extinct -- without this, a 2-tribe trial would burn the
    rest of cycle_budget's real Ollama inference on a lone survivor with
    nothing left to interact with. _only_one_tribe_left is mocked directly
    (rather than trying to force a real extinction/merge through the fake
    scheduler) since it's already covered on its own above -- this test is
    only about run_trial actually acting on it."""
    db_path = str(tmp_path / "benchmark.db")
    monkeypatch.setattr(benchmark_db, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setitem(SCENARIOS, "cooperation", _small_budget_scenario("cooperation", cycles=10))
    monkeypatch.setattr(run_benchmark, "_only_one_tribe_left", lambda sim, scenario: sim.cycle >= 2)

    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)), \
         mock.patch("backend.simulation.breed_individuals", mock.AsyncMock(return_value=_FAKE_CHILD)), \
         mock.patch("backend.scheduler.ModelBatchScheduler.run_batch", _fake_run_batch), \
         mock.patch("backend.ollama_client.OllamaClient.unload_model", mock.AsyncMock()):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        trial = await run_benchmark.run_trial("cooperation", ["gemma2:2b", "qwen2.5:3b"], trial_seed=0)

    assert trial["cycles_run"] == 2  # not the full 10-cycle budget
    assert trial["ended_reason"] == "single_tribe_remaining"


@run_async
async def test_run_trial_stops_at_the_cycle_budget_and_records_a_row(tmp_path, monkeypatch):
    db_path = str(tmp_path / "benchmark.db")
    monkeypatch.setattr(benchmark_db, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setitem(SCENARIOS, "survival", _small_budget_scenario("survival"))

    with mock.patch("backend.simulation.HardwareVRAMBoundaryGuard") as mock_guard_cls, \
         mock.patch("backend.simulation.elect_chief", mock.AsyncMock(return_value=_FAKE_CHIEF)), \
         mock.patch("backend.simulation.breed_individuals", mock.AsyncMock(return_value=_FAKE_CHILD)), \
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
         mock.patch("backend.simulation.breed_individuals", mock.AsyncMock(return_value=_FAKE_CHILD)), \
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
         mock.patch("backend.simulation.breed_individuals", mock.AsyncMock(return_value=_FAKE_CHILD)), \
         mock.patch("backend.scheduler.ModelBatchScheduler.run_batch", _fake_run_batch), \
         mock.patch("backend.ollama_client.OllamaClient.unload_model", mock.AsyncMock()):
        mock_guard_cls.return_value.verify_vram_safety_margin = mock.AsyncMock(return_value=(True, ""))
        trial = await run_benchmark.run_trial("cooperation", ["gemma2:2b", "qwen2.5:3b"], trial_seed=0)

    assert len(trial["tribes"]) == 2
    assert trial["tribes"][0]["model"] == "gemma2:2b"
    assert trial["tribes"][1]["model"] == "qwen2.5:3b"
