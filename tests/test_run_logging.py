"""2026-10-05: what a run leaves on disk for analysis afterward: which settings it ran with, the menu offered at each decision, and the end-of-run
report (kept even when the tab is closed instead of the run ending)."""
import asyncio
import json
from unittest import mock

from backend.simulation import Simulation


def _records(sim):
    return [json.loads(line) for line in open(sim.event_log.path, encoding="utf-8")]


def _sim():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    sim.tribes["tribe_0"].chief_name = "Ashgar"
    return sim


def _patches(sim):
    return (mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})),
            mock.patch("backend.simulation.generate_endgame_narrative", mock.AsyncMock(return_value="")),
            mock.patch.object(sim.client, "unload_model", mock.AsyncMock()))


def test_the_first_step_logs_the_run_config_once_with_the_settings_that_matter(monkeypatch):
    monkeypatch.setenv("NUDGES", "off")
    monkeypatch.setenv("MENU_CAP", "8")
    sim = _sim()

    async def go():
        a, b, c = _patches(sim)
        with a, b, c:
            await sim.step()
            await sim.step()

    asyncio.run(go())
    configs = [r for r in _records(sim) if r.get("kind") == "run_config"]
    assert len(configs) == 1
    data = configs[0]["data"]
    assert data["nudges"] == "off" and data["menu_cap"] == 8
    assert data["tribes"] == [{"id": "tribe_0", "name": "A", "model": "gemma2:2b"}]
    assert "git_commit" in data and data["vessel_cost"] == [2500, 2500]


def test_a_decision_record_carries_the_menu_that_was_offered():
    from backend.simulation import Tribe
    from tests.test_actions import _bare_simulation

    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.event_log = mock.Mock()
    before = sim._journal_snapshot(tribe)
    sim._journal_record(tribe, "SCOUT", before, None, menu_size=3, menu=["SCOUT", "GATHER_FOOD", "GATHER_WATER"])
    kind, data = sim.event_log.record_data.call_args[0][1], sim.event_log.record_data.call_args[0][2]
    assert kind == "decision" and data["menu"] == ["SCOUT", "GATHER_FOOD", "GATHER_WATER"] and data["menu_size"] == 3
    assert "menu" not in tribe.decision_journal[-1]  # the in-memory journal the read-back uses stays small


def test_the_report_is_written_to_disk_at_game_over_and_logged():
    sim = _sim()

    async def go():
        a, b, c = _patches(sim)
        with a, b, c:
            await sim.step()
            await sim._trigger_game_over("manual_quit")

    asyncio.run(go())
    path = sim.event_log.path.parent / f"report_{sim.run_id}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["reason"] == "manual_quit" and saved["report"]["headline"]["scene"] == "quiet"
    ends = [r for r in _records(sim) if r.get("kind") == "game_over_report"]
    assert len(ends) == 1 and ends[0]["data"]["file"] == path.name


def test_a_plain_stop_still_keeps_the_timeline_so_far():
    sim = _sim()

    async def go():
        a, b, c = _patches(sim)
        with a, b, c:
            await sim.step()
            await sim.step()
            await sim.shutdown()
            await sim.shutdown()  # called again (the tab closes after a quit): nothing is written twice

    asyncio.run(go())
    path = sim.event_log.path.parent / f"report_{sim.run_id}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["reason"] == "stopped" and saved["report"]["timeline"]["tribes"]["tribe_0"]["cycles"]
    assert len([r for r in _records(sim) if r.get("kind") == "game_over_report"]) == 1
