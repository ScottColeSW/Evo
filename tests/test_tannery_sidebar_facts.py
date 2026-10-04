"""2026-10-03: the Tannery sidebar figure changed only once a day and looked static; the snapshot now also carries the hunts' running
share and the last few days."""
import random
from unittest import mock

from backend import config
from backend.simulation import Simulation


def _tannery_tribe():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    sim._found_territory(tribe)
    tribe.cycles_since_relocate = config.SETTLEMENT_STABILITY_CYCLES
    tribe.tannery_built = True
    return sim, tribe


def test_the_snapshot_carries_the_running_hunt_share_and_a_history_of_days():
    sim, tribe = _tannery_tribe()
    tribe.tannery_fur_pending_from_hunts = 3
    assert sim.snapshot()["tribes"]["tribe_0"]["tannery_fur_pending"] == 3
    for day, hunts in enumerate((2, 0, 5), start=1):
        sim.cycle = day * config.DAY_LENGTH_CYCLES
        tribe.tannery_fur_pending_from_hunts = hunts
        with mock.patch.object(sim, "_is_camped", return_value=True):
            sim._advance_tannery_yield(tribe)
    assert tribe.tannery_fur_history == [2, 0, 5]
    assert sim.snapshot()["tribes"]["tribe_0"]["tannery_fur_history"] == [2, 0, 5]
    assert tribe.tannery_fur_pending_from_hunts == 0                          # the day's share was counted and cleared
