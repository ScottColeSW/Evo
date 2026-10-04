"""2026-10-03: the Tannery's daily Fur grows with the herd instead of capping at 6 a day."""
from backend import config
from backend.simulation import Simulation


def _day(deer):
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.tannery_built = tribe.deer_pen_built = True
    tribe.deer = deer
    sim.cycle = config.DAY_LENGTH_CYCLES
    sim._is_camped = lambda t: True
    sim._advance_tannery_yield(tribe)
    return tribe


def test_a_big_herd_yields_far_more_than_the_old_cap():
    tribe = _day(63)
    assert tribe.tannery_fur_today == 12 * config.FUR_PER_DEER_FED
    assert tribe.tannery_fur_today > 3 * config.FUR_PER_DEER_FED


def test_a_small_herd_still_gets_the_old_range_and_keeps_its_minimum():
    tribe = _day(5)
    assert 1 <= 5 - tribe.deer <= 2
    assert tribe.deer >= config.DEER_PEN_MINIMUM_HERD_SIZE
