"""2026-10-04: a TRADE with no rival at the target goes to the nearest rival already found, once the first wall ring stands."""
from backend import actions, city_layout
from backend.simulation import Simulation


def _sim(walled=True, discovered=True):
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    a, b = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    a.x, a.y, b.x, b.y = 20, 20, 58, 20  # 38 apart, like the recorded runs
    if discovered:
        a.discovered_rivals.add(b.id)
    if walled:
        a.wall_rings = [{"sections": []}]
        sim._wall_ok = True
    return sim, a, b


def _with_wall(monkeypatch):
    monkeypatch.setattr(city_layout, "ring_fully_built", lambda ring: True)


def test_an_unaimed_trade_reaches_the_known_rival(monkeypatch):
    _with_wall(monkeypatch)
    sim, a, b = _sim()
    a.food, b.food = 100, 100
    result = actions._trade(sim, a, "plains", (a.x, a.y))  # the Chief named no target: its own camp
    assert a.trades_completed == 1 and b.trades_completed == 1, result


def test_it_still_fails_without_a_finished_wall(monkeypatch):
    monkeypatch.setattr(city_layout, "ring_fully_built", lambda ring: False)
    sim, a, b = _sim()
    assert actions._trade(sim, a, "plains", (a.x, a.y)) == "found no rival encampment there to trade with"
    assert a.trades_completed == 0


def test_it_still_fails_with_no_rival_found(monkeypatch):
    _with_wall(monkeypatch)
    sim, a, b = _sim(discovered=False)
    assert actions._trade(sim, a, "plains", (a.x, a.y)) == "found no rival encampment there to trade with"
    assert a.trades_completed == 0


def test_a_rival_at_the_target_is_still_used_directly(monkeypatch):
    monkeypatch.setattr(city_layout, "ring_fully_built", lambda ring: False)  # no wall needed when the aim is true
    sim, a, b = _sim(discovered=False)
    actions._trade(sim, a, "plains", (b.x, b.y))
    assert a.trades_completed == 1
