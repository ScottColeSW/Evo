"""2026-10-04: a gather action whose stockpile is at the storage cap adds nothing, so it is not offered."""
from backend.simulation import Simulation, _storage_cap


def _actions(**stock):
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    sim._found_territory(tribe)
    for name, value in stock.items():
        setattr(tribe, name, value)
    _, ctx = sim._prepare_turn(tribe)
    return ctx["available_actions"], tribe


def test_a_full_stockpile_hides_its_gather():
    probe, tribe = _actions()
    cap = _storage_cap(tribe)
    actions, _ = _actions(stone=cap)
    assert "GATHER_STONE" not in actions


def test_a_stockpile_below_the_cap_keeps_its_gather():
    actions, tribe = _actions()
    assert "GATHER_WOOD" in actions or "GATHER_STONE" in actions or "GATHER_FOOD" in actions
