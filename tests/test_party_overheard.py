"""A traveling party overhears a rival's broadcast within the hearing radius and brings it home (2026-10-03)."""
from backend import config
from backend.simulation import Simulation


def _sim():
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}, {"name": "Mountain Tribe", "model": "qwen2.5:3b"}])
    a, b = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    b.x, b.y = a.x + 500, a.y  # far from the home camp: only a party in the field can hear
    b.last_broadcast, b.last_action = "KRA-ZUL", "HUNT_DEER"
    return sim, a, b


def test_party_within_radius_hears_and_reports_home():
    sim, a, b = _sim()
    exp = {"pos": (b.x - config.BROADCAST_HEARING_RADIUS + 5, b.y)}
    sim._party_listen(a, exp)
    assert [h["token"] for h in exp["overheard"].values()] == ["KRA-ZUL"]
    sim._party_report_overheard(a, exp)
    assert a.heard_by_parties[-1]["from"] == "Mountain Tribe"
    assert any("overheard" in line and "KRA-ZUL" in line for line in a.history)
    request, _ctx = sim._prepare_turn(a)
    assert "your travelers overheard Mountain Tribe broadcast 'KRA-ZUL' while performing HUNT_DEER" in request["prompt"]


def test_party_outside_radius_hears_nothing():
    sim, a, b = _sim()
    exp = {"pos": (b.x - config.BROADCAST_HEARING_RADIUS - 20, b.y)}
    sim._party_listen(a, exp)
    assert not exp.get("overheard")
    sim._party_report_overheard(a, exp)
    assert a.heard_by_parties == []


def test_heard_word_fades_from_view_after_window():
    sim, a, b = _sim()
    a.heard_by_parties.append({"token": "OLD-WORD", "from": "Mountain Tribe", "action": "HUNT_DEER", "cycle": 0})
    sim.cycle = config.PARTY_HEARD_VISIBLE_CYCLES + 5
    request, _ctx = sim._prepare_turn(a)
    assert "OLD-WORD" not in request["prompt"]
