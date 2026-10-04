"""2026-10-03: a spy reports the rival's battle readiness: soldiers and battalions, readiness, barracks, patrol, walls, defenses."""
from unittest import mock

from backend import config
from backend.actions import _spy, military_intel, military_intel_text
from backend.simulation import Simulation


def _two_tribes():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    a, b = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    a.barracks_built = 1
    a.discovered_rivals.add(b.id)
    return sim, a, b


def test_an_undetected_spy_brings_back_the_military_picture():
    sim, a, b = _two_tribes()
    b.battalion_size, b.battalions, b.battalion_readiness = 120, [{"leader": "X", "size": 60}, {"leader": "Y", "size": 60}], 0.63
    b.barracks_built, b.barracks_upgrades, b.keep_built, b.moat_built = 3, 2, True, True
    b.battalion_patrol = {"phase": "patrolling", "pos": [1, 1], "started_cycle": 1, "target": None}
    with mock.patch("backend.actions.random.random", return_value=0.99):      # not detected
        message = _spy(sim, a, "plains", (b.x, b.y))
    assert "undetected" in message
    m = a.rival_intel[b.id]["military"]
    assert m["soldiers"] == 120 and m["battalions"] == 2 and m["readiness"] == 0.63
    assert m["barracks"] == 3 and m["barracks_upgrades"] == 2 and m["patrol"] == "patrolling" and m["defenses"] == ["keep", "moat"]
    text = military_intel_text(m)
    assert "120 soldiers in 2 battalions at 63% readiness" in text and "3 barracks (2 upgrades)" in text
    assert "a patrol is out" in text and "defenses: keep, moat" in text


def test_the_report_reaches_the_chief_in_a_live_turn_and_at_night():
    sim, a, b = _two_tribes()
    a.has_ever_settled = True
    sim._found_territory(a)
    b.battalion_size = 40
    sim.cycle = 12
    a.rival_intel[b.id] = {"cycle": 12, "population": b.population, "era": b.era, "wood": 1, "stone": 2, "food": 3, "water": 4,
                           "long_houses_built": 0, "wall_ring_count": 0, "military": military_intel(b)}
    request, _ctx = sim._prepare_turn(a)
    assert "What your spy reported of B's forces (cycle 12): 40 soldiers" in request["prompt"]
    assert "Their forces: 40 soldiers" in sim._build_night_inventory(a)


def test_walls_are_counted_as_sections_built_and_reinforced():
    sim, a, b = _two_tribes()
    b.has_ever_settled = True
    sim._found_territory(b)
    sections = [s for ring in b.wall_rings for s in ring["sections"] if not s["natural_barrier"]]
    for s in sections[:3]:
        s["unlocked"] = True
    sections[0]["tier"] = config.WALL_MAX_LAYERS
    m = military_intel(b)
    assert m["wall_sections_built"] == 3 and m["wall_sections_reinforced"] >= 1 and m["wall_sections"] == len(sections)
