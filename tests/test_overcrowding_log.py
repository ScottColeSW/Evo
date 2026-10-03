"""Phase 0 of docs/OVERCROWDING-REBELLION-DESIGN.md: each overcrowding cull is recorded with every rival's room, and the cull
itself is unchanged."""
import json

from backend import config
from backend.simulation import Simulation, _sustainable_population


def _records(sim, kind):
    if not sim.event_log.path.exists():          # the file is only created by the first record
        return []
    return [json.loads(line) for line in sim.event_log.path.read_text(encoding="utf-8").splitlines()
            if f'"kind": "{kind}"' in line]


def test_a_cull_is_logged_with_each_rivals_room_and_the_cull_is_unchanged():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    sim.cycle = 10                                # past any immortality window, so the cull really takes people
    a, b = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    a.population, b.population = 5000, 10
    a.discovered_rivals.add("tribe_1")
    target = round(_sustainable_population(a) * config.POPULATION_CARRYING_CAPACITY_TARGET_FRACTION)
    expected_lost = max(1, round((5000 - target) * config.POPULATION_PRESSURE_CULL_FRACTION))

    sim._advance_population_pressure(a)

    assert a.population == 5000 - expected_lost                      # behavior unchanged
    (record,) = _records(sim, "overcrowding")
    data = record["data"]
    assert record["tribe"] == "A" and data["population"] == 5000 and data["lost"] == expected_lost
    assert data["excess"] == 5000 - target
    (rival,) = data["rivals"]
    b_target = round(_sustainable_population(b) * config.POPULATION_CARRYING_CAPACITY_TARGET_FRACTION)
    assert rival["id"] == "tribe_1" and rival["discovered"] is True
    assert rival["room"] == b_target - 10 and rival["population"] == 10 and "message" in record


def test_no_cull_means_no_record():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    sim.tribes["tribe_0"].population = 10
    sim._advance_population_pressure(sim.tribes["tribe_0"])
    assert _records(sim, "overcrowding") == []
