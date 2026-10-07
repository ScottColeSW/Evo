"""2026-10-07: a live run's tribe (departure era, vessel 2000/2500, a raider camp just inside its grown territory) was left with a menu of TRAIN_BATTALION and
DECLARE_CONQUEST. The territory lock strips every BUILD_* action while a camp is inside the clearing radius, CLEAR_TERRITORY is the way past it, and the final-era
lock then stripped CLEAR_TERRITORY as well."""
from backend.simulation import Simulation


def _final_era_pair():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}, {"name": "B", "model": "gemma2:2b", "x": 70, "y": 70}])
    a, b = sim.tribes["tribe_0"], sim.tribes["tribe_1"]
    a.has_ever_settled = True
    sim._found_territory(a)
    a.era = b.era = "departure_era"
    a.wood = a.stone = 3000
    a.vessel_wood_paid = a.vessel_stone_paid = 2000
    a.discovered_rivals.add("tribe_1")
    return sim, a


def test_a_raider_camp_at_the_gates_in_the_final_era_leaves_clear_territory_on_the_menu():
    sim, tribe = _final_era_pair()
    tx, ty = tribe.territory_center
    tribe.raider_sightings = [(tx + 1, ty)]
    _request, ctx = sim._prepare_turn(tribe)
    assert ctx["available_actions"] == ["CLEAR_TERRITORY"]


def test_with_no_camp_the_final_era_still_offers_the_unfinished_vessel():
    sim, tribe = _final_era_pair()
    _request, ctx = sim._prepare_turn(tribe)
    assert "BUILD_VESSEL" in ctx["available_actions"]
