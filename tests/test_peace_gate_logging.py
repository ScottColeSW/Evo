"""Phase 0 of docs/TRADE-GATE-DESIGN.md: the gate's state is tracked and logged, and nothing is locked."""
from unittest import mock

from backend import config, peace_gate
from backend.simulation import Simulation, _sustainable_population
from tests.conftest import run_async


def _sim():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    sim.cycle = 100
    return sim, sim.tribes["tribe_0"]


def test_a_cull_is_a_cost_and_resets_the_clean_count():
    sim, tribe = _sim()
    target = round(_sustainable_population(tribe) * config.POPULATION_CARRYING_CAPACITY_TARGET_FRACTION)
    tribe.population = target + 200
    sim._advance_population_pressure(tribe, night=True)
    sim._peace_gate_night(tribe)
    state = tribe.peace_gate
    assert state["felt"] and state["cost_events"][-1]["kind"] == "cull"
    assert state["clean_nights"] == 0 and peace_gate.scar(state) == 1


def test_hazard_losses_are_not_costs_and_small_war_losses_are_ignored():
    sim, tribe = _sim()
    tribe.population = 1000
    sim._lose_population(tribe, 300, cause="volcano")
    sim._lose_population(tribe, 5, cause="raid_losses")  # 0.5%, below the line
    assert not tribe.peace_gate["felt"]
    sim._lose_population(tribe, 100, cause="raid_losses")
    assert tribe.peace_gate["felt"] and tribe.peace_gate["cost_events"][-1]["kind"] == "war"


def test_lesson_path_needs_clean_nights_and_the_requirement_grows_with_culls():
    sim, tribe = _sim()
    state = tribe.peace_gate
    peace_gate.record_cost(state, 10, "starvation", 50, 500)
    for _ in range(config.PEACE_GATE_CLEAN_NIGHTS - 1):
        peace_gate.note_night(state, culled=False)
    assert peace_gate.evaluate(tribe, 40)["would_open"] is None
    peace_gate.note_night(state, culled=False)
    assert peace_gate.evaluate(tribe, 40)["would_open"] == "lesson"
    peace_gate.note_night(state, culled=True)
    peace_gate.note_night(state, culled=True)
    assert peace_gate.required_clean_nights(state) == 2 * config.PEACE_GATE_CLEAN_NIGHTS
    assert peace_gate.evaluate(tribe, 40)["would_open"] is None


def test_a_cull_habit_closes_the_stability_and_contact_paths():
    sim, tribe = _sim()
    assert peace_gate.evaluate(tribe, config.PEACE_GATE_STABILITY_CYCLES)["would_open"] == "stability"
    peace_gate.record_cost(tribe.peace_gate, 5, "overcrowding", 40, 500)
    peace_gate.note_night(tribe.peace_gate, culled=True)
    assert peace_gate.evaluate(tribe, config.PEACE_GATE_STABILITY_CYCLES)["would_open"] is None
    tribe.discovered_rivals.update({"x", "y"})
    for k in range(config.PEACE_GATE_CONTACT_NIGHTS):
        tribe.spy_missions_run += 1
        peace_gate.note_night(tribe.peace_gate, culled=True, raw_contacts=peace_gate.contacts(tribe))  # still culling
    assert peace_gate.evaluate(tribe, 20)["would_open"] is None


def test_outside_contacts_open_the_tier_for_a_clean_tribe():
    sim, tribe = _sim()
    assert peace_gate.evaluate(tribe, 20)["would_open"] is None
    for k in range(config.PEACE_GATE_CONTACT_NIGHTS):
        tribe.spy_missions_run += 1
        peace_gate.note_night(tribe.peace_gate, culled=False, raw_contacts=peace_gate.contacts(tribe))
    assert peace_gate.evaluate(tribe, 20)["would_open"] == "contact"


def test_a_burst_of_words_in_one_night_is_one_contact_night():
    sim, tribe = _sim()
    tribe.peace_gate["heard_reports"] += 40  # many reports before one night passes
    peace_gate.note_night(tribe.peace_gate, culled=False, raw_contacts=peace_gate.contacts(tribe))
    assert len(tribe.peace_gate["contact_nights"]) == 1
    peace_gate.note_night(tribe.peace_gate, culled=False, raw_contacts=peace_gate.contacts(tribe))  # nothing new
    assert len(tribe.peace_gate["contact_nights"]) == 1


def test_the_night_logs_the_state_and_changes_nothing_in_the_world():
    sim, tribe = _sim()
    calls = []
    sim.event_log.record_data = lambda *a, **k: calls.append(a[1])
    population, trades = tribe.population, tribe.trades_completed
    sim._peace_gate_night(tribe)
    assert "peace_gate" in calls
    assert (tribe.population, tribe.trades_completed) == (population, trades)


def test_a_trade_is_logged_with_whether_a_gate_would_have_blocked_it():
    sim, tribe = _sim()
    seen = []
    sim.event_log.record_data = lambda tribe_name, kind, data, **k: seen.append((kind, data))
    sim._peace_gate_note_peace(tribe, "trade")
    kind, data = seen[-1]
    assert kind == "peace_gate_peace_act" and data["act"] == "trade" and data["blocked"] is True
