"""2026-10-04: the project-wide nudge switch (config.NUDGES / the NUDGES environment variable, plus config.DISABLED_NUDGE_TAGS per tag).
A nudge is prompt text whose job is to steer the tribe toward an outcome. Off silences every site that goes through config.nudge_active,
and touches nothing else: not the menu, not a gate, not a world fact."""
from backend import config
from backend.instincts import survival_bias_string
from backend.prompts import _growth_pressure_text
from backend.simulation import Simulation


def _settled_sim():
    sim = Simulation([{"name": "Plains Tribe", "model": "gemma2:2b", "x": 65, "y": 65}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    sim._found_territory(tribe)
    tribe.era = "cognitive_horizon"
    tribe.food = tribe.water = 500
    tribe.wood = tribe.stone = 200
    return sim, tribe


def test_the_switch_reads_the_environment_each_call_and_the_tag_set(monkeypatch):
    monkeypatch.delenv("NUDGES", raising=False)
    assert config.nudge_active("build_hint") is True
    monkeypatch.setenv("NUDGES", "off")
    assert config.nudge_active("build_hint") is False
    assert config.nudge_active("anything_else") is False
    monkeypatch.setenv("NUDGES", "on")
    assert config.nudge_active("build_hint") is True
    monkeypatch.setattr(config, "DISABLED_NUDGE_TAGS", {"build_hint"})
    assert config.nudge_active("build_hint") is False and config.nudge_active("farm_hint") is True


def test_a_build_hint_appears_with_nudges_on_and_is_gone_with_them_off(monkeypatch):
    sim, tribe = _settled_sim()
    monkeypatch.setenv("NUDGES", "on")
    on_prompt = sim._prepare_turn(tribe)[0]["prompt"]
    assert "No long house stands yet" in on_prompt

    monkeypatch.setenv("NUDGES", "off")
    off_prompt = sim._prepare_turn(tribe)[0]["prompt"]
    assert "No long house stands yet" not in off_prompt


def test_switching_nudges_off_never_changes_the_menu(monkeypatch):
    sim, tribe = _settled_sim()
    monkeypatch.setenv("NUDGES", "on")
    on_actions = sim._prepare_turn(tribe)[1]["available_actions"]
    monkeypatch.setenv("NUDGES", "off")
    off_actions = sim._prepare_turn(tribe)[1]["available_actions"]
    assert on_actions == off_actions


def test_the_survival_warning_text_goes_but_the_crisis_flag_stays(monkeypatch):
    monkeypatch.setenv("NUDGES", "on")
    text, critical = survival_bias_string(food=0, water=500, population=100)
    assert "starving" in text and critical is True
    monkeypatch.setenv("NUDGES", "off")
    text, critical = survival_bias_string(food=0, water=500, population=100)
    assert text == "" and critical is True


def test_growth_pressure_keeps_the_fact_and_drops_the_framing(monkeypatch):
    note = "To reach Tribal Synapse, still short on: population 15/600."
    monkeypatch.setenv("NUDGES", "on")
    assert "Growing and advancing is how that fragility actually ends" in _growth_pressure_text(note, False)
    monkeypatch.setenv("NUDGES", "off")
    assert _growth_pressure_text(note, False) == note
    assert _growth_pressure_text(note, True) == note
    assert "NO ADVANCEMENT PENDING" in _growth_pressure_text("", False)  # no note: the neutral line is unchanged


def test_the_repetition_notice_is_a_nudge_but_the_menu_removal_it_comes_with_is_not(monkeypatch):
    """The Historian line asks for a different choice; removing the repeated action from the menu is a mechanic and stays."""
    sim, tribe = _settled_sim()
    tribe.era = "cognitive_horizon"
    tribe.throttled_actions = {"GATHER_STONE": 5}
    monkeypatch.setenv("NUDGES", "on")
    request, ctx = sim._prepare_turn(tribe)
    assert "The Historian has counseled against repeating" in request["prompt"]
    on_actions = ctx["available_actions"]
    monkeypatch.setenv("NUDGES", "off")
    request, ctx = sim._prepare_turn(tribe)
    assert "The Historian has counseled against repeating" not in request["prompt"]
    assert ctx["available_actions"] == on_actions and "GATHER_STONE" not in on_actions
