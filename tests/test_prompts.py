from backend.prompts import compile_live_state_prompt, get_prime_consciousness_prompt


def _world_state(**overrides):
    base = {
        "cycle": 1, "x": 50, "y": 60, "biome": "plains", "population": 8,
        "wood": 50, "stone": 50, "food": 40, "water": 30, "era": "stone_age",
        "available_actions": ["GATHER_WOOD", "IDLE"],
        "visible_entities": ["none"],
    }
    base.update(overrides)
    return base


def test_target_vector_placeholder_is_not_the_tribes_current_position():
    """Regression test: the schema example used to literally interpolate the tribe's own
    x,y as the sample target_vector, which small models would pattern-match and echo back
    verbatim -- producing tribes that never actually moved. The placeholder must not
    contain the real coordinates at all."""
    prompt = compile_live_state_prompt("base", _world_state(x=50, y=60), "", "")
    assert '"target_vector": [50, 60]' not in prompt
    assert '"target_vector": [x, y]' in prompt


def test_action_placeholder_does_not_repeat_select_strictly_one_pattern():
    """Regression test: the old "SELECT STRICTLY ONE: [...]" phrasing sat directly in the
    JSON value position, and models would copy that instruction text verbatim as their
    'visual_action' instead of substituting a real action name."""
    prompt = compile_live_state_prompt("base", _world_state(), "", "")
    assert "SELECT STRICTLY ONE" not in prompt
    assert "GATHER_WOOD" in prompt  # the actual action list is still shown, just as prose


def test_prompt_explicitly_encourages_movement():
    prompt = compile_live_state_prompt("base", _world_state(), "", "")
    assert "Standing still forever finds nothing new" in prompt


def test_system_prompt_includes_tribe_name_and_model():
    prompt = get_prime_consciousness_prompt("Forest Tribe", "gemma2:2b")
    assert "FOREST TRIBE" in prompt
    assert "gemma2:2b" in prompt
