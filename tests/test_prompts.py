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


def test_prompt_clarifies_only_relocate_moves_the_tribe():
    """Factual clarification, not a behavioral nudge: only RELOCATE moves the tribe, and
    SCOUT looks without moving -- earlier models assumed any action required staying put,
    then later a per-turn target_vector caused the tribe to drift on every single turn
    regardless of the action chosen. Both were architecture bugs, not model failures."""
    prompt = compile_live_state_prompt("base", _world_state(), "", "")
    assert "Only RELOCATE moves your tribe" in prompt
    assert "SCOUT looks at target_vector without moving" in prompt


def test_system_prompt_includes_tribe_name_and_model():
    prompt = get_prime_consciousness_prompt("Forest Tribe", "gemma2:2b")
    assert "FOREST TRIBE" in prompt
    assert "gemma2:2b" in prompt


def test_system_prompt_states_food_and_water_are_lethal_but_wood_and_stone_are_not():
    prompt = get_prime_consciousness_prompt("Forest Tribe", "gemma2:2b")
    assert "SURVIVAL PHYSIOLOGY" in prompt
    assert "prolonged shortage of either is lethal" in prompt
    assert "does not kill anyone" in prompt


def test_system_prompt_omits_leadership_block_without_a_chief():
    prompt = get_prime_consciousness_prompt("Forest Tribe", "gemma2:2b")
    assert "LEADERSHIP" not in prompt


def test_system_prompt_includes_chief_as_context_not_command():
    prompt = get_prime_consciousness_prompt("Forest Tribe", "gemma2:2b", "Ashgar", "expand aggressively")
    assert "Ashgar" in prompt
    assert "expand aggressively" in prompt
    assert "not a command" in prompt


def test_system_prompt_explains_what_each_available_action_does():
    """Regression test: live runs showed tribes repeatedly deciding they "must relocate
    to find water" while starving, despite GATHER_WATER already working wherever they
    stood -- because the prompt only ever listed bare action names, never what any of
    them actually do. A model has no way to infer GATHER_WATER works off-river from the
    name alone.

    This glossary lives in the system prompt, not compile_live_state_prompt, so it
    isn't sitting between the per-turn survival-crisis text and the JSON decision slot
    -- live runs also showed a tribe's own rationale correctly diagnosing a crisis and
    then picking an unrelated action, consistent with that crisis losing salience
    across several lines of generic reference text before the decision field."""
    prompt = get_prime_consciousness_prompt(
        "Forest Tribe", "gemma2:2b", available_actions=("GATHER_WATER", "RELOCATE")
    )
    assert "GATHER_WATER: Harvest water at your current tile -- works in any biome" in prompt
    assert "RELOCATE: Move your whole tribe" in prompt


def test_system_prompt_action_glossary_only_lists_currently_available_actions():
    """Era-gated actions not yet unlocked shouldn't get an explanation either -- the
    glossary should track available_actions exactly, not the full registry."""
    prompt = get_prime_consciousness_prompt("Forest Tribe", "gemma2:2b", available_actions=("IDLE",))
    assert "IDLE: Do nothing" in prompt
    assert "GATHER_WOOD:" not in prompt


def test_system_prompt_omits_glossary_when_no_actions_given():
    prompt = get_prime_consciousness_prompt("Forest Tribe", "gemma2:2b")
    assert "WHAT EACH OF YOUR CURRENT ACTIONS DOES" not in prompt


def test_live_state_prompt_no_longer_carries_the_action_glossary():
    """The glossary moved to the system prompt (see the tests above) specifically so it
    doesn't sit between the per-turn survival-crisis text and the JSON decision slot --
    compile_live_state_prompt should still list bare action names, just not explain them."""
    prompt = compile_live_state_prompt(
        "base", _world_state(available_actions=["GATHER_WATER", "RELOCATE"]), "", ""
    )
    assert "GATHER_WATER" in prompt  # the bare name list is still there
    assert "Harvest water at your current tile" not in prompt  # the explanation is not
