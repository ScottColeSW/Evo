from backend.prompts import _growth_pressure_text, compile_live_state_prompt, get_prime_consciousness_prompt


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


def test_leadership_block_orders_lineage_victory_responsibility_duty_philosophy():
    """Explicit request: the tribe should know its own leader's origin story and its
    own children, not just an abstract philosophy -- ordered lineage/ancestry first,
    then how this chief specifically rose, then the weight of the role itself
    (true regardless of who holds it), then today's standing order, then this
    chief's own personal belief last."""
    prompt = get_prime_consciousness_prompt(
        "Forest Tribe", "gemma2:2b",
        chief_name="Ashgar", chief_philosophy="expand aggressively",
        chief_decree="find water", chief_victory="won a wrestling match",
        lineage_note="Juni, child of Aila and RenKa, born cycle 12",
    )
    assert "LEADERSHIP - ACTIVE CHIEF" in prompt
    lineage_i = prompt.index("LINEAGE:")
    victory_i = prompt.index("VICTORY:")
    responsibility_i = prompt.index("RESPONSIBILITY:")
    duty_i = prompt.index("DUTY:")
    philosophy_i = prompt.index("PHILOSOPHY:")
    assert lineage_i < victory_i < responsibility_i < duty_i < philosophy_i
    assert "Juni, child of Aila and RenKa, born cycle 12" in prompt
    assert "Ashgar became chief. won a wrestling match" in prompt
    assert "find water" in prompt
    assert "expand aggressively" in prompt


def test_leadership_block_states_no_duty_or_lineage_when_absent():
    prompt = get_prime_consciousness_prompt(
        "Forest Tribe", "gemma2:2b", chief_name="Ashgar", chief_philosophy="expand aggressively",
    )
    assert "LINEAGE:" not in prompt  # omitted entirely rather than a blank line
    assert "no standing duty has been decreed" in prompt
    assert "Ashgar became chief." in prompt  # no victory story available, states the fact plainly


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
    prompt = get_prime_consciousness_prompt("Forest Tribe", "gemma2:2b", available_actions=("RELOCATE",))
    assert "RELOCATE: Move your whole tribe" in prompt
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


def test_growth_imperative_layer_is_the_last_thing_before_the_json_slot():
    """Explicit hypothesis: real data showed tribes plateau at low population and
    near-zero wood/stone for 70+ cycles despite era_gap_note stating exactly what
    was missing the whole time -- the same salience problem the survival-crisis fact
    had before it got moved to be the prompt's own dedicated last-thing-before-the-
    JSON-slot section. This applies that same fix one tier up."""
    prompt = compile_live_state_prompt(
        "base", _world_state(growth_note="To reach Bronze Age, still short on: wood 5/40."), "", "",
    )
    growth_i = prompt.index("GROWTH IMPERATIVE LAYER")
    survival_i = prompt.index("SURVIVAL INSTINCT LAYER")
    schema_i = prompt.index("MANDATORY REACTION SCHEMA")
    assert survival_i < growth_i < schema_i
    assert "still short on: wood 5/40" in prompt


def test_growth_pressure_text_defers_to_survival_when_critical():
    text = _growth_pressure_text("To reach Bronze Age, still short on: wood 5/40.", survival_critical=True)
    assert "Survival still comes first" in text
    assert "wood 5/40" in text


def test_growth_pressure_text_is_encouraging_and_honest_once_stable():
    """Explicit request: gratitude and real urgency, not a fabricated threat -- there
    is no actual danger once a tribe is settled and fed, so the stakes named here
    must be the real one (a small population's fragility to any ordinary setback),
    not an invented monster."""
    text = _growth_pressure_text("To reach Bronze Age, still short on: wood 5/40.", survival_critical=False)
    assert "grateful" in text.lower()
    assert "wood 5/40" in text
    assert "fragile" in text.lower()
    assert "monster" not in text.lower()
    assert "not safe" not in text.lower()


def test_growth_pressure_text_is_neutral_when_nothing_is_missing():
    text = _growth_pressure_text("", survival_critical=False)
    assert "GROWTH STATE" in text
    assert "grateful" not in text.lower()
