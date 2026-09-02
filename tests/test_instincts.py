from backend.instincts import survival_bias_string

# population=8 -> upkeep = max(1, 8//10) = 1/cycle -> critical <= 1, warning <= 4
SMALL_TRIBE = 8
# population=50 -> upkeep = max(1, 50//10) = 5/cycle -> critical <= 5, warning <= 20
LARGE_TRIBE = 50


def test_healthy_food_and_water_produce_no_bias():
    text, critical = survival_bias_string(food=50, water=50, population=SMALL_TRIBE)
    assert text == ""
    assert critical is False


def test_low_food_produces_warning_not_critical():
    text, critical = survival_bias_string(food=3, water=50, population=SMALL_TRIBE)
    assert "running low" in text
    assert critical is False


def test_survival_bias_now_names_a_concrete_response():
    """Regression/reversal: an earlier version deliberately described the crisis
    without naming an action, on the theory that a model connecting "starving" to "go
    hunt" on its own was more honest than scripting the outcome. Explicit request:
    nudge harder. This still isn't a forced action -- available_actions and the
    model's own choice are untouched -- but the fact block no longer pretends not to
    know what the tribe actually needs."""
    food_text, _ = survival_bias_string(food=1, water=50, population=SMALL_TRIBE)
    assert "hunting party" in food_text or "gather food" in food_text

    water_text, _ = survival_bias_string(food=50, water=1, population=SMALL_TRIBE)
    assert "gather water" in water_text or "scouts" in water_text


def test_critical_food_produces_urgent_text_and_critical_flag():
    text, critical = survival_bias_string(food=1, water=50, population=SMALL_TRIBE)
    assert "starving" in text
    assert critical is True


def test_critical_water_produces_urgent_text_and_critical_flag():
    text, critical = survival_bias_string(food=50, water=1, population=SMALL_TRIBE)
    assert "thirst" in text
    assert critical is True


def test_both_critical_combines_into_one_message():
    text, critical = survival_bias_string(food=1, water=1, population=SMALL_TRIBE)
    assert "starving" in text
    assert "thirst" in text
    assert critical is True


def test_critical_food_suggests_fishing_and_cooking_when_neither_is_learned():
    """Explicit request: "revise the 'your people are starving' messaging to be more
    inclusive of options that would help them fix it -- add Fishing or Cook in a
    Kitchen as additional ideas." Fishing and cooking are always real options, not
    just gather/hunt."""
    text, _ = survival_bias_string(food=1, water=50, population=SMALL_TRIBE)
    assert "fishing" in text
    assert "cook" in text.lower()


def test_warning_food_also_suggests_fishing_and_cooking_when_neither_is_learned():
    text, _ = survival_bias_string(food=3, water=50, population=SMALL_TRIBE)
    assert "fishing" in text
    assert "cook" in text.lower()


def test_critical_food_omits_fishing_suggestion_once_already_learned():
    text, _ = survival_bias_string(food=1, water=50, population=SMALL_TRIBE, fishing_learned=True)
    assert "fishing" not in text


def test_critical_food_omits_cooking_suggestion_once_already_learned():
    text, _ = survival_bias_string(food=1, water=50, population=SMALL_TRIBE, cooking_learned=True)
    assert "cook" not in text.lower()


def test_thresholds_scale_with_population_not_a_flat_stockpile_number():
    """The whole point of this design: the same absolute stockpile means something
    different depending on how many people it has to feed. food=5 is comfortably above
    a small tribe's warning threshold (upkeep 1/cycle -> warning at <=4) but already
    critical for a much larger tribe paying more per cycle (upkeep 5/cycle -> critical
    at <=5)."""
    small_text, small_critical = survival_bias_string(food=5, water=50, population=SMALL_TRIBE)
    large_text, large_critical = survival_bias_string(food=5, water=50, population=LARGE_TRIBE)

    assert small_text == ""
    assert small_critical is False
    assert "starving" in large_text
    assert large_critical is True
