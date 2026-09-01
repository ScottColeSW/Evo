from backend.instincts import effective_food_upkeep, survival_bias_string

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


def test_effective_food_upkeep_unaffected_without_cooking():
    assert effective_food_upkeep(5, cooking_learned=False) == 5


def test_effective_food_upkeep_divided_once_cooking_is_learned():
    """Explicit request: "cooked food is worth 3 raw food.\""""
    from backend import config

    assert effective_food_upkeep(6, cooking_learned=True) == max(1, round(6 / config.COOKING_UPKEEP_DIVISOR))


def test_effective_food_upkeep_never_drops_below_one():
    assert effective_food_upkeep(1, cooking_learned=True) == 1


def test_cooking_learned_raises_the_real_hunger_threshold():
    """A tribe that has learned to cook genuinely isn't as close to starving at the
    same raw food number -- the threshold itself should reflect the real, reduced
    drain, not just describe the old one more gently. LARGE_TRIBE's upkeep (5) is
    high enough that the /3 cooking divisor actually changes the effective number
    (unlike SMALL_TRIBE's upkeep=1, where max(1, round(1/3)) floors back to 1)."""
    same_food_with_cooking, _ = survival_bias_string(
        food=10, water=50, population=LARGE_TRIBE, cooking_learned=True
    )
    same_food_without_cooking, _ = survival_bias_string(
        food=10, water=50, population=LARGE_TRIBE, cooking_learned=False
    )

    assert same_food_with_cooking == ""  # 10 food comfortably clears cooking's lower effective upkeep
    assert "running low" in same_food_without_cooking  # but not the raw, uncooked rate


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
