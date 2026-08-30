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


def test_survival_bias_never_prescribes_a_specific_action():
    """A prior version named HUNT_DEER/GATHER_WATER directly as a near-command, which
    measurably changed behavior but amounts to scripting the outcome rather than letting
    the model reason its way there. This describes the crisis honestly and stops -- if a
    model can't connect "starving" to "go hunt" on its own, that's real information about
    that model, not something to paper over with a forced instruction."""
    for food, water in [(1, 50), (3, 50), (50, 1), (50, 3), (1, 1)]:
        text, _ = survival_bias_string(food=food, water=water, population=SMALL_TRIBE)
        assert "HUNT_DEER" not in text
        assert "GATHER_WATER" not in text
        assert "Choose visual_action" not in text


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
