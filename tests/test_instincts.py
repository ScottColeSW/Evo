from backend.instincts import survival_bias_string


def test_healthy_food_and_water_produce_no_bias():
    text, critical = survival_bias_string(food=50, water=50)
    assert text == ""
    assert critical is False


def test_low_food_produces_warning_not_critical():
    text, critical = survival_bias_string(food=15, water=50)
    assert "running low" in text
    assert critical is False


def test_survival_bias_never_prescribes_a_specific_action():
    """A prior version named HUNT_DEER/GATHER_WATER directly as a near-command, which
    measurably changed behavior but amounts to scripting the outcome rather than letting
    the model reason its way there. This describes the crisis honestly and stops -- if a
    model can't connect "starving" to "go hunt" on its own, that's real information about
    that model, not something to paper over with a forced instruction."""
    for food, water in [(3, 50), (15, 50), (50, 2), (50, 10), (1, 1)]:
        text, _ = survival_bias_string(food=food, water=water)
        assert "HUNT_DEER" not in text
        assert "GATHER_WATER" not in text
        assert "Choose visual_action" not in text


def test_critical_food_produces_urgent_text_and_critical_flag():
    text, critical = survival_bias_string(food=3, water=50)
    assert "starving" in text
    assert critical is True


def test_critical_water_produces_urgent_text_and_critical_flag():
    text, critical = survival_bias_string(food=50, water=2)
    assert "thirst" in text
    assert critical is True


def test_both_critical_combines_into_one_message():
    text, critical = survival_bias_string(food=1, water=1)
    assert "starving" in text
    assert "thirst" in text
    assert critical is True
