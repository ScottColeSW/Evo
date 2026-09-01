from backend.simulation import Tribe
from backend.wellbeing import TIER_SATISFIED_THRESHOLD, compute_wellbeing


def _tribe(**overrides) -> Tribe:
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for key, value in overrides.items():
        setattr(tribe, key, value)
    return tribe


def test_fresh_tribe_is_fed_but_unsettled_unconnected_and_unrecognized():
    """A brand-new tribe starts with a healthy stockpile but nothing else earned yet
    -- physiological should read satisfied while every other tier reads near zero."""
    tribe = _tribe()

    result = compute_wellbeing(tribe, wall_fraction=0.0)

    assert result["tiers"]["physiological"] == 1.0
    assert result["tiers"]["safety"] < TIER_SATISFIED_THRESHOLD
    assert result["tiers"]["belonging"] == 0.0
    assert result["tiers"]["esteem"] == 0.0
    assert result["tiers"]["self_actualization"] == 0.0


def test_physiological_tier_drops_as_the_food_water_buffer_shrinks():
    healthy = compute_wellbeing(_tribe(food=40, water=30, population=8), wall_fraction=0.0)
    starving = compute_wellbeing(_tribe(food=1, water=1, population=8), wall_fraction=0.0)

    assert healthy["tiers"]["physiological"] > starving["tiers"]["physiological"]
    assert starving["tiers"]["physiological"] < TIER_SATISFIED_THRESHOLD


def test_safety_tier_is_lower_for_a_nomadic_tribe_than_a_settled_undefended_one():
    nomadic = compute_wellbeing(_tribe(has_ever_settled=False), wall_fraction=0.0)
    settled_bare = compute_wellbeing(_tribe(has_ever_settled=True), wall_fraction=0.0)

    assert nomadic["tiers"]["safety"] < settled_bare["tiers"]["safety"]


def test_safety_tier_rises_with_wall_progress_and_repelled_raids():
    bare = compute_wellbeing(_tribe(has_ever_settled=True, raids_defended=0), wall_fraction=0.0)
    walled = compute_wellbeing(_tribe(has_ever_settled=True, raids_defended=0), wall_fraction=1.0)
    battle_tested = compute_wellbeing(_tribe(has_ever_settled=True, raids_defended=5), wall_fraction=1.0)

    assert walled["tiers"]["safety"] > bare["tiers"]["safety"]
    assert battle_tested["tiers"]["safety"] > walled["tiers"]["safety"]
    assert battle_tested["tiers"]["safety"] == 1.0


def test_belonging_tier_reflects_trade_contact_and_ever_having_broadcast():
    isolated = compute_wellbeing(_tribe(trades_completed=0, last_broadcast=""), wall_fraction=0.0)
    spoke_once = compute_wellbeing(_tribe(trades_completed=0, last_broadcast="KRA-ZUL"), wall_fraction=0.0)
    traded = compute_wellbeing(_tribe(trades_completed=3, last_broadcast="KRA-ZUL"), wall_fraction=0.0)

    assert isolated["tiers"]["belonging"] == 0.0
    assert spoke_once["tiers"]["belonging"] < traded["tiers"]["belonging"]
    assert traded["tiers"]["belonging"] == 1.0


def test_esteem_tier_scales_with_trophy_count_and_caps_at_one():
    few = compute_wellbeing(_tribe(trophies=[{"name": "First Fire", "chief": "x", "cycle": 1}]), wall_fraction=0.0)
    many = compute_wellbeing(
        _tribe(trophies=[{"name": str(i), "chief": "x", "cycle": 1} for i in range(10)]), wall_fraction=0.0
    )

    assert 0.0 < few["tiers"]["esteem"] < many["tiers"]["esteem"]
    assert many["tiers"]["esteem"] == 1.0


def test_self_actualization_tier_rises_with_era_and_city_growth():
    stone_age = compute_wellbeing(_tribe(era="stone_age", founded_city=False), wall_fraction=0.0)
    classical = compute_wellbeing(_tribe(era="classical_age", founded_city=False), wall_fraction=0.0)
    built_up_city = compute_wellbeing(_tribe(era="classical_age", founded_city=True, city_buildings=6), wall_fraction=0.0)

    assert stone_age["tiers"]["self_actualization"] < classical["tiers"]["self_actualization"]
    assert classical["tiers"]["self_actualization"] < built_up_city["tiers"]["self_actualization"]
    assert built_up_city["tiers"]["self_actualization"] == 1.0


def test_focus_is_the_lowest_unmet_tier_bottom_up_not_just_the_lowest_score():
    """Real Maslow logic: a lower unmet need is the pressing one even if a higher
    tier happens to score numerically lower still -- focus must walk the ladder
    bottom-up, not just argmin the tier scores."""
    tribe = _tribe(
        food=1, water=1,  # physiological unmet
        has_ever_settled=True, raids_defended=5,  # safety maxed
        trades_completed=3, last_broadcast="KRA-ZUL",  # belonging maxed
        trophies=[],  # esteem unmet too, but physiological comes first
    )

    result = compute_wellbeing(tribe, wall_fraction=1.0)

    assert result["focus"] == "physiological"


def test_focus_is_self_actualization_once_every_lower_tier_is_satisfied():
    tribe = _tribe(
        food=40, water=30,
        has_ever_settled=True, raids_defended=5,
        trades_completed=3, last_broadcast="KRA-ZUL",
        trophies=[{"name": str(i), "chief": "x", "cycle": 1} for i in range(5)],
        era="stone_age", founded_city=False,
    )

    result = compute_wellbeing(tribe, wall_fraction=1.0)

    assert result["focus"] == "self_actualization"


def test_summary_names_the_focus_tier_and_states_the_underlying_facts_not_a_directive():
    tribe = _tribe(has_ever_settled=True, raids_defended=0, trophies=[])

    result = compute_wellbeing(tribe, wall_fraction=0.0)

    assert "Safety" in result["summary"]
    assert "0 raid(s) repelled" in result["summary"]
    # Facts, not scripted directives: no imperative telling the tribe what to do.
    assert "should" not in result["summary"].lower()
    assert "must" not in result["summary"].lower()


def test_summary_names_still_nomadic_instead_of_a_wall_percentage_when_never_settled():
    tribe = _tribe(has_ever_settled=False)

    result = compute_wellbeing(tribe, wall_fraction=0.0)

    assert "still nomadic" in result["summary"]
    assert "% complete" not in result["summary"]
