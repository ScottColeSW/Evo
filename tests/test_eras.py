from backend.eras import ERAS, era_index, next_era, unlocked_actions_through

_EXPECTED_KEYS = [
    "primitive_dawn", "cognitive_horizon", "tribal_synapse", "monolithic_era",
    "object_creator_era", "war_and_world_domination_era",
]


def test_eras_are_defined_in_ascending_order():
    assert [e.key for e in ERAS] == _EXPECTED_KEYS


def test_era_index_finds_known_and_falls_back_on_unknown():
    assert era_index("primitive_dawn") == 0
    assert era_index("cognitive_horizon") == 1
    assert era_index("tribal_synapse") == 2
    assert era_index("war_and_world_domination_era") == 5
    assert era_index("not_a_real_era") == 0


def test_next_era_progresses_through_every_stage_and_ends_at_the_top():
    for current_key, expected_next_key in zip(_EXPECTED_KEYS, _EXPECTED_KEYS[1:]):
        assert next_era(current_key).key == expected_next_key
    assert next_era("war_and_world_domination_era") is None


def test_unlocked_actions_accumulate_across_eras():
    primitive = unlocked_actions_through("primitive_dawn")
    tribal = unlocked_actions_through("tribal_synapse")
    assert "CONSTRUCT_WALL" not in primitive
    assert "CONSTRUCT_WALL" in tribal
    assert primitive.issubset(tribal)  # nothing is ever un-learned by advancing


def test_only_monolithic_era_founds_a_city():
    for era in ERAS:
        assert era.founds_city == (era.key == "monolithic_era")


def test_population_and_resource_requirements_never_decrease_up_the_ladder():
    """Each era should be at least as demanding as the one before it -- a real
    ladder, not thresholds that wobble up and down."""
    for previous, current in zip(ERAS, ERAS[1:]):
        assert current.requires_population >= previous.requires_population


def test_final_era_sits_at_a_real_concrete_population_threshold():
    # config.POPULATION_GROWTH_CAP is infinite (explicit request: "we should not
    # put a cap on population"), so this is a real, concrete threshold, not an
    # enforced hard cap. Raised 80 -> 1500 (2026-09-07, "much bigger populations
    # going to war"), reachable in practice now that population growth itself
    # scales with tribe size and Well-Being instead of a flat +1/cycle -- see
    # config.POPULATION_GROWTH_SCALE_DIVISOR's own comment.
    assert ERAS[-1].requires_population == 1500


def test_cognitive_horizon_is_the_agrarian_infrastructure_tier():
    """Explicit request, 2026-09-08: "Cognitive Horizon needs to be an agrarian
    society kind of evolution... take about half of the action items from
    Tribal Synapse." Every settlement/subsistence-infrastructure action moved
    here -- none of them tied to war, diplomacy, or formal knowledge."""
    cognitive_horizon = next(e for e in ERAS if e.key == "cognitive_horizon")
    assert set(cognitive_horizon.unlocks_actions) == {
        "CONSTRUCT_WALL", "BUILD_LONG_HOUSE", "UPGRADE_LONG_HOUSE", "BUILD_DOCK", "BUILD_FISHERY",
        "BUILD_SAWMILL", "BUILD_QUARRY", "BUILD_KITCHEN", "BUILD_TANNERY", "BUILD_DEER_PEN", "BUILD_WAREHOUSE",
        "UPGRADE_WAREHOUSE", "BUILD_HATCHERY", "BUILD_COOP", "BUILD_BATH_HOUSE", "BUILD_WELL",
    }


def test_tribal_synapse_now_holds_only_military_diplomacy_and_knowledge():
    """The other half of the 2026-09-08 split -- what's left once the agrarian
    tier moved down to Cognitive Horizon is genuinely "true society" content:
    a standing military, foreign relations, escalated fortification, and
    formal knowledge institutions."""
    tribal_synapse = next(e for e in ERAS if e.key == "tribal_synapse")
    assert set(tribal_synapse.unlocks_actions) == {
        "STRIKE_RAIDER_CAMP", "EXPEL_RAIDERS_FROM_TERRITORY",
        "BUILD_BARRACKS", "UPGRADE_BARRACKS", "TRAIN_BATTALION",
        "DECLARE_ALLIANCE", "DECLARE_WAR", "SPY", "BUILD_MOAT", "BUILD_KEEP",
        "BUILD_LIBRARY", "RESEARCH",
    }


def test_wall_dependent_buildings_moved_down_together_with_the_wall():
    """BUILD_LONG_HOUSE needs a fully-built wall ring 0, and BUILD_KITCHEN needs
    a standing Long House -- moving CONSTRUCT_WALL down without its dependents
    would have made moving them a no-op, still transitively blocked on
    population 50 regardless of era."""
    cognitive_horizon_actions = unlocked_actions_through("cognitive_horizon")
    assert "CONSTRUCT_WALL" in cognitive_horizon_actions
    assert "BUILD_LONG_HOUSE" in cognitive_horizon_actions
    assert "BUILD_KITCHEN" in cognitive_horizon_actions
    # But the next tier of fortification (on top of an already-standing wall)
    # still reads as "true society" content, not founding-era survival.
    assert "BUILD_MOAT" not in cognitive_horizon_actions
    assert "BUILD_KEEP" not in cognitive_horizon_actions


def test_object_creator_and_war_domination_eras_replaced_the_old_empty_slots():
    """Explicit request, after a run reached the old ceiling with nothing left
    to do there ("we have to extend it now"): the three old empty reserved
    slots (mechanization_era/silicon_era/cosmic_post_human) are replaced
    outright by two real eras, not appended after them -- the ladder is 6
    stages now, not 7, and both new top eras actually unlock something."""
    assert [e.key for e in ERAS[-2:]] == ["object_creator_era", "war_and_world_domination_era"]
    object_creator, war_domination = ERAS[-2], ERAS[-1]
    assert set(object_creator.unlocks_actions) == {"BUILD_OBJECT_CREATOR", "CREATE_ITEM", "CREATE_USEFUL_STRUCTURE"}
    assert war_domination.unlocks_actions == ("DECLARE_CONQUEST",)
