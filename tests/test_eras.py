from backend.eras import ERAS, era_index, next_era, unlocked_actions_through

_EXPECTED_KEYS = [
    "primitive_dawn", "cognitive_horizon", "tribal_synapse", "monolithic_era",
    "mechanization_era", "silicon_era", "cosmic_post_human",
]


def test_eras_are_defined_in_ascending_order():
    assert [e.key for e in ERAS] == _EXPECTED_KEYS


def test_era_index_finds_known_and_falls_back_on_unknown():
    assert era_index("primitive_dawn") == 0
    assert era_index("cognitive_horizon") == 1
    assert era_index("tribal_synapse") == 2
    assert era_index("cosmic_post_human") == 6
    assert era_index("not_a_real_era") == 0


def test_next_era_progresses_through_every_stage_and_ends_at_the_top():
    for current_key, expected_next_key in zip(_EXPECTED_KEYS, _EXPECTED_KEYS[1:]):
        assert next_era(current_key).key == expected_next_key
    assert next_era("cosmic_post_human") is None


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


def test_top_two_eras_cap_population_at_the_hard_growth_ceiling():
    from backend import config

    assert ERAS[-1].requires_population == config.POPULATION_GROWTH_CAP
    assert ERAS[-2].requires_population == config.POPULATION_GROWTH_CAP
