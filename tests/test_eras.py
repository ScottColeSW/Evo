from backend.eras import ERAS, era_index, next_era, unlocked_actions_through


def test_eras_are_defined_in_ascending_order():
    assert [e.key for e in ERAS] == ["stone_age", "bronze_age", "classical_age"]


def test_era_index_finds_known_and_falls_back_on_unknown():
    assert era_index("stone_age") == 0
    assert era_index("bronze_age") == 1
    assert era_index("not_a_real_era") == 0


def test_next_era_progresses_and_ends_at_the_top():
    assert next_era("stone_age").key == "bronze_age"
    assert next_era("bronze_age").key == "classical_age"
    assert next_era("classical_age") is None


def test_unlocked_actions_accumulate_across_eras():
    stone = unlocked_actions_through("stone_age")
    bronze = unlocked_actions_through("bronze_age")
    assert "CONSTRUCT_WALL" not in stone
    assert "CONSTRUCT_WALL" in bronze
    assert stone.issubset(bronze)  # nothing is ever un-learned by advancing


def test_only_classical_age_founds_a_city():
    assert not ERAS[0].founds_city
    assert not ERAS[1].founds_city
    assert ERAS[2].founds_city
