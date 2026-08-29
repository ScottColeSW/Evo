from backend.ancestral_matrix import AncestralTraumaMatrix


def test_neutral_tile_has_no_bias():
    matrix = AncestralTraumaMatrix(grid_size=20)
    assert matrix.bias_string(10, 10) == ""


def test_positive_event_produces_pride_above_threshold():
    matrix = AncestralTraumaMatrix(grid_size=20)
    matrix.radiate_event_wave(10, 10, magnitude=0.5, radius=5)
    bias = matrix.bias_string(10, 10)
    assert "PRIDE" in bias


def test_negative_event_produces_dread_below_threshold():
    matrix = AncestralTraumaMatrix(grid_size=20)
    matrix.radiate_event_wave(10, 10, magnitude=-0.5, radius=5)
    bias = matrix.bias_string(10, 10)
    assert "DREAD" in bias


def test_falloff_weakens_with_distance_from_epicenter():
    matrix = AncestralTraumaMatrix(grid_size=20)
    matrix.radiate_event_wave(10, 10, magnitude=1.0, radius=5)
    epicenter = float(matrix.ghost_tensor[10, 10])
    edge = float(matrix.ghost_tensor[14, 10])
    assert epicenter > edge > 0


def test_outside_radius_is_untouched():
    matrix = AncestralTraumaMatrix(grid_size=20)
    matrix.radiate_event_wave(10, 10, magnitude=1.0, radius=3)
    assert matrix.ghost_tensor[19, 19] == 0.0


def test_score_is_clamped_to_unit_range():
    matrix = AncestralTraumaMatrix(grid_size=20)
    for _ in range(20):
        matrix.radiate_event_wave(10, 10, magnitude=1.0, radius=5)
    assert -1.0 <= float(matrix.ghost_tensor[10, 10]) <= 1.0
