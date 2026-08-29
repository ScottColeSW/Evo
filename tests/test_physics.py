from backend.physics import calculate_next_step


def test_moves_one_tile_toward_target():
    assert calculate_next_step(50, 50, 60, 40) == (51, 49)


def test_no_movement_when_already_at_target():
    assert calculate_next_step(10, 10, 10, 10) == (10, 10)


def test_clamps_to_lower_bound():
    assert calculate_next_step(0, 0, -5, -5) == (0, 0)


def test_clamps_to_upper_bound():
    assert calculate_next_step(99, 99, 500, 500, bound=99) == (99, 99)


def test_moves_along_single_axis():
    assert calculate_next_step(20, 20, 20, 30) == (20, 21)
    assert calculate_next_step(20, 20, 10, 20) == (19, 20)
