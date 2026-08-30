from backend.physics import calculate_next_step, extend_ray_to_grid_edge


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


def test_higher_speed_covers_more_ground_per_cycle():
    assert calculate_next_step(50, 50, 80, 50, speed=4) == (54, 50)


def test_higher_speed_never_overshoots_a_near_target():
    assert calculate_next_step(50, 50, 52, 50, speed=4) == (52, 50)


def test_higher_speed_respects_grid_bound():
    assert calculate_next_step(97, 97, 200, 200, bound=99, speed=10) == (99, 99)


def test_extend_ray_preserves_direction_to_the_grid_edge():
    # Due east from (50,50) through (56,50) should hit the eastern edge at (99,50).
    assert extend_ray_to_grid_edge(50, 50, 56, 50, 100) == (99, 50)


def test_extend_ray_handles_diagonal_direction():
    # 45-degree southeast ray from (0,0) through (10,10) hits the corner (99,99).
    assert extend_ray_to_grid_edge(0, 0, 10, 10, 100) == (99, 99)


def test_extend_ray_with_no_direction_returns_the_same_point():
    assert extend_ray_to_grid_edge(50, 50, 50, 50, 100) == (50, 50)


def test_extend_ray_toward_the_near_edge_shortens_correctly():
    # Heading west from (50,50) through (48,50) hits the western edge at (0,50).
    assert extend_ray_to_grid_edge(50, 50, 48, 50, 100) == (0, 50)
