from backend.physics import calculate_next_step, extend_ray_to_grid_edge, terrain_aware_step


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


def test_terrain_aware_step_moves_at_full_speed_on_plains():
    # (50, 50) is plains -- multiplier 1.0, no slowdown from the old flat behavior.
    assert terrain_aware_step(50, 50, 80, 50, base_speed=4) == (54, 50)


def test_terrain_aware_step_slows_down_in_mountains():
    """Regression test: RELOCATE/expeditions used to be a pure straight-line vector
    with zero regard for terrain -- a mountain crossing cost exactly the same as open
    plains. (17, 40) is mountains (multiplier 0.4) -- moved here from the original
    (10, 10) once map dream phase 2's west/north ocean insets swallowed that point --
    so a base speed of 4 should only actually cover round(4 * 0.4) = 2 tiles."""
    assert terrain_aware_step(17, 40, 47, 40, base_speed=4) == (19, 40)


def test_terrain_aware_step_never_drops_below_one_tile_of_progress():
    # Even in the slowest walkable terrain, movement never fully stalls out.
    assert terrain_aware_step(40, 37, 60, 37, base_speed=1) != (40, 37)  # (40,37) is river


def test_terrain_aware_step_boat_speeds_up_river_crossing():
    """Explicit request: "give the boat mobility in the clean water, not the
    sea" -- river is normally the slowest passable terrain (0.3x); a boat
    turns it into a real speed advantage instead."""
    from backend import config

    # (40, 37) is river. Without a boat: round(10 * 0.3) = 3 tiles.
    assert terrain_aware_step(40, 37, 60, 37, base_speed=10) == (43, 37)
    # With a boat: round(10 * BOAT_WATER_MOVEMENT_MULTIPLIER) = 12 tiles.
    expected = 40 + round(10 * config.BOAT_WATER_MOVEMENT_MULTIPLIER)
    assert terrain_aware_step(40, 37, 60, 37, base_speed=10, has_boat=True) == (expected, 37)


def test_terrain_aware_step_boat_does_not_help_on_dry_land():
    # (50, 50) is plains -- a boat shouldn't change anything off the water.
    assert terrain_aware_step(50, 50, 80, 50, base_speed=4, has_boat=True) == (54, 50)


def test_terrain_aware_step_boat_does_not_unlock_the_ocean():
    """Explicit request: this is deliberately NOT an ocean-crossing mechanic --
    only real, fresh water (river/lake) gets the boat's speed bonus."""
    from backend.world import biome_at

    # Find a real ocean-adjacent plains tile the same way the existing ocean
    # deflection test below does, then confirm a boat still can't cross it.
    ox, oy = None, None
    for x in range(100):
        if biome_at(x, 50) == "ocean" and biome_at(x - 1, 50) != "ocean":
            ox, oy = x, 50
            break
    assert ox is not None, "expected to find an ocean tile for this test"
    result = terrain_aware_step(ox - 1, oy, ox + 5, oy, base_speed=4, has_boat=True)
    assert biome_at(*result) != "ocean"


def test_terrain_aware_step_deflects_around_the_ocean_on_a_diagonal():
    """The one real "obstacle": stepping directly into open ocean is impassable (no
    boats yet), so a party heading southeast from forest into the sea gets deflected
    along the y-axis instead of teleported into the water -- still real progress since
    the target has a y-component to move along."""
    from backend.world import biome_at

    nx, ny = terrain_aware_step(85, 50, 99, 80, base_speed=8)
    assert (nx, ny) == (85, 58)
    assert biome_at(nx, ny) != "ocean"


def test_terrain_aware_step_makes_no_progress_when_the_only_path_is_straight_into_the_sea():
    """A due-east target with no y-component gives the y-axis deflection nothing to
    work with -- staying put is the only ocean-free option, not a bug."""
    nx, ny = terrain_aware_step(85, 50, 99, 50, base_speed=8)
    assert (nx, ny) == (85, 50)


def test_terrain_aware_step_falls_back_to_the_y_axis_when_x_is_blocked():
    """A target out to sea on a diagonal still finds a way to make progress along
    whichever axis the coastline doesn't block at this exact latitude -- the coast is
    wavy (see world.py's _coast_boundary_x), not a flat line, so which axis is blocked
    depends on where you're standing, not just on OCEAN_X_START."""
    from backend.world import biome_at

    nx, ny = terrain_aware_step(85, 50, 99, 99, base_speed=8)
    assert biome_at(nx, ny) != "ocean"
    assert (nx, ny) == (85, 58)  # x held at the shoreline, y-axis progress instead
