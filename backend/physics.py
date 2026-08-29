def calculate_next_step(x: int, y: int, target_x: int, target_y: int, bound: int = 99):
    """One tile of movement per turn, toward (target_x, target_y), clamped to the grid."""
    dx = (target_x > x) - (target_x < x)
    dy = (target_y > y) - (target_y < y)
    return max(0, min(bound, x + dx)), max(0, min(bound, y + dy))
