def calculate_next_step(x: int, y: int, target_x: int, target_y: int, bound: int = 99, speed: int = 1):
    """Moves up to `speed` tiles per axis per cycle toward (target_x, target_y), clamped
    to the grid and never overshooting the target."""
    dx = max(-speed, min(speed, target_x - x))
    dy = max(-speed, min(speed, target_y - y))
    return max(0, min(bound, x + dx)), max(0, min(bound, y + dy))
