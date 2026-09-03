def calculate_next_step(x: int, y: int, target_x: int, target_y: int, bound: int = 99, speed: int = 1):
    """Moves up to `speed` tiles per axis per cycle toward (target_x, target_y), clamped
    to the grid and never overshooting the target."""
    dx = max(-speed, min(speed, target_x - x))
    dy = max(-speed, min(speed, target_y - y))
    return max(0, min(bound, x + dx)), max(0, min(bound, y + dy))


def terrain_aware_step(
    x: int, y: int, target_x: int, target_y: int, bound: int = 99, base_speed: float = 1.0, has_boat: bool = False,
):
    """Like calculate_next_step, but movement speed depends on the terrain currently
    being crossed (config.TERRAIN_MOVEMENT_MULTIPLIER -- mountains and rivers are slow
    going, plains are easy), and stepping directly into the ocean stays impassable
    outright (deflected along a single axis instead of blocked outright, the simplest
    way to route around an obstacle without full pathfinding) -- explicit request: a
    Boat (Simulation._advance_automatic_boat) grants real mobility only in fresh
    water (config.BOAT_WATER_BIOMES -- river/lake), never the sea, so ocean crossing
    stays exactly as impassable as ever even once a tribe has one. Used by RELOCATE
    and scouting expeditions alike (backend/actions.py, backend/simulation.py.
    _advance_expedition) so a straight-line journey isn't perfectly linear regardless
    of what's actually in the way."""
    from . import config
    from .world import biome_at

    biome = biome_at(x, y)
    if has_boat and biome in config.BOAT_WATER_BIOMES:
        multiplier = config.BOAT_WATER_MOVEMENT_MULTIPLIER
    else:
        multiplier = config.TERRAIN_MOVEMENT_MULTIPLIER.get(biome, 1.0)
    speed = max(1, round(base_speed * multiplier))

    candidates = (
        calculate_next_step(x, y, target_x, target_y, bound=bound, speed=speed),
        calculate_next_step(x, y, target_x, y, bound=bound, speed=speed),  # x-axis only
        calculate_next_step(x, y, x, target_y, bound=bound, speed=speed),  # y-axis only
    )
    for nx, ny in candidates:
        if biome_at(nx, ny) != "ocean":
            return nx, ny
    return x, y  # boxed in by ocean on every axis -- stay put rather than swim


def extend_ray_to_grid_edge(ox: int, oy: int, tx: int, ty: int, grid_size: int) -> tuple[int, int]:
    """Projects the ray from (ox, oy) through (tx, ty) out to wherever it first hits
    the edge of the grid, preserving direction. Used when a scouting expedition
    (simulation.py._advance_expedition) reaches its declared target_vector without
    finding water but still has days left -- rather than treating "arrived" as "search
    over," it keeps walking the same heading using the days it has left, instead of a
    nearby target_vector (the common case) ending the search on day one."""
    dx, dy = tx - ox, ty - oy
    if dx == 0 and dy == 0:
        return tx, ty
    bound = grid_size - 1
    scales = []
    if dx > 0:
        scales.append((bound - ox) / dx)
    elif dx < 0:
        scales.append((0 - ox) / dx)
    if dy > 0:
        scales.append((bound - oy) / dy)
    elif dy < 0:
        scales.append((0 - oy) / dy)
    scale = min(scales)
    ex = max(0, min(bound, round(ox + dx * scale)))
    ey = max(0, min(bound, round(oy + dy * scale)))
    return ex, ey
