"""The City Architect role: decides *where* a building goes within a tribe's own
territory. Deliberately a cheap, deterministic function today, not a reasoning agent --
but named and scoped as its own role (mirroring backend/breeding.py, backend/
reflection.py) specifically so a future version could replace the body below with a
real LLM call reasoning about layout, without any caller needing to change. Do not add
a hook/interface for that now; this is a placeholder for later, not a build target.

The tribe's own model still decides *what* to build and *when* -- untouched. Only the
exact (x, y) a completed build lands at is automated, the same category of decision
this project already keeps out of small local models' hands for the same reason
(SCOUT's target_vector, the RELOCATE self-targeting bug both showed small models can't
reliably reason about exact coordinates).
"""

from . import config


def _occupied_rects(tribe) -> list[tuple[int, int, int, int]]:
    """Every rect a new building must not overlap: everything already placed, plus
    every wall section's footprint from every ring -- reserved the instant a ring's
    geometry exists, whether built or not, so buildings never get placed on top of
    where a wall is going to stand."""
    rects = [(b["x"], b["y"], b["w"], b["h"]) for b in tribe.buildings]
    for ring in tribe.wall_rings:
        rects.extend((s["x"], s["y"], s["w"], s["h"]) for s in ring["sections"])
    return rects


def _overlaps(ax, ay, aw, ah, bx, by, bw, bh, padding: int) -> bool:
    return not (
        ax + aw + padding <= bx
        or bx + bw + padding <= ax
        or ay + ah + padding <= by
        or by + bh + padding <= ay
    )


def _ring_perimeter_offsets(d: int):
    """Every (dx, dy) offset on the Chebyshev-distance-d square ring around the
    origin, in a fixed clockwise order starting due east -- deterministic so the
    same territory state always searches candidates in the same order."""
    if d == 0:
        yield (0, 0)
        return
    for dy in range(-d, d + 1):
        yield (d, dy)
    for dx in range(d - 1, -d - 1, -1):
        yield (dx, d)
    for dy in range(d - 1, -d - 1, -1):
        yield (-d, dy)
    for dx in range(-d + 1, d):
        yield (dx, -d)


def find_free_slot(world, tribe, building_type: str) -> tuple[int, int] | None:
    """Square-spiral scan outward from tribe.territory_center for a free,
    non-overlapping anchor for building_type's footprint. Returns the top-left
    (x, y) of the footprint, or None if the whole territory has no room left."""
    if tribe.territory_center is None:
        return None
    w, h = config.BUILDING_FOOTPRINTS[building_type]
    cx, cy = tribe.territory_center
    occupied = _occupied_rects(tribe)
    padding = config.BUILDING_PLACEMENT_PADDING

    for d in range(0, tribe.territory_radius + 1):
        for dx, dy in _ring_perimeter_offsets(d):
            x = cx + dx - w // 2
            y = cy + dy - h // 2
            if x < 0 or y < 0 or x + w > world.grid_size or y + h > world.grid_size:
                continue
            if max(abs((x + w / 2) - cx), abs((y + h / 2) - cy)) > tribe.territory_radius:
                continue
            if any(world.biome(tx, ty) in config.UNBUILDABLE_BIOMES for tx in range(x, x + w) for ty in range(y, y + h)):
                continue
            if any(_overlaps(x, y, w, h, ox, oy, ow, oh, padding) for ox, oy, ow, oh in occupied):
                continue
            return x, y
    return None


def record_building(tribe, building_type: str, x: int, y: int, w: int, h: int, cycle: int) -> None:
    tribe.buildings.append({"type": building_type, "x": x, "y": y, "w": w, "h": h, "cycle_built": cycle})
