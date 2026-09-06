"""Standalone, one-off analysis: does the new eroded river/lake shape change
wall/territory outcomes at this map's real spawn points, compared to the old
pure-sine-wave/circle shape? Answers the explicit request: "I'd like to see
the in-game analysis of if it would save us given our fixed map at this
time." Not a pytest test -- run by hand:

    python scripts/compare_hydrology_wall_impact.py

Exercises the real production path (Simulation._choose_territory_center ->
city_layout.build_ring -> city_layout._is_natural_barrier) against two biome
implementations: the current one (baked, eroded hydrology) and the pre-rework
one (recomputed directly from world.py's still-present formula pieces --
_river_center_y, LAKE_CENTER, etc. -- the same ones
scripts/generate_hydrology.py treats as its outer footprint), so the
comparison runs the identical territory-search logic against both, not a
reimplementation of it.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import city_layout, config
from backend.simulation import SPAWN_POINTS, Simulation
from backend.world import (
    LAKE_CENTER,
    LAKE_RADIUS,
    LAKE_TRIBUTARY_BRANCH_X,
    LAKE_TRIBUTARY_HALF_WIDTH,
    OCEAN_X_START,
    RIVER_HALF_WIDTH,
    RIVER_SOURCE_X,
    Landscape,
    _coast_boundary_x,
    _coast_is_headland,
    _desert_north_boundary,
    _forest_east_boundary,
    _forest_north_boundary,
    _is_headland_like,
    _is_volcano,
    _mountain_x_boundary,
    _mountain_y_boundary,
    _north_coast_boundary,
    _river_center_y,
    _south_coast_boundary,
    _west_coast_boundary,
    COAST_BAND_WIDTH,
)


def _old_is_river(x: int, y: int) -> bool:
    if x < max(RIVER_SOURCE_X, _west_coast_boundary(y)) or x >= min(OCEAN_X_START, _coast_boundary_x(y)):
        return False
    return abs(y - _river_center_y(x)) <= RIVER_HALF_WIDTH


def _old_is_lake(x: int, y: int) -> bool:
    lx, ly = LAKE_CENTER
    if math.hypot(x - lx, y - ly) <= LAKE_RADIUS:
        return True
    bx, by = LAKE_TRIBUTARY_BRANCH_X, _river_center_y(LAKE_TRIBUTARY_BRANCH_X)
    dx, dy = lx - bx, ly - by
    length_sq = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((x - bx) * dx + (y - by) * dy) / length_sq))
    proj_x, proj_y = bx + t * dx, by + t * dy
    return math.hypot(x - proj_x, y - proj_y) <= LAKE_TRIBUTARY_HALF_WIDTH


def _old_biome_at(x: int, y: int) -> str:
    """The pre-rework biome_at chain -- identical to world.biome_at's current
    body except for the two river/lake calls, which use the sine-wave/circle
    formulas above instead of the baked, eroded lookup."""
    if _old_is_river(x, y):
        return "river"
    east_boundary = _coast_boundary_x(y)
    west_boundary = _west_coast_boundary(y)
    north_boundary = _north_coast_boundary(x)
    south_boundary = _south_coast_boundary(x)
    if x >= east_boundary or x <= west_boundary or y <= north_boundary or y >= south_boundary:
        return "ocean"
    if east_boundary - x <= COAST_BAND_WIDTH:
        return "cliffs" if _coast_is_headland(y) else "shoals"
    if x - west_boundary <= COAST_BAND_WIDTH:
        return "cliffs" if _is_headland_like(_west_coast_boundary, y, ocean_on_increasing_side=False) else "shoals"
    if y - north_boundary <= COAST_BAND_WIDTH:
        return "cliffs" if _is_headland_like(_north_coast_boundary, x, ocean_on_increasing_side=False) else "shoals"
    if south_boundary - y <= COAST_BAND_WIDTH:
        return "cliffs" if _is_headland_like(_south_coast_boundary, x, ocean_on_increasing_side=True) else "shoals"
    if _old_is_lake(x, y):
        return "lake"
    if _is_volcano(x, y):
        return "volcano"
    if x < _mountain_x_boundary(y) and y < _mountain_y_boundary(x):
        return "mountains"
    if y >= _desert_north_boundary(x):
        return "desert"
    if y < _forest_north_boundary(x) or x >= _forest_east_boundary(y):
        return "forest"
    return "plains"


class _OldLandscapeStub:
    """Same interface as world.Landscape, for the pieces build_ring/
    _choose_territory_center actually touch (just .biome)."""

    def biome(self, x: int, y: int) -> str:
        return _old_biome_at(x, y)


class _SimStub:
    def __init__(self, world_obj):
        self.world = world_obj


class _TribeStub:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y


def _report_for(world_obj, label: str) -> None:
    print(f"\n=== {label} ===")
    for (sx, sy) in SPAWN_POINTS:
        cx, cy = Simulation._choose_territory_center(_SimStub(world_obj), _TribeStub(sx, sy))
        ring = city_layout.build_ring(world_obj, (cx, cy), ring_index=0)
        count = sum(1 for sec in ring["sections"] if sec["natural_barrier"])
        moved = (cx, cy) != (sx, sy)
        print(
            f"spawn {(sx, sy)!s:>10} -> center {(cx, cy)!s:>10} "
            f"| natural_barriers={count} | backed_away={moved}"
        )


if __name__ == "__main__":
    _report_for(_OldLandscapeStub(), "OLD sine-wave river / perfect-circle lake")
    _report_for(Landscape(config.GRID_SIZE), "NEW eroded river / organic lake basin")
