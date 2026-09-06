"""One-time generator for the river/lake tile sets baked into
backend/world_hydrology_data.py and the generated data block inside
frontend/index.html.

Not on the running app's import path -- rerun by hand whenever the hydrology
shape needs retuning, then commit the two regenerated files:

    python scripts/generate_hydrology.py

Technique: adapt the heightfield+erosion idea from a mewo2-style procedural
terrain generator (Gaussian-peak heightfield -> box-blur relaxation -> sqrt
"peaky" sharpening -- see the terrain.js reference this was ported from) to
this project's plain 100x100 grid. A fully free erosion pass could relocate
the river/lake anywhere, but the exact centerline/branch/lake-center tiles of
today's fixed sine-wave shape (world._river_center_y, world.LAKE_CENTER) are
load-bearing for ~60 existing test fixtures and one spawn point -- so this
generator treats today's shape as a PROTECTED CORE plus an OUTER FOOTPRINT,
and only erodes the shell in between. Erosion survival is decided by the
heightfield, biased so upstream stretches erode harder (narrower, per the
explicit "the river can be less wide" request) than the stretch near the
mouth. Net effect: real, organic, non-uniform width and a non-circular lake
basin that is never wider than or relocated from today's shape -- only ever a
subset of it -- so gameplay-facing fixtures that depend on the exact
centerline/anchor tiles keep passing untouched.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.world import (
    LAKE_CENTER,
    LAKE_RADIUS,
    LAKE_TRIBUTARY_BRANCH_X,
    LAKE_TRIBUTARY_HALF_WIDTH,
    OCEAN_X_START,
    RIVER_HALF_WIDTH,
    RIVER_SOURCE_X,
    _coast_boundary_x,
    _river_center_y,
    _west_coast_boundary,
)

GRID = config.GRID_SIZE
SEED = config.HYDROLOGY_SEED

BACKEND_DATA_PATH = Path(__file__).resolve().parent.parent / "backend" / "world_hydrology_data.py"
FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
MARKER_START = "// === GENERATED HYDROLOGY DATA START (scripts/generate_hydrology.py, do not hand-edit) ==="
MARKER_END = "// === GENERATED HYDROLOGY DATA END ==="


def _old_is_river(x: int, y: int) -> bool:
    """Today's fixed sine-wave river, exactly as world._is_river computed it
    before this rework -- the OUTER FOOTPRINT this generator only ever
    subtracts from, never exceeds."""
    if x < max(RIVER_SOURCE_X, _west_coast_boundary(y)) or x >= min(OCEAN_X_START, _coast_boundary_x(y)):
        return False
    return abs(y - _river_center_y(x)) <= RIVER_HALF_WIDTH


def _old_is_lake(x: int, y: int) -> bool:
    """Today's perfect-circle lake + straight tributary, exactly as
    world._is_lake computed it before this rework."""
    lx, ly = LAKE_CENTER
    if math.hypot(x - lx, y - ly) <= LAKE_RADIUS:
        return True
    bx, by = LAKE_TRIBUTARY_BRANCH_X, _river_center_y(LAKE_TRIBUTARY_BRANCH_X)
    dx, dy = lx - bx, ly - by
    length_sq = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((x - bx) * dx + (y - by) * dy) / length_sq))
    proj_x, proj_y = bx + t * dx, by + t * dy
    return math.hypot(x - proj_x, y - proj_y) <= LAKE_TRIBUTARY_HALF_WIDTH


def _build_roughness_field(rng: np.random.Generator) -> np.ndarray:
    """mountains()+relax()+peaky() analogue from the terrain.js reference:
    Gaussian peaks, box-blur smoothing, sqrt sharpening, normalized to [0, 1].
    Used purely as an erosion-resistance texture over the eroded shell tiles
    below (high value = resists erosion, stays dry land) -- not as a whole-map
    elevation model, since the coastline/mountains/desert/forest are fixed and
    out of scope for this phase."""
    xs, ys = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    h = np.zeros((GRID, GRID))
    for _ in range(40):
        px, py = rng.uniform(0, GRID), rng.uniform(0, GRID)
        amp, r = rng.uniform(0.4, 1.0), rng.uniform(4, 10)
        h += amp * np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * r ** 2))
    for _ in range(2):  # relax(): box-blur via edge-padded 4-neighbor average
        padded = np.pad(h, 1, mode="edge")
        h = (
            padded[1:-1, 1:-1] + padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) / 5
    h = np.sqrt(h - h.min())  # peaky(): sharpen, keep non-negative
    return (h - h.min()) / (h.max() - h.min())


def _river_tiles(roughness: np.ndarray) -> frozenset[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    span = OCEAN_X_START - RIVER_SOURCE_X
    for x in range(GRID):
        core_y = round(_river_center_y(x))
        if not (0 <= core_y < GRID) or not _old_is_river(x, core_y):
            continue  # outside today's valid (coast-clipped) x-range entirely
        tiles.add((x, core_y))  # protected core: the exact centerline tile
        progress = max(0.0, min(1.0, (x - RIVER_SOURCE_X) / span))
        # Erosive reach grows toward the mouth -- a trickle at the source,
        # its full historical width only near the sea ("the river can be less
        # wide" -- upstream stretches erode away almost the whole old ribbon).
        reach = progress * RIVER_HALF_WIDTH
        for offset in range(1, RIVER_HALF_WIDTH + 1):
            if offset > reach + 0.5:
                continue
            survive_chance = max(0.0, (reach - offset + 1) / (offset + 1))
            for sign in (-1, 1):
                y = core_y + sign * offset
                if not (0 <= y < GRID) or not _old_is_river(x, y):
                    continue
                if roughness[x, y] < survive_chance:
                    tiles.add((x, y))
    return frozenset(tiles)


def _lake_tiles(roughness: np.ndarray) -> frozenset[tuple[int, int]]:
    lx, ly = LAKE_CENTER
    bx, by = LAKE_TRIBUTARY_BRANCH_X, _river_center_y(LAKE_TRIBUTARY_BRANCH_X)
    inner_radius = LAKE_RADIUS - 2  # protected core disk
    tiles: set[tuple[int, int]] = set()
    for x in range(max(0, lx - LAKE_RADIUS - 1), min(GRID, lx + LAKE_RADIUS + 2)):
        for y in range(max(0, ly - LAKE_RADIUS - 1), min(GRID, ly + LAKE_RADIUS + 2)):
            dist = math.hypot(x - lx, y - ly)
            if dist > LAKE_RADIUS:
                continue
            if dist <= inner_radius:
                tiles.add((x, y))
                continue
            # Outer shell: erode based on roughness, biased to erode more near
            # the old edge -- "the lake less rounded": a perfect circle becomes
            # an irregular, sometimes-receding shoreline, never wider than before.
            shell_progress = (dist - inner_radius) / (LAKE_RADIUS - inner_radius)
            if roughness[x, y] > shell_progress:
                tiles.add((x, y))
    # Tributary: protected core is the exact line (perpendicular distance <=
    # 1), the outer skin out to LAKE_TRIBUTARY_HALF_WIDTH erodes the same way.
    dx, dy = lx - bx, ly - by
    length = math.hypot(dx, dy)
    steps = max(1, int(length) * 2)
    for i in range(steps + 1):
        t = i / steps
        cx, cy = bx + t * dx, by + t * dy
        span = LAKE_TRIBUTARY_HALF_WIDTH + 1
        for ox in range(-span, span + 1):
            for oy in range(-span, span + 1):
                x, y = round(cx) + ox, round(cy) + oy
                if not (0 <= x < GRID and 0 <= y < GRID):
                    continue
                perp_dist = math.hypot(x - cx, y - cy)
                if perp_dist > LAKE_TRIBUTARY_HALF_WIDTH:
                    continue
                if perp_dist <= 1.0 or roughness[x, y] < 0.5:
                    tiles.add((x, y))
    return frozenset(tiles)


def generate() -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    rng = np.random.default_rng(SEED)
    roughness = _build_roughness_field(rng)
    return _river_tiles(roughness), _lake_tiles(roughness)


def _write_backend_module(river_tiles: frozenset, lake_tiles: frozenset) -> None:
    def fmt(tiles: frozenset) -> str:
        return ",\n    ".join(str(t) for t in sorted(tiles))

    BACKEND_DATA_PATH.write_text(
        '"""Generated by scripts/generate_hydrology.py -- do not hand-edit.\n\n'
        "Baked river/lake tile membership for the fixed map's natural-hydrology "
        'rework. See that script for the generation technique.\n"""\n\n'
        f"RIVER_TILES = frozenset([\n    {fmt(river_tiles)},\n])\n\n"
        f"LAKE_TILES = frozenset([\n    {fmt(lake_tiles)},\n])\n",
        encoding="utf-8",
    )


def _write_frontend_block(river_tiles: frozenset, lake_tiles: frozenset) -> None:
    html = FRONTEND_PATH.read_text(encoding="utf-8")
    start = html.index(MARKER_START)
    end = html.index(MARKER_END) + len(MARKER_END)
    river_js = ", ".join(str(x * GRID + y) for x, y in sorted(river_tiles))
    lake_js = ", ".join(str(x * GRID + y) for x, y in sorted(lake_tiles))
    block = (
        f"{MARKER_START}\n"
        f"const HYDROLOGY_RIVER_TILES = new Set([{river_js}]);\n"
        f"const HYDROLOGY_LAKE_TILES = new Set([{lake_js}]);\n"
        f"{MARKER_END}"
    )
    FRONTEND_PATH.write_text(html[:start] + block + html[end:], encoding="utf-8")


if __name__ == "__main__":
    river_tiles, lake_tiles = generate()
    _write_backend_module(river_tiles, lake_tiles)
    _write_frontend_block(river_tiles, lake_tiles)
    old_river = sum(1 for x in range(GRID) for y in range(GRID) if _old_is_river(x, y))
    old_lake = sum(1 for x in range(GRID) for y in range(GRID) if _old_is_lake(x, y))
    print(f"river tiles: {len(river_tiles)} (was {old_river})")
    print(f"lake tiles:  {len(lake_tiles)} (was {old_lake})")
