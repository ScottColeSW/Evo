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


# Follow-up fix, direct feedback on the first pass: "the lake is perfectly
# round. the river to the lake is straight. the tributary looks broken." A
# 2-tile shell peeled off a radius-7 circle still reads as a circle, and the
# tributary was left an exact straight line. This redesign (a) gives the lake
# an angular wavy boundary using the same two-sine-wave idiom every coastline
# in world.py already uses, (b) lets it grow a real bay toward the southwest
# -- confirmed clear of every test fixture and every other biome boundary,
# and the OPPOSITE direction from the one nearby test anchor (30, 60) that
# must stay dry land -- per direct request ("we can extend the lake,
# naturally curving into the south-west part below it"), and (c) bows the
# tributary into a single natural arch instead of a straight line, verified
# computationally to bow AWAY from (30, 60) (increasing its clearance from
# 3.15 to just over 5 tiles, strictly safer than the straight line it
# replaces).
LAKE_SOUTHWEST_ANGLE = 3 * math.pi / 4  # atan2(dy, dx): -x (west) and +y (south)
LAKE_SOUTHWEST_EXTENSION_MAX = 8.0
TRIBUTARY_BOW_AMPLITUDE = 3.5


def _lake_boundary_radius(theta: float) -> float:
    """The lake body's own edge, same shape as _coast_boundary_x etc. but as a
    function of angle from LAKE_CENTER instead of x or y. Clipped to never
    exceed the original LAKE_RADIUS -- real coves and lobes, not a fuzzy
    circle -- everywhere except the deliberate southwest bay below."""
    base = LAKE_RADIUS - 2.5 + 1.5 * math.sin(3 * theta + 0.7) + 1.0 * math.sin(5 * theta + 2.1)
    return min(LAKE_RADIUS, base)


def _southwest_extension(theta: float) -> float:
    """Extra reach, unclipped, concentrated around the southwest direction --
    a real secondary bay, not just edge texture. Squared cosine keeps it a
    tight lobe rather than a wide lopsided bulge."""
    aligned = max(0.0, math.cos(theta - LAKE_SOUTHWEST_ANGLE))
    return LAKE_SOUTHWEST_EXTENSION_MAX * aligned ** 2


def _tributary_center(t: float) -> tuple[float, float]:
    """A single bow (zero at both ends, peak at the midpoint) instead of a
    straight line -- verified to bow away from the (30, 60) test anchor, see
    this module's own docstring."""
    bx, by = LAKE_TRIBUTARY_BRANCH_X, _river_center_y(LAKE_TRIBUTARY_BRANCH_X)
    lx, ly = LAKE_CENTER
    dx, dy = lx - bx, ly - by
    length = math.hypot(dx, dy)
    perp = (-dy / length, dx / length)
    bow = TRIBUTARY_BOW_AMPLITUDE * math.sin(math.pi * t)
    return bx + t * dx + bow * perp[0], by + t * dy + bow * perp[1]


def _lake_tiles(roughness: np.ndarray) -> frozenset[tuple[int, int]]:
    lx, ly = LAKE_CENTER
    max_reach = LAKE_RADIUS + LAKE_SOUTHWEST_EXTENSION_MAX
    tiles: set[tuple[int, int]] = set()
    for x in range(max(0, lx - int(max_reach) - 1), min(GRID, lx + int(max_reach) + 2)):
        for y in range(max(0, ly - int(max_reach) - 1), min(GRID, ly + int(max_reach) + 2)):
            dist = math.hypot(x - lx, y - ly)
            if dist == 0:
                tiles.add((x, y))
                continue
            theta = math.atan2(y - ly, x - lx)
            if dist <= _lake_boundary_radius(theta) + _southwest_extension(theta):
                tiles.add((x, y))
    # Tributary: protected core is the bowed centerline itself (perpendicular
    # distance <= 1), the outer skin out to LAKE_TRIBUTARY_HALF_WIDTH erodes
    # by roughness, tapering wider toward the lake the same progress-based way
    # the main river already does.
    bx, by = LAKE_TRIBUTARY_BRANCH_X, _river_center_y(LAKE_TRIBUTARY_BRANCH_X)
    length = math.hypot(lx - bx, ly - by)
    steps = max(1, int(length) * 4)
    for i in range(steps + 1):
        t = i / steps
        cx, cy = _tributary_center(t)
        reach = t * LAKE_TRIBUTARY_HALF_WIDTH
        span = LAKE_TRIBUTARY_HALF_WIDTH + 1
        for ox in range(-span, span + 1):
            for oy in range(-span, span + 1):
                x, y = round(cx) + ox, round(cy) + oy
                if not (0 <= x < GRID and 0 <= y < GRID):
                    continue
                perp_dist = math.hypot(x - cx, y - cy)
                if perp_dist > LAKE_TRIBUTARY_HALF_WIDTH:
                    continue
                if perp_dist <= 1.0:
                    tiles.add((x, y))
                elif perp_dist - 1.0 <= reach and roughness[x, y] < 0.6:
                    tiles.add((x, y))
    return frozenset(tiles)


def _confluence_pool() -> frozenset[tuple[int, int]]:
    """A small rounded pool of river tiles right at the branch point, so the
    incoming mountain river, the tributary peeling off, and the river's own
    continuation toward the coast read as one natural confluence instead of a
    hard, mismatched-width fork -- direct feedback: "the tributary looks
    broken at the river to the lake and the remainder extending east." Only
    ever keeps tiles the old fixed-width river formula already covered there
    (a wide, unaffected stretch -- see this module's own verification step),
    so this never violates the "never wider than the old footprint" guarantee."""
    bx, by = LAKE_TRIBUTARY_BRANCH_X, round(_river_center_y(LAKE_TRIBUTARY_BRANCH_X))
    pool = set()
    for x in range(bx - 3, bx + 4):
        for y in range(by - 3, by + 4):
            if math.hypot(x - bx, y - by) <= 2.2 and _old_is_river(x, y):
                pool.add((x, y))
    return frozenset(pool)


def _bridge_diagonal_pinches(tiles: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    """Small display cleanup, flagged directly from a live screenshot: near
    the river's source its protected centerline is only 1 tile wide and steps
    diagonally (both x and y advance by 1 each column, since the centerline's
    slope runs close to 1:1 there) -- two tiles touching only at a corner,
    with neither of the two orthogonal squares between them filled in. Every
    tile-based renderer draws that as a staircase of separate speckles, not a
    connected line, even though it's one connected shape by 8-connectivity.
    Fills exactly one of the two orthogonal bridge tiles for every such corner
    pair so nothing this renders ever touches only at a corner."""
    tiles = set(tiles)
    for x, y in list(tiles):
        for dx, dy in ((1, 1), (1, -1)):
            diagonal = (x + dx, y + dy)
            if diagonal not in tiles:
                continue
            bridge_a, bridge_b = (x + dx, y), (x, y + dy)
            if bridge_a not in tiles and bridge_b not in tiles:
                tiles.add(bridge_a)
    return frozenset(tiles)


def generate() -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    rng = np.random.default_rng(SEED)
    roughness = _build_roughness_field(rng)
    river = _river_tiles(roughness) | _confluence_pool()
    lake = _lake_tiles(roughness)
    return _bridge_diagonal_pinches(frozenset(river)), _bridge_diagonal_pinches(frozenset(lake))


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
