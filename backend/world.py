import functools
import math
import random

from . import config
from . import world_hydrology_data as _hydrology

BIOME_LABELS = {
    "forest": "Whispering Wilds",
    "mountains": "Crags of Oros",
    "river": "Serpent's Vein",
    "lake": "Stillwater Mere",
    "plains": "Sunken Basin",
    "ocean": "The Boundless Deep",
    "cliffs": "The Shattered Brink",
    "shoals": "The Glass Shallows",
    "desert": "The Sunbaked Wastes",
    "volcano": "The Smoldering Maw",
}

# Explicit request: "Mines can contain the Unique Resource of the Biome... you can
# make unique names for each." One named resource per biome a Mine can be built
# on, each name drawing on that biome's own BIOME_LABELS flavor above rather than
# a single generic "ore" -- see Simulation._advance_one_expedition (discovery) and
# actions.py._build_mine (excavation). Trade finally has something a tribe would
# actually want to hold onto or offer, not just the same four generic resources
# (see the "Mine & unique resource" design note this was scoped from).
# Explicit request: "are the scouts finding Wolves Dens and Bear Caves and Deer
# Stands? if not, they should be... be creative. We might add Rabbit Warrens
# (for food and fur) etc." A forest wildlife discovery used to only ever be one
# generic "confirmed wildlife-rich area" -- now it's one of these three, chosen
# at random each time (see Simulation._advance_one_expedition). Bear Caves
# deliberately left out for now, per direct confirmation -- there's no bear
# encounter/hazard anywhere in the game yet to hang one on, unlike Deer Stand
# (HUNT_DEER's own prey) and Wolf Den (the existing wolf-pack hunting hazard).
WILDLIFE_SITE_TYPES = ("Deer Stand", "Wolf Den", "Rabbit Warren")

UNIQUE_RESOURCE_BY_BIOME = {
    "forest": "Whisperwood Amber",
    "mountains": "Orosite Ore",
    "river": "Serpent's Gold",
    "lake": "Mere Pearl",
    "plains": "Basin Loamstone",
    "ocean": "Abyssal Pearl",
    "cliffs": "Brinkspar Crystal",
    "shoals": "Shoalglass",
    "desert": "Duneglass",
    # Dead code today, same as ocean/cliffs/shoals above -- volcano is in
    # UNBUILDABLE_BIOMES (real hazard, see config.VOLCANO_HAZARD_CHANCE), so no
    # Mine site can ever seed there. Kept only for symmetry with every other entry.
    "volcano": "Cindermarrow",
}

# Earth-like hydrology: the river originates in the mountains (west) and winds its way
# down through plains and forest to a coastline (east), rather than being an arbitrary
# diagonal band unrelated to anything else on the map.
OCEAN_X_START = 90
# A real mountain range is a long spine, not a squat corner block -- narrower in x than
# the original (30) but reaching much further south in y (was 35) so the range actually
# runs most of the length of the west edge. See SPAWN_POINTS in simulation.py: the
# Mountain Tribe used to spawn one tile from the river running through the range's
# original northern corner; this shape gives it somewhere to spawn further down the
# range's eastern (grassy) edge, genuinely distant from water instead of standing on it.
MOUNTAIN_X_END = 24
MOUNTAIN_Y_END = 55
RIVER_SOURCE_X = 15
RIVER_HALF_WIDTH = 3

# The coast itself used to be a perfectly straight vertical line -- real coastlines
# aren't. Two overlapping sine waves (different periods, so the shape doesn't just
# repeat) push the ocean boundary in and out of OCEAN_X_START; the river's own mouth
# stays anchored to the flat OCEAN_X_START (see _is_river) so it isn't dragged around by
# the same waviness. A narrow band just inland of the wavy boundary gets real texture
# instead of instantly becoming plains/forest: a headland (the coast bulging out into
# the sea, convex) reads as rocky cliffs, a bay (the coast recessed inland, concave)
# reads as sandy shoals -- the same geological logic real coastlines follow.
COAST_BAND_WIDTH = 3


def _coast_boundary_x(y: float) -> float:
    return OCEAN_X_START + 5 * math.sin(y * 0.08) + 2 * math.sin(y * 0.23 + 1.7)


def _coast_is_headland(y: float) -> bool:
    """True where the coastline bulges out into the ocean (a local peak in
    boundary_x -- land juts further east than its neighbors), false where it's
    recessed into a bay (a local trough -- the sea intrudes further inland). A local
    peak has negative second-derivative curvature, a trough positive -- crude
    finite-difference check, but the sign is what matters here, not precision."""
    step = 1.0
    curvature = _coast_boundary_x(y + step) - 2 * _coast_boundary_x(y) + _coast_boundary_x(y - step)
    return curvature < 0


# Map dream, phase 2: "another attempt at increasing the Ocean/unplayable
# area" -- a real island, ocean wrapping the north/south/west edges too, not
# just the existing east coast above. Same "fixed inset + two sine waves"
# shape, different phase/frequency per side so the four coastlines don't all
# look identical. West/north measure inward from x=0/y=0; south measures
# inward from GRID_SIZE-1, mirroring how the east coast measures inward from
# OCEAN_X_START.
#
# Explicit correction: "the Ocean was only supposed to be a 'frame'" -- the
# first attempt reused the east coast's own +-7 wave amplitude on top of a
# much deeper base (18), which ate into the mountains and produced sandbar-
# like spits near the corners. WEST/NORTH/SOUTH_COAST_INSET_BASE=6 stayed, but
# amplitude alone isn't what makes the east coast read as a real coastline
# with distinct capes and inlets -- mixing three sine frequencies instead of
# two does that, without deepening the envelope at all. Explicit follow-up
# request: "clean up the Ocean edges all around the Island" with richer
# texture at the SAME depth (confirmed computationally: still 4-8 tiles
# everywhere, same as the two-term version, no relocations needed for the
# volcano/spawns/river source).
def _west_coast_boundary(y: float) -> float:
    return (
        config.WEST_COAST_INSET_BASE
        + 1.1 * math.sin(y * 0.05 + 0.4) + 0.6 * math.sin(y * 0.14 + 2.3) + 0.3 * math.sin(y * 0.33 + 1.0)
    )


def _north_coast_boundary(x: float) -> float:
    return (
        config.NORTH_COAST_INSET_BASE
        + 1.1 * math.sin(x * 0.04 + 1.1) + 0.6 * math.sin(x * 0.12 + 0.3) + 0.3 * math.sin(x * 0.29 + 2.6)
    )


def _south_coast_boundary(x: float) -> float:
    return (
        (config.GRID_SIZE - 1) - config.SOUTH_COAST_INSET_BASE
        - 1.1 * math.sin(x * 0.045 + 2.5) - 0.6 * math.sin(x * 0.11 + 0.9) - 0.3 * math.sin(x * 0.27 + 1.5)
    )


def _is_headland_like(boundary_fn, coord: float, ocean_on_increasing_side: bool) -> bool:
    """Generalizes _coast_is_headland to any of the four coastlines. A headland
    is where land locally pushes further toward the ocean than its neighbors --
    which curvature sign that means depends on which side of the boundary the
    ocean actually is: increasing coordinate for the east/south coasts (a
    larger boundary value means land reaches further that way), decreasing for
    west/north (a smaller boundary value means land reaches further that way)."""
    step = 1.0
    curvature = boundary_fn(coord + step) - 2 * boundary_fn(coord) + boundary_fn(coord - step)
    return curvature < 0 if ocean_on_increasing_side else curvature > 0

# A tributary forking off the main river toward the lower-middle of the map, ending in
# a lake -- the only drinkable fresh water on the whole map used to be that single river
# ribbon, which left most of the grid genuinely far from any water no matter how well a
# tribe reasoned about it. Same drinkable status as the river (see actions.py's water
# handling), but calmer -- no drowning hazard, unlike a river's current.
LAKE_TRIBUTARY_BRANCH_X = 35
LAKE_CENTER = (25, 65)
LAKE_RADIUS = 7
LAKE_TRIBUTARY_HALF_WIDTH = 2

# Map dream, phase 1: "the volcano is a Hazard they will die if they go there" --
# a single, small, fixed decorative-but-lethal feature near the mountains, not a
# wavy zone boundary (this is one place, not an organic terrain type that should
# look different every map). Same circle-test shape as the lake above,
# deliberately simpler than the sine-wave boundaries -- a one-off feature reads
# better as a clean circle than an "organic" wobbly one.
#
# Map dream, phase 2: relocated from the original (10, 12), which the first
# (too-deep, since-corrected) west/north inset attempt swallowed. Once that
# inset was corrected down to a real 4-8 tile frame (see WEST/NORTH_COAST_
# INSET_BASE), (13, 13) -- found by sampling _west_coast_boundary/
# _north_coast_boundary directly, not estimated -- clears both by ~5.5-6 tiles
# (VOLCANO_RADIUS=4 plus real margin), just barely off the original spot and
# still squarely inside the mountain range.
VOLCANO_CENTER = (13, 13)
VOLCANO_RADIUS = 4


def _river_center_y(x: int) -> float:
    span = OCEAN_X_START - RIVER_SOURCE_X
    progress = max(0.0, min(1.0, (x - RIVER_SOURCE_X) / span))
    drift = 18 + progress * 50  # highlands (~y=18-24) down to the coast (~y=67-73)
    meander = 6 * math.sin(x * 0.07)
    return drift + meander


def _is_river(x: int, y: int) -> bool:
    # Natural river/lake rework: "I'd love the river and lake to look better and
    # more natural." The old body here was a live formula -- the river's mouth
    # clipped to whichever was closer, the flat OCEAN_X_START cutoff or the real
    # (wavy) coastline (so it never stuck out past a receded bay), and a fixed
    # RIVER_HALF_WIDTH ribbon around _river_center_y(x). That formula is now the
    # generator's OUTER FOOTPRINT (see scripts/generate_hydrology.py) rather than
    # the live check: a heightfield+erosion pass only ever subtracts from it
    # (never widens or relocates it), baked once into world_hydrology_data.py so
    # this stays an O(1) lookup despite biome_at running thousands of times per
    # tick. _river_center_y/RIVER_SOURCE_X/RIVER_HALF_WIDTH etc. are kept below,
    # still real and still load-bearing -- the generator's protected "core" is
    # exactly this centerline, which is why tests deriving expected points from
    # it still pass unchanged.
    return (x, y) in _hydrology.RIVER_TILES


def _is_lake(x: int, y: int) -> bool:
    # See _is_river's comment -- same rework, same baked-lookup treatment. The
    # old perfect-circle-plus-straight-tributary formula is the generator's
    # outer footprint; LAKE_CENTER and the exact tributary line are its
    # protected core.
    return (x, y) in _hydrology.LAKE_TILES


def river_tiles() -> frozenset[tuple[int, int]]:
    """The baked river tile set -- for tests and scripts/compare_hydrology_wall_impact.py
    rather than reaching into world_hydrology_data directly."""
    return _hydrology.RIVER_TILES


def lake_tiles() -> frozenset[tuple[int, int]]:
    """The baked lake (+tributary) tile set -- see river_tiles()."""
    return _hydrology.LAKE_TILES


def _is_volcano(x: int, y: int) -> bool:
    vx, vy = VOLCANO_CENTER
    return math.hypot(x - vx, y - vy) <= VOLCANO_RADIUS


# Mountains/forest/plains used to meet along perfectly straight, axis-aligned lines --
# a real range or treeline never does. Same technique as the coastline above (two
# overlapping sine waves of different periods, so the edge doesn't just repeat itself):
# each boundary gets its own low-amplitude wobble, small enough that no existing
# interior point (a spawn, a test fixture deep inside one biome) flips to another, but
# enough that the edge itself reads as a natural, uneven line instead of a ruler-drawn
# one.
def _mountain_x_boundary(y: float) -> float:
    return MOUNTAIN_X_END + 4 * math.sin(y * 0.1) + 2 * math.sin(y * 0.27 + 0.9)


def _mountain_y_boundary(x: float) -> float:
    return MOUNTAIN_Y_END + 4 * math.sin(x * 0.09 + 2.1) + 2 * math.sin(x * 0.22)


def _forest_north_boundary(x: float) -> float:
    return 18 + 3 * math.sin(x * 0.08) + 1.5 * math.sin(x * 0.19 + 1.3)


def _forest_east_boundary(y: float) -> float:
    return 70 + 4 * math.sin(y * 0.07 + 0.5) + 2 * math.sin(y * 0.21)


def _desert_north_boundary(x: float) -> float:
    """Map dream, phase 1: a real Desert zone in the south of the map. Same
    "fixed constant + two sine waves" shape as every other wavy boundary above --
    south of this line is desert, carved out of what would otherwise be
    forest/plains fallthrough (checked before those two in biome_at, below)."""
    return config.DESERT_NORTH_BOUNDARY_BASE + 5 * math.sin(x * 0.06) + 2 * math.sin(x * 0.17 + 0.6)


def biome_at(x: int, y: int) -> str:
    # River is checked before the coast texture so its mouth cuts straight through to
    # the sea rather than being interrupted by a cliff/shoal band -- real river mouths
    # do exactly this. _is_river has its own flat OCEAN_X_START cutoff, unaffected by
    # the coastline's waviness below.
    if _is_river(x, y):
        return "river"
    # Map dream, phase 2 correction: "we shouldn't have any legs on the
    # island." Each of the four edges used to be checked ocean-then-band in one
    # pass before moving to the next edge -- right next to a corner, an edge's
    # own coast-band check ("within COAST_BAND_WIDTH of MY boundary") has no
    # way to know a DIFFERENT edge's ocean already claims that same tile, so it
    # would commit to "cliffs/shoals" (a strip of land-adjacent texture) on
    # ground that a different edge's check -- never reached, since the first
    # edge already returned -- would have correctly called open ocean. That
    # produced a visible tendril of coastal texture jutting out past the real
    # coastline at every corner (the "sandbars" spotted in a live screenshot).
    # Checking EVERY edge's plain ocean condition first, before any edge's
    # texture band gets a chance to commit, closes that gap: a tile within
    # reach of two coastlines' bands only ever gets a texture verdict once
    # every one of the four has agreed it isn't just open ocean.
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
    if _is_lake(x, y):
        return "lake"
    # Checked before mountains -- the volcano sits inside the mountain region and
    # must win there (see VOLCANO_CENTER/_RADIUS's own comment).
    if _is_volcano(x, y):
        return "volcano"
    if x < _mountain_x_boundary(y) and y < _mountain_y_boundary(x):
        return "mountains"
    # Checked before forest -- desert claims the southern band out of what would
    # otherwise be forest/plains fallthrough; mountains (above) still wins
    # regardless of geography since it's checked first.
    if y >= _desert_north_boundary(x):
        return "desert"
    if y < _forest_north_boundary(x) or x >= _forest_east_boundary(y):
        return "forest"
    return "plains"


def sector_of(x: int, y: int) -> tuple[int, int]:
    """Buckets a tile into its Tribe Map sector -- see
    config.TRIBE_MAP_SECTOR_SIZE's own comment."""
    size = config.TRIBE_MAP_SECTOR_SIZE
    return x // size, y // size


def mark_visited_sector(tribe, x: int, y: int) -> None:
    """Records that a tribe's own people have actually walked through this
    ground. Called everywhere Landscape.wear_trail already is -- the same real
    "someone was physically here" moments -- distinct from a positive-find list
    (lumber_sites etc.), which only records a discovery, not mere passage."""
    tribe.visited_sectors.add(sector_of(x, y))


# 2026-09-02 rework: resource sites (lumber/wildlife/quarry/mine) used to be decided
# fresh on every single scouting report, rolling an independent chance on whatever
# exact tile the scout happened to land on. That has two real problems: the world has
# no actual geography of "where things are" (a site can spontaneously not-exist one
# scout's report and then exist for the next tribe's report on the same tile), and
# nothing prevents two different site types stacking on the identical coordinate by
# construction (they used to). Explicit follow-up request: "a twisted sparse matrix
# assignment based on the existing map."
#
# This scatters a real, fixed set of site locations across the map ONCE, up front,
# deterministic per (seed_type, grid_size) via a string-seeded RNG -- same "pure
# function of coordinates" philosophy biome_at itself already follows, so this needs
# no persisted state anywhere.
#
# 2026-09-12 rework: replaced the original coarse-grid-plus-jitter scatter (one slot
# per SITE_SEED_GRID_CELL_SIZE cell) with real Poisson-disc sampling (Bridson's
# algorithm). Live report, with a screenshot: "objects landed along the boards of the
# map... in a line" -- confirmed as the grid system showing its own seams. With only
# ~5-6 cells per axis on a 100-tile map, a whole row of independent per-cell hits near
# an edge reads as a straight line, not organic scatter -- inherent to a grid-cell
# scatter, not something the earlier same-day edge-clamping fix (still a real, separate
# bug) could ever fully solve. Poisson-disc sampling guarantees real minimum spacing
# between every pair of points with genuinely organic (non-gridded) coverage,
# everywhere, including edges.
#
# Also folds in a second live request: "no Mine location" turned out to be bad luck
# the old system did nothing to prevent (sites had zero awareness of where any tribe
# actually spawns), plus an explicit follow-up to bias density toward each spawn
# point with a smooth falloff rather than a hard "guarantee N nearby" cliff ("fair
# distribution not just a bunch locally easy"). _site_spacing_radius below makes the
# minimum spacing itself a smooth function of distance to the nearest spawn point --
# small (dense) near a spawn, relaxing to a larger (sparse) background value as that
# distance grows -- so every candidate is checked against its own local density
# requirement, not one constant. One continuous gradient, the same shape for every
# spawn, no special-cased zone.
#
# Explicit steer, matching the earlier discovery-chance fairness fix: seed points are
# placed on ANY buildable biome, not tied to matching terrain (forest for
# lumber/wildlife, mountains for quarry) -- a tribe scouting the "wrong" biome is no
# longer structurally locked out. Mine sites are included in the same sparse system;
# each pre-seeded mine's resource name is read off whatever real biome the point
# itself sits on (world.UNIQUE_RESOURCE_BY_BIOME), preserving the one deliberate
# exception -- ore is still biome-tied, just the location is now a real place, not a
# fresh roll.
SITE_SEED_TYPES = ("lumber", "wildlife", "quarry", "mine", "landmark")
# Per-type minimum spacing in tiles: (r_near, r_far) -- r_near applies right at a
# spawn point, r_far once far enough from every spawn that the bias has fully
# relaxed to background density. Smaller r = denser. Landmark's own much larger
# pair keeps it the sparsest type, matching the earlier explicit "reduce the
# number" request.
#
# Live-run finding, 2026-09-20: "add more Mines, Timber, Quarry as they seem to
# struggle with some needs more than others... scale this based on data."
# Grounded against 13 tribe-runs' final snapshots (board_history.db, a mix of
# short and 400+/600+-cycle games): lumber sites were found in 85% of them
# (avg 1.23 discovered), but quarry only 31% (avg 0.62) and mine only 38%
# (avg 0.54) -- both were quietly the same sparse (14, 30) pair, well behind
# lumber's own (10, 22). Mine is the severe case specifically: unlike Quarry/
# Sawmill/Tannery (all explicitly de-gated from needing a real discovered
# site a while back -- see actions.py._build_quarry/_build_sawmill's own
# comments), actions.py._build_mine still hard-requires tribe.mine_sites --
# zero discoveries means zero Mines, zero Forge, zero crafted items, for the
# entire game, which is exactly what happened in 8 of the 13 sampled
# tribe-runs. Mine's spacing is cut the most for that reason (denser than
# lumber's own, not just closing the gap to it); quarry gets most of the way
# there; lumber gets a smaller bump too, per the explicit request, even
# though it wasn't the actual bottleneck in this data. Still invented
# first-pass numbers past this point -- watch discovery rates in a few fresh
# runs before treating these three as tuned.
SITE_DENSITY_BY_TYPE = {
    "lumber": (7, 15),   # was (8, 18): a few more Timber Groves for the wood gate, 2026-10-03 (about 40% more sites)
    "wildlife": (12, 26),
    "quarry": (9, 20),
    "mine": (7, 16),
    "landmark": (22, 48),
}
# How quickly the spawn bias relaxes to background, in tiles -- roughly "how far
# from home the map still feels noticeably richer." Loosely anchored to
# EXPLORATION_PARTY_PATROL_DISTANCE (45): comfortably reachable in one real trip.
SPAWN_BIAS_FALLOFF_DISTANCE = 60
# Bridson's algorithm's own "how many tries before giving up on this active point"
# constant -- higher finds tighter packings but costs more rejected candidates.
_POISSON_DISC_CANDIDATE_ATTEMPTS = 30
SITE_DISCOVERY_RADIUS = 8


def _site_spacing_radius(x: float, y: float, seed_type: str, spawn_points: tuple[tuple[int, int], ...]) -> float:
    r_near, r_far = SITE_DENSITY_BY_TYPE[seed_type]
    nearest = min(math.hypot(x - sx, y - sy) for sx, sy in spawn_points)
    falloff = math.exp(-nearest / SPAWN_BIAS_FALLOFF_DISTANCE)
    return r_far - (r_far - r_near) * falloff


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Where resource sites go (2026-10-06, the owner: "I think it is probably lacking in game smarts"). The old placement was a Poisson-disc scatter that knew nothing about
# what a site is for: on the real map only 28% of timber groves stood in forest (19% were in desert), 6% of stone-rich sites and mines stood on mountains, and nothing
# clustered. The rules now:
#   - Affinity: how well a tile suits a site, read off the game's own yield table (actions.BIOME_YIELD_MULTIPLIER). Groves stand in forest, and on plains within
#     a few tiles of forest; hunting grounds in forest, and on plains near forest or water; stone-rich sites and mines on mountains, and on the foothills (poorer ground within a few
#     tiles of mountains). Desert is barren for wood and game. A site pays by its ground, so a foothill quarry pays little.
#   - Clusters: groves come in woods of two or three, hunting grounds near forest edges, and a mine usually stands beside a quarry (a range).
#   - A fair start: every spawn point gets one of each type close by whatever its biome (the best ground within reach), and a settling tribe is guaranteed the same in its own
#     territory (Simulation._ensure_homeland), because tribes settle far from where they spawn.
#   - The count of sites of each type still thins out with distance from the spawn points, as before (_site_spacing_radius).
# ------------------------------------------------------------------------------------------------------------------------------------------------------
_AFFINITY_RESOURCE = {"lumber": "wood", "wildlife": "game", "quarry": "stone", "mine": "stone"}
_NEAR_FOREST_TILES = 6        # plains this close to forest can hold a grove
_NEAR_COVER_TILES = 8         # plains this close to forest or water can hold a hunting ground
_FOOTHILL_TILES = 6           # poorer ground this close to mountains can hold stone or ore
_CLUSTER_CHANCE = {"lumber": 0.0, "wildlife": 0.35, "quarry": 0.25, "mine": 0.0}  # timber is spread over any valid ground, not clustered (the owner, 2026-10-06)
_CLUSTER_RADIUS = (3, 6)      # satellites stand this far from the site they cluster around
_SATELLITE_GAP = 3            # and never closer than this to anything
_MINE_BESIDE_QUARRY = (5, 8)  # a mine's distance from the quarry it is paired with (never on top of it: see _CROSS_TYPE_GAP)
_MINE_PAIR_CHANCE = 0.85
_STARTER_RING = (4, 12)       # a spawn point's guaranteed site stands this far from it
_DART_CANDIDATES = 4000
# Sites of different types never overlap (the owner, 2026-10-06: "I hate seeing overlapping resources on the map"): no two are closer than this. The types are generated in this
# order and each avoids the ones before it, so the layout stays the same from run to run.
_CROSS_TYPE_GAP = 5
_GENERATION_ORDER = ("lumber", "wildlife", "quarry", "mine")
# A site needs dry ground around it (the owner: sites were "out of bounds" along the ocean, and the map draws waterfalls on river heads and tails): no unbuildable tile
# within 1 tile, and at most this share of the tiles within 3 tiles unbuildable.
_INLAND_RADIUS = 3
_INLAND_MAX_WET_SHARE = 0.25


@functools.lru_cache(maxsize=None)
def _inland_map(grid_size: int) -> tuple[tuple[bool, ...], ...]:
    """For every tile, whether it has dry ground around it (see _INLAND_RADIUS): nothing unbuildable within 1 tile and not much within 3."""
    unbuildable = [[biome_at(x, y) in config.UNBUILDABLE_BIOMES for y in range(grid_size)] for x in range(grid_size)]
    out = []
    for x in range(grid_size):
        row = []
        for y in range(grid_size):
            if unbuildable[x][y]:
                row.append(False)
                continue
            near_wet = wet = total = 0
            for dx in range(-_INLAND_RADIUS, _INLAND_RADIUS + 1):
                for dy in range(-_INLAND_RADIUS, _INLAND_RADIUS + 1):
                    a, b = x + dx, y + dy
                    if 0 <= a < grid_size and 0 <= b < grid_size:
                        total += 1
                        if unbuildable[a][b]:
                            wet += 1
                            if abs(dx) <= 1 and abs(dy) <= 1:
                                near_wet += 1
            row.append(near_wet == 0 and wet / total <= _INLAND_MAX_WET_SHARE)
        out.append(tuple(row))
    return tuple(out)


def is_inland(x: int, y: int, grid_size: int = 100) -> bool:
    return 0 <= x < grid_size and 0 <= y < grid_size and _inland_map(grid_size)[x][y]


@functools.lru_cache(maxsize=None)
def _distance_to_biomes(grid_size: int, biomes: frozenset) -> tuple[tuple[float, ...], ...]:
    """For every tile, the distance in tiles to the nearest tile of any of these biomes (a multi-source sweep over 8-neighbors, close enough to straight-line for a
    few tiles' worth of 'near')."""
    inf = float("inf")
    dist = [[0.0 if biome_at(x, y) in biomes else inf for y in range(grid_size)] for x in range(grid_size)]
    frontier = [(x, y) for x in range(grid_size) for y in range(grid_size) if dist[x][y] == 0.0]
    while frontier:
        nxt = []
        for x, y in frontier:
            d = dist[x][y] + 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    a, b = x + dx, y + dy
                    if 0 <= a < grid_size and 0 <= b < grid_size and dist[a][b] > d:
                        dist[a][b] = d
                        nxt.append((a, b))
        frontier = nxt
    return tuple(tuple(row) for row in dist)


def site_affinity(seed_type: str, x: int, y: int, grid_size: int = 100) -> float:
    """How well the tile (x, y) suits a site of this type, 0 (never) to 1 (the best ground), from the game's own yield table (see the notes above)."""
    from .actions import BIOME_YIELD_MULTIPLIER  # deferred: actions imports this module

    if not is_inland(x, y, grid_size):
        return 0.0
    biome = biome_at(x, y)
    if seed_type == "lumber":
        return 1.0  # timber stands anywhere valid: forest, plains, desert, the foothills (the owner, 2026-10-06; wood was the bottleneck, so it is plentiful)
    value = BIOME_YIELD_MULTIPLIER.get(_AFFINITY_RESOURCE[seed_type], {}).get(biome, 0.0)
    if seed_type == "wildlife":
        if biome == "forest":
            return value
        near = _distance_to_biomes(grid_size, frozenset({"forest", "river", "lake"}))[x][y]
        return value if biome == "plains" and near <= _NEAR_COVER_TILES else 0.0
    # quarry and mine: mountains, and the foothills around them
    if biome == "mountains":
        return value
    if value > 0 and _distance_to_biomes(grid_size, frozenset({"mountains"}))[x][y] <= _FOOTHILL_TILES:
        return value
    return 0.0


@functools.lru_cache(maxsize=None)
def _scatter_site_points(seed_type: str, grid_size: int) -> tuple[tuple[int, int], ...]:
    """The original placement: an even Poisson-disc scatter over buildable ground with no idea what the site is for. Kept for types that have no ground
    to suit (the landmarks); resource sites use site_seed_points' affinity rules."""
    # Deferred import: simulation.py imports from this module at load time, so a
    # top-level import here would be circular -- safe deferred to call time, well
    # after both modules have finished loading (this is never called from either
    # module's own top-level init).
    from .simulation import SPAWN_POINTS

    rng = random.Random(f"site_seed:{seed_type}:{grid_size}")

    def spacing_at(x: float, y: float) -> float:
        return _site_spacing_radius(x, y, seed_type, SPAWN_POINTS)

    def valid(x: int, y: int, placed: list[tuple[int, int, float]]) -> bool:
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return False
        if biome_at(x, y) in config.UNBUILDABLE_BIOMES:
            return False
        r_here = spacing_at(x, y)
        return all(math.hypot(x - ox, y - oy) >= max(r_here, o_r) for ox, oy, o_r in placed)

    # Bridson's algorithm: seed with one valid point, then grow outward -- each
    # active point tries a bounded number of random candidates in its own local
    # annulus [r, 2r] before giving up, guaranteeing every accepted point is at
    # least its own (and its neighbor's) minimum spacing away from anything else.
    placed: list[tuple[int, int, float]] = []
    active: list[tuple[int, int]] = []
    for _ in range(200):  # bounded search for a valid starting point
        x, y = rng.randrange(grid_size), rng.randrange(grid_size)
        if valid(x, y, placed):
            r = spacing_at(x, y)
            placed.append((x, y, r))
            active.append((x, y))
            break

    while active:
        idx = rng.randrange(len(active))
        px, py = active[idx]
        found = False
        for _ in range(_POISSON_DISC_CANDIDATE_ATTEMPTS):
            angle = rng.uniform(0, 2 * math.pi)
            r_here = spacing_at(px, py)
            dist = rng.uniform(r_here, 2 * r_here)
            x = round(px + math.cos(angle) * dist)
            y = round(py + math.sin(angle) * dist)
            if valid(x, y, placed):
                r = spacing_at(x, y)
                placed.append((x, y, r))
                active.append((x, y))
                found = True
                break
        if not found:
            active.pop(idx)

    return tuple((x, y) for x, y, _ in placed)


@functools.lru_cache(maxsize=None)
def site_seed_points(seed_type: str, grid_size: int) -> tuple[tuple[int, int], ...]:
    # Deferred import: simulation.py imports from this module at load time, so a
    # top-level import here would be circular -- safe deferred to call time, well
    # after both modules have finished loading (this is never called from either
    # module's own top-level init).
    if seed_type not in _AFFINITY_RESOURCE:
        return _scatter_site_points(seed_type, grid_size)
    from .simulation import SPAWN_POINTS

    rng = random.Random(f"site_seed:{seed_type}:{grid_size}")
    placed: list[tuple[int, int, float]] = []  # x, y, the spacing this site demands of its neighbors
    earlier = [p for t in _GENERATION_ORDER[:_GENERATION_ORDER.index(seed_type)] for p in site_seed_points(t, grid_size)]  # other types already on the map

    def far_enough(x: int, y: int, gap: float, parent: tuple[int, int] | None = None) -> bool:
        if any(math.hypot(x - a, y - b) < _CROSS_TYPE_GAP for a, b in earlier):
            return False
        # A satellite keeps only a short gap from the site it clusters around (`parent`) and no more than 60% of the spacing any other site demands.
        for ox, oy, o_gap in placed:
            if parent is not None and (ox, oy) == parent:
                if math.hypot(x - ox, y - oy) < _SATELLITE_GAP:
                    return False
            elif math.hypot(x - ox, y - oy) < max(gap, o_gap * (0.6 if parent is not None else 1.0)):
                return False
        return True

    def add(x: int, y: int, gap: float) -> None:
        placed.append((x, y, gap))

    # 1. A fair start: one site close to every spawn point, on the best ground within reach (any buildable ground if there is none that suits).
    lo, hi = _STARTER_RING
    for sx, sy in SPAWN_POINTS:
        ring = [(x, y) for x in range(sx - hi, sx + hi + 1) for y in range(sy - hi, sy + hi + 1)
                if lo <= math.hypot(x - sx, y - sy) <= hi and is_inland(x, y, grid_size)]
        ring.sort(key=lambda p: (-site_affinity(seed_type, p[0], p[1], grid_size), math.hypot(p[0] - sx, p[1] - sy), rng.random()))
        for x, y in ring:
            if far_enough(x, y, _SATELLITE_GAP):
                add(x, y, _SATELLITE_GAP)
                break

    # 2. A mine usually stands beside a quarry: the mountains hold both.
    if seed_type == "mine":
        for qx, qy in site_seed_points("quarry", grid_size):
            if rng.random() > _MINE_PAIR_CHANCE:
                continue
            for _ in range(20):
                angle, dist = rng.uniform(0, 2 * math.pi), rng.uniform(*_MINE_BESIDE_QUARRY)
                x, y = round(qx + math.cos(angle) * dist), round(qy + math.sin(angle) * dist)
                if site_affinity("mine", x, y, grid_size) > 0 and far_enough(x, y, _SATELLITE_GAP):
                    add(x, y, _SATELLITE_GAP)
                    break

    # 3. The rest: darts thrown over the map, kept with the chance the ground suits them (best ground first), at the usual spacing.
    darts = []
    for _ in range(_DART_CANDIDATES):
        x, y = rng.randrange(grid_size), rng.randrange(grid_size)
        a = site_affinity(seed_type, x, y, grid_size)
        if a > 0 and rng.random() < a:
            darts.append((a * rng.random(), x, y))
    darts.sort(reverse=True)
    core = []
    tight = SITE_DENSITY_BY_TYPE[seed_type][0]
    for _priority, x, y in darts:
        # Prime ground (a forest, a mountain range) is packed at the near spacing everywhere; the thinning with distance from spawn applies to the poorer ground.
        gap = tight if (seed_type != "lumber" and site_affinity(seed_type, x, y, grid_size) >= 0.9) else _site_spacing_radius(x, y, seed_type, SPAWN_POINTS)
        if far_enough(x, y, gap):
            add(x, y, gap)
            core.append((x, y))

    # 4. Clusters: a grove has neighbors, a hunting ground sits near others along an edge, a quarry may have a second face.
    chance = _CLUSTER_CHANCE.get(seed_type, 0.0)
    for cx, cy in list(core):
        if rng.random() >= chance:
            continue
        for _ in range(rng.choice((1, 1, 2))):
            for _try in range(12):
                angle, dist = rng.uniform(0, 2 * math.pi), rng.uniform(*_CLUSTER_RADIUS)
                x, y = round(cx + math.cos(angle) * dist), round(cy + math.sin(angle) * dist)
                if site_affinity(seed_type, x, y, grid_size) > 0.3 and far_enough(x, y, _SATELLITE_GAP, parent=(cx, cy)):
                    add(x, y, _SATELLITE_GAP)
                    break

    return tuple((x, y) for x, y, _ in placed)


def find_nearby_site(
    seed_type: str, x: int, y: int, grid_size: int, known: set, radius: int = SITE_DISCOVERY_RADIUS, extra_points: tuple = ()
) -> tuple[int, int] | None:
    """The nearest real, pre-seeded site of this type within `radius` of (x, y) that
    isn't already in `known` -- how a scout's report turns into an actual new
    discovery now, instead of an independent chance roll on their exact tile.
    `extra_points` are sites that came into being after the seeding (a spent node's
    respawn, see Landscape.respawned_sites); spent seed points arrive in `known`."""
    best, best_dist = None, None
    for px, py in list(site_seed_points(seed_type, grid_size)) + list(extra_points):
        if (px, py) in known:
            continue
        dist = (px - x) ** 2 + (py - y) ** 2
        if dist <= radius * radius and (best is None or dist < best_dist):
            best, best_dist = (px, py), dist
    return best


class Landscape:
    """Tracks terrain, built structures, and per-tile resource depletion. Emotional/
    ancestral bias lives separately in ancestral_matrix.AncestralTraumaMatrix — terrain
    and memory are different axes."""

    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.constructions: dict[tuple[int, int], dict] = {}
        # (resource_name, x, y) -> depletion level in [0, MAX_SCARCITY]. Repeatedly
        # harvesting the same resource at the same spot drives this up, which scales
        # down yield there -- a real, mechanical reason to move on, not a scripted one.
        self.depletion: dict[tuple[str, int, int], float] = {}
        # (x, y) -> {"wear": float in [0, 1], "color": str | None}. The inverse of
        # depletion: repeatedly relocating through a tile wears a path that speeds up
        # later travel through it, fading if it falls out of use. See config.
        # TRAIL_WEAR_PER_PASS. `color` is whichever tribe most recently wore this
        # exact tile (any tribe passing through benefits from the speed bonus, not
        # just whoever wore it first -- a trail is a feature of the ground, not a
        # private memory -- but the frontend renders it in that tribe's own color so
        # multiple tribes' paths stay visually distinct instead of blending into one
        # shared amber-to-gold gradient).
        self.trails: dict[tuple[int, int], dict] = {}
        # Resource nodes (docs/RESOURCE-NODES-DESIGN.md): how many times each pre-seeded site has been drawn from (shared by every tribe), the sites that are
        # spent for good, and the sites that appeared elsewhere when one was spent. Keys are (type, x, y) with type "lumber", "quarry" or "wildlife".
        self.site_uses: dict[tuple[str, int, int], int] = {}
        self.exhausted_sites: set[tuple[str, int, int]] = set()
        self.site_last_draw: dict[tuple[str, int, int], tuple[str, int]] = {}  # (type, x, y) -> (tribe id, cycle) of the latest draw, for field contests
        self.respawned_sites: dict[str, list[tuple[int, int]]] = {}

    def site_affinity(self, seed_type: str, x: int, y: int) -> float:
        return site_affinity(seed_type, x, y, self.grid_size)

    def is_inland(self, x: int, y: int) -> bool:
        return is_inland(x, y, self.grid_size)

    def all_live_sites(self) -> list[tuple[int, int]]:
        """Every live site of every type: what a new site keeps its distance from."""
        return [p for t in _GENERATION_ORDER for p in self.live_sites(t)]

    def live_sites(self, seed_type: str) -> list[tuple[int, int]]:
        """Every site of this type that can still be drawn from: the seeded points and the respawned ones, minus the spent."""
        points = list(site_seed_points(seed_type, self.grid_size)) + list(self.respawned_sites.get(seed_type, []))
        return [p for p in points if (seed_type, p[0], p[1]) not in self.exhausted_sites]

    def spent_of(self, seed_type: str) -> set[tuple[int, int]]:
        return {(x, y) for (t, x, y) in self.exhausted_sites if t == seed_type}

    def biome(self, x: int, y: int) -> str:
        return biome_at(x, y)

    def nearby_structures(self, x: int, y: int, radius: int = 6) -> list[dict]:
        out = []
        for (sx, sy), info in self.constructions.items():
            if abs(sx - x) <= radius and abs(sy - y) <= radius:
                out.append({"x": sx, "y": sy, **info})
        return out

    def add_construction(self, x: int, y: int, kind: str, cycle: int, progress: int = 100) -> None:
        # progress < 100 means "under construction" -- only CONSTRUCT_WALL builds in
        # stages (actions.py._construct_wall); BUILD_FIRE stays instant via the
        # default, no call-site changes needed anywhere else.
        self.constructions[(x, y)] = {"type": kind, "cycle": cycle, "progress": progress}

    def scarcity(self, resource: str, x: int, y: int) -> float:
        return self.depletion.get((resource, x, y), 0.0)

    def deplete(self, resource: str, x: int, y: int, amount: float, max_scarcity: float) -> None:
        key = (resource, x, y)
        self.depletion[key] = min(max_scarcity, self.depletion.get(key, 0.0) + amount)

    def regenerate(self, rate: float) -> None:
        """Called once per simulation tick, independent of who's standing where --
        the land recovers on its own schedule, not just when a tribe leaves."""
        for key in list(self.depletion):
            remaining = self.depletion[key] - rate
            if remaining <= 0:
                del self.depletion[key]
            else:
                self.depletion[key] = remaining

    def wear_trail(self, x: int, y: int, amount: float, color: str | None = None, tribe_id: str | None = None) -> None:
        """A tribe just relocated through (x, y) -- wear the path a little more. Any
        tribe passing through benefits, not just whoever wore it first: a trail is a
        feature of the ground, not a private memory. `color` (whichever tribe wore it
        just now) overwrites the tile's displayed color -- the most recent walker's
        color wins on a shared tile, rather than trying to blend multiple tribes'
        colors together.

        Explicit request: "trails that have been traversed more than 5 times by
        anyone will automatically evolve into visible and owned roads... The
        first trailblazer gets the ownership and tolls." `crossings` is a
        separate, never-decaying lifetime counter from `wear` (which fades on
        disuse and is purely cosmetic/speed-bonus) -- see is_toll_road/
        road_owner below, and Simulation._resolve_toll for where a toll is
        actually charged. `owner` is set once, from whichever tribe_id first
        ever wore this exact tile, and never changes after that even if a
        different tribe wears it far more since."""
        key = (x, y)
        existing = self.trails.get(key, {"wear": 0.0, "color": None, "crossings": 0, "owner": None})
        wear = min(1.0, existing["wear"] + amount)
        self.trails[key] = {
            "wear": wear,
            "color": color if color is not None else existing["color"],
            "crossings": existing["crossings"] + 1,
            "owner": existing["owner"] if existing["owner"] is not None else tribe_id,
        }

    def trail_speed_bonus(self, x: int, y: int, max_bonus: float) -> float:
        """Extra movement speed from standing on a worn trail, scaled linearly by wear."""
        entry = self.trails.get((x, y))
        return (entry["wear"] if entry else 0.0) * max_bonus

    def is_toll_road(self, x: int, y: int) -> bool:
        """A trail that's been crossed enough times to have evolved into a real,
        owned road -- see wear_trail's own docstring."""
        entry = self.trails.get((x, y))
        return bool(entry) and entry.get("crossings", 0) > config.ROAD_EVOLVE_CROSSINGS

    def road_owner(self, x: int, y: int) -> str | None:
        """The tribe_id of whoever first ever wore this tile, if anyone has."""
        entry = self.trails.get((x, y))
        return entry.get("owner") if entry else None

    def decay_trails(self, rate: float) -> None:
        """Called once per tick alongside regenerate() -- an unused trail fades back
        into open ground rather than staying fast forever once worn.

        A tile that's already evolved into a real, owned road is exempt -- a
        road doesn't revert to open ground just because no one's walked it
        this week, the same way a built wall doesn't un-build itself from
        disuse. Its speed-bonus wear can still fade toward a lower (but
        nonzero-crossings) floor; only the deletion is skipped."""
        for key in list(self.trails):
            if self.trails[key].get("crossings", 0) > config.ROAD_EVOLVE_CROSSINGS:
                continue
            remaining = self.trails[key]["wear"] - rate
            if remaining <= 0:
                del self.trails[key]
            else:
                self.trails[key]["wear"] = remaining

    def nearest_water(
        self, x: int, y: int, kinds: tuple[str, ...] = ("river", "ocean")
    ) -> tuple[int, int] | None:
        """Full-grid search for the closest tile among `kinds` by true Euclidean
        distance. A one-time fact supplied to a tribe's leadership election (see
        leadership.py) -- legitimate map knowledge for the simulation to hand over, the
        way a game master would tell players what's nearby, not a live gameplay check
        run every tick. Deliberately a plain scan rather than an outward ring search:
        an early version of the latter stopped at the first ring containing any match,
        which can be farther in true Euclidean distance than a match in a nominally
        "later" ring along a shallower angle -- caught by checking against a brute-force
        reference before trusting it. Grid is only 100x100 and this runs once per tribe,
        so the simple, obviously-correct version costs nothing that matters.

        `kinds` defaults to both river and ocean, but a caller specifically after
        drinkable fresh water should pass `("river",)` -- seawater doesn't quench
        thirst, so treating a coastal tribe as already having solved its water problem
        would be wrong. What else the ocean might be good for (fishing, salt, a raft
        eventually) is left entirely open, not decided here."""
        if biome_at(x, y) in kinds:
            return (x, y)
        best, best_dist = None, None
        for cx in range(self.grid_size):
            for cy in range(self.grid_size):
                if biome_at(cx, cy) not in kinds:
                    continue
                dist = (cx - x) ** 2 + (cy - y) ** 2
                if best_dist is None or dist < best_dist:
                    best, best_dist = (cx, cy), dist
        return best
