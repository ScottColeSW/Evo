import functools
import math
import random

from . import config

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
# a single, small, fixed decorative-but-lethal feature inside the existing
# mountain region, not a wavy zone boundary (this is one place, not an organic
# terrain type that should look different every map). Same circle-test shape as
# the lake above, deliberately simpler than the sine-wave boundaries -- a one-off
# feature reads better as a clean circle than an "organic" wobbly one.
VOLCANO_CENTER = (10, 12)
VOLCANO_RADIUS = 4


def _river_center_y(x: int) -> float:
    span = OCEAN_X_START - RIVER_SOURCE_X
    progress = max(0.0, min(1.0, (x - RIVER_SOURCE_X) / span))
    drift = 18 + progress * 50  # highlands (~y=18-24) down to the coast (~y=67-73)
    meander = 6 * math.sin(x * 0.07)
    return drift + meander


def _is_river(x: int, y: int) -> bool:
    # The river's mouth used to always cut off at the flat OCEAN_X_START regardless of
    # the wavy coastline -- wherever the coast recedes into a bay (its own boundary_x
    # drops below OCEAN_X_START), the river kept extending to the old fixed line
    # anyway, sticking out several tiles past the real coast into open ocean. Clipping
    # to whichever is closer keeps the strip river/highland stretch (mouth nowhere
    # near the coast, boundary_x always >= OCEAN_X_START there) exactly as before, and
    # only pulls the mouth itself in to match the real shoreline.
    if x < RIVER_SOURCE_X or x >= min(OCEAN_X_START, _coast_boundary_x(y)):
        return False
    return abs(y - _river_center_y(x)) <= RIVER_HALF_WIDTH


def _is_lake(x: int, y: int) -> bool:
    lx, ly = LAKE_CENTER
    if math.hypot(x - lx, y - ly) <= LAKE_RADIUS:
        return True
    # Distance from (x, y) to the tributary's line segment, branch point to lake center.
    bx, by = LAKE_TRIBUTARY_BRANCH_X, _river_center_y(LAKE_TRIBUTARY_BRANCH_X)
    dx, dy = lx - bx, ly - by
    length_sq = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((x - bx) * dx + (y - by) * dy) / length_sq))
    proj_x, proj_y = bx + t * dx, by + t * dy
    return math.hypot(x - proj_x, y - proj_y) <= LAKE_TRIBUTARY_HALF_WIDTH


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
    boundary = _coast_boundary_x(y)
    if x >= boundary:
        return "ocean"
    if boundary - x <= COAST_BAND_WIDTH:
        return "cliffs" if _coast_is_headland(y) else "shoals"
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
# This scatters a real, fixed set of site locations across the map ONCE, up front --
# a coarse grid of cells (SITE_SEED_GRID_CELL_SIZE), each with SITE_SEED_FILL_
# PROBABILITY odds of holding exactly one site, placed at a random jittered offset
# within its own cell rather than a fixed grid intersection (the "twist" -- the same
# organic-irregularity idea the coastline/mountain-boundary sine distortion above
# already uses, just via jitter instead of a wave). Sparse because most cells end up
# empty; a matrix because it's still grid-structured, not a purely uniform random
# scatter that could clump by chance. Deterministic per (seed_type, grid_size) via a
# string-seeded RNG -- same "pure function of coordinates" philosophy biome_at
# itself already follows, so this needs no persisted state anywhere.
#
# Explicit steer, matching the earlier discovery-chance fairness fix: seed points are
# placed on ANY buildable biome, not tied to matching terrain (forest for
# lumber/wildlife, mountains for quarry) -- a tribe scouting the "wrong" biome is no
# longer structurally locked out. Mine sites are included in the same sparse system;
# each pre-seeded mine's resource name is read off whatever real biome the point
# itself sits on (world.UNIQUE_RESOURCE_BY_BIOME), preserving the one deliberate
# exception -- ore is still biome-tied, just the location is now a real place, not a
# fresh roll.
SITE_SEED_TYPES = ("lumber", "wildlife", "quarry", "mine")
SITE_SEED_GRID_CELL_SIZE = 18
SITE_SEED_FILL_PROBABILITY = 0.55
SITE_DISCOVERY_RADIUS = 8


@functools.lru_cache(maxsize=None)
def site_seed_points(seed_type: str, grid_size: int) -> tuple[tuple[int, int], ...]:
    rng = random.Random(f"site_seed:{seed_type}:{grid_size}")
    cell = SITE_SEED_GRID_CELL_SIZE
    points = []
    for cell_y in range(0, grid_size, cell):
        for cell_x in range(0, grid_size, cell):
            if rng.random() >= SITE_SEED_FILL_PROBABILITY:
                continue
            x = min(grid_size - 1, cell_x + rng.randint(0, cell - 1))
            y = min(grid_size - 1, cell_y + rng.randint(0, cell - 1))
            if biome_at(x, y) in config.UNBUILDABLE_BIOMES:
                continue
            points.append((x, y))
    return tuple(points)


def find_nearby_site(
    seed_type: str, x: int, y: int, grid_size: int, known: set, radius: int = SITE_DISCOVERY_RADIUS
) -> tuple[int, int] | None:
    """The nearest real, pre-seeded site of this type within `radius` of (x, y) that
    isn't already in `known` -- how a scout's report turns into an actual new
    discovery now, instead of an independent chance roll on their exact tile."""
    best, best_dist = None, None
    for px, py in site_seed_points(seed_type, grid_size):
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
