import math

BIOME_LABELS = {
    "forest": "Whispering Wilds",
    "mountains": "Crags of Oros",
    "river": "Serpent's Vein",
    "plains": "Sunken Basin",
    "ocean": "The Boundless Deep",
}

# Earth-like hydrology: the river originates in the mountains (west) and winds its way
# down through plains and forest to a coastline (east), rather than being an arbitrary
# diagonal band unrelated to anything else on the map.
OCEAN_X_START = 90
MOUNTAIN_X_END = 30
MOUNTAIN_Y_END = 35
RIVER_SOURCE_X = 15
RIVER_HALF_WIDTH = 3


def _river_center_y(x: int) -> float:
    span = OCEAN_X_START - RIVER_SOURCE_X
    progress = max(0.0, min(1.0, (x - RIVER_SOURCE_X) / span))
    drift = 18 + progress * 50  # highlands (~y=18-24) down to the coast (~y=67-73)
    meander = 6 * math.sin(x * 0.07)
    return drift + meander


def _is_river(x: int, y: int) -> bool:
    if x < RIVER_SOURCE_X or x >= OCEAN_X_START:
        return False
    return abs(y - _river_center_y(x)) <= RIVER_HALF_WIDTH


def biome_at(x: int, y: int) -> str:
    if x >= OCEAN_X_START:
        return "ocean"
    if _is_river(x, y):
        return "river"
    if x < MOUNTAIN_X_END and y < MOUNTAIN_Y_END:
        return "mountains"
    if y < 18 or x >= 70:
        return "forest"
    return "plains"


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

    def biome(self, x: int, y: int) -> str:
        return biome_at(x, y)

    def nearby_structures(self, x: int, y: int, radius: int = 6) -> list[dict]:
        out = []
        for (sx, sy), info in self.constructions.items():
            if abs(sx - x) <= radius and abs(sy - y) <= radius:
                out.append({"x": sx, "y": sy, **info})
        return out

    def add_construction(self, x: int, y: int, kind: str, cycle: int) -> None:
        self.constructions[(x, y)] = {"type": kind, "cycle": cycle}

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
