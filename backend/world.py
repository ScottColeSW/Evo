import numpy as np

from . import config

BIOME_LABELS = {
    "forest": "Whispering Wilds",
    "mountains": "Crags of Oros",
    "river": "Serpent's Vein",
    "plains": "Sunken Basin",
}


def biome_at(x: int, y: int) -> str:
    if x > 70 or y < 30:
        return "forest"
    if x < 25 and y < 60:
        return "mountains"
    if abs(x - y) < 4:
        return "river"
    return "plains"


class Landscape:
    """Tracks terrain, built structures, and an 'ancestral ghost' bias matrix.

    The ghost matrix is not a game mechanic the tribes see directly — it's radiated
    outward from triumphant or traumatic events and quietly injected into future
    prompts as an unexplained cultural instinct tied to that coordinate.
    """

    def __init__(self, grid_size: int = config.GRID_SIZE):
        self.grid_size = grid_size
        self.constructions: dict[tuple[int, int], dict] = {}
        self.ghost_matrix = np.zeros((grid_size, grid_size), dtype=np.float32)

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

    def record_event(self, x: int, y: int, valence: float, radius: int = 5) -> None:
        x0, x1 = max(0, x - radius), min(self.grid_size, x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.grid_size, y + radius + 1)
        for gx in range(x0, x1):
            for gy in range(y0, y1):
                dist = ((gx - x) ** 2 + (gy - y) ** 2) ** 0.5
                if dist <= radius:
                    self.ghost_matrix[gx, gy] += valence * (1 - dist / radius)

    def ancestral_bias(self, x: int, y: int) -> str:
        score = float(self.ghost_matrix[x, y])
        if score <= -0.4:
            return (
                "Your ancestors suffered near this ground. An unexplained dread "
                "urges caution, fortification, or retreat."
            )
        if score >= 0.4:
            return (
                "This ground carries ancestral triumph. You feel emboldened to "
                "settle, gather, and defend it."
            )
        return ""
