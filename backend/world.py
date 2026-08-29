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
    """Tracks terrain and built structures. Emotional/ancestral bias lives separately
    in ancestral_matrix.AncestralTraumaMatrix — terrain and memory are different axes."""

    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.constructions: dict[tuple[int, int], dict] = {}

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
