"""The world's authored layers, loaded from world/world_layers.json (2026-10-08).

Why this exists (see docs/WORLD-LAYERS.md): reshaping the river took far more than the edit, because its tiles lived in a generated Python module and a hand-mirrored copy
inside the frontend. The river, lake, field and spawn points are now one document, edited on its own:

    metadata   hidden tags (where tribes may start)
    terrain    the river, the lake and the field, laid over the computed base terrain

validate() holds the rules the layers must keep together (spawn points on usable ground, one connected body of water, ...). It runs in the tests and in
scripts/world_layers_report.py, not at import, so a bad edit fails loudly where you are working and never takes down a running game. The frontend is not sent this document:
it is sent the finished terrain grid (world.terrain_grid), so it has nothing to keep in step with.
"""
import json
from pathlib import Path

LAYERS_PATH = Path(__file__).resolve().parent.parent / "world" / "world_layers.json"

# What the loaded document holds, as module attributes so the hot paths (biome_at runs thousands of times a tick) cost one lookup. Reassigned by reload().
RIVER_TILES: frozenset = frozenset()
LAKE_TILES: frozenset = frozenset()
FIELD_TILES: frozenset = frozenset()
SPAWN_POINTS: tuple = ()
DOCUMENT: dict = {}

def _tiles(items) -> frozenset:
    return frozenset((int(x), int(y)) for x, y in items)


def load(path: Path | None = None) -> dict:
    return json.loads((path or LAYERS_PATH).read_text(encoding="utf-8"))


def _apply(doc: dict) -> None:
    global RIVER_TILES, LAKE_TILES, FIELD_TILES, SPAWN_POINTS, DOCUMENT
    terrain = doc["layers"]["terrain"]
    RIVER_TILES, LAKE_TILES, FIELD_TILES = _tiles(terrain["river"]), _tiles(terrain["lake"]), _tiles(terrain["field"])
    SPAWN_POINTS = tuple((int(x), int(y)) for x, y in doc["layers"]["metadata"]["spawn_points"])
    DOCUMENT = doc


def validate(doc: dict | None = None, biome_fn=None) -> list[str]:
    """The rules the layers must keep together. Returns a list of problems (empty when the document is sound). `biome_fn(x, y)` is world.biome_at for the checks that need
    the computed base terrain; they are skipped without it."""
    doc = doc or DOCUMENT
    size = doc.get("size", 100)
    terrain, meta = doc["layers"]["terrain"], doc["layers"]["metadata"]
    river, lake, field = _tiles(terrain["river"]), _tiles(terrain["lake"]), _tiles(terrain["field"])
    spawns = [tuple(p) for p in meta["spawn_points"]]
    problems: list[str] = []

    for name, tiles in (("river", river), ("lake", lake), ("field", field), ("spawn_points", set(spawns))):
        out = sorted(t for t in tiles if not (0 <= t[0] < size and 0 <= t[1] < size))
        if out:
            problems.append(f"{name}: {len(out)} tile(s) outside the {size}x{size} map, e.g. {out[0]}")
    if river & lake:
        problems.append(f"river and lake share {len(river & lake)} tile(s); a tile is one or the other, e.g. {sorted(river & lake)[0]}")
    if field & (river | lake):
        problems.append(f"field overlaps water on {len(field & (river | lake))} tile(s), e.g. {sorted(field & (river | lake))[0]}")

    water = river | lake
    if water:
        seen, stack = set(), [next(iter(sorted(water)))]
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend((p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (p[0] + dx, p[1] + dy) in water and (p[0] + dx, p[1] + dy) not in seen)
        if len(seen) != len(water):
            problems.append(f"the river and lake are not one connected body of water: {len(water) - len(seen)} tile(s) are cut off from the rest")

    if biome_fn is not None:
        # a river tile with open ocean directly east is where the frontend draws the waterfall, so a river that never reaches the sea has none
        if river and not any(biome_fn(x + 1, y) == "ocean" for x, y in river):
            problems.append("the river never reaches the ocean (no river tile has ocean directly to its east)")
        usable = {"river"}  # a tribe may start in the river (one does); never in the sea, on cliffs or shoals, or in the volcano
        for p in spawns:
            b = biome_fn(*p)
            if b in ("ocean", "cliffs", "shoals", "volcano") or (b == "lake"):
                problems.append(f"spawn point {p} is on {b}, not usable ground")
            elif b in usable:
                pass
    return problems


_apply(load())
