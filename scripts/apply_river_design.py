"""Applies the traced river (scripts/river_design.json) to the world document, world/world_layers.json.

2026-10-08: the owner drew a new river on a screenshot (a course from the northwest mountains to a junction, a west arm into the lake and an east arm to the sea, with a
field of plain land where the old course ran); it was traced into tiles and saved as scripts/river_design.json. This turns that design into the document's terrain layer.
Since the layered world (docs/WORLD-LAYERS.md) the document is the source and can simply be edited; this script is how the traced design got in, and can redo it:

    python scripts/apply_river_design.py

What it does beyond copying the traced tiles:
  - the river that overlaps the lake body becomes lake (a lake has no drowning hazard);
  - the mouth is extended east through the coastal cliffs/shoals to the first ocean tile, so the river meets the sea and the frontend's waterfall (a river tile with ocean
    directly east) draws, as the old river's mouth did (world.biome_at checks river before the coast band for the same reason);
  - the field tiles (the owner's yellow patch) become the field layer, which both biome functions turn into plains.
It leaves the metadata layer (spawn points) as it is, and validates the result before writing.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import world_layers  # noqa: E402
from backend.world import _coast_boundary_x, biome_at  # noqa: E402  (the coast functions do not read the layers)

DESIGN = ROOT / "scripts" / "river_design.json"
GRID = 100


def build():
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    river = {tuple(p) for p in design["river"]}
    lake = {tuple(p) for p in design["lake"]}
    field = {tuple(p) for p in design["field"]}
    river -= lake
    # extend the mouth east to the sea: every row of the easternmost columns runs on to the last tile before the ocean
    east = max(x for x, _ in river)
    rows = sorted({y for x, y in river if x >= east - 1})
    for y in rows:
        x = east
        while x + 1 < _coast_boundary_x(y) and x + 1 < GRID:
            x += 1
            river.add((x, y))
    field -= river | lake
    return frozenset(river), frozenset(lake), frozenset(field)


def dump(obj, indent=0):
    """Compact JSON, one line per tile list, so the document diffs and edits sensibly."""
    pad = "  " * indent
    if isinstance(obj, dict):
        items = [f'{pad}  {json.dumps(k)}: {dump(v, indent + 1).lstrip()}' for k, v in obj.items()]
        return pad + "{\n" + ",\n".join(items) + "\n" + pad + "}"
    return pad + json.dumps(obj, separators=(",", ":"))


if __name__ == "__main__":
    river, lake, field = build()
    doc = world_layers.load()
    doc["layers"]["terrain"]["river"] = sorted(map(list, river))
    doc["layers"]["terrain"]["lake"] = sorted(map(list, lake))
    doc["layers"]["terrain"]["field"] = sorted(map(list, field))
    problems = world_layers.validate(doc, biome_fn=biome_at)
    if problems:
        print("not written; the design breaks these rules:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    world_layers.LAYERS_PATH.write_text(dump(doc) + "\n", encoding="utf-8")
    print(f"world/world_layers.json written: river {len(river)} tiles, lake {len(lake)}, field {len(field)}")
