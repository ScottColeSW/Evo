"""Prints what the world document holds, what is derived from it, and whether its layers keep their rules:

    python scripts/world_layers_report.py

Run it after editing world/world_layers.json (by hand, or with a map editor that writes the same shape). Exit status 1 when a rule is broken."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import world, world_layers  # noqa: E402


def main() -> int:
    doc = world_layers.DOCUMENT
    terrain, meta = doc["layers"]["terrain"], doc["layers"]["metadata"]
    print(f"world document: {world_layers.LAYERS_PATH}  (format {doc['format']}, {doc['size']}x{doc['size']})")
    print("layers:")
    print(f"  metadata   spawn points {meta['spawn_points']}")
    print(f"  terrain    river {len(terrain['river'])} tiles, lake {len(terrain['lake'])}, field {len(terrain['field'])}")
    print("derived from them (cleared when a layer they depend on changes):")
    for name, deps in world_layers.derived_registry():
        print(f"  {name:22s} depends on {', '.join(deps)}")
    problems = world_layers.validate(biome_fn=world.biome_at)
    if problems:
        print("rules broken:")
        for p in problems:
            print("  -", p)
        return 1
    print("rules: all kept (tiles in range, river and lake do not overlap, the field is dry land, one connected body of water, the river reaches the sea, spawn points on usable ground)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
