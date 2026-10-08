# World layers

Written 2026-10-08, after reshaping the river showed what was weak. The edit itself was a few lines. The cost was in finding out what depended on the river: a
generated Python module, a hand-mirrored copy inside the frontend, caches derived from the terrain (the inland map, the resource-site seeds), tests that pinned the
old shape, a spawn point, and a latent placement bug. Nothing declared those links, and a browser kept an old copy of the page.

## The idea

Authored data lives in **layers** in one document, `world/world_layers.json`. Each layer can be edited on its own. Everything else is **derived** from the layers, and
says so. The rule of the weave: a layer reads the ones beneath it, never the reverse, and changes flow upward as invalidation.

| Layer | Holds | Status |
|---|---|---|
| metadata | hidden tags: the spawn points | in the document |
| terrain | the river, the lake, the field, laid over the computed base terrain (coast, mountains, forest, desert) | in the document |
| resources | site seeds, spent and respawned sites | derived from terrain and metadata (`site_seed_points`); the runtime part (spent, respawned) stays in `Landscape` |
| hazards | cliffs, river drowning, hazard markers | derived from terrain |
| builds | what each tribe has built | runtime state, per tribe |
| action | what a tribe chooses | not a layer: the rules that read every layer and change them through events |

The action pipeline is separate on purpose. Its own failure (a menu lock that removed another lock's exit) is checked by a test of the property "a lock never removes
its own exit", not by layering.

## How it works

- `backend/world_layers.py` loads the document. `RIVER_TILES`, `LAKE_TILES`, `FIELD_TILES` and `SPAWN_POINTS` are module attributes, so `biome_at` pays one lookup.
- `@derived("terrain", ...)` is `functools.lru_cache` plus a declaration of which layers the function depends on. `reload()` re-reads the document and clears the caches
  that depend on a layer that changed; `invalidate()` clears all or some. Today `_inland_map`, `_distance_to_biomes`, `_scatter_site_points` and `site_seed_points`
  are declared.
- `validate()` holds the rules the layers keep together: tiles inside the map, river and lake do not overlap, the field is dry, the water is one connected body, the
  river reaches the sea, spawn points are on usable ground (a tribe may start in the river). It runs in the tests and in `scripts/world_layers_report.py`, never at
  import, so a bad edit fails where you are working and cannot take down a running game.
- The server fills a placeholder in `frontend/index.html` with the same document on every request (`render_index` in `backend/app.py`), and sends the page with
  `Cache-Control: no-cache`. There is no second copy to keep in step, and no generated block.

## Editing the map

Edit `world/world_layers.json` (tile lists are `[x, y]`), run `python scripts/world_layers_report.py`, then restart the server. The traced river came in through
`scripts/apply_river_design.py` from `scripts/river_design.json`; that script can redo it. `scripts/generate_hydrology.py` is superseded and refuses to run.

## Not done yet

- The resources, hazards and builds layers are named here but not yet separate documents. Only terrain and metadata are.
- A map editor that writes this shape (a Tiled-style tile layer export, say) would let the river be painted directly. The document is plain JSON, so a small converter is
  all it needs.
- More rules in `validate()` as they are found: no site within a tile of water, different site types at least 5 apart, a lock never removes its own exit.
