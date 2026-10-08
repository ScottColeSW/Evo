# World layers and the terrain grid

Written 2026-10-08, after reshaping the river showed what was weak. The edit itself was a few lines. The cost was in finding out what depended on the river: a generated
Python module, a hand-mirrored copy of the terrain logic inside the frontend, tests that pinned the old shape, a spawn point, and a latent placement bug. A browser also
kept an old copy of the page.

## What is in place

**One document for the authored map data.** `world/world_layers.json` holds two layers:

| Layer | Holds |
|---|---|
| metadata | hidden tags: the spawn points |
| terrain | the river, the lake and the field (plain land), laid over the base terrain that `backend/world.py` still computes (coast, mountains, forest, desert) |

Tile lists are `[x, y]`. Edit the file, run `python scripts/world_layers_report.py`, restart the server. `backend/world_layers.py` loads it and holds `validate()`:
tiles inside the map, river and lake do not overlap, the field is dry, the water is one connected body, the river reaches the sea, spawn points are on usable ground (a
tribe may start in the river). The rules run in the tests and in the report script, never at import, so a bad edit fails where you are working and cannot take down a running
game. The traced river came in through `scripts/apply_river_design.py` from `scripts/river_design.json`; `scripts/generate_hydrology.py` is superseded and refuses to run.

**The frontend is sent the finished terrain, not the recipe.** `world.terrain_grid()` is the 100 x 100 result of `biome_at`, one letter per tile. The server fills it into
`frontend/index.html` on every request (`render_index` in `backend/app.py`), and the page only draws it. The page used to carry a hand-mirrored copy of the backend's coast,
mountain, forest, desert, river and lake logic; that copy is deleted, so there is nothing left to drift. Pages are sent with `Cache-Control: no-cache`.

## What was tried and removed

A first version also tracked, for each cached derived function (the inland map, the resource-site seeds), which layers it depended on, and cleared the right caches when a
layer was reloaded. Nothing in the running game reloads a layer (a map edit needs a restart), so it only helped in tests and added vocabulary. It was removed. "Layers" for
resources, hazards and builds were never built: they are derived from the terrain or are runtime state, and naming them layers adds ceremony.

## Not done

- A map editor that writes this shape (a Tiled-style tile layer export, say) would let the river be painted directly. The document is plain JSON, so a small converter is
  all it needs. Another route is to bake the whole biome grid to an image once and paint on that; the cost is losing the generated coastline and boundary formulas.
- More rules in `validate()` as they are found: no site within a tile of water, different site types at least 5 apart, and, on the action side, a property test that a menu
  lock never removes its own exit.
