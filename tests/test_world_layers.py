"""2026-10-08: the world's authored layers (world/world_layers.json) are the one source for the river, lake, field and spawn points; the layers keep their rules together;
the frontend is sent the finished terrain grid and carries no terrain logic of its own (docs/WORLD-LAYERS.md)."""
import asyncio
import copy
import json

from aiohttp.test_utils import TestClient, TestServer

from backend import world, world_layers
from backend.app import create_app, render_index
from backend.simulation import SPAWN_POINTS


def test_the_document_holds_the_layers_the_game_runs_on():
    doc = world_layers.load()
    assert doc["format"] == 1 and doc["size"] == 100
    assert set(doc["layers"]) == {"metadata", "terrain"}
    assert len(world_layers.RIVER_TILES) == len(doc["layers"]["terrain"]["river"]) > 300
    assert world.river_tiles() == world_layers.RIVER_TILES and world.lake_tiles() == world_layers.LAKE_TILES and world.field_tiles() == world_layers.FIELD_TILES
    assert tuple(SPAWN_POINTS) == world_layers.SPAWN_POINTS and len(SPAWN_POINTS) == 4


def test_the_real_document_keeps_every_rule():
    assert world_layers.validate(biome_fn=world.biome_at) == []


def _doc():
    return copy.deepcopy(world_layers.DOCUMENT)


def test_validation_catches_each_kind_of_broken_edit():
    d = _doc(); d["layers"]["terrain"]["river"].append([150, 3])
    assert any("outside the 100x100 map" in p for p in world_layers.validate(d))
    d = _doc(); d["layers"]["terrain"]["lake"].append(d["layers"]["terrain"]["river"][0])
    assert any("share" in p for p in world_layers.validate(d))
    d = _doc(); d["layers"]["terrain"]["field"].append(d["layers"]["terrain"]["river"][0])
    assert any("field overlaps water" in p for p in world_layers.validate(d))
    d = _doc(); d["layers"]["terrain"]["river"].append([2, 2])  # a stranded pool
    assert any("not one connected body" in p for p in world_layers.validate(d))
    d = _doc(); d["layers"]["metadata"]["spawn_points"][0] = [0, 0]  # open ocean
    assert any("spawn point (0, 0)" in p or "spawn point (0, 0)" in p.replace("[0, 0]", "(0, 0)") for p in world_layers.validate(d, biome_fn=world.biome_at))
    d = _doc(); d["layers"]["terrain"]["river"] = [t for t in d["layers"]["terrain"]["river"] if t[0] < 60]  # never reaches the sea
    assert any("never reaches the ocean" in p for p in world_layers.validate(d, biome_fn=world.biome_at))


def test_a_tribe_may_start_in_the_river_but_not_in_the_sea():
    d = _doc(); d["layers"]["metadata"]["spawn_points"] = [[40, 37]]  # a river tile in the real document
    assert world.biome_at(40, 37) == "river" and world_layers.validate(d, biome_fn=world.biome_at) == []


def test_the_terrain_grid_is_exactly_what_biome_at_says():
    grid = world.terrain_grid()
    assert len(grid) == 100 and all(len(row) == 100 for row in grid)
    letters = {v: k for k, v in world.BIOME_LETTERS.items()}
    assert all(letters[grid[y][x]] == world.biome_at(x, y) for y in range(100) for x in range(100))


def test_the_page_is_sent_the_grid_and_carries_no_terrain_logic_of_its_own():
    html = render_index()
    assert "/*__TERRAIN_GRID__*/null" not in html
    marker = "const TERRAIN_GRID = "
    start = html.index(marker) + len(marker)
    assert tuple(json.loads(html[start:html.index(";\n", start)])) == world.terrain_grid()
    # the mirrored copy is gone: none of the functions that used to recompute the terrain in the browser is left
    for name in ("coastBoundaryX", "westCoastBoundary", "mountainXBoundary", "forestNorthBoundary", "desertNorthBoundary", "isHeadlandLike", "HYDROLOGY_RIVER_TILES"):
        assert name not in html, name


def test_the_page_script_still_parses():
    import re
    import shutil
    import subprocess
    import tempfile

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", render_index(), re.S)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write("\n".join(scripts))
    assert subprocess.run(["node", "--check", f.name], capture_output=True).returncode == 0


def test_the_served_page_is_not_cached_and_carries_the_grid():
    async def fetch():
        async with TestClient(TestServer(create_app())) as client:
            r = await client.get("/")
            return r.status, r.headers.get("Cache-Control"), await r.text()

    status, cache, text = asyncio.run(fetch())
    assert status == 200 and cache == "no-cache" and world.terrain_grid()[0] in text
