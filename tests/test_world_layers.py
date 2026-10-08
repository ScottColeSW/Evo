"""2026-10-08: the world's authored layers (world/world_layers.json) are the one source for the river, lake, field and spawn points; derived data declares what it depends
on; the layers keep their rules together; the frontend is served the same document (docs/WORLD-LAYERS.md)."""
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
    assert set(doc["layers"]) == set(world_layers.LAYER_NAMES)
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


def test_derived_data_declares_its_layers_and_is_cleared_when_they_change(tmp_path):
    names = dict(world_layers.derived_registry())
    assert names["_inland_map"] == ("terrain",) and set(names["site_seed_points"]) == {"terrain", "metadata"}
    world.site_seed_points("lumber", 100)
    assert world.site_seed_points.cache_info().currsize > 0
    # an unchanged reload clears nothing; a changed terrain layer clears what depends on it and the new tile is live
    assert world_layers.reload() == []
    assert world.site_seed_points.cache_info().currsize > 0
    doc = _doc(); doc["layers"]["terrain"]["field"] = doc["layers"]["terrain"]["field"][:-1]
    path = tmp_path / "w.json"; path.write_text(json.dumps(doc), encoding="utf-8")
    original = world_layers.LAYERS_PATH
    try:
        assert world_layers.reload(path) == ["terrain"]
        assert world.site_seed_points.cache_info().currsize == 0 and len(world_layers.FIELD_TILES) == len(doc["layers"]["terrain"]["field"])
    finally:
        assert world_layers.reload(original) == ["terrain"]  # put the real document back
    assert world_layers.validate(biome_fn=world.biome_at) == []


def test_the_page_is_served_the_same_document_with_no_second_copy():
    html = render_index()
    assert "/*__WORLD_LAYERS__*/null" not in html and "GENERATED HYDROLOGY DATA" not in html
    marker = "const WORLD_LAYERS = "
    start = html.index(marker) + len(marker)
    injected = json.loads(html[start:html.index(";\n", start)])
    assert injected == world_layers.DOCUMENT

    async def fetch():
        async with TestClient(TestServer(create_app())) as client:
            r = await client.get("/")
            return r.status, r.headers.get("Cache-Control"), await r.text()

    status, cache, text = asyncio.run(fetch())
    assert status == 200 and cache == "no-cache" and '"layers":{"metadata"' in text
