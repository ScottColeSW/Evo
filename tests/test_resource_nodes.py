"""2026-10-06: resource nodes (docs/RESOURCE-NODES-DESIGN.md). A gather draws from a node within reach of the tribe's territory (radius + NODE_REACH_BEYOND_TERRITORY),
for NODE_USES uses shared by every tribe; the last use spends it, a replacement appears elsewhere undiscovered, and the tribe is told. Without a node in reach the gather
harvests the tile exactly as before."""
import math
from unittest import mock

from backend import config
from backend.actions import ACTION_REGISTRY
from backend.simulation import Tribe
from backend.world import find_nearby_site
from tests.test_actions import _NO_TARGET, _bare_simulation, _settle


def _sim_with(node_type, points):
    """A sim with one settled tribe and only these sites of `node_type` in the world (every real seed point removed so the test controls what is in reach)."""
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    _settle(sim, tribe)
    sim.tribes = {"tribe_0": tribe}
    cx, cy = tribe.territory_center
    sites = tuple((cx + dx, cy + dy) for dx, dy in points)
    sim.world.exhausted_sites = {(t, x, y) for t in ("lumber", "quarry", "wildlife") for x, y in __import__("backend.world", fromlist=["x"]).site_seed_points(t, sim.world.grid_size)}
    sim.world.respawned_sites = {node_type: list(sites)}
    return sim, tribe, sites


def _reach(tribe):
    return tribe.territory_radius + config.NODE_REACH_BEYOND_TERRITORY


def test_a_gather_draws_from_a_homeland_node_ignoring_the_tile_and_counts_the_use():
    sim, tribe, sites = _sim_with("lumber", [(3, 0)])
    sim.world.deplete("wood", tribe.x, tribe.y, 0.8, 0.8)  # the tile is worked out; the node is not
    with mock.patch.object(sim.world, "biome", return_value="forest"):
        result = ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.wood == 50 + 10  # the starting 50 plus the base 10 at labor 1.0 and a forest node: nothing taken off for the depleted tile
    assert f"timber grove at ({sites[0][0]},{sites[0][1]})" in result and "2 uses left" in result
    assert sim.world.site_uses[("lumber", *sites[0])] == 1


def test_the_third_use_spends_the_node_tells_the_tribe_and_places_an_undiscovered_replacement():
    sim, tribe, sites = _sim_with("lumber", [(3, 0)])
    node = sites[0]
    with mock.patch.object(sim.world, "biome", return_value="forest"):
        for _ in range(config.NODE_USES - 1):
            ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "plains", _NO_TARGET)
        assert ("lumber", *node) not in sim.world.exhausted_sites
        result = ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "plains", _NO_TARGET)
    assert ("lumber", *node) in sim.world.exhausted_sites and "gives out for good" in result
    assert any("gives out for good" in str(line) for line in tribe.history)
    replacement = [p for p in sim.world.respawned_sites["lumber"] if p != node]
    assert len(replacement) == 1
    rx, ry = replacement[0]
    assert math.hypot(rx - node[0], ry - node[1]) >= config.RAIDER_SIGHTING_MIN_OFFSET - 1  # it moved, by the same rule raiders and hunting grounds use
    assert not sim._inside_any_territory(rx, ry)  # wild ground, not inside anyone's territory
    assert (rx, ry) not in set(tribe.lumber_sites)  # nobody knows it: a scout has to find it
    assert sim.homeland_nodes(tribe, "lumber") == []  # and it is not within reach, so the homeland is now spent


def test_stone_and_game_draw_from_their_own_nodes():
    for action, node_type, resource, nominal in (("GATHER_STONE", "quarry", "stone", 10), ("HUNT_DEER", "wildlife", "food", 15)):
        sim, tribe, sites = _sim_with(node_type, [(2, 2)])
        start = getattr(tribe, resource)
        with mock.patch.object(sim.world, "biome", return_value={"quarry": "mountains", "wildlife": "forest"}[node_type]), \
             mock.patch("backend.actions.random.random", return_value=0.99):  # no wolf pack
            result = ACTION_REGISTRY[action](sim, tribe, "plains", _NO_TARGET)
        assert getattr(tribe, resource) - start == nominal, action
        assert sim.world.site_uses[(node_type, *sites[0])] == 1 and "2 uses left" in result


def test_a_node_pays_by_its_own_ground():
    sim, tribe, sites = _sim_with("lumber", [(3, 0)])
    with mock.patch.object(sim.world, "biome", return_value="plains"):  # a plains-edge grove: BIOME_YIELD_MULTIPLIER wood plains is 0.4
        ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "plains", _NO_TARGET)
    assert tribe.wood == 50 + 4


def test_no_node_in_reach_means_the_tile_as_before():
    sim, tribe, _ = _sim_with("lumber", [])
    cx, cy = tribe.territory_center
    sim.world.respawned_sites = {"lumber": [(cx + _reach(tribe) + 2, cy)]}  # just beyond reach (radius + 3)
    assert sim.homeland_nodes(tribe, "lumber") == []
    before = tribe.wood
    ACTION_REGISTRY["GATHER_WOOD"](sim, tribe, "forest", _NO_TARGET)
    assert tribe.wood > before and not sim.world.site_uses  # the tile paid, no node was touched
    sim.world.respawned_sites = {"lumber": [(cx + _reach(tribe) - 1, cy)]}  # just inside reach
    assert len(sim.homeland_nodes(tribe, "lumber")) == 1


def test_a_tribe_with_no_territory_yet_has_no_homeland():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    assert tribe.territory_center is None and sim.homeland_nodes(tribe, "lumber") == []


def test_the_use_count_is_shared_between_tribes():
    sim, tribe, sites = _sim_with("lumber", [(3, 0)])
    rival = Tribe("tribe_1", "Rival", "qwen2.5:3b", 52, 52, "#fb923c")
    rival.territory_center, rival.territory_radius = tribe.territory_center, tribe.territory_radius  # the same ground
    sim.tribes["tribe_1"] = rival
    sim.use_node(tribe, "lumber", *sites[0])
    sim.use_node(rival, "lumber", *sites[0])
    note = sim.use_node(tribe, "lumber", *sites[0])
    assert "gives out for good" in note and ("lumber", *sites[0]) in sim.world.exhausted_sites


def test_a_spent_site_is_not_rediscovered_and_its_replacement_can_be_found():
    sim, tribe, sites = _sim_with("lumber", [(3, 0)])
    sx, sy = sites[0]
    far = (95, 95)
    sim.world.respawned_sites = {"lumber": [(sx, sy), far]}
    sim.world.exhausted_sites.add(("lumber", sx, sy))
    got = find_nearby_site("lumber", sx, sy, 100, sim.world.spent_of("lumber"), extra_points=tuple(sim.world.respawned_sites["lumber"]))
    assert got is None  # the spent site is skipped
    found = find_nearby_site("lumber", far[0], far[1], 100, sim.world.spent_of("lumber"), extra_points=tuple(sim.world.respawned_sites["lumber"]))
    assert found == far  # a respawned point can be discovered like any other


def test_a_tribe_still_listing_a_spent_site_is_told_it_is_worked_out_and_the_entry_goes():
    sim, tribe, sites = _sim_with("lumber", [(3, 0)])
    other = Tribe("tribe_1", "Rival", "qwen2.5:3b", 10, 10, "#fb923c")
    sim.tribes["tribe_1"] = other
    other.lumber_sites = [sites[0], (90, 90)]
    other.wildlife_sites = [{"x": 70, "y": 70, "type": "Deer Stand"}]
    for _ in range(config.NODE_USES):
        sim.use_node(tribe, "lumber", *sites[0])
    sim._prune_spent_sites(other)
    assert sites[0] not in other.lumber_sites and (90, 90) in other.lumber_sites and other.wildlife_sites
    assert any("is worked out" in str(line) for line in other.history)
