"""2026-10-06 (the owner: "I think it is probably lacking in game smarts"): where resource sites go. Groves in forest, quarries and mines on mountains, hunting grounds on forest and
the plains beside it, desert barren for wood and game; woods and mountain ranges cluster, a mine stands beside a quarry; a settling tribe is guaranteed a grove, a hunting
ground and a stone-rich site within reach; a respawned site obeys the same ground rules."""
import math

from backend.simulation import Tribe
from backend.world import biome_at, site_affinity, site_seed_points
from tests.test_actions import _bare_simulation, _settle

G = 100


def _share_on(seed_type, biomes):
    points = site_seed_points(seed_type, G)
    return sum(1 for x, y in points if biome_at(x, y) in biomes) / len(points)


def test_sites_stand_on_the_ground_they_are_for():
    assert _share_on("lumber", {"forest"}) >= 0.7          # was 28%
    assert _share_on("wildlife", {"forest", "plains"}) >= 0.9
    assert _share_on("quarry", {"mountains"}) >= 0.5       # was 6%
    assert _share_on("mine", {"mountains"}) >= 0.4         # was 6%
    for seed_type in ("lumber", "wildlife"):
        assert _share_on(seed_type, {"desert", "mountains"}) <= 0.05, seed_type  # barren for wood and game (a spawn point's guaranteed site may be the exception)


def test_every_site_outside_a_fair_start_suits_its_ground():
    for seed_type in ("lumber", "wildlife", "quarry", "mine"):
        unsuited = [p for p in site_seed_points(seed_type, G) if site_affinity(seed_type, p[0], p[1]) <= 0]
        assert len(unsuited) <= 4, (seed_type, unsuited)  # only a spawn point's guaranteed site, where nothing suits within reach


def test_groves_and_hunting_grounds_come_in_clusters():
    for seed_type, minimum in (("lumber", 0.4), ("wildlife", 0.3)):
        points = site_seed_points(seed_type, G)
        clustered = sum(1 for i, (x, y) in enumerate(points) if any(j != i and math.hypot(x - a, y - b) <= 7 for j, (a, b) in enumerate(points)))
        assert clustered / len(points) >= minimum, seed_type  # an even scatter gave 0 to a few percent


def test_a_mine_usually_stands_beside_a_quarry():
    quarries, mines = site_seed_points("quarry", G), site_seed_points("mine", G)
    paired = sum(1 for x, y in mines if any(math.hypot(x - a, y - b) <= 7 for a, b in quarries))
    assert paired / len(mines) >= 0.8


def test_placement_is_deterministic():
    assert site_seed_points("lumber", G) == site_seed_points("lumber", G)


def test_affinity_follows_the_yield_table():
    from backend.actions import BIOME_YIELD_MULTIPLIER

    forest = next((x, y) for x in range(G) for y in range(G) if biome_at(x, y) == "forest" and site_affinity("lumber", x, y) > 0)
    assert site_affinity("lumber", *forest) == BIOME_YIELD_MULTIPLIER["wood"]["forest"]
    desert = next((x, y) for x in range(G) for y in range(G) if biome_at(x, y) == "desert")
    assert site_affinity("lumber", *desert) == 0 and site_affinity("wildlife", *desert) == 0
    ocean = next((x, y) for x in range(G) for y in range(G) if biome_at(x, y) == "ocean")
    assert all(site_affinity(t, *ocean) == 0 for t in ("lumber", "wildlife", "quarry", "mine"))


def _settled(sim_x=50, sim_y=50):
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", sim_x, sim_y, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    _settle(sim, tribe)
    return sim, tribe


def test_a_settling_tribe_is_guaranteed_a_grove_a_hunting_ground_and_a_stone_site_within_reach():
    sim, tribe = _settled()
    # take every real site of those types out of the world, then ask for the guarantee again: it has to supply them
    for t in ("lumber", "wildlife", "quarry"):
        sim.world.exhausted_sites |= {(t, x, y) for x, y in site_seed_points(t, G)}
        sim.world.respawned_sites[t] = []
        assert sim.homeland_nodes(tribe, t) == []
    sim._ensure_homeland(tribe)
    for t in ("lumber", "wildlife", "quarry"):
        nodes = sim.homeland_nodes(tribe, t)
        assert len(nodes) == 1, t
        assert math.hypot(nodes[0][0] - tribe.territory_center[0], nodes[0][1] - tribe.territory_center[1]) <= tribe.territory_radius


def test_the_guarantee_adds_nothing_where_the_map_already_has_the_sites():
    sim, tribe = _settled()
    before = {t: list(sim.world.respawned_sites.get(t, [])) for t in ("lumber", "wildlife", "quarry")}
    present = {t for t in before if sim.homeland_nodes(tribe, t)}
    sim._ensure_homeland(tribe)
    for t in present:
        assert sim.world.respawned_sites.get(t, []) == before[t], t


def test_a_respawned_site_stands_on_ground_that_suits_it():
    import random

    sim, tribe = _settled()
    for seed in range(8):
        random.seed(seed)
        for node_type in ("lumber", "wildlife", "quarry"):
            sim.world.respawned_sites[node_type] = []
            placed = sim._respawn_node(node_type, tribe.territory_center[0], tribe.territory_center[1])
            assert placed is not None, (node_type, seed)
            assert sim.world.site_affinity(node_type, *placed) > 0, (node_type, seed, placed)
            assert not sim._inside_any_territory(*placed)


def test_a_sawmill_before_any_grove_was_scouted_still_counts_as_wood_mastered_once_one_is():
    from backend.simulation import _is_wood_secure

    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    tribe.sawmill_built = True
    assert _is_wood_secure(tribe) is False  # no grove known yet
    tribe.lumber_sites.append((70, 70))
    assert _is_wood_secure(tribe) is True   # the sawmill was built first; the scouted grove still counts
