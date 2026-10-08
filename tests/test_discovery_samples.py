"""2026-10-08 (the owner: "let's make 'discover' a first 'gather' of anything encountered. They will have to deliver a small amount and mark it."): a scout that marks a site
brings home a small sample of what is there, and that counts as having gathered it once. A live run's two tribes sat 500+ cycles in the monolithic era because the mine's one
manual GATHER_ORE was never chosen; discovering a vein now delivers its ore and sets the flag."""
from unittest import mock

from backend import config
from backend.simulation import AFFORDABILITY_CHECKS, Simulation

PATH = [[57, 40], [45, 38], [30, 36], [17, 34]]  # a route with every kind of site in reach (see tests/test_scout_site_cap.py)


def _sim():
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 57, "y": 40}, {"name": "B", "model": "gemma2:2b", "x": 5, "y": 95}])
    return sim, sim.tribes["tribe_0"]


def _exp(path=PATH):
    return {"kind": "scout", "launched": 5, "pos": list(path[0]), "origin": list(path[0]), "target": list(path[-1]), "day": 2, "phase": "returning", "found": None,
            "terrain_report": None, "food_gathered": 0, "water_gathered": 0, "lead_scout": "Ashgar", "determination": 0.5, "max_days": 3, "path": path}


def _home(sim, tribe):
    exp = _exp()
    tribe.expeditions = [exp]
    with mock.patch.object(sim, "_celebrate_game_discovery"):
        sim._advance_one_expedition(tribe, exp)
    return [str(line) for line in tribe.history if "is home and gives" in str(line)]


def test_a_marked_site_delivers_a_small_sample_and_counts_as_a_first_gather():
    sim, tribe = _sim()
    tribe.wood = tribe.stone = tribe.food = 50
    assert not (tribe.wood_ever_gathered or tribe.stone_ever_gathered or tribe.ore_ever_gathered)
    (line,) = _home(sim, tribe)
    n = config.DISCOVERY_SAMPLE_AMOUNT
    assert (tribe.wood, tribe.stone) == (50 + n, 50 + n) and tribe.food >= 50 + n  # timber, stone and game on this route, each delivered
    assert tribe.wood_ever_gathered and tribe.stone_ever_gathered and tribe.ore_ever_gathered
    assert f"{n} wood from a timber grove at (" in line and f"{n} stone from a stone-rich site at (" in line and f"{n} food from a game-rich site at (" in line


def test_a_vein_delivers_a_little_of_its_own_ore_and_the_report_names_it():
    sim, tribe = _sim()
    (line,) = _home(sim, tribe)
    (name, amount), = tribe.unique_resources.items()
    assert amount == config.DISCOVERY_ORE_SAMPLE and f"{amount} {name} from a vein at (" in line
    assert tribe.mine_sites[0]["resource"] == name


def test_after_discovering_a_vein_the_forge_chain_that_stalled_a_live_run_opens():
    """The mine's yield flows once ore has been gathered, and the forge needs one of the mine's own ore: both now follow from the scout's sample plus building the mine."""
    sim, tribe = _sim()
    _home(sim, tribe)
    site = tribe.mine_sites[0]
    tribe.mine_built, tribe.mine_resource_name, tribe.mine_site = True, site["resource"], (site["x"], site["y"])
    tribe.has_ever_settled = True
    sim._found_territory(tribe)  # the forge needs room for its footprint
    tribe.wood = tribe.stone = 500
    assert tribe.ore_ever_gathered and tribe.unique_resources.get(site["resource"], 0) >= config.FORGE_ITEM_ORE_COST
    with mock.patch.object(sim, "_is_camped", return_value=True):
        before = tribe.unique_resources[site["resource"]]
        sim._advance_mine_yield(tribe)
    assert tribe.unique_resources[site["resource"]] == before + config.MINE_YIELD_PER_CYCLE  # the passive yield now flows with no manual GATHER_ORE
    assert AFFORDABILITY_CHECKS["BUILD_FORGE"](tribe, sim.world) is True


def test_a_full_store_delivers_nothing_but_the_gather_still_counts():
    from backend.actions import _storage_cap

    sim, tribe = _sim()
    tribe.wood = _storage_cap(tribe)
    (line,) = _home(sim, tribe)
    assert tribe.wood == _storage_cap(tribe) and tribe.wood_ever_gathered
    assert "wood from a timber grove" not in line and "a timber grove at (" in line  # named as marked, no sample to report
