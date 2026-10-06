"""2026-10-06: a party's resource-site discovery covers the ground it actually crossed, however the trip ended (it used to run only for a party that
reached its target and surveyed; 40 of 43 scouting reports in a live run came from trips that ended some other way and came home having found nothing)."""
import math
from unittest import mock

from backend.simulation import Tribe, party_ground_points
from backend.world import SITE_DISCOVERY_RADIUS, site_seed_points
from tests.test_actions import _bare_simulation


def test_the_ground_between_two_recorded_positions_is_filled_in():
    points = party_ground_points([[0, 0], [10, 0], [20, 0]])
    assert (0, 0) in points and (20, 0) in points
    xs = sorted(x for x, y in points)
    assert all(b - a <= 4 for a, b in zip(xs, xs[1:])), xs  # no gap wider than 4 tiles
    assert party_ground_points([]) == [] and party_ground_points(None) == []
    assert party_ground_points([[5, 5]]) == [(5, 5)]
    assert len(party_ground_points([[0, 0], [0, 0], [0, 0]])) == 1  # standing still adds nothing


def _sim_and_tribe():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    return sim, tribe


def _isolated_wildlife_site(sim):
    """A wildlife site outside every territory and at least 18 tiles from every other one, so nothing else along its route can be reported in its place
    (a ground point reports one new site per type; with a neighbour close by, a different one can be found first)."""
    seeds = site_seed_points("wildlife", sim.world.grid_size)
    for x, y in seeds:
        if sim._inside_any_territory(x, y):
            continue
        if all(math.hypot(x - a, y - b) >= 18 for a, b in seeds if (a, b) != (x, y)):
            return x, y
    raise AssertionError("no isolated wildlife site outside territory")


def _route_just_missing_a_site(site):
    """Straight past `site` 7 tiles to the side, recorded points 10 apart with the site midway between two of them: every recorded point is about 8.6 tiles away
    (outside the 8-tile discovery radius) while the line itself passes within 7."""
    x, y = site
    return [[x - 15, y + 7], [x - 5, y + 7], [x + 5, y + 7], [x + 15, y + 7]]


def test_a_site_beside_the_route_is_found_even_though_no_recorded_point_is_near_it():
    sim, tribe = _sim_and_tribe()
    site = _isolated_wildlife_site(sim)
    path = _route_just_missing_a_site(site)
    assert all(math.hypot(px - site[0], py - site[1]) > SITE_DISCOVERY_RADIUS for px, py in path)  # the old check at recorded points could not see it
    with mock.patch.object(sim, "_celebrate_game_discovery"):
        sim._discover_along_party_ground(tribe, {"path": path, "kind": "scout"}, "Ashgar")
    assert (site[0], site[1]) in {(s["x"], s["y"]) for s in tribe.wildlife_sites}


def test_a_trip_that_ends_by_finding_water_or_turning_back_still_reports_the_ground_it_crossed():
    """The two ways almost every live trip ended, driven through the real homecoming code."""
    sim, tribe = _sim_and_tribe()
    site = _isolated_wildlife_site(sim)
    for found in ((55, 55), None):  # found water; nothing new found (what a turn-back reports)
        tribe.wildlife_sites = []
        exp = {"pos": [50, 50], "origin": [50, 50], "target": [site[0], site[1]], "day": 2, "phase": "returning", "found": found, "terrain_report": None,
               "food_gathered": 0, "water_gathered": 0, "lead_scout": "Ashgar", "determination": 0.5, "max_days": 3, "kind": "scout",
               "path": _route_just_missing_a_site(site)}
        tribe.expeditions = [exp]
        with mock.patch.object(sim, "_celebrate_game_discovery"):
            sim._advance_one_expedition(tribe, exp)
        assert (site[0], site[1]) in {(s["x"], s["y"]) for s in tribe.wildlife_sites}, found


def test_a_new_timber_grove_or_stone_site_is_reported_to_the_chief():
    sim, tribe = _sim_and_tribe()
    with mock.patch("backend.simulation.find_nearby_site", side_effect=[(60, 60), None, (70, 70), None]):
        sim._discover_sites_along_route(tribe, 60, 60, "Ashgar")
    text = " ".join(str(line) for line in tribe.history)
    assert "timber grove at (60,60)" in text and "stone-rich site at (70,70)" in text


def test_a_site_inside_a_territory_is_still_not_discovered():
    sim, tribe = _sim_and_tribe()
    tribe.territory_center, tribe.territory_radius = (50, 50), 1000  # everything is inside
    sim._discover_along_party_ground(tribe, {"path": [[10, 10], [20, 10], [30, 10]], "kind": "scout"}, "Ashgar")
    assert tribe.lumber_sites == [] and tribe.quarry_sites == [] and tribe.wildlife_sites == [] and tribe.mine_sites == []
