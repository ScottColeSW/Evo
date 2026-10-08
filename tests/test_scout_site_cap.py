"""2026-10-08 (the owner): a scouting trip captures at most one new site of each type, and a later scout is never shown a location the tribe already has. A live run had
scouts reporting up to 7 timber groves or 7 quarries from a single trip."""
from unittest import mock

from backend.simulation import Tribe, party_ground_points
from backend.world import SITE_DISCOVERY_RADIUS, site_seed_points
from tests.test_actions import _bare_simulation

# a long route across ground the live runs' tribes crossed, rich in every type of site
PATH = [[57, 40], [45, 38], [30, 36], [17, 34]]


def _sim_and_tribe():
    sim = _bare_simulation()
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 57, 40, "#c084fc")
    sim.tribes = {"tribe_0": tribe}
    return sim, tribe


def _trip(sim, tribe, scout="Ashgar"):
    before = (set(tribe.lumber_sites), {(s["x"], s["y"]) for s in tribe.wildlife_sites}, set(tribe.quarry_sites), {(s["x"], s["y"]) for s in tribe.mine_sites})
    with mock.patch.object(sim, "_celebrate_game_discovery"):
        sim._discover_along_party_ground(tribe, {"path": PATH, "kind": "scout"}, scout)
    after = (set(tribe.lumber_sites), {(s["x"], s["y"]) for s in tribe.wildlife_sites}, set(tribe.quarry_sites), {(s["x"], s["y"]) for s in tribe.mine_sites})
    return [a - b for a, b in zip(after, before)]  # the new lumber, wildlife, quarry, mine sites this trip added


def _in_reach(ty):
    pts = party_ground_points(PATH)
    return [s for s in site_seed_points(ty, 100) if any((s[0] - x) ** 2 + (s[1] - y) ** 2 <= SITE_DISCOVERY_RADIUS ** 2 for x, y in pts)]


def test_the_route_really_has_several_sites_of_a_type_so_the_cap_is_what_limits_the_trip():
    assert len(_in_reach("lumber")) >= 3 and len(_in_reach("quarry")) >= 2


def test_one_trip_captures_at_most_one_new_site_of_each_type():
    sim, tribe = _sim_and_tribe()
    new = _trip(sim, tribe)
    assert all(len(n) <= 1 for n in new), new
    assert len(new[0]) == 1 and len(new[2]) == 1  # and it does capture one where the route has some


def test_a_second_scout_over_the_same_ground_reports_new_sites_only():
    sim, tribe = _sim_and_tribe()
    first = _trip(sim, tribe, "Ashgar")
    second = _trip(sim, tribe, "Tikmen")
    for one, two in zip(first, second):
        assert not (one & two), (one, two)         # nothing the first scout reported is reported again
        assert len(two) <= 1
    assert len(second[0]) == 1                      # there was more timber on the route, so the second scout found the next grove


def test_scouts_keep_going_until_the_route_is_exhausted_and_then_report_nothing():
    sim, tribe = _sim_and_tribe()
    seen = set()
    for i in range(12):
        new = _trip(sim, tribe, f"Scout{i}")
        flat = {("lumber" if k == 0 else "wildlife" if k == 1 else "quarry" if k == 2 else "mine", s) for k, group in enumerate(new) for s in group}
        assert not (flat & seen), "a location was reported twice"
        seen |= flat
    totals = [len(_in_reach(t)) for t in ("lumber", "wildlife", "quarry", "mine")]
    mine_counts = [len(tribe.lumber_sites), len(tribe.wildlife_sites), len(tribe.quarry_sites), len(tribe.mine_sites)]
    assert mine_counts == totals                    # eventually every site on the route is known, each reported once
    assert all(len(n) == 0 for n in _trip(sim, tribe, "Last"))  # and the next scout comes home with nothing new
