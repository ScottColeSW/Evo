from backend.simulation import Tribe
from backend.tribe_fixtures import apply_tribe_fixture, load_fixture


def _fixture(cycle=100, **tribe_overrides):
    tribe = {
        "x": 40, "y": 37, "population": 5000, "era": "war_and_world_domination_era",
        "barracks_built": 5, "battalion_size": 100,
        "unique_resources": {"Iron Ore": 200},
        "visited_sectors": ["a", "b"],
        "last_harvest_cycle": 90,
        # Deliberately present but must never be copied -- see FIXTURE_FIELDS'
        # own comment.
        "name": "Should Not Copy", "model": "should-not-copy", "color": "#000000",
        "discovered_rivals": ["tribe_9"], "stance_toward": {"tribe_9": "WAR"},
        "chief_name": "Should Not Copy Either",
    }
    tribe.update(tribe_overrides)
    return {"run_id": "run_test", "cycle": cycle, "tribe": tribe}


def test_apply_tribe_fixture_copies_the_safe_fields():
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 0, 0, "#c084fc")

    apply_tribe_fixture(tribe, _fixture(), new_cycle=100)

    assert tribe.x == 40 and tribe.y == 37
    assert tribe.population == 5000
    assert tribe.era == "war_and_world_domination_era"
    assert tribe.barracks_built == 5
    assert tribe.battalion_size == 100
    assert tribe.unique_resources == {"Iron Ore": 200}


def test_apply_tribe_fixture_never_overrides_identity_or_relationship_fields():
    """Explicit design: identity (name/model/color) always comes from the
    scenario's own tribe_configs, and relationship state (discovered_rivals/
    stance_toward/chief_*) is never copied -- a fixture is about a tribe's own
    earned capability, not a replay of one specific run's diplomatic history."""
    tribe = Tribe("tribe_0", "Real Name", "real-model", 0, 0, "#ffffff")

    apply_tribe_fixture(tribe, _fixture(), new_cycle=100)

    assert tribe.name == "Real Name"
    assert tribe.model == "real-model"
    assert tribe.color == "#ffffff"
    assert tribe.discovered_rivals == set()
    assert tribe.stance_toward == {}
    assert tribe.chief_name == ""


def test_apply_tribe_fixture_converts_list_back_to_a_real_set():
    """Tribe.to_dict() renders visited_sectors as list(...) for JSON -- must
    come back as a real set, not a list, on the way in."""
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 0, 0, "#c084fc")

    apply_tribe_fixture(tribe, _fixture(), new_cycle=100)

    assert tribe.visited_sectors == {"a", "b"}
    assert isinstance(tribe.visited_sectors, set)


def test_apply_tribe_fixture_shifts_cycle_relative_fields_by_the_offset():
    """last_harvest_cycle=90 at the fixture's own source cycle (100) means 'it
    happened 10 cycles before the snapshot' -- starting a new trial at cycle
    500 must preserve that same 10-cycle gap, not the raw number 90 (which
    would misreport it as 410 cycles ago, or worse, go negative and break a
    '> 0'-as-boolean read like _is_food_secure's)."""
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 0, 0, "#c084fc")

    apply_tribe_fixture(tribe, _fixture(cycle=100, last_harvest_cycle=90), new_cycle=500)

    assert tribe.last_harvest_cycle == 490  # 90 + (500 - 100)


def test_apply_tribe_fixture_does_not_shift_a_zero_cycle_field():
    """A cycle field still at its Tribe.__init__ default (0 -- 'never
    happened') must stay 0, not become a small positive number that would
    misreport a real event that never occurred."""
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 0, 0, "#c084fc")

    apply_tribe_fixture(tribe, _fixture(cycle=100, last_harvest_cycle=0), new_cycle=500)

    assert tribe.last_harvest_cycle == 0


def test_the_shipped_war_ready_fixtures_load_and_look_real():
    """Grounds the two checked-in fixtures (backend/fixtures/war_ready_a.json,
    war_ready_b.json) rather than just testing apply_tribe_fixture's logic in
    the abstract -- these are the real tribe states run_benchmark.py's
    "war_ready_5000" scenario actually ships with."""
    a = load_fixture("war_ready_a")
    b = load_fixture("war_ready_b")

    assert a["cycle"] == b["cycle"]  # both tribes are from the same real moment
    for fixture in (a, b):
        tribe = fixture["tribe"]
        assert tribe["population"] > 3000
        assert tribe["barracks_built"]
        assert tribe["battalion_size"] > 0
