from backend.simulation import Tribe
from backend.threat import threat_assessment_string


def _tribe(tid, x, y) -> Tribe:
    return Tribe(tid, f"Tribe {tid}", "gemma2:2b", x, y, "#c084fc")


def test_no_rivals_produces_no_assessment():
    tribe = _tribe("tribe_0", 50, 50)
    assert threat_assessment_string(tribe, []) == ""


def test_neutral_rival_produces_no_assessment_regardless_of_distance():
    tribe = _tribe("tribe_0", 50, 50)
    rival = _tribe("tribe_1", 51, 51)  # adjacent, but no declared stance

    assert threat_assessment_string(tribe, [rival]) == ""


def test_allied_rival_produces_no_assessment_even_when_close():
    tribe = _tribe("tribe_0", 50, 50)
    rival = _tribe("tribe_1", 51, 51)
    tribe.stance_toward["tribe_1"] = "ALLIED"

    assert threat_assessment_string(tribe, [rival]) == ""


def test_close_declared_enemy_produces_a_real_assessment():
    tribe = _tribe("tribe_0", 50, 50)
    rival = _tribe("tribe_1", 51, 51)
    tribe.stance_toward["tribe_1"] = "WAR"

    text = threat_assessment_string(tribe, [rival])

    assert "THREAT ASSESSMENT" in text
    assert rival.name in text
    assert "declared enemy" in text


def test_distant_declared_enemy_decays_below_the_assessment_floor():
    tribe = _tribe("tribe_0", 0, 0)
    rival = _tribe("tribe_1", 99, 99)  # far across the grid
    tribe.stance_toward["tribe_1"] = "WAR"

    assert threat_assessment_string(tribe, [rival]) == ""


def test_never_mentions_a_forced_response():
    """Explicit design decision: Module A's own override behavior (STATE_OVERRIDE_RED:
    FLEE_OR_PHALANX) is rejected -- this is a fact only, never a directive."""
    tribe = _tribe("tribe_0", 50, 50)
    rival = _tribe("tribe_1", 51, 51)
    tribe.stance_toward["tribe_1"] = "WAR"

    text = threat_assessment_string(tribe, [rival]).lower()

    assert "must" not in text
    assert "flee" not in text
    assert "your own decision" in text


def test_reports_the_most_dangerous_of_multiple_declared_enemies():
    tribe = _tribe("tribe_0", 50, 50)
    near_enemy = _tribe("tribe_1", 51, 51)
    far_enemy = _tribe("tribe_2", 90, 90)
    tribe.stance_toward["tribe_1"] = "WAR"
    tribe.stance_toward["tribe_2"] = "WAR"

    text = threat_assessment_string(tribe, [near_enemy, far_enemy])

    assert near_enemy.name in text
    assert far_enemy.name not in text
