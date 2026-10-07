"""2026-10-07: a live run's tribe answered FORTIFY or BUILD_WALL (15 times across the runs on record, always meaning the wall); the parser did not know them and a
text-similarity guess applied RAID six times and a hatchery or barracks twice."""
from backend.simulation import ACTION_ALIASES, _resolve_action

MENU = ["BUILD_BARRACKS", "RAID", "CONSTRUCT_WALL", "BUILD_HATCHERY", "SCOUT"]


def test_fortify_and_build_wall_mean_the_wall_when_it_is_on_the_menu():
    for word in ("FORTIFY", "fortify", "Fortify Wall", "BUILD_WALL", "build-wall", "FORTIFY_WALL", "REINFORCE_WALL"):
        assert _resolve_action(word, MENU) == ("CONSTRUCT_WALL", None), word


def test_with_no_wall_on_the_menu_the_old_ladder_runs_unchanged():
    menu = ["BUILD_BARRACKS", "RAID", "SCOUT"]
    action, unresolved = _resolve_action("FORTIFY", menu)
    assert action in menu and action != "CONSTRUCT_WALL"


def test_every_alias_points_at_a_real_action():
    from backend.actions import ACTION_REGISTRY

    assert set(ACTION_ALIASES.values()) <= set(ACTION_REGISTRY)


def test_real_actions_still_win_over_aliases():
    assert _resolve_action("RAID", MENU) == ("RAID", None)
