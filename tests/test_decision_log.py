from backend.board_history import record_board_state
from backend.decision_log import materialize, query_tribe_decisions


def _snapshot(action="GATHER_WOOD", target=(5, 6), **overrides):
    tribe = {
        "name": "Forest Tribe",
        "era": "stone_age",
        "population": 12,
        "wood": 3.0,
        "stone": 1.0,
        "food": 4.0,
        "water": 2.0,
        "x": 10,
        "y": 20,
        "confirmed_water_sites": [[8, 8]],
        "lumber_sites": [],
        "wildlife_sites": [],
        "quarry_sites": [],
        "mine_sites": [],
        "raider_sightings": [],
        "last_action": action,
        "last_decision_target": list(target),
    }
    tribe.update(overrides)
    return {"cycle": 0, "tribes": {"tribe_0": tribe}}


def test_materialize_extracts_one_row_per_tribe_per_cycle(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 1, _snapshot(), path=path)

    written = materialize("run_1", path=path)

    assert written == 1
    rows = query_tribe_decisions("run_1", "Forest Tribe", path=path)
    assert len(rows) == 1
    row = rows[0]
    assert row["cycle"] == 1
    assert row["tribe_id"] == "tribe_0"
    assert row["era"] == "stone_age"
    assert row["population"] == 12
    assert row["x"] == 10
    assert row["y"] == 20
    assert row["action"] == "GATHER_WOOD"
    assert row["target_x"] == 5
    assert row["target_y"] == 6
    assert row["confirmed_water_sites"] == "[[8, 8]]"


def test_materialize_handles_missing_last_decision_target(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 1, _snapshot(last_decision_target=None), path=path)

    materialize("run_1", path=path)

    row = query_tribe_decisions("run_1", "Forest Tribe", path=path)[0]
    assert row["target_x"] is None
    assert row["target_y"] is None


def test_materialize_is_idempotent_on_rerun(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 1, _snapshot(), path=path)

    materialize("run_1", path=path)
    written_again = materialize("run_1", path=path)

    assert written_again == 1
    assert len(query_tribe_decisions("run_1", "Forest Tribe", path=path)) == 1


def test_materialize_without_run_id_covers_all_runs(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_a", 1, _snapshot(action="SCOUT"), path=path)
    record_board_state("run_b", 1, _snapshot(action="RELOCATE"), path=path)

    written = materialize(path=path)

    assert written == 2
    assert query_tribe_decisions("run_a", "Forest Tribe", path=path)[0]["action"] == "SCOUT"
    assert query_tribe_decisions("run_b", "Forest Tribe", path=path)[0]["action"] == "RELOCATE"


def test_query_tribe_decisions_respects_cycle_range(tmp_path):
    path = str(tmp_path / "board_history.db")
    for cycle in range(1, 6):
        record_board_state("run_1", cycle, _snapshot(), path=path)
    materialize("run_1", path=path)

    rows = query_tribe_decisions("run_1", "Forest Tribe", path=path, start_cycle=2, end_cycle=4)

    assert [r["cycle"] for r in rows] == [2, 3, 4]
