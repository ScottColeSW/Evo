from backend.board_history import list_runs, read_cycle, read_run, record_board_state


def test_record_and_read_run_round_trips(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 0, {"cycle": 0, "tribes": {"tribe_0": {"population": 8}}}, path=path)
    record_board_state("run_1", 1, {"cycle": 1, "tribes": {"tribe_0": {"population": 9}}}, path=path)

    rows = read_run("run_1", path=path)

    assert len(rows) == 2
    assert rows[0]["cycle"] == 0
    assert rows[0]["snapshot"]["tribes"]["tribe_0"]["population"] == 8
    assert rows[1]["snapshot"]["tribes"]["tribe_0"]["population"] == 9


def test_read_run_returns_empty_list_for_unknown_run(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 0, {"cycle": 0}, path=path)

    assert read_run("nonexistent_run", path=path) == []


def test_record_board_state_overwrites_a_duplicate_cycle_rather_than_duplicating(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 5, {"cycle": 5, "value": "first"}, path=path)
    record_board_state("run_1", 5, {"cycle": 5, "value": "second"}, path=path)

    rows = read_run("run_1", path=path)

    assert len(rows) == 1
    assert rows[0]["snapshot"]["value"] == "second"


def test_read_cycle_returns_one_specific_cycle(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 0, {"cycle": 0}, path=path)
    record_board_state("run_1", 1, {"cycle": 1, "marker": "here"}, path=path)

    result = read_cycle("run_1", 1, path=path)

    assert result["snapshot"]["marker"] == "here"


def test_read_cycle_returns_none_when_missing(tmp_path):
    path = str(tmp_path / "board_history.db")
    assert read_cycle("run_1", 99, path=path) is None


def test_list_runs_returns_distinct_sorted_run_ids(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_b", 0, {}, path=path)
    record_board_state("run_a", 0, {}, path=path)
    record_board_state("run_a", 1, {}, path=path)

    assert list_runs(path=path) == ["run_a", "run_b"]


def test_a_snapshot_can_be_written_while_another_connection_is_reading(tmp_path):
    """2026-10-08: a live run's write failed with 'database is locked' while analysis scripts read the history. WAL mode lets a reader and the writer work together."""
    import sqlite3

    from backend.board_history import read_cycle, record_board_state

    db = str(tmp_path / "board.db")
    record_board_state("run_a", 1, {"cycle": 1}, path=db)
    reader = sqlite3.connect(db)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM board_snapshots").fetchall()  # an open read transaction
    record_board_state("run_a", 2, {"cycle": 2}, path=db)       # would raise 'database is locked' in the default journal mode
    reader.close()
    assert read_cycle("run_a", 2, path=db)["snapshot"] == {"cycle": 2}
