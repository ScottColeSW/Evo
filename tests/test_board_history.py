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
