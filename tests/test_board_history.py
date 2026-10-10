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


def test_the_writer_keeps_one_connection_open_between_writes_and_close_all_releases_the_file(tmp_path):
    """2026-10-10: opening, setting up and closing a connection every cycle cost about 120 ms of file flushing on the thread that serves the page."""
    import os

    from backend import board_history

    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 0, {"cycle": 0}, path=path)
    first = list(board_history._writers.values())
    record_board_state("run_1", 1, {"cycle": 1}, path=path)
    assert len(board_history._writers) == 1 and list(board_history._writers.values()) == first  # the same connection

    assert [r["cycle"] for r in read_run("run_1", path=path)] == [0, 1]  # a reader sees rows while the writer stays open
    board_history.close_all()
    assert board_history._writers == {}
    os.remove(path)  # raises on Windows if the file were still held open


def test_a_snapshot_already_serialized_is_stored_as_given(tmp_path):
    path = str(tmp_path / "board_history.db")
    record_board_state("run_1", 3, {"cycle": "ignored"}, path=path, snapshot_json='{"cycle": 3, "from": "the caller"}')
    assert read_cycle("run_1", 3, path=path)["snapshot"] == {"cycle": 3, "from": "the caller"}


def test_the_server_tick_sends_the_page_first_and_serializes_the_snapshot_once(monkeypatch):
    """The text sent to the page is the text stored in the history: one json.dumps a cycle, and a slow write cannot delay the page."""
    import asyncio

    from backend import app

    order = []

    class FakeSim:
        paused = False
        game_over = False
        cycle = 7
        run_id = "run_x"

        async def step(self):
            pass

        def snapshot(self):
            return {"cycle": 7, "tribes": {}}

    class FakeWs:
        async def send_str(self, text):
            order.append(("sent", text))

    def fake_record(run_id, cycle, snapshot, path=None, snapshot_json=None):
        order.append(("recorded", snapshot_json))

    monkeypatch.setattr(app, "record_board_state", fake_record)
    asyncio.run(app._tick_session(FakeWs(), {"sim": FakeSim(), "observer": False}))

    assert [kind for kind, _ in order] == ["sent", "recorded"]
    assert order[0][1] is order[1][1]  # the very same string object
