import json

from backend.event_log import RunEventLog


def _lines(log):
    return [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]


def test_data_records_are_written_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("RUN_DATA_LOG", raising=False)
    log = RunEventLog(str(tmp_path))
    log.record("A", "chronicle line")
    log.record_data("A", "night_watch", {"x": 1})
    assert [r.get("kind") for r in _lines(log)] == [None, "night_watch"]


def test_switching_data_records_off_keeps_the_chronicle(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_DATA_LOG", "off")
    log = RunEventLog(str(tmp_path))
    log.record("A", "chronicle line")
    log.record_data("A", "night_watch", {"x": 1})
    assert [r["message"] for r in _lines(log)] == ["chronicle line"]
