"""scripts/compare_nights.py: the night timing and reflection summary used to compare REFLECTION_MODEL settings (2026-10-10)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compare_nights  # noqa: E402


def _write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_summarize_reads_timing_relations_and_cycle_durations(tmp_path):
    records = [{"cycle": 0, "tribe": "Run", "kind": "run_config", "data": {"reflection_model": "tribe", "git_commit": "abc", "tribes": [{"model": "qwen2.5:3b"}, {"model": "qwen2.5:3b"}]}}]
    ts = 1000.0
    for cycle in range(1, 62):
        ts += 3.0 + (12.0 if cycle > 1 and (cycle - 1) % 30 == 0 else 0.0)  # the night's extra seconds fall between cycle 30 and cycle 31
        records.append({"cycle": cycle, "tribe": "A", "ts": ts, "message": "x"})
    for cycle in (30, 60):
        for tribe in ("A", "B"):
            records.append({"cycle": cycle, "tribe": tribe, "kind": "night_timing", "data": {"model": "qwen2.5:3b", "reflect_s": 4.0, "embed_s": 0.5}})
            records.append({"cycle": cycle, "tribe": tribe, "kind": "reflection_memory", "data": {"relation": "new", "text": "a thought"}})
        records.append({"cycle": cycle, "tribe": "Run", "kind": "night_phase_timing", "data": {"model_phase_s": 4.6, "apply_s": {"A": 0.1, "B": 0.2}}})
    path = tmp_path / "run.jsonl"
    _write(path, records)
    s = compare_nights.summarize(str(path))
    assert s["reflection_setting"] == "tribe" and s["tribe_models"] == ["qwen2.5:3b"]
    assert s["reflect_call_s"]["median"] == 4.0 and s["reflect_by_model"]["qwen2.5:3b"]["n"] == 4
    assert s["model_phase_s"]["median"] == 4.6 and abs(s["apply_s"]["median"] - 0.3) < 1e-9
    assert s["reflections_stored"] == 4 and s["relations"] == {"new": 4} and s["reflection_text_chars"]["median"] == len("a thought")
    assert round(s["normal_cycle_s"]["median"], 1) == 3.0 and round(s["night_cycle_s"]["median"], 1) == 15.0


def test_a_run_logged_before_the_timing_records_says_so(tmp_path, capsys):
    path = tmp_path / "old.jsonl"
    _write(path, [{"cycle": 1, "tribe": "A", "ts": 1.0, "message": "x"}, {"cycle": 2, "tribe": "A", "ts": 4.0, "message": "y"}])
    compare_nights.main([str(path)])
    out = capsys.readouterr().out
    assert "no timing records" in out and "(not recorded)" in out
