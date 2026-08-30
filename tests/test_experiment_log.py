from backend.experiment_log import read_all_experiment_runs, record_experiment_run, summarize_experiment


def test_record_and_read_round_trips(tmp_path):
    path = str(tmp_path / "experiments.jsonl")

    record_experiment_run(
        "hunt_deer_wording", "baseline", "qwen2.5:3b",
        metrics={"hunt_deer_count": 1, "total_actions": 100, "extinct": False},
        run_id=0, notes="Attempt to harvest food...", path=path,
    )
    runs = read_all_experiment_runs(path)

    assert len(runs) == 1
    assert runs[0]["experiment"] == "hunt_deer_wording"
    assert runs[0]["variant"] == "baseline"
    assert runs[0]["model"] == "qwen2.5:3b"
    assert runs[0]["metrics"]["hunt_deer_count"] == 1
    assert runs[0]["run_id"] == 0
    assert runs[0]["notes"] == "Attempt to harvest food..."


def test_read_all_experiment_runs_returns_empty_list_when_file_does_not_exist(tmp_path):
    assert read_all_experiment_runs(str(tmp_path / "nonexistent.jsonl")) == []


def test_record_appends_rather_than_overwrites(tmp_path):
    path = str(tmp_path / "experiments.jsonl")
    record_experiment_run("exp_a", "v1", "gemma2:2b", metrics={"n": 1}, path=path)
    record_experiment_run("exp_a", "v2", "gemma2:2b", metrics={"n": 2}, path=path)

    runs = read_all_experiment_runs(path)
    assert len(runs) == 2
    assert [r["variant"] for r in runs] == ["v1", "v2"]


def test_summarize_experiment_groups_by_variant_and_averages():
    runs = [
        {"experiment": "hunt_deer_wording", "variant": "baseline",
         "metrics": {"hunt_deer_count": 1, "total_actions": 100}},
        {"experiment": "hunt_deer_wording", "variant": "baseline",
         "metrics": {"hunt_deer_count": 0, "total_actions": 100}},
        {"experiment": "hunt_deer_wording", "variant": "confident",
         "metrics": {"hunt_deer_count": 0, "total_actions": 100}},
        {"experiment": "other_experiment", "variant": "baseline",
         "metrics": {"hunt_deer_count": 99, "total_actions": 1}},
    ]

    summary = summarize_experiment("hunt_deer_wording", runs)

    assert set(summary.keys()) == {"baseline", "confident"}
    assert summary["baseline"]["n_runs"] == 2
    assert summary["baseline"]["totals"]["hunt_deer_count"] == 1
    assert summary["baseline"]["means"]["hunt_deer_count"] == 0.5
    assert summary["confident"]["n_runs"] == 1
    assert summary["confident"]["totals"]["hunt_deer_count"] == 0


def test_summarize_experiment_ignores_non_numeric_metrics():
    runs = [
        {"experiment": "e", "variant": "v", "metrics": {"count": 4, "label": "not a number", "extinct": False}},
    ]
    summary = summarize_experiment("e", runs)
    assert summary["v"]["totals"] == {"count": 4}


def test_summarize_experiment_defaults_to_reading_from_disk(tmp_path, monkeypatch):
    from backend import experiment_log
    monkeypatch.setattr(experiment_log, "DEFAULT_EXPERIMENT_LOG_PATH", str(tmp_path / "experiments.jsonl"))
    record_experiment_run("e", "v", "gemma2:2b", metrics={"count": 3})

    summary = summarize_experiment("e")
    assert summary["v"]["totals"]["count"] == 3
