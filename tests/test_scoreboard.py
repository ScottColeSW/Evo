from backend.scoreboard import read_all_results, record_tribe_result, summarize_by_model
from backend.simulation import Tribe


def _tribe(model="gemma2:2b", **overrides):
    t = Tribe("tribe_0", "Forest Tribe", model, 50, 50, "#c084fc")
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def test_record_and_read_round_trips(tmp_path):
    path = str(tmp_path / "scoreboard.jsonl")
    tribe = _tribe(era="bronze_age", max_population=15, chiefs_elected=2)

    record_tribe_result(tribe, cause="starvation", cycles_survived=42, path=path)
    results = read_all_results(path)

    assert len(results) == 1
    assert results[0]["model"] == "gemma2:2b"
    assert results[0]["cause_of_death"] == "starvation"
    assert results[0]["cycles_survived"] == 42
    assert results[0]["era_reached"] == "bronze_age"
    assert results[0]["max_population"] == 15
    assert results[0]["chiefs_elected"] == 2


def test_read_all_results_returns_empty_list_when_file_does_not_exist(tmp_path):
    assert read_all_results(str(tmp_path / "nonexistent.jsonl")) == []


def test_record_appends_rather_than_overwrites(tmp_path):
    path = str(tmp_path / "scoreboard.jsonl")
    record_tribe_result(_tribe(), cause="starvation", cycles_survived=10, path=path)
    record_tribe_result(_tribe(), cause="thirst", cycles_survived=20, path=path)

    results = read_all_results(path)
    assert len(results) == 2
    assert [r["cycles_survived"] for r in results] == [10, 20]


def test_summarize_by_model_groups_and_averages():
    results = [
        {"model": "gemma2:2b", "cycles_survived": 30, "era_reached": "stone_age", "max_population": 8,
         "chiefs_elected": 1, "chief_deaths": 0, "expeditions_launched": 2, "expeditions_succeeded": 1,
         "raids_won": 0, "raids_lost": 0, "raids_defended": 0, "trades_completed": 0},
        {"model": "gemma2:2b", "cycles_survived": 50, "era_reached": "stone_age", "max_population": 9,
         "chiefs_elected": 2, "chief_deaths": 1, "expeditions_launched": 3, "expeditions_succeeded": 2,
         "raids_won": 1, "raids_lost": 0, "raids_defended": 0, "trades_completed": 1},
        {"model": "qwen2.5:3b", "cycles_survived": 100, "era_reached": "bronze_age", "max_population": 20,
         "chiefs_elected": 1, "chief_deaths": 0, "expeditions_launched": 4, "expeditions_succeeded": 3,
         "raids_won": 0, "raids_lost": 1, "raids_defended": 2, "trades_completed": 3},
    ]

    rows = summarize_by_model(results)

    assert rows[0]["model"] == "qwen2.5:3b"  # best avg survival sorts first
    assert rows[0]["avg_cycles_survived"] == 100.0
    gemma_row = next(r for r in rows if r["model"] == "gemma2:2b")
    assert gemma_row["runs"] == 2
    assert gemma_row["avg_cycles_survived"] == 40.0
    assert gemma_row["best_cycles_survived"] == 50
    assert gemma_row["total_chiefs_elected"] == 3
    assert gemma_row["total_raids_won"] == 1
    assert gemma_row["total_trades_completed"] == 1


def test_summarize_by_model_handles_empty_results():
    assert summarize_by_model([]) == []
