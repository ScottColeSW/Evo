from backend.benchmark_db import extract_tribe_facts, list_trials, new_trial_id, read_trial, record_trial
from backend.simulation import Tribe


def _fake_tribe(tid="tribe_0", model="gemma2:2b"):
    t = Tribe(tid, "Trial-0", model, 50, 50, "#c084fc")
    return t


def _trial_facts(trial_id, scenario_key="survival", seed=0):
    return {
        "trial_id": trial_id,
        "scenario_key": scenario_key,
        "scenario_version": 1,
        "trial_seed": seed,
        "cycle_budget": 200,
        "cycles_run": 150,
        "ended_reason": "extinction",
        "git_commit": "abc1234",
        "run_id": "run_test",
        "started_ts": 1.0,
        "finished_ts": 2.0,
    }


def test_extract_tribe_facts_reads_the_tribes_own_state():
    tribe = _fake_tribe()
    tribe.era = "tribal_synapse"
    tribe.max_population = 80
    tribe.population = 40
    tribe.raids_won = 2

    facts = extract_tribe_facts(tribe)

    assert facts["model"] == "gemma2:2b"
    assert facts["era_reached"] == "tribal_synapse"
    assert facts["max_population"] == 80
    assert facts["final_population"] == 40
    assert facts["raids_won"] == 2
    assert facts["discovered_rival"] is False
    assert facts["allied_with_rival"] is False


def test_extract_tribe_facts_reads_the_relationship_to_a_given_rival():
    tribe = _fake_tribe("tribe_0")
    rival = _fake_tribe("tribe_1", "qwen2.5:3b")
    tribe.discovered_rivals.add(rival.id)
    tribe.stance_toward[rival.id] = "ALLIED"
    tribe.castle_built = True

    facts = extract_tribe_facts(tribe, rival=rival)

    assert facts["discovered_rival"] is True
    assert facts["allied_with_rival"] is True
    assert facts["at_war_with_rival"] is False
    assert facts["joint_castle_completed"] is True


def test_record_and_list_trials_round_trips(tmp_path):
    path = str(tmp_path / "benchmark.db")
    trial_id = new_trial_id("survival", 0)
    tribe_facts = [extract_tribe_facts(_fake_tribe())]

    record_trial(_trial_facts(trial_id), tribe_facts, path=path)

    trials = list_trials(path=path)
    assert len(trials) == 1
    assert trials[0]["trial_id"] == trial_id
    assert trials[0]["scenario_key"] == "survival"
    assert len(trials[0]["tribes"]) == 1
    assert trials[0]["tribes"][0]["model"] == "gemma2:2b"


def test_list_trials_filters_by_scenario_key(tmp_path):
    path = str(tmp_path / "benchmark.db")
    record_trial(_trial_facts(new_trial_id("survival", 0), "survival"), [extract_tribe_facts(_fake_tribe())], path=path)
    record_trial(_trial_facts(new_trial_id("conflict", 0), "conflict"), [extract_tribe_facts(_fake_tribe())], path=path)

    assert len(list_trials(path=path)) == 2
    assert len(list_trials(scenario_key="conflict", path=path)) == 1


def test_list_trials_filters_by_model(tmp_path):
    path = str(tmp_path / "benchmark.db")
    record_trial(
        _trial_facts(new_trial_id("survival", 0)),
        [extract_tribe_facts(_fake_tribe(model="gemma2:2b"))],
        path=path,
    )
    record_trial(
        _trial_facts(new_trial_id("survival", 1), seed=1),
        [extract_tribe_facts(_fake_tribe(model="qwen2.5:3b"))],
        path=path,
    )

    assert len(list_trials(model="qwen2.5:3b", path=path)) == 1


def test_read_trial_returns_none_for_unknown_id(tmp_path):
    path = str(tmp_path / "benchmark.db")
    assert read_trial("nonexistent", path=path) is None


def test_record_trial_overwrites_a_duplicate_trial_id_rather_than_duplicating(tmp_path):
    path = str(tmp_path / "benchmark.db")
    trial_id = new_trial_id("survival", 0)
    record_trial(_trial_facts(trial_id), [extract_tribe_facts(_fake_tribe())], path=path)
    record_trial({**_trial_facts(trial_id), "cycles_run": 999}, [extract_tribe_facts(_fake_tribe())], path=path)

    result = read_trial(trial_id, path=path)
    assert result["cycles_run"] == 999
    assert len(list_trials(path=path)) == 1
