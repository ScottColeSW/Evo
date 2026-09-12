from backend.benchmark_scoring import score_conflict, score_cooperation, score_settlement, score_survival, score_trial


def _tribe(**overrides):
    base = {
        "model": "gemma2:2b", "extinct": False, "extinction_cause": None,
        "era_reached": "primitive_dawn", "max_population": 10, "final_population": 10,
        "chiefs_elected": 1, "chief_deaths": 0,
        "expeditions_launched": 0, "expeditions_succeeded": 0,
        "raids_won": 0, "raids_lost": 0, "raids_defended": 0,
        "trades_completed": 0, "spy_missions_run": 0, "spy_missions_caught": 0,
        "discovered_rival": False, "allied_with_rival": False, "at_war_with_rival": False,
        "joint_castle_completed": False, "settled_permanently_near_water": False,
    }
    base.update(overrides)
    return base


def test_survival_score_scales_with_fraction_of_budget_survived():
    short = score_survival(_tribe(cycles_run=50), cycle_budget=200)
    long = score_survival(_tribe(cycles_run=150), cycle_budget=200)
    full = score_survival(_tribe(cycles_run=200), cycle_budget=200)

    assert short < long < full
    assert full == 100


def test_survival_score_gives_a_bonus_for_genuinely_settling_but_caps_at_100():
    settled = score_survival(_tribe(cycles_run=200, settled_permanently_near_water=True), cycle_budget=200)
    assert settled == 100  # capped, not 110


def test_settlement_score_rewards_era_progress():
    early = score_settlement(_tribe(era_reached="primitive_dawn", max_population=10))
    late = score_settlement(_tribe(era_reached="monolithic_era", max_population=200))
    assert late > early


def test_settlement_score_penalizes_extinction():
    survived = score_settlement(_tribe(era_reached="tribal_synapse", max_population=60, extinct=False))
    died = score_settlement(_tribe(era_reached="tribal_synapse", max_population=60, extinct=True))
    assert survived > died


def test_cooperation_score_ordering_discovered_less_than_allied_less_than_joint_castle():
    nothing = score_cooperation(_tribe(), _tribe())
    discovered = score_cooperation(_tribe(discovered_rival=True), _tribe())
    allied = score_cooperation(_tribe(allied_with_rival=True), _tribe())
    joint = score_cooperation(_tribe(joint_castle_completed=True), _tribe())

    assert nothing == (0, 0)
    assert nothing[0] < discovered[0] < allied[0] < joint[0]


def test_cooperation_score_is_symmetric_between_both_tribes():
    a_score, b_score = score_cooperation(_tribe(allied_with_rival=True), _tribe())
    assert a_score == b_score


def test_conflict_score_rewards_winning_raids():
    passive = score_conflict(_tribe(), _tribe())[0]
    aggressive_winner = score_conflict(_tribe(raids_won=2), _tribe())[0]
    assert aggressive_winner > passive


def test_conflict_score_penalizes_a_losing_raid_against_a_similar_rival_more_than_a_stronger_one():
    """Explicit design goal: judgment matters, not just aggression -- losing a
    raid against a rival with no real population edge is a worse decision than
    losing one against a genuinely stronger rival."""
    lost_to_similar = score_conflict(_tribe(raids_lost=1, max_population=50), _tribe(max_population=55))[0]
    lost_to_stronger = score_conflict(_tribe(raids_lost=1, max_population=50), _tribe(max_population=100))[0]

    assert lost_to_similar < lost_to_stronger


def test_score_trial_dispatches_by_scenario_key():
    trial = {
        "scenario_key": "survival",
        "cycle_budget": 200,
        "cycles_run": 200,
        "tribes": [_tribe()],
    }
    assert score_trial(trial) == [100]


def test_score_trial_rejects_an_unknown_scenario_key():
    import pytest

    with pytest.raises(ValueError):
        score_trial({"scenario_key": "nonexistent", "tribes": []})
