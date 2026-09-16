from backend.benchmark_scoring import score_conflict, score_cooperation, score_settlement, score_survival, score_trial
from backend.eras import ERAS


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


def test_survival_score_reaches_100_only_with_full_survival_and_real_progress():
    """era_fraction (backend/benchmark_scoring.py) is era_index(era) /
    (len(ERAS) - 1) -- generic, self-adjusts whenever the era ladder changes
    length (e.g. departure_era, added 2026-09-14). Uses ERAS[-1].key (the real
    top era) rather than a hardcoded era name, so this test keeps testing its
    actual intent -- "reached the top of everything" -- instead of going stale
    the next time a new era is appended."""
    thriving = score_survival(
        _tribe(cycles_run=200, era_reached=ERAS[-1].key, max_population=5000),
        cycle_budget=200,
    )
    assert thriving == 100


def test_survival_score_weighs_population_and_era_not_just_cycles_survived():
    """Explicit fix, 2026-09-12: a real batch run showed two trials that both
    survived the full cycle_budget landing on very different real outcomes
    underneath that (population 215 at cognitive_horizon vs. 1253 at
    tribal_synapse) -- the old formula scored both 100, blind to the
    difference. Reproduces that exact pair."""
    weaker_survivor = score_survival(_tribe(cycles_run=200, era_reached="cognitive_horizon", max_population=215), cycle_budget=200)
    stronger_survivor = score_survival(_tribe(cycles_run=200, era_reached="tribal_synapse", max_population=1253), cycle_budget=200)

    assert weaker_survivor < stronger_survivor
    assert weaker_survivor < 100
    assert stronger_survivor < 100


def test_survival_score_still_credits_progress_made_before_going_extinct():
    """Extinction should still hurt (dominant 0.6 weight on cycles_run/budget), but
    a tribe that reached real progress before dying shouldn't score identically to
    one that achieved nothing in the same number of cycles."""
    died_with_progress = score_survival(
        _tribe(cycles_run=100, era_reached="tribal_synapse", max_population=800, extinct=True), cycle_budget=200,
    )
    died_with_nothing = score_survival(_tribe(cycles_run=100, era_reached="primitive_dawn", max_population=10), cycle_budget=200)

    assert died_with_progress > died_with_nothing


def test_survival_score_gives_a_bonus_for_genuinely_settling_but_caps_at_100():
    settled = score_survival(
        _tribe(
            cycles_run=200, era_reached="war_and_world_domination_era", max_population=5000,
            settled_permanently_near_water=True,
        ),
        cycle_budget=200,
    )
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


def test_conflict_score_rewards_a_clean_spy_mission():
    """Explicit request, 2026-09-16: "we need to measure spy too." SPY was
    already recorded (spy_missions_run/spy_missions_caught) but never read by
    score_conflict -- real espionage got zero credit."""
    passive = score_conflict(_tribe(), _tribe())[0]
    spied = score_conflict(_tribe(spy_missions_run=2, spy_missions_caught=0), _tribe())[0]
    assert spied > passive


def test_conflict_score_penalizes_a_caught_spy_mission():
    passive = score_conflict(_tribe(), _tribe())[0]
    caught = score_conflict(_tribe(spy_missions_run=1, spy_missions_caught=1), _tribe())[0]
    assert caught < passive


def test_score_trial_dispatches_by_scenario_key():
    """Compares against a direct score_survival(...) call rather than a
    hardcoded literal -- era_fraction's own denominator shifts whenever the
    era ladder changes length (see test_survival_score_reaches_100_only_with_
    full_survival_and_real_progress's own comment), so a literal expected
    score here would go stale for the same reason, without this test's own
    real purpose (dispatch-by-key) actually caring what the number is."""
    tribe = _tribe(era_reached=ERAS[-1].key, max_population=5000, cycles_run=200)
    trial = {
        "scenario_key": "survival",
        "cycle_budget": 200,
        "cycles_run": 200,
        "tribes": [tribe],
    }
    assert score_trial(trial) == [score_survival(tribe, cycle_budget=200)]


def test_score_trial_rejects_an_unknown_scenario_key():
    import pytest

    with pytest.raises(ValueError):
        score_trial({"scenario_key": "nonexistent", "tribes": []})
