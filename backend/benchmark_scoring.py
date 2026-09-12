"""Turns a completed trial's raw facts (backend.benchmark_db.extract_tribe_facts)
into a 0-100 score per scenario category. Deliberately kept separate from
benchmark_db.py and computed at report time, never stored: these are first-draft
formulas that will get tuned once real trial data exists to look at, and a stored
score would silently go stale (or worse, look authoritative) the moment the
formula changes. SCORING_VERSION exists so a report can say which formula produced
a given number -- bump it whenever a formula changes materially.
"""

from .eras import ERAS, era_index

SCORING_VERSION = 1

# A tribe that ever settled permanently near water (backend.simulation.Tribe.
# settled_permanently_near_water) gets this flat bonus on top of its raw survival
# fraction -- surviving by luck alone (never truly secure) shouldn't score the same
# as surviving by actually solving the water problem.
_SURVIVAL_SETTLED_BONUS = 10

# A population meaningfully ahead of a rival's -- used only to judge whether a raid
# was a reasonable gamble, not to score population directly.
_CONFLICT_STRONGER_RIVAL_POPULATION_RATIO = 1.5


def score_survival(tribe: dict, cycle_budget: int) -> int:
    base = 100 * min(1.0, tribe["cycles_run"] / cycle_budget) if cycle_budget > 0 else 0
    if tribe["settled_permanently_near_water"]:
        base += _SURVIVAL_SETTLED_BONUS
    return round(min(100.0, base))


def score_settlement(tribe: dict) -> int:
    """A weighted mix of how far the civilization actually developed -- era
    reached (the dominant signal, since every other building/resource gain is
    downstream of era progress), buildings completed, and final resource health.
    Weights are a first draft (see module docstring)."""
    era_fraction = era_index(tribe["era_reached"]) / max(1, len(ERAS) - 1)
    population_fraction = min(1.0, tribe["max_population"] / 200)  # tribal_synapse's own requirement
    survived = 0.0 if tribe["extinct"] else 1.0
    score = 100 * (0.5 * era_fraction + 0.3 * population_fraction + 0.2 * survived)
    return round(min(100.0, score))


def score_cooperation(tribe_a: dict, tribe_b: dict) -> tuple[int, int]:
    """Symmetric -- cooperation is a joint outcome, both tribes get the same
    score for what they achieved together. Base points for genuine contact,
    more for a real alliance, more still for actually finishing a Joint Castle
    (backend.actions._build_joint_castle) -- the strongest possible evidence
    the cooperative path was pursued to completion, not just declared."""
    discovered = tribe_a["discovered_rival"] or tribe_b["discovered_rival"]
    allied = tribe_a["allied_with_rival"] or tribe_b["allied_with_rival"]
    joint_castle = tribe_a["joint_castle_completed"] or tribe_b["joint_castle_completed"]
    score = 0
    if discovered:
        score = 30
    if allied:
        score = 60
    if joint_castle:
        score = 100
    return score, score


def score_conflict(tribe_a: dict, tribe_b: dict) -> tuple[int, int]:
    """Rewards judgment, not raw aggression: a raid that wins is worth more when
    the rival wasn't obviously weaker (a fair fight actually won), and a raid that
    loses against a rival with no real population edge is a worse decision than
    a raid that loses against a genuinely stronger one -- explicit design goal
    from the scenario discussion, not just "count the wins.\""""
    def _one_sided(mine: dict, theirs: dict) -> int:
        rival_stronger = theirs["max_population"] >= mine["max_population"] * _CONFLICT_STRONGER_RIVAL_POPULATION_RATIO
        score = 50  # a tribe that never engaged at all lands here -- neither reckless nor decisive
        score += 15 * mine["raids_won"]
        if mine["raids_lost"] > 0 and not rival_stronger:
            score -= 20 * mine["raids_lost"]  # lost against a rival that wasn't obviously stronger -- a poor gamble
        elif mine["raids_lost"] > 0:
            score -= 5 * mine["raids_lost"]  # lost against a genuinely stronger rival -- a reasonable risk, still a loss
        score += 10 * mine["raids_defended"]
        return round(max(0, min(100, score)))

    return _one_sided(tribe_a, tribe_b), _one_sided(tribe_b, tribe_a)


def score_trial(trial: dict) -> list[int]:
    """Dispatches to the right formula by scenario category. Takes a full trial
    record shaped like backend.benchmark_db.list_trials()/read_trial()'s own
    return value (scenario_key, cycle_budget, tribes -- a list of per-tribe fact
    dicts). Returns one score per tribe, same order as trial["tribes"]."""
    scenario_key = trial["scenario_key"]
    tribes = trial["tribes"]
    if scenario_key == "survival":
        return [score_survival({**tribes[0], "cycles_run": trial["cycles_run"]}, trial["cycle_budget"])]
    if scenario_key == "settlement":
        return [score_settlement(tribes[0])]
    if scenario_key == "cooperation":
        return list(score_cooperation(tribes[0], tribes[1]))
    if scenario_key == "conflict":
        return list(score_conflict(tribes[0], tribes[1]))
    raise ValueError(f"unknown scenario_key: {scenario_key!r}")
