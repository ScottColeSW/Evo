"""A tribe's current military strength -- Military branch, step 5 (plan file
valiant-forging-falcon.md).

A **computed** stat, never a stored field, so it can't go stale -- same
reasoning wellbeing.compute_wellbeing and Simulation.expedition_capacity
already use for their own derived numbers. Stacks independent, named
multipliers on tribe.battalion_size (the base), the same "stack multiple
named multipliers" shape actions._food_multiplier already uses for
kitchen/cooking/created-object bonuses. See config.py's own comments on each
MIGHT_* constant for why its magnitude was picked -- all invented first-pass
defaults, not yet tuned against live data.

Kept free of any Simulation import (mirrors wellbeing.py's own reasoning) to
avoid a circular dependency -- everything read here already lives on the
Tribe object itself.
"""

from . import config

_WELLBEING_TIER_NAMES = ("physiological", "safety", "belonging", "esteem", "self_actualization")


def compute_might(tribe) -> int:
    """0 with no Battalion at all -- Might only exists once there's a real
    force to measure. Each term below is confirmed decisions from the
    brainstorm this plan was built from:

    - Forge weapons "counted, not consumed" -- tribe.items stays untouched,
      this just checks how many are weapons.
    - The tier idea (Keep/Fortress/Castle) is worth integrating.
    - Trophy count (specifically the Warrior's own, not the whole tribe's)
      is worth adding.
    - Well-Being "might be included some" -- deliberately the
      smallest-magnitude term (config.MIGHT_WELLBEING_WEIGHT).
    - Training/readiness: explicit follow-up, "not overpowered, more like
      bolster and upkeep" (tribe.battalion_readiness, see
      Simulation._advance_battalion_readiness_upkeep and
      actions._train_battalion).
    """
    if tribe.battalion_size <= 0:
        return 0

    weapon_count = sum(1 for item in tribe.items if item["type"] == "weapon")
    armed_fraction = min(1.0, weapon_count / tribe.battalion_size)
    equipment = 1 + config.MIGHT_WEAPON_BONUS * armed_fraction

    tier = 3 if tribe.castle_built else 2 if tribe.fortress_built else 1 if tribe.keep_built else 0
    tier_bonus = 1 + config.MIGHT_TIER_BONUS_PER_TIER * tier

    warrior_trophies = (
        sum(1 for t in tribe.trophies if t["chief"] == tribe.warrior_name) if tribe.warrior_name else 0
    )
    warrior_bonus = 1 + config.MIGHT_TROPHY_BONUS_PER_TROPHY * warrior_trophies

    # tribe.wellbeing is whatever Simulation._prepare_turn last computed (runs every
    # turn already) -- an empty/partial dict (before the first turn has run) reads
    # every missing tier as a neutral 0.5, the same graceful-default shape
    # Simulation._grow_population's own wellbeing read already uses. No single
    # tier here is "the" wellbeing score (unlike population growth, which is
    # deliberately keyed on physiological alone) -- Might wants a broad read on
    # how the tribe as a whole is doing, so this averages all five real Maslow
    # tiers (fame is prestige, not a need, and stays out per wellbeing.py's own
    # FAME_LABEL comment).
    tiers = tribe.wellbeing.get("tiers", {})
    wellbeing_overall = sum(tiers.get(name, 0.5) for name in _WELLBEING_TIER_NAMES) / len(_WELLBEING_TIER_NAMES)
    wellbeing_bonus = 1 + config.MIGHT_WELLBEING_WEIGHT * (wellbeing_overall - 0.5)

    training_bonus = 1 + config.MIGHT_TRAINING_BONUS * tribe.battalion_readiness

    return round(
        tribe.battalion_size * equipment * tier_bonus * warrior_bonus * wellbeing_bonus * training_bonus
    )
