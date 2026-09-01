"""Immediate physiological survival pressure.

Distinct from ancestral_matrix.py's location-based bias (what happened *here*,
historically) -- this is about a tribe's own current condition: are they starving or
dehydrated *right now*, regardless of where they're standing or what happened there
before. This is the basic self-preservation signal nothing in the sim modeled until
food/water actually had upkeep consumption (see Simulation._apply_upkeep).
"""

from . import config


def effective_food_upkeep(base_upkeep: int, cooking_learned: bool, kitchen_built: bool = False) -> int:
    """Explicit request: "cooked food is worth 3 raw food." Applied at the single
    point food is actually consumed (config.COOKING_UPKEEP_DIVISOR) rather than at
    every scattered food-gain call site -- economically equivalent (the same
    stockpile now covers 3x the need) but one shared calculation instead of six.
    Shared by Simulation._apply_upkeep (the real drain), this module's own hunger
    threshold below, and wellbeing.compute_wellbeing's physiological tier, so all
    three agree on how far a cooking tribe's food actually goes. Water is
    unaffected -- cooking doesn't change how much a tribe needs to drink.

    Explicit follow-up: "we might have to let them build a kitchen which improves
    cooked food to excellent food yielding 3 per cooked item" -- a kitchen only
    means anything once cooking is already known (kitchen_built alone, without
    cooking_learned, changes nothing), and stacks a further config.
    KITCHEN_UPKEEP_MULTIPLIER on top of the cooking divisor rather than replacing
    it -- excellent food is 3x as good as cooked food, not just 3x raw."""
    if not cooking_learned:
        return base_upkeep
    divisor = config.COOKING_UPKEEP_DIVISOR
    if kitchen_built:
        divisor *= config.KITCHEN_UPKEEP_MULTIPLIER
    return max(1, round(base_upkeep / divisor))


def survival_bias_string(
    food: int, water: int, population: int, cooking_learned: bool = False, kitchen_built: bool = False
) -> tuple[str, bool]:
    """Returns (bias_text, is_critical). is_critical raises inference temperature the
    same way ancestral dread does -- panic should read as less predictable model
    output, not just differently worded prompt text.

    Thresholds scale with population rather than being a flat stockpile number -- the
    same per-cycle upkeep formula _apply_upkeep actually charges (see config.
    UPKEEP_POPULATION_DIVISOR), so "warning"/"critical" mean the same thing (a
    consistent number of cycles of real buffer left) regardless of how large the tribe
    has grown. `cooking_learned` adjusts the food-specific threshold the same way
    _apply_upkeep's real drain is adjusted (see effective_food_upkeep) -- a cooking
    tribe genuinely isn't as close to starving as the raw upkeep number would suggest."""
    upkeep = max(1, population // config.UPKEEP_POPULATION_DIVISOR)
    food_upkeep = effective_food_upkeep(upkeep, cooking_learned, kitchen_built)
    urgent: list[str] = []
    critical = False

    # NUDGE (2026-08-30, explicit "nudge harder" request): this used to describe the
    # crisis without naming a specific action, on the theory that a model connecting
    # "starving" to "go hunt" on its own was more honest than scripting the outcome.
    # Naming GATHER_FOOD/HUNTING_PARTY and GATHER_WATER/SCOUT directly is still a
    # suggestion inside a fact block, not a forced action -- available_actions and the
    # model's own choice are untouched -- but it no longer pretends not to know what
    # the tribe actually needs. Revisit if this proves too heavy-handed later --
    # grep "# NUDGE" across backend/ to find every place this line was crossed.
    if food <= food_upkeep * config.HUNGER_CRITICAL_CYCLES_LEFT:
        urgent.append("Your people are starving -- gather food or send a hunting party now.")
        critical = True
    elif food <= food_upkeep * config.HUNGER_WARNING_CYCLES_LEFT:
        urgent.append("Food stores are running low -- gathering food or hunting soon would help.")

    if water <= upkeep * config.THIRST_CRITICAL_CYCLES_LEFT:
        urgent.append("Your people are dying of thirst -- gather water or dispatch scouts to find a source now.")
        critical = True
    elif water <= upkeep * config.THIRST_WARNING_CYCLES_LEFT:
        urgent.append("Water stores are running low -- gathering water or scouting for more soon would help.")

    if not urgent:
        return "", False
    return "[SURVIVAL INSTINCT]: " + " ".join(urgent), critical
