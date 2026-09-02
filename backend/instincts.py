"""Immediate physiological survival pressure.

Distinct from ancestral_matrix.py's location-based bias (what happened *here*,
historically) -- this is about a tribe's own current condition: are they starving or
dehydrated *right now*, regardless of where they're standing or what happened there
before. This is the basic self-preservation signal nothing in the sim modeled until
food/water actually had upkeep consumption (see Simulation._apply_upkeep).
"""

from . import config


def survival_bias_string(
    food: int, water: int, population: int,
    fishing_learned: bool = False, cooking_learned: bool = False,
) -> tuple[str, bool]:
    """Returns (bias_text, is_critical). is_critical raises inference temperature the
    same way ancestral dread does -- panic should read as less predictable model
    output, not just differently worded prompt text.

    Thresholds scale with population rather than being a flat stockpile number -- the
    same per-cycle upkeep formula _apply_upkeep actually charges (see config.
    UPKEEP_POPULATION_DIVISOR), so "warning"/"critical" mean the same thing (a
    consistent number of cycles of real buffer left) regardless of how large the tribe
    has grown. Cooking no longer adjusts this -- see config.COOKING_FOOD_MULTIPLIER's
    own comment: cooking now multiplies food *production* at the harvest point
    (actions._food_multiplier), the same shape Sawmill/Quarry/Dock already use,
    instead of shrinking *consumption* here."""
    upkeep = max(1, population // config.UPKEEP_POPULATION_DIVISOR)
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
    #
    # Explicit follow-up: "revise the 'your people are starving' messaging to be
    # more inclusive of options that would help them fix it -- add Fishing or Cook
    # in a Kitchen as additional ideas." The original three options (gather/hunt)
    # aren't the only real ways to fix a food shortage -- fishing is always a real
    # option, and cooking permanently stretches every future harvest further, not
    # just this one crisis. Only mentioned once fishing_learned/cooking_learned are
    # actually false, so a tribe that's already mastered them doesn't get told to
    # go learn something it already knows.
    if food <= upkeep * config.HUNGER_CRITICAL_CYCLES_LEFT:
        message = "Your people are starving -- gather food, send a hunting party"
        message += ", or try fishing now." if not fishing_learned else " now."
        if not cooking_learned:
            message += " Learning to cook (build a fire, then cook after a successful hunt) would make every future harvest go much further."
        urgent.append(message)
        critical = True
    elif food <= upkeep * config.HUNGER_WARNING_CYCLES_LEFT:
        message = "Food stores are running low -- gathering food, hunting"
        message += ", or fishing soon would help." if not fishing_learned else " soon would help."
        if not cooking_learned:
            message += " Learning to cook would help stored food last much longer too."
        urgent.append(message)

    if water <= upkeep * config.THIRST_CRITICAL_CYCLES_LEFT:
        urgent.append("Your people are dying of thirst -- gather water or dispatch scouts to find a source now.")
        critical = True
    elif water <= upkeep * config.THIRST_WARNING_CYCLES_LEFT:
        urgent.append("Water stores are running low -- gathering water or scouting for more soon would help.")

    if not urgent:
        return "", False
    return "[SURVIVAL INSTINCT]: " + " ".join(urgent), critical
