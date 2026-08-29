"""Immediate physiological survival pressure.

Distinct from ancestral_matrix.py's location-based bias (what happened *here*,
historically) -- this is about a tribe's own current condition: are they starving or
dehydrated *right now*, regardless of where they're standing or what happened there
before. This is the basic self-preservation signal nothing in the sim modeled until
food/water actually had upkeep consumption (see Simulation._apply_upkeep).
"""

from . import config


def survival_bias_string(food: int, water: int) -> tuple[str, bool]:
    """Returns (bias_text, is_critical). is_critical raises inference temperature the
    same way ancestral dread does -- panic should read as less predictable model
    output, not just differently worded prompt text."""
    urgent: list[str] = []
    critical = False

    if food <= config.HUNGER_CRITICAL_THRESHOLD:
        # Naming the exact action, not just describing the crisis, is deliberate: models
        # reliably articulated "we are starving" in their own rationale while still
        # choosing an unrelated action (GATHER_WOOD) in practice. A named directive
        # closes that gap between what a model says and what it actually does.
        urgent.append("Your people are starving. Choose visual_action HUNT_DEER this cycle unless you have a specific, better reason not to.")
        critical = True
    elif food <= config.HUNGER_WARNING_THRESHOLD:
        urgent.append("Food stores are running low.")

    if water <= config.THIRST_CRITICAL_THRESHOLD:
        urgent.append("Your people are dying of thirst. Choose visual_action GATHER_WATER this cycle unless you have a specific, better reason not to.")
        critical = True
    elif water <= config.THIRST_WARNING_THRESHOLD:
        urgent.append("Water stores are running low.")

    if not urgent:
        return "", False
    return "[SURVIVAL INSTINCT]: " + " ".join(urgent), critical
