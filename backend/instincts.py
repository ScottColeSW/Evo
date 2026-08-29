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

    # Describes the crisis honestly without naming a specific action to take. An earlier
    # version of this named HUNT_DEER/GATHER_WATER directly as a near-command, which
    # measurably fixed the starvation loop -- but that's scripting the outcome, not
    # letting the model reason its way there. If a model can't connect "starving" to
    # "go hunt" on its own, that's real information about that model, worth seeing
    # honestly rather than papering over.
    if food <= config.HUNGER_CRITICAL_THRESHOLD:
        urgent.append("Your people are starving.")
        critical = True
    elif food <= config.HUNGER_WARNING_THRESHOLD:
        urgent.append("Food stores are running low.")

    if water <= config.THIRST_CRITICAL_THRESHOLD:
        urgent.append("Your people are dying of thirst.")
        critical = True
    elif water <= config.THIRST_WARNING_THRESHOLD:
        urgent.append("Water stores are running low.")

    if not urgent:
        return "", False
    return "[SURVIVAL INSTINCT]: " + " ".join(urgent), critical
