"""Immediate physiological survival pressure.

Distinct from ancestral_matrix.py's location-based bias (what happened *here*,
historically) -- this is about a tribe's own current condition: are they starving or
dehydrated *right now*, regardless of where they're standing or what happened there
before. This is the basic self-preservation signal nothing in the sim modeled until
food/water actually had upkeep consumption (see Simulation._apply_upkeep).
"""

from . import config


def survival_bias_string(food: int, water: int, population: int) -> tuple[str, bool]:
    """Returns (bias_text, is_critical). is_critical raises inference temperature the
    same way ancestral dread does -- panic should read as less predictable model
    output, not just differently worded prompt text.

    Thresholds scale with population rather than being a flat stockpile number -- the
    same per-cycle upkeep formula _apply_upkeep actually charges (see config.
    UPKEEP_POPULATION_DIVISOR), so "warning"/"critical" mean the same thing (a
    consistent number of cycles of real buffer left) regardless of how large the tribe
    has grown."""
    upkeep = max(1, population // config.UPKEEP_POPULATION_DIVISOR)
    urgent: list[str] = []
    critical = False

    # Describes the crisis honestly without naming a specific action to take. An earlier
    # version of this named HUNT_DEER/GATHER_WATER directly as a near-command, which
    # measurably fixed the starvation loop -- but that's scripting the outcome, not
    # letting the model reason its way there. If a model can't connect "starving" to
    # "go hunt" on its own, that's real information about that model, worth seeing
    # honestly rather than papering over.
    if food <= upkeep * config.HUNGER_CRITICAL_CYCLES_LEFT:
        urgent.append("Your people are starving.")
        critical = True
    elif food <= upkeep * config.HUNGER_WARNING_CYCLES_LEFT:
        urgent.append("Food stores are running low.")

    if water <= upkeep * config.THIRST_CRITICAL_CYCLES_LEFT:
        urgent.append("Your people are dying of thirst.")
        critical = True
    elif water <= upkeep * config.THIRST_WARNING_CYCLES_LEFT:
        urgent.append("Water stores are running low.")

    if not urgent:
        return "", False
    return "[SURVIVAL INSTINCT]: " + " ".join(urgent), critical
