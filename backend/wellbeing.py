"""A Maslow's-hierarchy-of-needs read on a tribe's overall condition.

Distinct from instincts.py's survival_bias_string (moment-to-moment starvation/thirst
alarm) and prompts.py's growth-pressure text (population fragility) -- this is a
slower-moving, five-tier summary meant to answer "how is this tribe doing, broadly?"
Every tier is computed from a real signal the simulation already tracks (see each
tier below); nothing here is invented. Per explicit design note, this is not
viewer-only -- compute_wellbeing's summary text is meant to reach the tribe's own
prompt as a fact the chief can reason about, the same as ancestral bias or the
survival instinct layer, not just rendered for the human audience.

Kept free of any World/Simulation import (wall_fraction is passed in, computed by
Simulation._wall_fraction) to avoid a circular dependency with simulation.py.
"""

from . import config
from .eras import ERAS, era_index

TIER_SATISFIED_THRESHOLD = 0.6

TIER_LABELS = {
    "physiological": "Physiological",
    "safety": "Safety",
    "belonging": "Belonging",
    "esteem": "Esteem",
    "self_actualization": "Self-Actualization",
}


def compute_wellbeing(tribe, wall_fraction: float) -> dict:
    """Returns {"tiers": {name: score 0..1}, "focus": tier_name, "summary": fact text}.

    focus is the lowest unmet tier (first below TIER_SATISFIED_THRESHOLD, bottom-up),
    the real Maslow logic that a lower unmet need is what's actually pressing --
    self_actualization only becomes "focus" once every tier below it is satisfied.
    """
    # Physiological: reuses instincts.py's own upkeep-buffer formula (the same "how
    # many cycles of food/water are actually left" this tribe's own survival-instinct
    # layer already alarms on) rather than a second, different notion of hunger.
    # Cooking no longer changes this buffer -- it multiplies food production at the
    # harvest point instead (config.COOKING_FOOD_MULTIPLIER, actions._food_multiplier),
    # which already shows up here as a larger tribe.food stockpile. A Bath House
    # (Simulation._apply_upkeep) genuinely lowers the same real per-cycle drain --
    # mirrored here so this score reflects the real number being charged, not a
    # stale unmultiplied one.
    upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
    if tribe.bath_house_built:
        upkeep = max(1, round(upkeep * config.BATH_HOUSE_UPKEEP_MULTIPLIER))
    buffer_cycles = min(tribe.food / upkeep, tribe.water / upkeep)
    physiological = min(1.0, buffer_cycles / (config.HUNGER_WARNING_CYCLES_LEFT * 2))

    # Safety: a nomadic tribe has no fixed camp to defend at all (a real, if
    # different, kind of exposure than a settled-but-wall-less one). Once settled,
    # wall progress and a track record of repelled raids are the only two real
    # anti-raider signals the sim tracks.
    if not tribe.has_ever_settled:
        safety = 0.2
    else:
        safety = min(1.0, 0.3 + 0.4 * wall_fraction + 0.3 * min(1.0, tribe.raids_defended / 3))

    # Belonging: contact with other tribes (trade) plus having ever spoken at all
    # (a broadcast is this tribe's own cultural voice, however small a signal).
    belonging = min(1.0, tribe.trades_completed / 3) * 0.7 + (0.3 if tribe.last_broadcast else 0.0)

    # Esteem: trophies are the sim's only earned-recognition mechanic.
    esteem = min(1.0, len(tribe.trophies) / 5)

    # Self-Actualization: era progression plus, once a city is founded, how much of
    # it has actually been built out -- real placed buildings now (backend/
    # architect.py), not the old abstract city_buildings counter.
    era_fraction = era_index(tribe.era) / max(1, len(ERAS) - 1)
    city_fraction = min(1.0, len(tribe.buildings) / config.SELF_ACTUALIZATION_BUILDING_REFERENCE) if tribe.founded_city else 0.0
    self_actualization = min(1.0, era_fraction * 0.6 + city_fraction * 0.4)

    tiers = {
        "physiological": round(physiological, 2),
        "safety": round(safety, 2),
        "belonging": round(belonging, 2),
        "esteem": round(esteem, 2),
        "self_actualization": round(self_actualization, 2),
    }

    focus = next(
        (name for name in TIER_LABELS if tiers[name] < TIER_SATISFIED_THRESHOLD),
        "self_actualization",
    )
    summary = _summary_text(tribe, tiers, focus, buffer_cycles, wall_fraction)
    return {"tiers": tiers, "focus": focus, "summary": summary}


def _summary_text(tribe, tiers: dict, focus: str, buffer_cycles: float, wall_fraction: float) -> str:
    era_label = next((e.label for e in ERAS if e.key == tribe.era), tribe.era)
    lines = [
        f"Physiological: {'stable' if tiers['physiological'] >= TIER_SATISFIED_THRESHOLD else 'strained'} "
        f"(~{buffer_cycles:.1f} cycles of food/water buffer at current population).",
        "Safety: "
        + (
            "still nomadic, no fixed camp to defend."
            if not tribe.has_ever_settled
            else f"{'secure' if tiers['safety'] >= TIER_SATISFIED_THRESHOLD else 'exposed'} "
            f"(wall {round(wall_fraction * 100)}% complete, {tribe.raids_defended} raid(s) repelled)."
        ),
        f"Belonging: {'connected' if tiers['belonging'] >= TIER_SATISFIED_THRESHOLD else 'isolated'} "
        f"({tribe.trades_completed} trade contact(s) made with other tribes).",
        f"Esteem: {len(tribe.trophies)} trophy/trophies earned.",
        f"Self-Actualization: {era_label}"
        + (f", {len(tribe.buildings)} building(s) raised" if tribe.founded_city else ", no city founded yet")
        + ".",
    ]
    return (
        f"[COMMUNITY WELL-BEING]: Most pressing unmet need right now: {TIER_LABELS[focus]}. "
        + " ".join(lines)
    )
