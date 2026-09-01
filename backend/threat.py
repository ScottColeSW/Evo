"""Threat-proximity assessment from hostile rival tribes.

This is the reconciled version of the Agentic Evolution spec's Module A
(calculate_threat_proximity): the same distance-weighted exponential-decay math,
but rendered as an honest prompt fact instead of a hard override. Module A's own
version computed this score and then FORCED an action (STATE_OVERRIDE_RED:
FLEE_OR_PHALANX) once a threshold crossed -- directly conflicting with this
project's own, repeatedly-reinforced "no scripted directives" rule (a hardcoded
HUNT_DEER hunger nudge was already tried and reverted for exactly this reason,
confirmed again when this reconciliation was scoped). The math survives; the
override doesn't. This never touches available_actions or forces anything -- a new
prompt fact, same shape as instincts.py's SURVIVAL INSTINCT layer.

Scoped specifically to declared-WAR rival tribes -- raider proximity already has
its own honest fact (Simulation._advance_raider_approach's "RAIDERS ARE RIDING IN"
line), and an ALLIED or NEUTRAL rival, however close, isn't a threat.
"""

import math

from . import config


def threat_assessment_string(tribe, rival_tribes) -> str:
    """`rival_tribes`: non-extinct tribes other than `tribe` itself. Only tribes
    `tribe` has declared WAR on contribute -- see actions.py._declare_war."""
    threats = []
    for other in rival_tribes:
        if tribe.stance_toward.get(other.id) != "WAR":
            continue
        distance = math.hypot(other.x - tribe.x, other.y - tribe.y)
        level = math.exp(-config.THREAT_DECAY_RATE * distance)
        threats.append((level, other, distance))

    if not threats:
        return ""

    level, other, distance = max(threats, key=lambda entry: entry[0])
    if level < config.THREAT_ASSESSMENT_MIN_LEVEL:
        return ""

    if level >= 0.7:
        posture = "close and immediate"
    elif level >= 0.3:
        posture = "elevated"
    else:
        posture = "distant but real"

    return (
        f"[THREAT ASSESSMENT]: {other.name}, a declared enemy, is {distance:.0f} tiles away -- "
        f"{posture} danger ({level:.2f}/1.0). This is informational only; how to respond "
        "(fortify, retaliate, seek peace, ignore it) is your own decision."
    )
