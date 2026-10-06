"""The record of one fight, for the board's skirmish card (2026-10-06, the owner: "I'd like to see the battle scene when raiders strike or any battle really").

DECLARE_CONQUEST keeps its full-screen replay (it is the climactic ending, with its own multi-round payload). Everything else that is a fight (a raider strike on a
camp, a tribe raiding a rival, a strike on a raider camp, a wolf pack, the passive and active defenses) rides on its `recent_encounters` entry as a `skirmish`
dict, which the frontend plays as a small card that does not block the board. Only facts the simulation actually computed go in: the two sides, the odds the roll was
made against (None where the event has no roll, like a wolf pack), who prevailed, and the losses.
"""


def skirmish(title: str, attacker: str, defender: str, attacker_won: bool, attacker_chance: float | None = None,
             attacker_force: int | None = None, defender_force: int | None = None,
             attacker_lost: int = 0, defender_lost: int = 0, outcome: str = "") -> dict:
    return {
        "title": title,
        "attacker": attacker,
        "defender": defender,
        "attacker_won": bool(attacker_won),
        "attacker_chance": None if attacker_chance is None else round(max(0.0, min(1.0, attacker_chance)), 3),
        "attacker_force": attacker_force,
        "defender_force": defender_force,
        "attacker_lost": int(attacker_lost),
        "defender_lost": int(defender_lost),
        "outcome": outcome,
    }
