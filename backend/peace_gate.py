"""Phase 0 of docs/TRADE-GATE-DESIGN.md (2026-10-03): track and log whether a tribe has earned the peace tier (TRADE,
SEND_TRADE_EMISSARY, DECLARE_ALLIANCE). Nothing here changes what a tribe can do. It records the state the gate would use, so the
real numbers (when a first cost lands, how often tribes cull, how many outside contacts there are) can be read before any lock is
built.

The state lives on the tribe (`tribe.peace_gate`), not on the chief, so a new chief inherits it.

Three ways to open the tier, all closed to a tribe with a cull habit (any cull in the last PEACE_GATE_SCAR_WINDOW_NIGHTS nights):
  - "lesson":    a cost was felt (a cull, a loss in a raid or war, a famine) and then enough clean nights passed;
  - "stability": a long stretch with no cull, for tribes that were simply well managed;
  - "contact":   contact with the outside (a rival found, a spy sent, words brought home) on enough separate nights.
"""
from __future__ import annotations

from . import config

# Loss causes (the `cause` given to Simulation._lose_population) that count as a cost of war or of mismanagement. Environmental
# hazards (volcano, ocean, lightning, wolves and so on) are not: they teach nothing about governing or fighting.
CULL_CAUSES = frozenset({"overcrowding"})
WAR_CAUSES = frozenset({
    "raider_attack", "raider_ambush", "raid_losses", "failed_raid", "failed_raider_strike", "expel_raiders_failed",
    "conquest_round_loss", "spy_caught",
})
FAMINE_CAUSES = frozenset({"starvation", "thirst"})


def new_state() -> dict:
    return {
        "felt": False,            # a cost event has been felt
        "first_cost_cycle": None,
        "cost_events": [],        # the last few: {"cycle", "kind", "cause", "amount", "population"}
        "cull_nights": [],        # night numbers on which a cull happened (the night counter, not the cycle)
        "nights": 0,              # nights seen
        "clean_nights": 0,        # consecutive nights since the latest cull (counted once a cost has been felt)
        "heard_reports": 0,       # returning parties that brought words home (one per party, not per word)
        "contact_nights": [],     # nights on which any new outside contact happened
        "contact_seen": 0,        # the raw contact count at the last night
        "would_open": None,       # the path that would currently open the tier, or None
        "would_open_cycle": None, # the cycle it first would have opened
    }


def classify(cause: str) -> str | None:
    if cause in CULL_CAUSES:
        return "cull"
    if cause in WAR_CAUSES:
        return "war"
    if cause in FAMINE_CAUSES:
        return "famine"
    return None


def record_cost(state: dict, cycle: int, cause: str, amount: int, population_before: int) -> dict | None:
    """Notes a population loss. Returns the cost-event record when the loss counts, else None. A loss counts when it is of a kind
    above and at least PEACE_GATE_MIN_LOSS_FRACTION of the population (a lost scout is not a lesson)."""
    kind = classify(cause)
    if kind is None or amount <= 0:
        return None
    if population_before > 0 and amount / population_before < config.PEACE_GATE_MIN_LOSS_FRACTION and kind != "cull":
        return None
    event = {"cycle": cycle, "kind": kind, "cause": cause, "amount": amount, "population": population_before}
    state["cost_events"].append(event)
    del state["cost_events"][:-20]
    if not state["felt"]:
        state["felt"] = True
        state["first_cost_cycle"] = cycle
        state["clean_nights"] = 0
    return event


def note_night(state: dict, culled: bool, raw_contacts: int | None = None) -> None:
    """Called once per night, after the night's cull has happened (or not). raw_contacts is contacts(tribe): a night counts as a
    contact night when the raw count grew since the last one."""
    state["nights"] += 1
    if raw_contacts is not None:
        if raw_contacts > state["contact_seen"]:
            state["contact_nights"].append(state["nights"])
            del state["contact_nights"][:-30]
        state["contact_seen"] = raw_contacts
    if culled:
        state["cull_nights"].append(state["nights"])
        del state["cull_nights"][:-30]
        state["clean_nights"] = 0
    elif state["felt"]:
        state["clean_nights"] += 1


def scar(state: dict) -> int:
    """Culls in the last PEACE_GATE_SCAR_WINDOW_NIGHTS nights."""
    floor = state["nights"] - config.PEACE_GATE_SCAR_WINDOW_NIGHTS
    return sum(1 for n in state["cull_nights"] if n > floor)


def required_clean_nights(state: dict) -> int:
    """3, 6, 9, ... capped: the more recent culls, the longer the clean stretch has to be."""
    return min(config.PEACE_GATE_MAX_CLEAN_NIGHTS, config.PEACE_GATE_CLEAN_NIGHTS * max(1, scar(state)))


def contacts(tribe) -> int:
    """The raw outside-contact count: rivals found, spy missions run, and parties that brought words home. The first run showed that
    counting every word (about 25 per returning party) opened the tier at cycle 30, so the gate counts contact *nights* instead
    (see note_night): sustained contact, not a burst."""
    return len(tribe.discovered_rivals) + tribe.spy_missions_run + tribe.peace_gate["heard_reports"]


def evaluate(tribe, cycle: int) -> dict:
    """The gate's state for this tribe now, and which path (if any) would open the tier. Pure: reads, never changes the world."""
    state = tribe.peace_gate
    habit = scar(state) > 0
    required = required_clean_nights(state)
    n_contacts = len(state["contact_nights"])
    path = None
    if not habit:
        if state["felt"] and state["clean_nights"] >= required:
            path = "lesson"
        elif cycle >= config.PEACE_GATE_STABILITY_CYCLES:
            path = "stability"
        elif n_contacts >= config.PEACE_GATE_CONTACT_NIGHTS:
            path = "contact"
    elif state["felt"] and state["clean_nights"] >= required:
        path = "lesson"  # a long enough clean stretch after the habit still counts
    return {
        "felt": state["felt"], "clean_nights": state["clean_nights"], "required_clean_nights": required, "scar": scar(state),
        "contacts": n_contacts, "cycle": cycle, "would_open": path,
    }
