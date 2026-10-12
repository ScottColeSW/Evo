"""2026-10-11 (the owner): the relationship ladder. Each tribe keeps one score per rival, its own chief's view of that rival (so two tribes can disagree), plus one for the raiders. The score moves with
what happens between them and sinks back toward zero a little every day; the band it falls in decides what is on offer (TRADE needs Neutral or better on both sides), how likely an envoy is to backfire,
and what the sidebar shows. Three caught spies make a rival an Enemy for good: locked, no decay, no trust, war is all that is left on the table.

Low to high: ENEMY (locked) < UNTRUSTWORTHY < UNTRUSTED (a rival just discovered starts here) < NEUTRAL < TRUSTED_TRADE_PARTNER. War is a separate stance (Tribe.stance_toward) that sits over the band.
Pure functions over the two Tribe attributes `standing` (id -> score) and `standing_locked` (ids), and `spies_caught_of` (id -> spies of theirs this tribe has caught)."""
from . import config

ENEMY = "ENEMY"
UNTRUSTWORTHY = "UNTRUSTWORTHY"
UNTRUSTED = "UNTRUSTED"
NEUTRAL = "NEUTRAL"
TRUSTED = "TRUSTED_TRADE_PARTNER"
ORDER = [ENEMY, UNTRUSTWORTHY, UNTRUSTED, NEUTRAL, TRUSTED]
LABELS = {
    ENEMY: "Enemy (never trusts)", UNTRUSTWORTHY: "Untrustworthy", UNTRUSTED: "Untrusted",
    NEUTRAL: "Neutral", TRUSTED: "Trusted Trade Partner",
}
RAIDERS = "raiders"  # the key the raiders' standing is kept under (never locked, never a trade partner)


def score(tribe, other_id: str) -> float:
    return tribe.standing.get(other_id, 0.0)


def is_enemy(tribe, other_id: str) -> bool:
    return other_id in tribe.standing_locked


def band(tribe, other_id: str) -> str:
    if is_enemy(tribe, other_id):
        return ENEMY
    s = score(tribe, other_id)
    if s <= config.STANDING_UNTRUSTWORTHY_AT:
        return UNTRUSTWORTHY
    if s >= config.STANDING_TRUSTED_AT:
        return TRUSTED
    if s >= config.STANDING_NEUTRAL_AT:
        return NEUTRAL
    return UNTRUSTED


def rank(band_name: str) -> int:
    return ORDER.index(band_name)


def adjust(tribe, other_id: str, delta: float) -> None:
    """Moves one tribe's view of another; an Enemy never changes."""
    if is_enemy(tribe, other_id):
        return
    tribe.standing[other_id] = max(config.STANDING_MIN, min(config.STANDING_MAX, score(tribe, other_id) + delta))


def trusts_enough_to_trade(a, b) -> bool:
    """TRADE needs Neutral or better in both directions."""
    return rank(band(a, b.id)) >= rank(NEUTRAL) and rank(band(b, a.id)) >= rank(NEUTRAL)


def note_spy_caught(catcher, spier) -> bool:
    """The catcher's view of the spying tribe drops config.SPY_CAUGHT_STANDING_DROP; the SPIES_CAUGHT_FOR_ENEMY-th spy caught makes it an Enemy for good. Returns True when it just became one."""
    adjust(catcher, spier.id, -config.SPY_CAUGHT_STANDING_DROP)
    catcher.spies_caught_of[spier.id] = catcher.spies_caught_of.get(spier.id, 0) + 1
    if catcher.spies_caught_of[spier.id] >= config.SPIES_CAUGHT_FOR_ENEMY and spier.id not in catcher.standing_locked:
        catcher.standing_locked.add(spier.id)
        catcher.standing[spier.id] = config.STANDING_MIN
        return True
    return False


def decay_day(tribe) -> None:
    """Once a day every score moves config.STANDING_DECAY_PER_DAY toward zero (trust fades, grudges soften); an Enemy does not."""
    for other_id in list(tribe.standing):
        if other_id in tribe.standing_locked:
            continue
        s = tribe.standing[other_id]
        s = max(0.0, s - config.STANDING_DECAY_PER_DAY) if s > 0 else min(0.0, s + config.STANDING_DECAY_PER_DAY)
        if s == 0.0:
            del tribe.standing[other_id]
        else:
            tribe.standing[other_id] = s


def summary(tribe) -> dict:
    """What the page shows: every rival with a non-zero score or a lock, as {id: {"score", "band"}} (the raiders too, under "raiders")."""
    ids = set(tribe.standing) | set(tribe.standing_locked)
    return {i: {"score": round(score(tribe, i), 2), "band": band(tribe, i)} for i in ids}
