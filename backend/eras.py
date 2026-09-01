"""The progression ladder tribes climb as they grow.

Kept as an ordered, data-driven table rather than a hardcoded if/elif chain in
Simulation -- adding an era, a new resource, or a new unlockable action means editing
this table, not simulation logic. This is also what would let a themed reskin (e.g. a
cyberspace variant: compute/bandwidth/power instead of wood/stone/water) swap
vocabulary without touching the advancement logic itself.

Advancement is automatic once a tribe's population and stockpiles clear an era's
requirements -- it is deliberately NOT gated behind the model choosing a special
"advance" action. Relying on a small quantized model to correctly reason its way to a
meta-progression action would make the payoff moment unreliable; the tribes' own
actions (what they gather, hunt, and build) still fully determine *when* they cross
the threshold, they just don't have to also realize to declare it.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Era:
    key: str
    label: str
    requires_population: int
    requires_resources: dict[str, int]  # resource attr name -> minimum stockpile
    advancement_cost: dict[str, int]  # resource attr name -> amount spent on advancing
    unlocks_actions: tuple[str, ...]
    announcement: str  # "{tribe}" is substituted with the tribe's name
    founds_city: bool = False


ERAS: tuple[Era, ...] = (
    Era(
        key="stone_age",
        label="Stone Age",
        requires_population=0,
        requires_resources={},
        advancement_cost={},
        unlocks_actions=(
            # Explicit request: a tribe always has something worth doing, and a
            # parse failure now falls back to a real available action instead of a
            # no-op (see backend/simulation.py._resolve_action) -- there's no IDLE
            # concept anywhere anymore, defensive fallback or otherwise.
            "GATHER_WOOD", "GATHER_STONE", "GATHER_WATER", "GATHER_FOOD", "HUNT_DEER",
            "BUILD_FIRE", "SCOUT", "HUNTING_PARTY", "RELOCATE", "RAID", "TRADE", "SEND_TRADE_EMISSARY", "BREED",
        ),
        announcement="{tribe} has awakened as a Stone Age tribe.",
    ),
    Era(
        key="bronze_age",
        label="Bronze Age",
        requires_population=20,
        # Wood used to be spent on advancing (advancement_cost below) without ever
        # being required beforehand -- a tribe with 0 wood could still advance, it
        # just floored at 0 instead of actually paying the cost. Real requirement now,
        # with the same buffer-above-cost pattern stone/water already use (40
        # required vs. 30 spent -- advancing doesn't zero the tribe out).
        requires_resources={"water": 40, "stone": 40, "wood": 40},
        advancement_cost={"wood": 30, "stone": 30, "water": 20},
        unlocks_actions=("CONSTRUCT_WALL", "PLANT_CROP", "GATHER_EGGS", "GATHER_FISH", "STRIKE_RAIDER_CAMP", "COOK_FOOD"),
        announcement="{tribe} has forged its way into the Bronze Age!",
    ),
    Era(
        key="classical_age",
        label="Classical Age",
        requires_population=40,
        requires_resources={"water": 60, "stone": 30, "wood": 50},
        advancement_cost={"wood": 40, "stone": 40, "water": 40},
        unlocks_actions=(),  # reserved for trade/diplomacy actions in a later phase
        announcement="{tribe} has founded a lasting city and entered the Classical Age!",
        founds_city=True,
    ),
)

_BY_KEY = {era.key: era for era in ERAS}


def era_index(key: str) -> int:
    for i, era in enumerate(ERAS):
        if era.key == key:
            return i
    return 0


def next_era(current_key: str) -> Era | None:
    idx = era_index(current_key)
    if idx + 1 < len(ERAS):
        return ERAS[idx + 1]
    return None


def unlocked_actions_through(current_key: str) -> set[str]:
    idx = era_index(current_key)
    actions: set[str] = set()
    for era in ERAS[: idx + 1]:
        actions.update(era.unlocks_actions)
    return actions
