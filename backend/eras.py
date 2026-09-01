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

Reworked into a 7-stage ladder (was 3: stone_age/bronze_age/classical_age) to
reconcile with the user's "Agentic Evolution Architecture" spec -- see the plan at
the time of this change for the full reconciliation (what already existed vs. what
was rejected as conflicting with this project's own "no scripted directives" rule).
Thresholds extrapolate the original 3-era curve (population 0 -> 20 -> 40); the top
two eras cap population at config.POPULATION_GROWTH_CAP (80), the hard ceiling
_grow_population already enforces, and lean on higher resource requirements instead.
Mechanization/Silicon/Cosmic Post-Human currently unlock no new actions -- their real
content (self-modification, symbolic doctrine-sharing, the endgame epilogue) is
deliberately sequenced as separate follow-up work, not invented here just to fill
the slot.
"""

from dataclasses import dataclass


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
        key="primitive_dawn",
        label="Primitive Dawn",
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
            # Explicit request: "this can happen early" -- COOK_FOOD moved out of
            # Tribal Synapse's era gate entirely. Its own real prerequisites (a
            # successful hunt and a successfully-built fire, ever -- see
            # Simulation._prepare_turn) are what actually gate it now, and both of
            # those are only reachable post-settling anyway (see config.
            # PRE_SETTLEMENT_ACTIONS), so this can't fire before a tribe has a camp.
            "COOK_FOOD",
        ),
        announcement="{tribe} has awakened at the dawn of the Primitive Age.",
    ),
    Era(
        key="cognitive_horizon",
        label="Cognitive Horizon",
        requires_population=12,
        requires_resources={"water": 20, "stone": 20, "wood": 20},
        advancement_cost={"wood": 15, "stone": 15, "water": 10},
        # No new discrete action -- small models essentially never reach for a newly
        # unlocked one anyway (see _check_for_celebration's own docstring on this
        # exact finding). This era's real content is a passive system, not an
        # action: the tribe-level genetics.breed() cultural crossover, sequenced as
        # separate follow-up work rather than invented here just to fill the slot.
        unlocks_actions=(),
        announcement="{tribe} crosses into the Cognitive Horizon -- reflection begins to compound into wisdom.",
    ),
    Era(
        key="tribal_synapse",
        label="Tribal Synapse",
        requires_population=20,
        # Wood used to be spent on advancing (advancement_cost below) without ever
        # being required beforehand -- a tribe with 0 wood could still advance, it
        # just floored at 0 instead of actually paying the cost. Real requirement now,
        # with the same buffer-above-cost pattern stone/water already use (40
        # required vs. 30 spent -- advancing doesn't zero the tribe out).
        requires_resources={"water": 40, "stone": 40, "wood": 40},
        advancement_cost={"wood": 30, "stone": 30, "water": 20},
        unlocks_actions=(
            "CONSTRUCT_WALL", "PLANT_CROP", "GATHER_EGGS", "CATCH_FISH", "STRIKE_RAIDER_CAMP",
            "BUILD_LONG_HOUSE", "DECLARE_ALLIANCE", "DECLARE_WAR",
        ),
        announcement="{tribe} has forged the Tribal Synapse -- true society begins!",
    ),
    Era(
        key="monolithic_era",
        label="Monolithic Era",
        requires_population=40,
        requires_resources={"water": 60, "stone": 30, "wood": 50},
        advancement_cost={"wood": 40, "stone": 40, "water": 40},
        unlocks_actions=(),  # reserved: BUILD_CASTLE/BUILD_ROAD/EXPAND_TERRITORY, not yet implemented
        announcement="{tribe} has founded a lasting city and entered the Monolithic Era!",
        founds_city=True,
    ),
    Era(
        key="mechanization_era",
        label="Mechanization Era",
        requires_population=60,
        requires_resources={"water": 80, "stone": 60, "wood": 70},
        advancement_cost={"wood": 50, "stone": 50, "water": 50},
        unlocks_actions=(),  # reserved: flips config.ENABLE_SELF_MODIFICATION on, not yet implemented
        announcement="{tribe} enters the Mechanization Era!",
    ),
    Era(
        key="silicon_era",
        label="Silicon Era",
        requires_population=80,  # config.POPULATION_GROWTH_CAP -- the hard ceiling
        requires_resources={"water": 100, "stone": 80, "wood": 90},
        advancement_cost={"wood": 60, "stone": 60, "water": 60},
        unlocks_actions=(),  # reserved: symbolic doctrine-sharing crossover, not yet implemented
        announcement="{tribe} enters the Silicon Era!",
    ),
    Era(
        key="cosmic_post_human",
        label="Cosmic Post-Human Age",
        requires_population=80,
        requires_resources={"water": 120, "stone": 100, "wood": 110},
        advancement_cost={"wood": 70, "stone": 70, "water": 70},
        unlocks_actions=(),  # reserved: the endgame epilogue/Transcendent Monument, not yet implemented
        announcement="{tribe} transcends into the Cosmic Post-Human Age!",
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
