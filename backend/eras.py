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
two eras both require population 80 (real live runs reached this), the same
population every subsequent era leans on higher resource requirements to keep
climbing past rather than a raised population floor -- there's no longer a
population ceiling at all (config.POPULATION_GROWTH_CAP, explicit request: "we
should not put a cap on population").

Later narrowed to 6 real stages: a run finally reached the old ceiling
("cosmic_post_human") for the first time with nothing left to do there --
explicit request, "we have to extend it now." The old top three
(mechanization_era/silicon_era/cosmic_post_human) were all empty reserved
slots (unlocks_actions=()); they're replaced outright by two real eras
(object_creator_era, war_and_world_domination_era) rather than kept around
alongside new ones appended after. ENABLE_SELF_MODIFICATION (the old reason
mechanization_era existed) and symbolic doctrine-sharing (silicon_era's old
reserved purpose) both stay explicitly deferred, not folded into either
replacement -- each needs its own future design/sign-off pass.
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
            "BUILD_FIRE", "SCOUT", "EXPLORATION_PARTY", "HUNTING_PARTY", "RELOCATE", "RAID", "TRADE", "SEND_TRADE_EMISSARY", "BREED",
            # Explicit request: "this can happen early" -- COOK_FOOD moved out of
            # Tribal Synapse's era gate entirely. Its own real prerequisites (a
            # successful hunt and a successfully-built fire, ever -- see
            # Simulation._prepare_turn) are what actually gate it now, and both of
            # those are only reachable post-settling anyway (see config.
            # PRE_SETTLEMENT_ACTIONS), so this can't fire before a tribe has a camp.
            "COOK_FOOD",
            # Same move, same reasoning (explicit request, 2026-09-05): PLANT_CROP/
            # GATHER_EGGS/CATCH_FISH used to wait for Tribal Synapse, which itself
            # requires banking 20 food (cognitive_horizon, below) -- but a tribe
            # stuck foraging the same settled tile can rarely bank that much food in
            # the first place (see _harvest's scarcity/depletion mechanic). That's a
            # real chicken-and-egg: farming/fishing/eggs are supposed to be the way
            # OUT of a food-scarcity trap, not something locked behind first
            # escaping it. Confirmed live: two tribes sat in Primitive Dawn for a
            # full 128-cycle run, food oscillating 0-16 the whole time, wood/stone/
            # water all comfortably past the threshold -- food alone blocked every
            # era advance. Simulation._prepare_turn's own "Settled gate, not a real-
            # water one" filter already gates these three on has_ever_settled
            # regardless of era, so moving them here is the same real-prerequisite
            # swap COOK_FOOD already got, not a loosening of any actual constraint.
            "PLANT_CROP", "GATHER_EGGS", "CATCH_FISH",
            # Explicit request, 2026-09-09: "Territory boundaries must be cleared
            # of threats before they can really start building anything really."
            # Has to unlock this early -- Cognitive Horizon (below) is where real
            # construction (CONSTRUCT_WALL, BUILD_LONG_HOUSE, ...) begins, and the
            # menu-lock this action exists to clear (Simulation._prepare_turn)
            # would otherwise block every one of those from the very first cycle
            # a tribe could ever reach them.
            "CLEAR_TERRITORY",
        ),
        announcement="{tribe} has awakened at the dawn of the Primitive Age.",
    ),
    Era(
        key="cognitive_horizon",
        label="Cognitive Horizon",
        requires_population=12,
        # Live-run correction (2026-09-02): "some later game options are coming up
        # too early... they don't even have food under control." No era, at any
        # tier, ever required food -- a tribe could clear every threshold here
        # while genuinely food-fragile. Food gets water's own buffer ratio (2x
        # required vs. spent), not wood/stone's looser 1.33x -- food actually
        # drains on its own via upkeep between now and whenever it's spent, the
        # same real risk water carries, unlike wood/stone which only move when a
        # tribe chooses to spend them.
        requires_resources={"water": 20, "stone": 20, "wood": 20, "food": 20},
        advancement_cost={"wood": 15, "stone": 15, "water": 10, "food": 10},
        # Redesigned 2026-09-08 (explicit request: "Cognitive Horizon needs to be
        # an agrarian society kind of evolution... take about half of the action
        # items from Tribal Synapse"). Originally unlocked nothing of its own on
        # the theory that a small model essentially never reaches for a newly
        # unlocked action anyway (see _check_for_celebration's own docstring) --
        # true, but that theory was about a SINGLE new action landing in a big
        # already-crowded menu, not about giving a whole tier real content. Every
        # action moved here is real settlement/subsistence infrastructure with no
        # tie to war, diplomacy, or formal knowledge -- an agrarian village's own
        # wall, houses, water, storage, and food/goods processing, not yet a
        # standing army or a library. Each one keeps its own existing real
        # prerequisite (BUILD_DOCK still needs fishing_learned, BUILD_SAWMILL
        # still needs wood_ever_gathered, etc.) -- moving the ERA gate earlier
        # just stops it from ALSO waiting on population 50 once its real
        # prerequisite is already met, the same "real prerequisite, not era
        # progression" fix COOK_FOOD/PLANT_CROP/GATHER_EGGS/CATCH_FISH already
        # got out of this exact era, one tier further down the ladder.
        #
        # CONSTRUCT_WALL and BUILD_LONG_HOUSE moved together deliberately: every
        # other agrarian building here is independent, but BUILD_KITCHEN needs a
        # standing Long House. Bringing the whole chain down means a tribe can
        # fully wall in, house, and feed itself well before Tribal Synapse, then
        # arrive there already ready for the next real tier of defense
        # (BUILD_MOAT/BUILD_KEEP, still gated on this same housing progress)
        # instead of starting fortification from zero at population 50.
        # BUILD_MOAT/BUILD_KEEP themselves stay in Tribal Synapse -- reinforcing
        # and escalating an already-secure settlement's defense reads as "true
        # society" content, not founding-era survival.
        #
        # Explicit correction, 2026-09-09: "I'm very tempted to remove the Wall
        # restriction on it" -- BUILD_LONG_HOUSE no longer needs the wall at
        # all (see actions._build_long_house's own docstring for why); it's
        # kept alongside CONSTRUCT_WALL here purely because they were already
        # grouped, not because either still depends on the other.
        unlocks_actions=(
            "CONSTRUCT_WALL", "BUILD_LONG_HOUSE", "UPGRADE_LONG_HOUSE", "BUILD_DOCK", "BUILD_FISHERY",
            "BUILD_SAWMILL", "BUILD_QUARRY", "BUILD_KITCHEN", "BUILD_TANNERY", "BUILD_DEER_PEN", "BUILD_WAREHOUSE",
            "UPGRADE_WAREHOUSE", "BUILD_HATCHERY", "BUILD_COOP", "BUILD_BATH_HOUSE", "BUILD_WELL",
        ),
        announcement="{tribe} crosses into the Cognitive Horizon -- reflection begins to compound into wisdom.",
    ),
    Era(
        key="tribal_synapse",
        label="Tribal Synapse",
        # Explicit request (2026-09-07): "much bigger populations going to war,"
        # reachable in practice now that population growth itself scales with
        # tribe size and Well-Being (config.POPULATION_GROWTH_SCALE_DIVISOR/
        # _WELLBEING_FLOOR) instead of a flat +1/cycle -- raised 20 -> 50 without
        # the real-time cost that would have meant under the old flat rate.
        requires_population=50,
        # Wood used to be spent on advancing (advancement_cost below) without ever
        # being required beforehand -- a tribe with 0 wood could still advance, it
        # just floored at 0 instead of actually paying the cost. Real requirement now,
        # with the same buffer-above-cost pattern stone/water already use (40
        # required vs. 30 spent -- advancing doesn't zero the tribe out).
        # Food added for the same reason as Cognitive Horizon above -- water's 2x
        # buffer ratio, not wood/stone's 1.33x, since food keeps draining via
        # upkeep on its own.
        requires_resources={"water": 40, "stone": 40, "wood": 40, "food": 40},
        advancement_cost={"wood": 30, "stone": 30, "water": 20, "food": 20},
        # Redesigned 2026-09-08 alongside Cognitive Horizon above -- the 12
        # agrarian-infrastructure actions that used to unlock here (BUILD_LONG_
        # HOUSE, BUILD_DOCK/_FISHERY/_SAWMILL/_QUARRY/_KITCHEN/_TANNERY/
        # _WAREHOUSE/_HATCHERY/_BATH_HOUSE/_WELL, CONSTRUCT_WALL) moved down a
        # tier. What's left here is genuinely "true society" content: a standing
        # military (the whole Military branch, plan file valiant-forging-falcon.md),
        # foreign relations (ALLIANCE/WAR), escalated fortification on top of an
        # already-standing wall (MOAT/KEEP), and formal knowledge institutions
        # (LIBRARY/RESEARCH) -- organized defense, diplomacy, and scholarship, not
        # founding-era survival.
        unlocks_actions=(
            "STRIKE_RAIDER_CAMP", "EXPEL_RAIDERS_FROM_TERRITORY",
            "BUILD_BARRACKS", "UPGRADE_BARRACKS", "TRAIN_BATTALION",
            "DECLARE_ALLIANCE", "DECLARE_WAR", "SPY", "BUILD_MOAT", "BUILD_KEEP",
            "BUILD_LIBRARY", "RESEARCH",
        ),
        announcement="{tribe} has forged the Tribal Synapse -- true society begins!",
    ),
    Era(
        key="monolithic_era",
        label="Monolithic Era",
        requires_population=200,  # see tribal_synapse's own comment on the 2026-09-07 rescale
        # Fur (Tannery/Mine -- actions.py._build_tannery, world.
        # UNIQUE_RESOURCE_BY_BIOME) is the first requires_resources entry that
        # isn't a core Tribe attribute -- see simulation.py._era_resource_amount/
        # _spend_era_resource, added the same pass specifically so this works
        # (getattr/setattr alone would have silently always read/spent 0 against
        # tribe.unique_resources). Tannery yields TANNERY_YIELD_PER_CYCLE (4)
        # Fur/cycle once built, so 20 is a handful of cycles, not a bottleneck --
        # population is the real pacing lever at this tier, not Fur.
        requires_resources={"water": 65, "stone": 65, "wood": 65, "Fur": 20},
        advancement_cost={"wood": 45, "stone": 45, "water": 45, "Fur": 15},
        unlocks_actions=(
            "BUILD_FORTRESS", "BUILD_CASTLE", "BUILD_ROAD", "BUILD_MINE",
            "BUILD_FORGE", "FORGE_ITEM", "USE_ITEM", "GATHER_ORE",
        ),
        # Founding itself is a separate, real milestone (Simulation._advance_city_founding
        # requires at least one Long House first) -- this only announces the era
        # threshold being reached, not the city actually standing yet.
        announcement="{tribe} enters the Monolithic Era, ready to found a lasting city!",
        founds_city=True,
    ),
    # Explicit request, after a run first reached the old era ceiling with
    # nothing left to do there: "we have to extend it now." Two ideas floated
    # earlier the same session -- an "Object Creator" manufacturing factory
    # ("they can create anything they want and we have to somehow support
    # it") as a second-to-last era, "War and World Domination" as the true
    # final one -- replace the three old empty reserved slots
    # (mechanization_era/silicon_era/cosmic_post_human, all unlocks_actions=())
    # below rather than getting appended after them; the ladder goes from 7
    # stages to 6, all of them real. ENABLE_SELF_MODIFICATION (the old reason
    # mechanization_era existed) stays explicitly deferred -- it needs its own
    # separate sign-off given the risk profile (autonomous filesystem writes),
    # not folded into this pass.
    Era(
        key="object_creator_era",
        label="Object Creator Era",
        requires_population=800,  # see tribal_synapse's own comment on the 2026-09-07 rescale
        requires_resources={"water": 100, "stone": 100, "wood": 100, "Fur": 50},
        advancement_cost={"wood": 70, "stone": 70, "water": 70, "Fur": 35},
        unlocks_actions=("BUILD_OBJECT_CREATOR", "CREATE_ITEM", "CREATE_USEFUL_STRUCTURE"),
        announcement="{tribe} enters the Object Creator Era -- they can build anything they can imagine!",
    ),
    Era(
        key="war_and_world_domination_era",
        label="War and World Domination",
        requires_population=1500,  # see tribal_synapse's own comment on the 2026-09-07 rescale
        requires_resources={"water": 125, "stone": 125, "wood": 125, "Fur": 100},
        advancement_cost={"wood": 90, "stone": 90, "water": 90, "Fur": 70},
        unlocks_actions=("DECLARE_CONQUEST",),
        announcement="{tribe} enters the age of War and World Domination!",
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
