import asyncio
import difflib
import importlib
import math
import random

from . import architect, city_layout, config, physics
from .actions import (
    ACTION_REGISTRY, BIOME_YIELD_MULTIPLIER, GAME_SPECIES_BY_BIOME, GAME_SPECIES_LABEL,
    _created_object_bonus, _eligible_breeding_pair, _eligible_warrior_candidate, _food_multiplier,
    _item_storage_cap, _labor_multiplier, _long_house_fur_discount, _push_past_visited_ground,
    _record_combat, _storage_cap,
    expedition_capacity,
)
from .ancestral_matrix import AncestralTraumaMatrix
from .breeding import breed_individuals
from .genetics import breed, hatch
from .reflection import AWARD_CATEGORIES, reflect_on_history
from .eras import ERAS, era_index, next_era, unlocked_actions_through
from .event_log import RunEventLog, TribeHistory
from .scoreboard import record_tribe_result
from .instincts import survival_bias_string
from .threat import threat_assessment_string
from .wellbeing import compute_wellbeing
from .leadership import elect_chief, name_settlement
from .memory import TribeMemory
from .might import compute_might
from .ollama_client import OllamaClient
from .prompts import compile_live_state_prompt, get_prime_consciousness_prompt
from .scheduler import ModelBatchScheduler
from .self_mod import SelfModEngine
from .translation_matrix import TranslationConfidenceMatrix
from .vram_guard import HardwareVRAMBoundaryGuard
from .world import (
    BIOME_LABELS, UNIQUE_RESOURCE_BY_BIOME, WILDLIFE_SITE_TYPES, Landscape, biome_at, find_nearby_site,
    mark_visited_sector,
)

# Historically one spawn per land biome (forest, mountains, plains, river), each chosen
# to sit within real reach of fresh water (roughly 11-16 tiles by nearest_water) and with
# real clearance from every grid edge (see actions.py._reflect_into_grid's own comment --
# a spawn too close to an edge compresses two genuinely different SCOUT headings back
# toward each other when both clip the same boundary, read live as "the scouts basically
# follow the first one").
#
# Map dream, phase 2: turning the map into a real island (ocean insets on all four
# sides, not just the east) originally forced all four spawns into a tiny interior
# band -- corrected down to a real 4-8 tile ocean frame (see world.config's
# WEST/NORTH/SOUTH_COAST_INSET_BASE) after "the Ocean was only supposed to be a
# 'frame'." With that shallower frame, the original two landmark-relative slots
# only needed a small nudge to still clear SCOUT_PATROL_DISTANCE=25 from every
# coast, confirmed each time by directly sampling the real boundary functions,
# not estimated.
#
# Explicit follow-up requests, after watching live runs on the corrected map:
# first, swap which slot sits north vs. south (slot 0 "Tribe 1" north, slot 1
# "Tribe 2" south); then, told the first attempt's compliant-but-conservative
# points ((48, 30)/(51, 68), chosen to keep SCOUT_PATROL_DISTANCE=25 clearance
# from the coast) were "still too close" together, and to push both closer to
# their respective edges. The exact points originally marked on a screenshot,
# (48, 19)/(51, 77), go one real constraint too far, though: see
# test_every_spawn_point_keeps_scout_patrol_clearance_from_every_grid_edge's
# own comment -- a documented live bug (two scouts leaving in nearly the same
# direction) means every spawn needs at least 20 tiles of clearance from the
# literal grid edge (0 or 99), not just the coastline, and y=19 undercuts that
# by one tile.
#
# Slot 1 ("Tribe 2") moved again after the lake's natural-hydrology rework
# grew a real southwest bay (see scripts/generate_hydrology.py): the old
# (51, 78) ended up only 23.3 tiles from the lake's new nearest shore, inside
# SCOUT_PATROL_DISTANCE=25 -- "instant range of Scouts," trivializing the
# water search that spawn was meant to pose. (64, 70), confirmed
# computationally, is 33.2 tiles from the lake (clear of patrol range) while
# staying within a single expedition's reach of real water (19.1 tiles,
# comfortably under EXPEDITION_SPEED*EXPEDITION_MAX_DAYS=50 -- see
# test_every_spawn_point_is_within_a_single_expeditions_reach_of_water),
# still solid plains, 21.1 tiles clear of the nearest coastline (the map's
# geometry doesn't leave room for the full 25 once the lake-distance and
# edge-clearance constraints are both satisfied here), and at least 20 tiles
# from every other spawn.
#
# Slot 0 ("Tribe 1") moved too: "the scout is too close for a fair takeoff."
# The old (48, 21) cleared the west/east/south coastlines comfortably but had
# only 15.2 tiles of north clearance -- well under SCOUT_PATROL_DISTANCE=25,
# so any northward-leaning heading in the rotation sweep launched a scout
# straight toward the coast band instead of a real, comparable patrol. (67,
# 31), confirmed computationally, clears all four coastlines by 25+ tiles
# (a genuinely fair takeoff in every direction), the literal grid edge by 31,
# the volcano by ~57, stays solid plains, sits 12.5 tiles from real water
# (well within a single expedition's reach), and 27.7+ tiles from every
# other spawn.
# Slots 2/3 are untouched fallbacks for a 3rd/4th tribe -- their spacing
# against the new slot 0/1 positions is tighter than the original four-corner
# layout, a direct result of pulling 0/1 toward the coast, but all four remain
# distinct, non-clustered starting points.
SPAWN_POINTS = [(72, 17), (54, 70), (50, 55), (40, 37)]
COLORS = ["#c084fc", "#fb923c", "#34d399", "#60a5fa"]
# Design intent, not just "whichever body is closest": Tribe 1 settles the
# river, Tribe 2 the lake -- each gets a distinct natural-barrier wall ring
# (city_layout._is_natural_barrier) against a different water feature instead
# of two tribes converging on the same one. Passed to Tribe.
# seed_scout_heading_toward_water per spawn slot (Simulation.__init__); slots
# 2/3 have no assigned preference yet (untouched 3rd/4th-tribe fallback), so
# they keep searching for either.
SPAWN_WATER_TARGET_KINDS = [("river",), ("lake",)]

# See _prepare_turn's own use of this -- every one-time structure with a single
# boolean "already built" flag is retired from a tribe's available_actions the
# moment that flag is set, the same treatment BUILD_FIRE/COOK_FOOD/GATHER_FOOD
# already get individually. Long House/Warehouse/farm plots are deliberately
# excluded -- genuinely repeatable, gated by their own real capacity check.
ONE_TIME_BUILD_FLAGS = {
    "BUILD_DOCK": "dock_built", "BUILD_FISHERY": "fishery_built",
    "BUILD_SAWMILL": "sawmill_built", "BUILD_QUARRY": "quarry_built",
    "BUILD_KITCHEN": "kitchen_built", "BUILD_MOAT": "moat_built",
    "BUILD_KEEP": "keep_built", "BUILD_FORTRESS": "fortress_built",
    "BUILD_CASTLE": "castle_built", "BUILD_TANNERY": "tannery_built",
    "BUILD_MINE": "mine_built", "BUILD_FORGE": "forge_built",
    "BUILD_ROAD": "road_built", "BUILD_HATCHERY": "hatchery_built",
    "BUILD_BATH_HOUSE": "bath_house_built", "BUILD_LIBRARY": "library_built",
    "BUILD_WELL": "well_built", "BUILD_OBJECT_CREATOR": "object_creator_built",
}

# See _prepare_turn's survival-crisis filter. A live run showed a tribe stay at 0
# food for ~24 consecutive cycles, correctly naming "we are starving" in its own
# rationale while still choosing EXPAND_TERRITORY/GATHER_STONE/BUILD_LONG_HOUSE/
# BUILD_WAREHOUSE -- consistent with this project's documented "a prompt fact
# doesn't reliably redirect a small model" pattern (instincts.py's
# survival_bias_string already names this exact set of actions as a fact, and
# that clearly wasn't enough on its own). Once a real crisis is active, the menu
# itself is cut down to only actions that can plausibly help -- building,
# expansion, trade, and family plans can wait. RELOCATE is deliberately excluded
# here even though it's exempt from the repetition throttle: it carries a real
# food/water cost of its own (see its ACTION_DESCRIPTIONS entry), which could
# make an active crisis worse rather than better.
#
# PLANT_CROP included (2026-09-05, live bug report: a tribe with cooking learned
# still starved two members to death). Root cause: GATHER_FOOD's own yield runs
# through _harvest's scarcity/depletion mechanic, which a tribe repeatedly
# foraging the same settled tile drives toward MAX_SCARCITY -- at that point even
# cooking's multiplier is stretching an already-tiny raw harvest. _advance_farming
# bypasses scarcity entirely (a flat per-plot yield, see config.CROP_HARVEST_YIELD),
# making it the real fix for ground foraged into the ground, not just another food
# action -- but the observed tribe never once chose it in 94 cycles. Without this,
# a survival crisis cuts PLANT_CROP from the menu at the exact moment the visible_
# entities nudge below (in _prepare_turn) is trying to point the tribe at it --
# the same "never dangle an action that isn't actually offered" reasoning
# AFFORDABILITY_CHECKS already follows.
SURVIVAL_CRISIS_ACTIONS = {
    "GATHER_FOOD", "HUNT_DEER", "CATCH_FISH", "HUNTING_PARTY", "GATHER_EGGS", "COOK_FOOD",
    "GATHER_WATER", "SCOUT", "PLANT_CROP",
}

# Explicit request: "if they choose Wall, they have to complete it, no changing
# orders other than to collect what is needed to complete it. gather, build,
# gather, build, and so forth, until the Wall is 100%." Once tribe.
# wall_commitment_active is set (actions.py._construct_wall, the moment
# CONSTRUCT_WALL is chosen for a still-incomplete section), the menu narrows to
# just the wall itself, its own two resources, and the same survival-crisis set
# above -- a wall push still shouldn't be able to starve a tribe. BUILD_LONG_HOUSE
# is added back in separately, only while a banked credit exists (see
# wall_lock_long_house_credits).
WALL_LOCK_ACTIONS = {"CONSTRUCT_WALL", "GATHER_WOOD", "GATHER_STONE"} | SURVIVAL_CRISIS_ACTIONS

# See _prepare_turn's affordability filter. A live run showed a tribe stuck at
# wood=1 for 150+ cycles, cycling BUILD_WAREHOUSE/BUILD_FISHERY/BREED without
# ever choosing GATHER_WOOD -- BUILD_WAREHOUSE and BUILD_FISHERY both cost wood
# it never had, so those two were silently failing (each action's own guard
# clause already returns None below the cost) every single time they were
# picked. Never crossed the repetition throttle because it alternates between
# actions instead of repeating one 4x straight. Same category of bug as the
# very first diagnostic run's finding ("EXPAND_TERRITORY failed all 80 attempts
# on affordability") -- this closes it generally instead of one action at a
# time: a flat, guaranteed-no-op cost is checked here and the action is hidden
# from the menu entirely, the same "don't dangle an impossible choice" logic
# ONE_TIME_BUILD_FLAGS already applies to a satisfied one-time flag. Each
# lambda reads the exact same config constant its own action function already
# guards on, so the two can't drift out of sync.
#
# Live-run correction: this originally left CONSTRUCT_WALL out on the theory
# that its per-section, progress-scaled cost "degrades gracefully" instead of
# being a flat all-or-nothing amount -- wrong at wood=0 specifically, where it
# fails outright exactly like every other action here (confirmed live: a wall
# section's progress was bit-for-bit identical 100 cycles apart while
# CONSTRUCT_WALL kept getting chosen, the exact same no-op-oscillation bug this
# whole table exists to close). _wall_next_afford_cost mirrors actions.
# _construct_wall's own cost computation exactly, without mutating any state.
def _is_food_secure(tribe) -> bool:
    """A Kitchen plus a genuinely proven, passive food source (a Fishery, or at
    least one real harvest ever brought in) -- see Simulation._advance_food_supply's
    own docstring for the full reasoning. Module-level (not a Simulation method) so
    both Tribe.to_dict (self) and every Simulation method (tribe) can share the one
    real definition instead of repeating the same boolean expression -- confirmed
    live why that matters: tribe.food_crisis_active used to compute this same
    "is food actually a crisis" question from raw numbers alone, correct only by the
    incidental fact that a secure tribe's food happens to already be huge by the
    time it runs, not by an explicit guarantee."""
    return tribe.kitchen_built and (tribe.fishery_built or tribe.last_harvest_cycle > 0)


def _is_water_secure(tribe) -> bool:
    """config.WATER_SECURITY_SITE_THRESHOLD distinct confirmed water sources -- see
    Simulation._advance_water_supply's own docstring. Module-level for the same
    shared-definition reason as _is_food_secure above."""
    return len(tribe.confirmed_water_sites) >= config.WATER_SECURITY_SITE_THRESHOLD


def _is_wood_secure(tribe) -> bool:
    """A Sawmill plus a real, discovered-and-in-use Timber Grove -- see
    Simulation._advance_wood_supply's own docstring. Module-level for the same
    shared-definition reason as _is_food_secure/_is_water_secure above."""
    return tribe.sawmill_built and tribe.lumber_site is not None


def _is_stone_secure(tribe) -> bool:
    """Stone's own bar, set higher than wood's by explicit request: a Quarry
    alone isn't enough -- real stone mastery means a Mine too. BUILD_MINE
    already requires quarry_built as its own prerequisite (see AFFORDABILITY_
    CHECKS), so checking both here is belt-and-suspenders, not two independent
    facts -- but stated explicitly to match the request and to stay correct
    even if BUILD_MINE's own prerequisites ever change. A Mine also already
    requires a real discovered mine_sites entry to build in the first place
    (see actions._build_mine), so this still carries the same "a genuinely
    discovered, real resource site" spirit Sawmill/Timber Grove's own
    lumber_site check has, just via the Mine's site instead of the Quarry's.
    See Simulation._advance_stone_supply."""
    return tribe.quarry_built and tribe.mine_built


def _wall_next_afford_cost(tribe) -> tuple[int, int] | None:
    target = city_layout.next_wall_work_section(tribe)
    if target is None:
        return None  # nothing left to build/reinforce -- not a cost problem
    ring_i, sec_i = target
    section = tribe.wall_rings[ring_i]["sections"][sec_i]
    if section["progress"] >= 100:
        return config.WALL_LAYER_WOOD_COST, config.WALL_LAYER_STONE_COST
    added = min(100 - section["progress"], round(config.WALL_PROGRESS_PER_ACTION_BASE * _labor_multiplier(tribe.population)))
    return round(config.WALL_WOOD_COST_TOTAL * added / 100), round(config.WALL_STONE_COST_TOTAL * added / 100)


def _warehouse_capacity_note(tribe: "Tribe") -> str:
    """Live trace finding (run_20260908_082234): a tribe that grew past population
    1600 on the original STORAGE_CAP_BASE of 150 (never having built a warehouse)
    spent 70+ cycles oscillating between 0 food and a harvest overflowing back out --
    "the farm plots yield a harvest -- 0 food gathered in (stores nearly full, 225
    wasted)" repeating every cycle or two, real waste, not a one-off. 178 dispatched
    turns across that same run never once chose BUILD_WAREHOUSE. Same "real, computed
    fact, not a scripted directive" shape as _prepare_turn's diversification_note --
    population outgrowing the cap is true and checkable the moment it happens, well
    before the boom-bust actually starts costing anything.

    A standalone function (unlike diversification_note's own inline block) so an A/B
    test can monkeypatch it off for a baseline run without touching real game
    mechanics -- see scripts/ab_test_growth_facts.py."""
    if tribe.warehouses_built == 0 and tribe.population >= config.STORAGE_CAP_BASE:
        return (
            f"Population ({tribe.population}) has already outgrown the storage cap "
            f"({config.STORAGE_CAP_BASE} of any one resource, with no warehouse built yet) -- a "
            "harvest that arrives faster than it's spent overflows and is lost for good. Building "
            "a warehouse raises that ceiling for good."
        )
    return ""



def _can_afford_construct_wall(tribe, world) -> bool:
    # Live bug ("Walls didn't unlock for some reason and they wasted cycles"):
    # this used to return True here on the theory that letting the action's
    # own "nothing to build" message surface would redirect the tribe to a
    # separate EXPAND_TERRITORY action -- confirmed live it does not: one
    # tribe chose CONSTRUCT_WALL well over 100 times in a row against a ring
    # with zero sections ever unlocked, the message never once causing it to
    # pick EXPAND_TERRITORY instead. A second, independent live trace found
    # the same failure from the other side: EXPAND_TERRITORY itself sat
    # affordable and alone on the menu and still got picked in roughly 1 of 8
    # real test runs. 2026-09-08: rather than a third attempt at nudging the
    # model toward the "other" action, CONSTRUCT_WALL and EXPAND_TERRITORY
    # were merged into one action (see actions._construct_wall's own
    # docstring) -- there's no longer a second action to redirect to, so this
    # only needs to ask "is there ANY next wall-related step (progress,
    # reinforce, unlock, or open a new ring) the tribe can actually afford
    # right now," same "hide the guaranteed no-op" shape every other entry in
    # this table already uses.
    cost = _wall_next_afford_cost(tribe)
    if cost is not None:
        wood_cost, stone_cost = cost
        return tribe.wood >= wood_cost and tribe.stone >= stone_cost
    if not tribe.wall_rings:
        return False
    if tribe.wood < config.TERRITORY_EXPANSION_WOOD_COST or tribe.stone < config.TERRITORY_EXPANSION_STONE_COST:
        return False
    if city_layout.next_unlockable_section(tribe) is not None:
        return True
    # Nothing left to unlock in any existing ring -- next_wall_work_section
    # already being None (we're only here because it was) guarantees every
    # unlocked section is both built and maxed, which in turn guarantees the
    # outermost ring is fully reinforced -- so the only thing left to check is
    # whether there's room for one more ring under the cap.
    return len(tribe.wall_rings) < config.MAX_WALL_RINGS


def _can_place(tribe, world, building_type: str) -> bool:
    """Live-run correction (2026-09-03): "good calls but they are failing,
    probably resources... maybe something else." Turned out to be something
    else -- a live run showed Forest Tribe repeat BUILD_WAREHOUSE ~23 times
    with wood/stone in the hundreds the whole time, never spending a thing,
    because its territory (25 buildings already packed into radius 12) had no
    room left for a 15th warehouse. Every build action already no-ops for free
    when architect.find_free_slot returns None (see each one's own guard
    clause in actions.py) -- this just runs that same cheap, deterministic,
    side-effect-free scan here too, so a territory with no room left hides the
    option instead of dangling a guaranteed no-op, the same treatment an
    unaffordable wood/stone cost already gets below."""
    return architect.find_free_slot(world, tribe, building_type) is not None


def _can_afford_build_long_house(tribe, world) -> bool:
    """Explicit request: "the option to build a long house should not even
    come up if they don't have a full Wall built" -- mirrors actions.
    _build_long_house's own real prerequisite (ring_fully_built, or a banked
    wall_lock_long_house_credits -- see that field's own comment) and repeat
    gate (real housing need, not a flat one-time flag) exactly, so a
    guaranteed no-op never dangles in the menu. Cost check reuses _long_house_
    fur_discount (same Fur-discounted cost _build_long_house actually charges)
    rather than the flat base cost, so banked Fur can be the difference
    between this showing as affordable or not."""
    ring0_done = bool(tribe.wall_rings) and city_layout.ring_fully_built(tribe.wall_rings[0])
    if not ring0_done and tribe.wall_lock_long_house_credits <= 0:
        return False
    houses_needed = max(1, -(-tribe.population // config.HOUSING_POPULATION_PER_LONG_HOUSE))
    if tribe.long_houses_built >= houses_needed:
        return False
    wood_cost, stone_cost, _ = _long_house_fur_discount(tribe)
    return tribe.wood >= wood_cost and tribe.stone >= stone_cost and _can_place(tribe, world, "long_house")


AFFORDABILITY_CHECKS = {
    # Live-run correction ("the Ore collection did not trigger the Forge" led
    # to auditing every entry below against its action's own real guard clause):
    # several of these only ever checked wood/stone, silently missing the same
    # real structural prerequisite (a building or proven success) their own
    # action function already gates on -- the exact guaranteed-no-op class this
    # whole table exists to close, just missed on these specific entries.
    "BUILD_DOCK": lambda t, w: t.fishing_learned and t.wood >= config.DOCK_WOOD_COST and _can_place(t, w, "dock"),
    "CONSTRUCT_WALL": _can_afford_construct_wall,
    "BUILD_LONG_HOUSE": _can_afford_build_long_house,
    "BUILD_FISHERY": lambda t, w: (
        t.dock_built and t.wood >= config.FISHERY_WOOD_COST and t.stone >= config.FISHERY_STONE_COST
        and _can_place(t, w, "fishery")
    ),
    # Real prerequisite AND cost checked together -- see actions.py._build_sawmill/
    # _build_quarry/_build_tannery's own simplified gates (a proven success, not a
    # Long House/scouted site).
    "BUILD_SAWMILL": lambda t, w: (
        t.wood_ever_gathered and t.wood >= config.SAWMILL_WOOD_COST and t.stone >= config.SAWMILL_STONE_COST
        and _can_place(t, w, "sawmill")
    ),
    "BUILD_QUARRY": lambda t, w: (
        t.stone_ever_gathered and t.wood >= config.QUARRY_WOOD_COST and t.stone >= config.QUARRY_STONE_COST
        and _can_place(t, w, "quarry")
    ),
    "BUILD_TANNERY": lambda t, w: (
        t.hunt_ever_succeeded and t.wood >= config.TANNERY_WOOD_COST and t.stone >= config.TANNERY_STONE_COST
        and _can_place(t, w, "tannery")
    ),
    "BUILD_HATCHERY": lambda t, w: (
        t.eggs_ever_gathered and t.wood >= config.HATCHERY_WOOD_COST and t.stone >= config.HATCHERY_STONE_COST
        and _can_place(t, w, "hatchery")
    ),
    "BUILD_BATH_HOUSE": lambda t, w: (
        t.wood >= config.BATH_HOUSE_WOOD_COST and t.stone >= config.BATH_HOUSE_STONE_COST
        and _can_place(t, w, "bath_house")
    ),
    "BUILD_LIBRARY": lambda t, w: (
        t.long_houses_built > 0
        and t.wood >= config.LIBRARY_WOOD_COST and t.stone >= config.LIBRARY_STONE_COST
        and _can_place(t, w, "library")
    ),
    "BUILD_WELL": lambda t, w: (
        t.wood >= config.WELL_WOOD_COST and t.stone >= config.WELL_STONE_COST
        and _can_place(t, w, "well")
    ),
    # Deliberately NOT gated on the tribe having any memory yet -- same as
    # CONSTRUCT_WALL's own "let the action's own message surface instead" case:
    # a library with nothing worth studying yet is a real, informative state,
    # not a guaranteed no-op this table exists to hide.
    "RESEARCH": lambda t, w: t.library_built and t.wood >= config.RESEARCH_WOOD_COST,
    # GATHER_ORE has no wood/stone cost of its own -- the real prerequisite is
    # a mine existing at all (see actions.py._gather_ore's own guard clause).
    "GATHER_ORE": lambda t, w: t.mine_built,
    "BUILD_KITCHEN": lambda t, w: (
        t.cooking_learned and t.long_houses_built > 0
        and t.wood >= config.KITCHEN_WOOD_COST and t.stone >= config.KITCHEN_STONE_COST
        and _can_place(t, w, "kitchen")
    ),
    "BUILD_MOAT": lambda t, w: (
        bool(t.wall_rings) and city_layout.ring_fully_reinforced(t.wall_rings[0])
        and t.wood >= config.MOAT_WOOD_COST and t.stone >= config.MOAT_STONE_COST
    ),
    "BUILD_WAREHOUSE": lambda t, w: (
        t.wood >= config.WAREHOUSE_WOOD_COST and t.stone >= config.WAREHOUSE_STONE_COST
        and _can_place(t, w, "warehouse")
    ),
    # Military branch, step 2 (plan file valiant-forging-falcon.md) -- real
    # prerequisite (Keep) AND cost checked together, same shape BUILD_SAWMILL/
    # BUILD_QUARRY/BUILD_TANNERY already use.
    "BUILD_BARRACKS": lambda t, w: (
        t.keep_built and t.wood >= config.BARRACKS_WOOD_COST and t.stone >= config.BARRACKS_STONE_COST
        and _can_place(t, w, "barracks")
    ),
    # Military branch, steps 3 and 5 (plan file valiant-forging-falcon.md) --
    # hides the guaranteed no-ops (no Warrior/Barracks yet, can't afford even
    # one soldier's food cost while still recruiting, or -- once at full
    # headcount -- readiness already maxed or can't afford the cheaper
    # maintenance-drill cost) the same way every other entry in this table
    # already does. Two real branches, matching actions._train_battalion's
    # own recruit-vs-drill split.
    "TRAIN_BATTALION": lambda t, w: (
        t.warrior_name is not None and t.barracks_built > 0
        and (
            (t.battalion_size < config.BATTALION_CAPACITY_PER_BARRACKS * t.barracks_built
             and t.food >= config.BATTALION_TRAINING_FOOD_COST_PER_SOLDIER)
            or (t.battalion_size >= config.BATTALION_CAPACITY_PER_BARRACKS * t.barracks_built
                and t.battalion_readiness < 1.0 and t.food >= config.BATTALION_READINESS_UPKEEP_FOOD_COST)
        )
    ),
    "BUILD_KEEP": lambda t, w: (
        t.long_houses_built >= config.KEEP_LONG_HOUSES_REQUIRED
        and t.wood >= config.KEEP_WOOD_COST and t.stone >= config.KEEP_STONE_COST
        and _can_place(t, w, "keep")
    ),
    "BUILD_FORTRESS": lambda t, w: (
        t.keep_built and t.long_houses_built >= config.FORTRESS_LONG_HOUSES_REQUIRED
        and t.wood >= config.FORTRESS_WOOD_COST and t.stone >= config.FORTRESS_STONE_COST
        and _can_place(t, w, "fortress")
    ),
    "BUILD_CASTLE": lambda t, w: (
        t.fortress_built and t.long_houses_built >= config.CASTLE_LONG_HOUSES_REQUIRED
        and t.wood >= config.CASTLE_WOOD_COST and t.stone >= config.CASTLE_STONE_COST
        and _can_place(t, w, "castle")
    ),
    "BUILD_MINE": lambda t, w: (
        t.quarry_built and bool(t.mine_sites)
        and t.wood >= config.MINE_WOOD_COST and t.stone >= config.MINE_STONE_COST
        and _can_place(t, w, "mine")
    ),
    "BUILD_FORGE": lambda t, w: (
        t.mine_built and t.unique_resources.get(t.mine_resource_name, 0) >= config.FORGE_ITEM_ORE_COST
        and t.wood >= config.FORGE_WOOD_COST and t.stone >= config.FORGE_STONE_COST
        and _can_place(t, w, "forge")
    ),
    "BUILD_ROAD": lambda t, w: t.wood >= config.ROAD_WOOD_COST and t.stone >= config.ROAD_STONE_COST,
    "PLANT_CROP": lambda t, w: (
        t.farm_plots < config.MAX_FARM_PLOTS and t.wood >= config.PLANT_CROP_WOOD_COST
        and _can_place(t, w, "farm_plot")
    ),
    "BREED": lambda t, w: t.food >= config.BREED_FOOD_COST and t.water >= config.BREED_WATER_COST,
    "NAME_WARRIOR": lambda t, w: t.warrior_name is None and _eligible_warrior_candidate(t) is not None,
    # Explicit request: "it's unwise to Trade before we have a full Wall" --
    # see actions.py._send_trade_emissary's matching real prerequisite. Instant
    # TRADE is left alone (a chance encounter, not a deliberate choice to
    # expose the tribe) -- only the deliberate reach-out to an already-known
    # rival is gated.
    "SEND_TRADE_EMISSARY": lambda t, w: bool(t.wall_rings) and city_layout.ring_fully_built(t.wall_rings[0]),
    # Both a real resource cost AND config.ITEM_STORAGE_CAP_BASE's own ceiling --
    # see _forge_item's matching "item stores are already full" no-op message.
    "FORGE_ITEM": lambda t, w: (
        len(t.items) < _item_storage_cap(t)
        and t.wood >= config.FORGE_ITEM_WOOD_COST
        and t.unique_resources.get(t.mine_resource_name, 0) >= config.FORGE_ITEM_ORE_COST
    ),
    "BUILD_OBJECT_CREATOR": lambda t, w: (
        t.wood >= config.OBJECT_CREATOR_WOOD_COST and t.stone >= config.OBJECT_CREATOR_STONE_COST
        and _can_place(t, w, "object_creator")
    ),
    # Both require the factory itself to actually stand -- same "structural
    # prerequisite, not just a resource cost" shape BUILD_FORGE gates FORGE_ITEM
    # on, made explicit here rather than relying only on the handler's own
    # no-op guard.
    "CREATE_ITEM": lambda t, w: (
        t.object_creator_built
        and t.wood >= config.CREATE_ITEM_WOOD_COST and t.stone >= config.CREATE_ITEM_STONE_COST
    ),
    "CREATE_USEFUL_STRUCTURE": lambda t, w: (
        t.object_creator_built
        and t.wood >= config.CREATE_USEFUL_STRUCTURE_WOOD_COST and t.stone >= config.CREATE_USEFUL_STRUCTURE_STONE_COST
        and _can_place(t, w, "created_structure")
    ),
    "DECLARE_CONQUEST": lambda t, w: (
        t.wood >= config.DECLARE_CONQUEST_WOOD_COST and t.stone >= config.DECLARE_CONQUEST_STONE_COST
    ),
    # Live report: "I keep seeing 'send hunting party'" -- confirmed against a
    # real run: SCOUT and HUNTING_PARTY were each chosen and rejected with "no
    # one left to send" roughly three times out of four (SCOUT 305/403,
    # HUNTING_PARTY 253/325, across both tribes). expedition_capacity(tribe) is
    # a hard, always-known ceiling (unlike a resource cost, which the model
    # could at least try to remedy) -- the exact "guaranteed no-op dangling in
    # the menu" class this whole table exists to close, just never extended to
    # expedition dispatch. EXPLORATION_PARTY gets the same treatment for
    # consistency even though it was rejected far less often (8/49) in this run.
    "SCOUT": lambda t, w: expedition_capacity(t) - len(t.expeditions) > 0,
    "HUNTING_PARTY": lambda t, w: expedition_capacity(t) - len(t.expeditions) > 0,
    "EXPLORATION_PARTY": lambda t, w: expedition_capacity(t) - len(t.expeditions) > 0,
}


def _interpolated_path(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Every whole tile on the straight line from (x0, y0) to (x1, y1), inclusive
    of both ends -- used by Simulation._advance_resource_trails to wear a real
    route between a settlement and a resource site it draws from, the same way
    a party's own footsteps wear one tile at a time via terrain_aware_step.
    Simple linear interpolation, not pathfinding around terrain -- a supply
    route between two known points, not a search for one."""
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return [(x0, y0)]
    return [
        (round(x0 + (x1 - x0) * i / steps), round(y0 + (y1 - y0) * i / steps))
        for i in range(steps + 1)
    ]


def _compass_direction(dx: float, dy: float) -> str:
    """An 8-point compass label for a (dx, dy) offset -- used for a distant rival
    sighting (see config.RIVAL_DISTANT_SIGHTING_RADIUS), where only a rough heading is
    plausible, not exact coordinates. y increases southward on this map (matching
    world.py's own north/south framing -- the mountain and forest bands sit at low y),
    so dy > 0 is south, dy < 0 is north."""
    angle = math.degrees(math.atan2(dy, dx)) % 360
    directions = ("east", "southeast", "south", "southwest", "west", "northwest", "north", "northeast")
    return directions[round(angle / 45) % 8]


# Explicit request: the model's "visual_action" text used to need an exact, case-
# sensitive match against available_actions or it silently collapsed to IDLE -- with
# no record of the mismatch anywhere, a genuine parse failure was indistinguishable
# from the tribe deliberately choosing to do nothing. Live data showed exact-match
# already succeeds the overwhelming majority of the time, so this stays a cheap,
# free-in-the-common-case ladder (no second LLM call): exact match, then a
# normalization pass for case/spacing/hyphen variance, then a fuzzy close-match for
# typos. Only a genuine miss falls through further, and even then a looser fuzzy pass
# records a best-guess for the correction nudge (_prepare_turn's last_confusion
# block) to name -- "Instant Enlightenment" for next cycle, not a forced action now.
def _resolve_action(raw: str, available_actions: list[str]) -> tuple[str, str | None]:
    """Returns (action_to_apply, unresolved_raw). unresolved_raw is None on any real
    match (exact, normalized, or a confident fuzzy match) -- including a syntactically
    real action name that just isn't unlocked/available right now (wrong era, not
    settled, etc.), which is a legitimate "can't do that here" case, not a parse
    failure, and gets no correction nudge. unresolved_raw is only the original raw
    text when nothing recognizable was said at all.

    Explicit request: "IDLE needs to be removed altogether, we should never need
    this" -- action_to_apply is now always a real, currently-available action, never
    a no-op; several seconds of real inference time doing nothing on a parse miss
    wasted the turn for no reason. Falls back to the same fuzzy guess already used
    for the correction fact (_guess_intended_action) at a looser cutoff, and only
    picks the first available action if even that finds nothing."""
    raw = str(raw)
    if raw in available_actions:
        return raw, None
    normalized = raw.strip().upper().replace(" ", "_").replace("-", "_")
    if normalized in available_actions:
        return normalized, None
    if normalized in ACTION_REGISTRY:
        # A real action name, just not unlocked/available right now -- a legitimate
        # "can't do that here," not confusion, so no correction nudge fires.
        return available_actions[0], None
    close = difflib.get_close_matches(normalized, available_actions, n=1, cutoff=0.6)
    if close:
        return close[0], None
    guess = _guess_intended_action(normalized, available_actions)
    if guess:
        return guess, raw
    return available_actions[0], raw


def _guess_intended_action(raw: str, available_actions: list[str]) -> str | None:
    """A looser, display-only fuzzy pass used only to name a possible intended action
    in the next cycle's correction fact -- never used to actually decide what
    happens. A wrong guess here costs nothing (it's a suggestion in a fact block, not
    an applied action), so a lower cutoff than _resolve_action's is fine."""
    normalized = str(raw).strip().upper().replace(" ", "_").replace("-", "_")
    close = difflib.get_close_matches(normalized, available_actions, n=1, cutoff=0.3)
    return close[0] if close else None


# Explicit request: a celebration should shout in the tribe's own invented language,
# not just plain English chronicle prose. Reuses whatever the tribe most recently
# actually broadcast (tribe.last_broadcast, see prompts.py's LINGUISTIC SYNTHESIS
# PROTOCOL) rather than inventing a fresh word for the occasion, which would have no
# real grounding -- this is the one place that private field actually gets heard,
# not just tracked by translation_matrix.py. A tribe that hasn't broadcast anything
# yet stays silent, not filled in with a placeholder.
def _celebration_shout(tribe: "Tribe") -> str:
    return f' -- "{tribe.last_broadcast}!"' if tribe.last_broadcast else ""


def _feast_word(tribe: "Tribe") -> str:
    return "potluck feast" if tribe.cooking_learned else "feast"


# Explicit finding: a flat 30%-of-current-food cost gets more expensive in absolute
# terms the wealthier a tribe gets, with no ceiling -- "we spend a lot of time on
# Parties." Shared by every _celebrate_* method so the cap applies uniformly.
# Explicit request: once cooking is learned (actions.py._cook_food), a feast costs
# less -- real food contributed and prepared efficiently, not just handed over.
def _celebration_cost(tribe: "Tribe") -> int:
    cost = min(round(tribe.food * config.CELEBRATION_RESOURCE_COST_FRACTION), config.CELEBRATION_MAX_COST)
    if tribe.cooking_learned:
        cost = round(cost * config.CELEBRATION_COOKING_COST_MULTIPLIER)
    # Object Creator era's celebration_discount effect -- see actions.py.
    # _created_object_bonus. Floored at 0 rather than letting it go negative.
    cost = max(0, round(cost * (1 - _created_object_bonus(tribe, "celebration_discount"))))
    return cost


class Tribe:
    def __init__(
        self, tribe_id: str, name: str, model: str, x: int, y: int, color: str,
        event_log: RunEventLog | None = None,
    ):
        self.id = tribe_id
        self.name = name
        self.model = model
        self.x, self.y = x, y
        self.color = color
        self.wood = 50
        self.stone = 50
        self.food = 40
        self.water = config.STARTING_WATER
        self.population = 8
        self.era = ERAS[0].key
        self.last_broadcast = ""
        self.last_action = ""
        # See config.ACTION_REPETITION_THROTTLE_THRESHOLD/COOLDOWN -- tracks how many
        # cycles in a row the SAME action has just been chosen (regardless of whether
        # it actually succeeded), and which actions are currently pulled from
        # available_actions as a result. throttled_actions maps action name -> the
        # cycle it becomes choosable again.
        self.action_streak_name = ""
        self.action_streak_count = 0
        self.throttled_actions: dict[str, int] = {}
        # See _prepare_turn's survival-crisis filter (SURVIVAL_CRISIS_ACTIONS above).
        # Hysteresis, not a plain threshold check: entering requires crossing the
        # critical line, but clearing requires climbing back past the (higher)
        # warning line, so the menu doesn't flicker in and out every cycle right at
        # the boundary.
        self.food_crisis_active = False
        self.water_crisis_active = False
        # Set by _apply_turn whenever _resolve_action couldn't match the model's raw
        # visual_action text to anything real; surfaced once as a correction fact by
        # _prepare_turn next cycle, then cleared. {"raw", "guess", "fallback"} or None
        # -- None means last cycle's answer was understood as a real action.
        self.last_confusion: dict | None = None
        self.last_target: list[int] | None = None
        # Unlike last_target (RELOCATE-only, drives the journey_note fact), this
        # records the target_vector submitted alongside *every* action, purely for
        # decision_log.py's offline analysis -- it has no effect on gameplay.
        self.last_decision_target: list[int] | None = None
        self.history: list[str] = TribeHistory(name, event_log)
        self.memory = TribeMemory(tribe_id)
        self.founded_city = False
        # Set the moment the tribe's era reaches Era.founds_city, even if it isn't
        # actually allowed to found yet (see _advance_city_founding) -- separates "the
        # era-progression milestone has been reached" from "a real city has actually
        # been founded," so a still-pending founding gets rechecked every cycle instead
        # of only at the one instant the era itself advanced.
        self.city_founding_eligible = False
        self.extinct = False
        # Set alongside self.extinct in _lose_population -- lets a game-over
        # summary (Simulation._generate_game_over_summary) say *why* a tribe
        # ended, not just that it did. Mirrors the same `cause` scoreboard.py
        # already records, just kept on the tribe itself too.
        self.extinction_cause: str | None = None
        self.chief_name = ""
        self.chief_philosophy = ""
        self.chief_decree = ""
        self.chief_victory = ""
        # Military branch, step 1 (plan file valiant-forging-falcon.md,
        # actions.NAME_WARRIOR) -- a real, permanently-appointed individual,
        # never the chief. Later steps (Barracks, Battalion, Might) build on
        # this once it's real.
        self.warrior_name: str | None = None
        # Lifetime counters for backend/scoreboard.py -- what an evaluator actually
        # wants to compare across models isn't just "did it survive," it's how it got
        # there: how often it needed a new leader, how often scouting actually paid
        # off, how it fared in conflict.
        self.max_population = self.population
        self.chiefs_elected = 0
        self.chief_deaths = 0
        self.expeditions_launched = 0
        self.expeditions_succeeded = 0
        # Split from expeditions_succeeded above so a scouting milestone ("Master
        # Pathfinder") and a hunting milestone ("Master Hunter") can be tracked and
        # credited to the specific person who earned them separately -- see
        # Simulation._award_trophy's `individual` param.
        self.scout_successes = 0
        self.hunt_successes = 0
        self.raids_won = 0
        self.raids_lost = 0
        self.raids_defended = 0
        self.trades_completed = 0
        # Explicit request: sidebar boxes for "an elastic and running total of
        # things traded away/received" and "each type of combat with W/L
        # totals" -- the scalar counters above already exist for the
        # end-of-run scoreboard (backend/scoreboard.py) but don't break down
        # by resource or by which kind of fight this actually was. Keyed
        # dicts instead of a fixed field per resource/combat-kind, so a newly
        # traded resource or a combat kind that's never happened yet doesn't
        # need its own hardcoded field.
        self.trade_given: dict[str, int] = {}
        self.trade_received: dict[str, int] = {}
        self.combat_record: dict[str, dict[str, int]] = {}
        # See actions.py._declare_alliance/_declare_war -- keyed by the other
        # tribe's id, value in {"ALLIED", "WAR"}. Absent = implicitly NEUTRAL.
        # Symmetric: set on both tribes at once, since only one side ever "chooses"
        # this in a given cycle but the declaration is real for both.
        self.stance_toward: dict[str, str] = {}
        # Credited to whichever chief is in power the moment each is first earned --
        # see Simulation._check_chief_trophies. [{"name", "chief", "cycle"}, ...]
        self.trophies: list[dict] = []
        # Fame: a new well-being measurement, distinct from esteem (trophy count)
        # -- prestige from real events (celebrations, Landmarks), not achievement
        # milestones. See config.FAME_PER_CELEBRATION/_PER_LANDMARK and
        # wellbeing.compute_wellbeing's own 6th tier.
        self.fame: float = 0.0
        # Set by actions.py._breed, resolved (an async LLM call -- see backend/
        # breeding.py) in the same Simulation.step() that set it, same pattern as
        # pending_chief_context/_install_chief. {"parent_a", "parent_b"} or None.
        self.pending_birth: dict | None = None
        # A real parent record per child, not just an anonymous population+1 -- the
        # "capture lineage" requirement. [{"child_name", "parents", "cycle", "note"}, ...]
        self.lineage: list[dict] = []
        # See Simulation._check_for_celebration -- very negative so a tribe's very
        # first celebration isn't blocked by a cooldown it never actually used yet.
        self.last_celebration_cycle: int = -config.CELEBRATION_COOLDOWN_CYCLES
        # Counts only the surplus-only celebration branch (not discovery-based ones)
        # -- see _check_for_celebration's retirement once this reaches
        # CELEBRATION_SURPLUS_RETIREMENT_COUNT.
        self.surplus_celebrations = 0
        # See Simulation._check_raider_attack -- very negative so a tribe's first
        # possible raid isn't blocked by a cooldown it never actually used yet.
        self.last_raider_attack_cycle: int = -config.RAIDER_HAZARD_COOLDOWN_CYCLES
        # A triggered attack in its multi-cycle "riding in" approach -- see
        # Simulation._advance_raider_approach. None means no attack is currently
        # approaching. {"start_x", "start_y", "x", "y", "cycles_left", "total_cycles"}.
        self.raiders_approaching: dict | None = None
        # See Simulation._advance_wall_security -- explicit request: "once the
        # Wall is complete all Raiders are kicked out of the area or absorbed."
        # One-way, like every other real-milestone flag here: once the first
        # wall ring is genuinely finished, this tribe stops being raided at all.
        self.raiders_repelled_by_wall = False
        # Honors a chief has personally proposed via the night cycle (see
        # reflection.py's AWARD_CATEGORIES). Simulation._check_custom_awards hands one
        # out to whoever first earns it after it's proposed.
        self.custom_awards: list[dict] = []
        # Innate daily tribal gathering (see Simulation._hold_tribal_gathering) -- state
        # for what's changed since the last one. gathering_brief is re-surfaced into the
        # tribe's own live turn context every cycle until the next gathering overwrites
        # it (see Simulation._build_visible_entities), not just narrated into the
        # chronicle and forgotten.
        self.last_gathering_cycle = 0
        self.population_at_last_gathering = self.population
        self.gathering_brief = ""
        # A confirmed water discovery used to only ever reach the tribe's own live
        # reasoning via tribe.memory.recall(f"{biome} at {x},{y}") -- a query about the
        # tribe's *current* location, which essentially never overlaps in vocabulary
        # with "Scouts confirmed fresh water at (fx,fy)" unless the tribe happens to
        # already be standing on those exact coordinates. In practice that meant a
        # hard-won discovery vanished from the model's own context the moment the
        # one-time chronicle line scrolled by, leaving RELOCATE's target_vector an
        # ungrounded guess even right after a successful scout. This persists real
        # confirmed sites (deduped, most recent last) so _build_visible_entities can
        # keep surfacing them the same durable way it does taboos -- and the frontend
        # can mark them permanently on the map (see index.html's drawLandmarks).
        #
        # lumber_sites/wildlife_sites/quarry_sites are the same idea for a scout's
        # terrain_report: set only when the reported biome is one BIOME_YIELD_MULTIPLIER
        # already marks as maxed out (1.0) for a resource -- forest for both wood and
        # game, mountains for stone -- a real, already-measured "considerable cluster",
        # not an arbitrary new threshold invented just for this.
        self.confirmed_water_sites: list[tuple[int, int]] = []
        # Counts consecutive cycles without choosing RELOCATE -- see
        # Simulation._is_camped/config.SETTLEMENT_STABILITY_CYCLES. Reset to 0 the
        # instant RELOCATE is chosen again, so "camped" means genuinely staying put,
        # not just having once paused for ten cycles somewhere.
        self.cycles_since_relocate = 0
        # The chief's own reasoning from the most recent night cycle (see
        # Simulation._run_night_cycle) -- kept even when the philosophy didn't
        # change, purely so the spectator UI has something real to show as a
        # night-time thought bubble.
        self.last_reflection = ""
        self.last_reflection_cycle = 0
        self.lumber_sites: list[tuple[int, int]] = []
        # Explicit request: "are the scouts finding Wolves Dens and Bear Caves
        # and Deer Stands? if not, they should be." {"x", "y", "type"} dicts,
        # most recent last -- type is one of world.WILDLIFE_SITE_TYPES, chosen
        # at random per discovery (see Simulation._advance_one_expedition).
        self.wildlife_sites: list[dict] = []
        self.quarry_sites: list[tuple[int, int]] = []
        self.raider_sightings: list[tuple[int, int]] = []
        # A fact for the next chief election to reason about, set when something more
        # specific than "just founded" is true (currently only a raid-conquest merge,
        # see Simulation._merge_tribes) -- consumed and cleared by _install_chief so an
        # election it's informing isn't indistinguishable from an ordinary founding.
        self.pending_chief_context: str = ""
        # Empty when no party is out; otherwise a list of {"pos", "origin", "target",
        # "day", "phase" ("outbound"/"returning"), "found", "terrain_report"} dicts --
        # see actions.py._scout/_hunting_party and Simulation._advance_expeditions. A
        # tribe can run more than one party at once (up to config.MAX_CONCURRENT_
        # EXPEDITIONS), any mix of scouting and hunting.
        self.expeditions: list[dict] = []
        # Tribe Map: coarse "ground we've actually walked" record -- see
        # config.TRIBE_MAP_SECTOR_SIZE/world.mark_visited_sector.
        self.visited_sectors: set[tuple[int, int]] = set()
        # See actions.py._scout -- explicit request: "scout directions rotate
        # on a 20 degree angle starting with the South East." Advances by one
        # step (config.SCOUT_ROTATION_STEP_DEGREES) every real SCOUT dispatch,
        # guaranteeing coverage spreads out over time regardless of the
        # model's own (frequently unreliable) sense of direction.
        # Live report: "the scouts went exact the same way" -- every tribe used
        # to start this at a flat 0, so any two tribes' opening (or later,
        # coincidentally-aligned) SCOUT computed the identical heading off their
        # own position. Seeded from the tribe's own spawn index instead (see
        # config.SCOUT_ROTATION_TRIBE_STAGGER_STEPS) so tribes rotate in
        # permanently offset lockstep rather than in unison.
        try:
            tribe_index = int(tribe_id.rsplit("_", 1)[-1])
        except ValueError:
            tribe_index = 0
        stagger = tribe_index * config.SCOUT_ROTATION_TRIBE_STAGGER_STEPS
        # A per-tribe hardcoded override used to live here ("make the Scout from
        # Tribe 2 go West first"), pointing spawn slot 1's opening heading at a
        # fixed compass direction on the theory that water always sits west of
        # it. That's a fact about one specific map layout baked into code --
        # live testing while retuning SPAWN_POINTS showed Tribe 2's scout
        # reliably NOT finding water that way while Tribe 1's generic-formula
        # heading did, every single run. seed_scout_heading_toward_water below
        # replaces it with a real per-run lookup (Landscape.nearest_water) once
        # a world exists to ask, so this stays correct across spawn-point
        # changes instead of silently going stale again. The generic formula
        # above is still the fallback for construction without a world (most
        # of this project's own tests build a bare Tribe directly).
        self.scout_rotation_index = stagger
        # See actions.py._exploration_party -- its own separate rotating heading
        # (offset from SCOUT's own sweep) so the two don't retrace each other's
        # ground. landmarks: {"x", "y", "resource"} entries, one per Landmark
        # actually found (Simulation._advance_exploration_party_outbound).
        self.explore_rotation_index = stagger
        self.landmarks: list[dict] = []
        # Explicit design spec: "if anyone discovers a hazard, even if no one
        # dies, they landmark it." Same {"x", "y", "name"} shape as landmarks
        # above, deliberately a separate list -- these are warnings, not
        # rewards, and need their own dedup check (Simulation._landmark_hazard)
        # so a lingering party doesn't re-report the same known-dangerous tile
        # every single day.
        self.hazard_landmarks: list[dict] = []
        # Farming (backend/actions.py PLANT_CROP, Simulation._advance_farming). Growth
        # is a passive per-cycle tick once at least one plot exists, not a discrete
        # action -- same category as upkeep/population growth.
        self.farm_plots = 0
        self.crop_growth = 0
        self.last_harvest_cycle = 0
        # Fishing (backend/actions.py CATCH_FISH, Simulation._advance_fish_supply):
        # not a separate knowledge/skill system -- the first successful catch just
        # flips this, and _advance_fish_supply checks nothing else to start a passive
        # daily food supply, the same "action unlocks a passive system" shape farming
        # and water already use.
        self.fishing_learned = False
        # See actions.py._cook_food -- one-way, like fishing_learned. Once true,
        # _celebration_cost charges less: real food contributed and prepared, not
        # just handed over from the stockpile, and every future food harvest goes
        # further (config.COOKING_FOOD_MULTIPLIER, actions._food_multiplier).
        self.cooking_learned = False
        # Real prerequisites for COOK_FOOD's own availability (see Simulation.
        # _prepare_turn) -- a successful hunt (instant HUNT_DEER or a HUNTING_PARTY
        # catch) and a successfully-built fire, ever. One-way, same shape as every
        # other "proven once" flag here.
        self.hunt_ever_succeeded = False
        # See Simulation._advance_automatic_fire -- the other real "found food"
        # prerequisite fire can ignite from, alongside a successful hunt. GATHER_FOOD
        # never has a hazard/failure branch (see actions.py._forage), so this is set
        # the first time it's ever chosen at all, not on some stricter "good yield"
        # bar.
        self.foraged_ever_succeeded = False
        # See actions.py._build_sawmill/_build_quarry -- explicit request, "the
        # Sawmill is also online easily if they Gather Wood successfully" (and the
        # same for Quarry/stone): a real proven success, not a scouted site or a
        # Long House, is what actually should gate these now. Live data showed both
        # tribes permanently blocked from Sawmill/Tannery/Quarry behind a Long House
        # that itself needs a completed wall ring neither tribe reliably finishes --
        # this removes that chain for these three specifically.
        self.wood_ever_gathered = False
        self.stone_ever_gathered = False
        self.fire_ever_built = False
        # See Simulation._advance_automatic_boat/config.BOAT_WATER_BIOMES --
        # automatic like fire, once both a Dock and fishing are real. Grants real
        # movement speed through river/lake tiles (backend/physics.py.
        # terrain_aware_step), never ocean access.
        self.boat_built = False
        # See actions.py._build_bath_house/Simulation._apply_upkeep -- explicit
        # request: "bath house bolsters Well-Being upkeep once built."
        self.bath_house_built = False
        # See actions.py._build_library/_research -- explicit request: a Library
        # summarizes the tribe's own TribeMemory (backend/memory.py) into permanent,
        # readable entries (own Library tab, not just a building icon), and unlocks
        # RESEARCH: a real, repeatable "boost growth and innovation" -- each
        # completed research discounts the next era's threshold (see
        # config.INNOVATION_ERA_DISCOUNT_PER_RESEARCH, Simulation._advance_era_if_ready),
        # a genuine payoff a spectator can watch compound, not a flat stat nudge.
        # library_entries: {"summary", "cycle"} dicts, oldest first.
        self.library_built = False
        self.library_entries: list[dict] = []
        self.research_completed = 0
        # See actions.py._build_well/Simulation._advance_water_supply -- explicit
        # request: water's passive income had no equivalent of Fishery/Dock's
        # stacking bonus. No prerequisite beyond being settled and affordable, same
        # shape as Bath House/Warehouse.
        self.well_built = False
        # See actions.py._build_moat -- one-way, gated on the first wall ring being
        # fully reinforced (backend/city_layout.py.ring_fully_reinforced). A cheaper
        # alternative defense investment, not a wall replacement.
        self.moat_built = False
        # See actions.py._build_long_house -- explicit correction: "most
        # structures they only need 1 of. but house builds are dependant on
        # population needs." Repeatable, not a one-time flag -- a count, gated on
        # real population need (config.HOUSING_POPULATION_PER_LONG_HOUSE) each
        # time, and the real proxy the Keep/Fortress/Castle tier below reads for
        # how established this settlement has become.
        self.long_houses_built = 0
        # The defensive tier ladder after Long House (backend/actions.py.
        # _build_keep/_build_fortress/_build_castle): explicit request -- "10
        # houses before they build a Keep, 40 until they reach a Fortress, 70
        # until they can build castles." Each one-way, each gated on the previous
        # stage already standing plus tribe.long_houses_built clearing its own
        # threshold. Each adds a real defense bonus on top of the wall's own
        # (Simulation._resolve_raider_attack).
        self.keep_built = False
        self.fortress_built = False
        self.castle_built = False
        # Military branch, step 2 (plan file valiant-forging-falcon.md,
        # actions.BUILD_BARRACKS) -- repeatable, like warehouses_built: each
        # one raises how large a Battalion can ever be trained
        # (config.BATTALION_CAPACITY_PER_BARRACKS per Barracks).
        self.barracks_built = 0
        # Military branch, step 3 (actions.TRAIN_BATTALION) -- current
        # trained headcount, staged up toward config.
        # BATTALION_CAPACITY_PER_BARRACKS * barracks_built the same way a
        # wall section's own progress builds up over several actions.
        self.battalion_size = 0
        # Military branch, step 5 (Might's Training factor) -- explicit
        # request: "not overpowered, more like bolster and upkeep," a real
        # meter that has to be earned AND maintained, not a one-time flip.
        # Bolstered by actions._train_battalion (config.
        # BATTALION_READINESS_BOLSTER_PER_ACTION per call), drained a little
        # every cycle by Simulation._advance_battalion_readiness_upkeep
        # regardless of whether the Battalion is currently out on patrol.
        self.battalion_readiness = 0.0
        # Military branch, step 4 (Simulation._advance_battalion_patrol) --
        # deliberately NOT stored in self.expeditions: a Battalion patrol is
        # autonomous (never chief-dispatched) and must never compete with
        # SCOUT/HUNTING_PARTY/etc. for expedition_capacity's own limited
        # slots. None when no battalion is out patrolling; while patrolling,
        # a dict: {"pos": (x, y), "phase": "patrolling"|"returning",
        # "started_cycle": int, "target": (x, y) | None} -- "target" is a
        # raider_sightings location currently being closed on, None while
        # idly holding near territory_center.
        self.battalion_patrol: dict | None = None
        # Explicit request: "Cooldown for 3 whole days" -- gates only the
        # *start* of a new patrol (config.BATTALION_PATROL_COOLDOWN_DAYS
        # after the previous one ends); TRAIN_BATTALION has its own
        # separate affordability gate untouched by this ("training has its
        # own controls"). 0 means no patrol has ever run yet.
        self.battalion_cooldown_until_cycle = 0
        # See actions.py._build_road -- one-way. Adds a flat speed bonus to every
        # future expedition (Simulation._advance_one_expedition), the same shape a
        # well-worn trail already grants.
        self.road_built = False
        # wellbeing.py's esteem tier (config.ESTEEM_POINTS_PER_TOLL_ROAD): counts
        # real completed toll roads, incremented once per Simulation.
        # _celebrate_road_complete (itself cooldown-gated to one per tribe per
        # config.CELEBRATION_COOLDOWN_CYCLES -- see that function's own comment
        # on why, after a live run saw ~40 tiles evolve and celebrate in a
        # single cycle) -- a reasonable proxy for "one whole toll road"
        # milestone, since individual tile evolution has no other natural
        # "this route is done" boundary to count against.
        self.toll_roads_completed = 0
        # See actions.py._build_dock -- one-way, gated on general settling.
        # Boosts every future CATCH_FISH catch.
        self.dock_built = False
        # See actions.py._build_sawmill/_build_quarry -- one-way, each gated on
        # long_house_built + fishing_learned (explicit request: "after they have
        # farming and fishing down and are building homes") plus a real
        # discovered site (lumber_sites/quarry_sites). Each permanently triples
        # every future GATHER_WOOD/GATHER_STONE yield respectively.
        self.sawmill_built = False
        self.quarry_built = False
        # The exact site coordinate locked in when each was built (from
        # lumber_sites/quarry_sites/mine_sites at that moment) -- a tribe with
        # more discoveries on record afterward still only ever draws from the
        # one it actually excavated. See Simulation._advance_resource_trails:
        # explicit request, "these are collectables that must be fetched and
        # so trails/roads to them should be established naturally."
        self.lumber_site: tuple[int, int] | None = None
        self.quarry_site: tuple[int, int] | None = None
        self.mine_site: tuple[int, int] | None = None
        # A discovered-but-unexcavated mine (see Simulation._advance_one_expedition),
        # same shape as quarry_sites/lumber_sites -- {"x", "y", "biome", "resource"}
        # dicts, most recent last. Scattered across any biome, not just mountains --
        # see world.UNIQUE_RESOURCE_BY_BIOME.
        self.mine_sites: list[dict] = []
        # See actions.py._build_mine -- one-way, gated on quarry_built (excavating a
        # named seam is a deeper extension of already knowing how to quarry) plus at
        # least one discovered mine site. mine_resource_name locks in which of the
        # tribe's discovered sites it actually excavated -- a tribe with more than
        # one on record still only ever works the one it chose. unique_resources is
        # a dict (not a fixed field per biome) since most tribes will only ever hold
        # zero or one named resource in their whole run.
        self.mine_built = False
        self.mine_resource_name: str | None = None
        self.unique_resources: dict[str, int] = {}
        # See actions.py._gather_ore/Simulation._advance_mine_yield -- a real
        # fetch, not automatic the instant mine_built is set (explicit
        # correction: "they do not harvest on a Discovery, so they have to
        # fetch it once"). Same "action unlocks a passive system" shape
        # fishing_learned already uses for _advance_fish_supply.
        self.ore_ever_gathered = False
        # See actions.py._build_kitchen -- one-way, gated on cooking_learned +
        # long_house_built. Stacks config.KITCHEN_FOOD_MULTIPLIER on top of
        # cooking's own harvest-point multiplier (see actions._food_multiplier).
        self.kitchen_built = False
        # See actions.py._build_tannery -- one-way, gated on a discovered Rabbit
        # Warren (tribe.wildlife_sites). Mirrors mine_built/mine_site/
        # mine_resource_name exactly, paying "Fur" into the same
        # unique_resources dict rather than a second parallel system.
        self.tannery_built = False
        self.tannery_site: tuple[int, int] | None = None
        # See actions.py._build_forge/_forge_item/_use_item -- the natural next step
        # once a Mine actually produces something (explicit request: "we skipped a
        # beat" between mining ore and doing anything with it). Gated on mine_built
        # plus at least one unit of the tribe's own mine_resource_name already in
        # stock, not a separate discovery mechanic. items holds crafted goods --
        # {"name", "type" (tool/weapon/innovation), "value", "cycle_made"} -- no
        # durability tracked, per explicit request; a flat value is all each one
        # carries, redeemable via USE_ITEM or handed over in a TRADE.
        self.forge_built = False
        self.items: list[dict] = []
        # Object Creator era (see actions.py._build_object_creator/_create_item/
        # _create_useful_structure, eras.py's object_creator_era): the factory
        # itself, plus every item/structure it's produced --
        # {"name", "category" (one of config.CREATED_OBJECT_CATEGORIES),
        # "kind": "item"|"structure"}. category is picked round-robin off this
        # list's own length at creation time, not a hidden roll -- see
        # _created_object_bonus for where each category's bounded effect is
        # actually read.
        self.object_creator_built = False
        self.created_objects: list[dict] = []
        # War and World Domination era (see actions.py._declare_conquest,
        # Simulation._merge_tribes, Simulation.step's world_domination check):
        # incremented every time this tribe fully absorbs a rival, whether via
        # an ordinary RAID grinding a defender to 0 population or a deliberate
        # DECLARE_CONQUEST campaign. The real signal for "this tribe won by
        # conquest," not just "this tribe is the only one left" (which could
        # also happen if every rival died of unrelated hazards).
        self.conquests_won = 0
        # _merge_tribes deletes the loser from Simulation.tribes entirely, so
        # by the time a world_domination game-over summary is generated,
        # nothing else on hand still names who was actually conquered --
        # tracked here instead so that summary can name them for real.
        self.conquered_tribe_names: list[str] = []
        # See actions.py._build_warehouse/_storage_cap -- explicit request after a
        # live run showed unbounded hoarding (200+ wood while starved on stone).
        # Repeatable, same shape as long_houses_built -- each one raises every
        # resource's storage cap by a further flat amount.
        self.warehouses_built = 0
        # See Simulation._prepare_turn's GATHER_FOOD retirement -- one-way, like
        # has_ever_settled, once a genuinely proven passive food source exists.
        self.foraging_retired = False
        # Same one-way retirement shape as foraging_retired, for GATHER_WATER once
        # _advance_water_supply's passive income exists (settled_near_water) -- see
        # Simulation._prepare_turn.
        self.watering_retired = False
        # Same one-way retirement shape, for CONSTRUCT_WALL once config.
        # MAX_WALL_RINGS is reached and fully reinforced -- see
        # Simulation._prepare_turn.
        self.walls_complete = False
        # Egg-gathering/flock genetics (backend/actions.py GATHER_EGGS, Simulation.
        # _resolve_hatch, backend/genetics.py hatch()) -- same pending_X/resolve shape
        # as pending_birth/lineage above, applied to a flock instead of the tribe's own
        # population. flock_lineage entries: {"trait", "parents", "cycle", "note"}.
        self.flock = 0
        self.flock_lineage: list[dict] = []
        self.pending_hatch: dict | None = None
        # See Simulation._advance_flock_eggs/_advance_livestock_feast -- a real,
        # separate stockpile a living flock lays into passively each cycle,
        # distinct from GATHER_EGGS finding a wild nest to hatch (which grows
        # flock directly, never touches this count).
        self.eggs = 0
        # See actions.py._gather_eggs/_build_hatchery -- a real wild find, the
        # Hatchery's own prerequisite (not flock size alone).
        self.eggs_ever_gathered = False
        self.hatchery_built = False
        # Set the first time this tribe genuinely settles next to real water (see
        # Simulation._is_settled_near_water) -- the chief names the place via a real
        # LLM call (backend/leadership.py's name_settlement), the same pending_X/
        # resolve shape as pending_birth/pending_hatch above.
        self.settlement_name = ""
        self.pending_settlement_naming = False
        # DECLARE_ALLIANCE's tribe-level cultural crossover (backend/actions.py.
        # _declare_alliance, Simulation._resolve_cultural_crossover, backend/
        # genetics.py's breed()) -- same pending_X/resolve shape as pending_birth/
        # pending_hatch above, applied to two whole tribes' cultures on first
        # becoming allies. Stores the rival's id (not a direct reference) so a
        # rival going extinct before this resolves is handled the same way
        # elsewhere in this file -- a fresh lookup, not a stale object.
        self.pending_cultural_crossover: str | None = None
        # Set once, the first time _is_settled_near_water is ever true, and never
        # cleared again even if the tribe later relocates away -- see config.
        # PRE_SETTLEMENT_ACTIONS. Distinct from currently-settled (which can toggle)
        # because the point is to have proven the tribe CAN settle, once.
        self.has_ever_settled = False
        # The cycle has_ever_settled flipped True -- explicit request ("suspend
        # crisis for 10 cycles beyond Territory lock"): the march to reach
        # confirmed water is itself expensive (RELOCATE's own food/water cost,
        # several cycles running), so a tribe often arrives and founds its city
        # already right at the survival-crisis threshold. See
        # SURVIVAL_CRISIS_ACTIONS's own grace-period comment in _prepare_turn.
        self.settled_at_cycle: int | None = None

        # Real territory + building footprints (2026-09-02 redesign): granted the
        # instant has_ever_settled becomes True, via Simulation._found_territory.
        # territory_center is fixed forever at the founding coordinate, deliberately
        # NOT the same as tribe.x/y -- RELOCATE can move a tribe anytime with no
        # gating, but buildings/wall rings must stay anchored to where the city was
        # actually founded, not wherever the tribe currently roams.
        self.territory_center: tuple[int, int] | None = None
        self.territory_radius = 0
        # One entry per concentric wall ring -- see backend/city_layout.py for the
        # shape of each ring/section dict.
        self.wall_rings: list[dict] = []
        # Explicit request: "if they choose Wall, they have to complete it, no
        # changing orders other than to collect what is needed to complete it."
        # Set the moment CONSTRUCT_WALL is chosen for a section that's still
        # incomplete afterward (actions.py._construct_wall), cleared the moment
        # that section finishes -- see Simulation._prepare_turn for the actual
        # available_actions narrowing this drives.
        self.wall_commitment_active = False
        # User's own refinement: locking out BUILD_LONG_HOUSE for an entire wall
        # ring would stall housing for too long, so the tribe banks the right to
        # build exactly one Long House per wall section completed, spendable any
        # time while still locked.
        self.wall_lock_long_house_credits = 0
        # One entry per placed structure (every Long House instance, Town Hall,
        # Sawmill, Quarry, Dock, Fishery, Kitchen, Tannery, Mine, Keep, Fortress,
        # Castle, each Farm plot, the Flock pen, Fire) -- positional metadata only,
        # placed by backend/architect.py; the flags below stay the source of truth
        # for gating/mechanics.
        self.buildings: list[dict] = []
        # See actions.py._build_fishery -- one-way, gated on dock_built.
        self.fishery_built = False

        # Set once per turn by Simulation._prepare_turn (see wellbeing.compute_wellbeing)
        # -- cached here rather than recomputed in to_dict() because the safety tier
        # needs a world.constructions lookup only Simulation has access to.
        self.wellbeing: dict = {}

    def seed_scout_heading_toward_water(self, world, kinds=("river", "lake")) -> None:
        """Replaces the generic tribe_index*stagger guess (set in __init__) with
        a heading aimed at this tribe's own real nearest fresh water, once a
        world actually exists to ask (Simulation.__init__, right after spawning
        each tribe -- never called by the many tests that build a bare Tribe with
        no world at all, which is exactly why __init__ still needs its own
        formula-only fallback). Reuses Landscape.nearest_water the same one-time,
        whole-grid way leadership election already does for its water_needed
        fact -- cheap on a 100x100 grid, and legitimate here for the same reason:
        aiming a search, not handing over the water's coordinates outright the
        way the old leadership fact used to.

        `kinds` narrows which water this tribe should actually aim for --
        Simulation.__init__ passes a single kind per spawn slot (design intent:
        Tribe 1 settles the river, Tribe 2 the lake) rather than "whichever body
        is closest," so two tribes near the same stretch of coastline don't both
        make for the same one. Defaults to either, for any slot with no assigned
        preference.

        Snaps the real angle to the nearest step on SCOUT's own rotation grid
        (SCOUT_ROTATION_START_ANGLE_DEGREES + STEP*index) rather than using the
        raw angle directly, so every later dispatch's sweep (scout_rotation_index
        += 1 per real dispatch) still lands on the same fixed set of headings
        this tribe would otherwise have used -- only the starting point changes,
        not the rest of the rotation's shape."""
        water = world.nearest_water(self.x, self.y, kinds=kinds)
        if water is None or tuple(water) == (self.x, self.y):
            return  # nothing to aim at, or already standing on it -- leave the generic stagger
        wx, wy = water
        angle_degrees = math.degrees(math.atan2(wy - self.y, wx - self.x)) % 360
        steps_per_full_rotation = round(360 / config.SCOUT_ROTATION_STEP_DEGREES)
        step = round(
            (angle_degrees - config.SCOUT_ROTATION_START_ANGLE_DEGREES) / config.SCOUT_ROTATION_STEP_DEGREES
        ) % steps_per_full_rotation
        self.scout_rotation_index = step
        self.explore_rotation_index = step

    def to_dict(self) -> dict:
        era_label = next((e.label for e in ERAS if e.key == self.era), self.era)
        survival_warning, _ = survival_bias_string(
            self.food, self.water, self.population, self.fishing_learned, self.cooking_learned,
            water_secure=_is_water_secure(self), food_secure=_is_food_secure(self),
        )
        nxt = next_era(self.era)
        next_era_info = None
        if nxt is not None:
            next_era_info = {
                "label": nxt.label,
                "requires_population": nxt.requires_population,
                "requires_resources": nxt.requires_resources,
            }
        return {
            "name": self.name,
            "model": self.model,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "wood": self.wood,
            "stone": self.stone,
            "food": self.food,
            "water": self.water,
            "population": self.population,
            "era": self.era,
            "era_label": era_label,
            "last_broadcast": self.last_broadcast,
            "last_action": self.last_action,
            "throttled_actions": list(self.throttled_actions.keys()),
            "food_crisis_active": self.food_crisis_active,
            "water_crisis_active": self.water_crisis_active,
            "last_decision_target": self.last_decision_target,
            "history": self.history[-6:],
            "cycles_since_relocate": self.cycles_since_relocate,
            "last_reflection": self.last_reflection,
            "last_reflection_cycle": self.last_reflection_cycle,
            "last_celebration_cycle": self.last_celebration_cycle,
            "scout_successes": self.scout_successes,
            "hunt_successes": self.hunt_successes,
            "founded_city": self.founded_city,
            "farm_plots": self.farm_plots,
            "crop_growth": self.crop_growth,
            "fishing_learned": self.fishing_learned,
            "cooking_learned": self.cooking_learned,
            "hunt_ever_succeeded": self.hunt_ever_succeeded,
            "foraged_ever_succeeded": self.foraged_ever_succeeded,
            "wood_ever_gathered": self.wood_ever_gathered,
            "stone_ever_gathered": self.stone_ever_gathered,
            "ore_ever_gathered": self.ore_ever_gathered,
            "fire_ever_built": self.fire_ever_built,
            "moat_built": self.moat_built,
            "long_houses_built": self.long_houses_built,
            # Explicit request: "It needs to say housed/unhoused... so the Tribe
            # knows how many more they will need to build until everyone is
            # housed comfortably." Computed here (not left for the frontend to
            # duplicate) from the same HOUSING_POPULATION_PER_LONG_HOUSE
            # _build_long_house's own repeat-gate already uses.
            "housing_capacity": self.long_houses_built * config.HOUSING_POPULATION_PER_LONG_HOUSE,
            "keep_built": self.keep_built,
            "fortress_built": self.fortress_built,
            "castle_built": self.castle_built,
            "barracks_built": self.barracks_built,
            "battalion_size": self.battalion_size,
            "battalion_readiness": round(self.battalion_readiness, 3),
            "might": compute_might(self),
            "battalion_patrol": self.battalion_patrol,
            "road_built": self.road_built,
            "toll_roads_completed": self.toll_roads_completed,
            "dock_built": self.dock_built,
            "sawmill_built": self.sawmill_built,
            "quarry_built": self.quarry_built,
            "mine_sites": self.mine_sites,
            "mine_built": self.mine_built,
            "mine_resource_name": self.mine_resource_name,
            "unique_resources": self.unique_resources,
            "visited_sectors": list(self.visited_sectors),
            "scout_rotation_index": self.scout_rotation_index,
            "landmarks": self.landmarks,
            "hazard_landmarks": self.hazard_landmarks,
            "kitchen_built": self.kitchen_built,
            "tannery_built": self.tannery_built,
            "forge_built": self.forge_built,
            "items": self.items,
            "object_creator_built": self.object_creator_built,
            "created_objects": self.created_objects,
            "conquests_won": self.conquests_won,
            "conquered_tribe_names": self.conquered_tribe_names,
            "warehouses_built": self.warehouses_built,
            "foraging_retired": self.foraging_retired,
            "watering_retired": self.watering_retired,
            "last_harvest_cycle": self.last_harvest_cycle,
            "flock": self.flock,
            "eggs": self.eggs,
            "livestock_surplus_threshold": _livestock_surplus_threshold(self),
            "hatchery_built": self.hatchery_built,
            "boat_built": self.boat_built,
            "bath_house_built": self.bath_house_built,
            "library_built": self.library_built,
            "library_entries": self.library_entries,
            "research_completed": self.research_completed,
            "well_built": self.well_built,
            "flock_lineage": self.flock_lineage,
            "settlement_name": self.settlement_name,
            "has_ever_settled": self.has_ever_settled,
            "territory_center": self.territory_center,
            "territory_radius": self.territory_radius,
            "wall_rings": self.wall_rings,
            "wall_commitment_active": self.wall_commitment_active,
            "buildings": self.buildings,
            "fishery_built": self.fishery_built,
            "survival_warning": survival_warning,
            "extinct": self.extinct,
            "chief_name": self.chief_name,
            "chief_philosophy": self.chief_philosophy,
            "chief_decree": self.chief_decree,
            "chief_victory": self.chief_victory,
            "warrior_name": self.warrior_name,
            "trophies": self.trophies,
            "fame": self.fame,
            "lineage": self.lineage,
            "custom_awards": self.custom_awards,
            "confirmed_water_sites": self.confirmed_water_sites,
            "lumber_sites": self.lumber_sites,
            "wildlife_sites": self.wildlife_sites,
            "quarry_sites": self.quarry_sites,
            "raider_sightings": self.raider_sightings,
            "last_raider_attack_cycle": self.last_raider_attack_cycle,
            "raiders_approaching": self.raiders_approaching,
            "raiders_repelled_by_wall": self.raiders_repelled_by_wall,
            "stance_toward": self.stance_toward,
            "trade_given": self.trade_given,
            "trade_received": self.trade_received,
            "combat_record": self.combat_record,
            "wellbeing": self.wellbeing,
            "next_era": next_era_info,
            "expeditions": [
                {
                    "kind": exp.get("kind", "scout"),
                    "pos": exp["pos"],
                    "day": exp["day"],
                    "max_days": exp["max_days"],
                    "phase": exp["phase"],
                    "lead_scout": exp["lead_scout"],
                    "food_gathered": exp["food_gathered"],
                    "water_gathered": exp["water_gathered"],
                    "path": exp["path"],
                }
                for exp in self.expeditions
            ],
        }


# (tribe.name, label) pairs, in the order the ending card lists them -- one-off
# structures a tribe either has or doesn't, checked directly off the same boolean
# flags every BUILD_* action already sets (see Tribe.__init__), not anything
# tallied fresh here. Long Houses/Warehouses/wall rings are counted separately in
# _final_build_summary since those are repeatable, not one-off.
_ONE_OFF_STRUCTURE_FLAGS: tuple[tuple[str, str], ...] = (
    ("sawmill_built", "Sawmill"), ("quarry_built", "Quarry"), ("dock_built", "Dock"),
    ("fishery_built", "Fishery"), ("kitchen_built", "Kitchen"), ("tannery_built", "Tannery"),
    ("mine_built", "Mine"), ("forge_built", "Forge"), ("keep_built", "Keep"),
    ("fortress_built", "Fortress"), ("castle_built", "Castle"), ("road_built", "Road"),
    ("hatchery_built", "Hatchery"), ("bath_house_built", "Bath House"),
    ("library_built", "Library"), ("well_built", "Well"),
    ("object_creator_built", "Object Creator"),
)


def _final_build_summary(tribe: "Tribe") -> str:
    """What a tribe actually ended the run having built -- see
    Simulation._generate_game_over_summary's own docstring for the live report
    this answers ("did they ever build a Castle/Object Creator" was previously
    unanswerable once a run ended). Reads existing one-way flags/counts only,
    the same data the sidebar and to_dict() already expose; nothing new is
    computed here."""
    parts = []
    if tribe.wall_rings:
        parts.append(f"{len(tribe.wall_rings)} wall ring(s)")
    if tribe.long_houses_built:
        parts.append(f"{tribe.long_houses_built} Long House(s)")
    parts.extend(label for flag, label in _ONE_OFF_STRUCTURE_FLAGS if getattr(tribe, flag, False))
    if tribe.warehouses_built:
        parts.append(f"{tribe.warehouses_built} Warehouse(s)")
    return ", ".join(parts) if parts else "no permanent structures"


def _conquest_record_summary(tribe: "Tribe") -> str | None:
    """War and World Domination era's one real action, DECLARE_CONQUEST (see
    actions.py._declare_conquest), only ever showed up in the ending card as a
    reached *era* -- a win merges the loser away entirely (a separate
    game_over_reason, "world_domination"), so a run that ended some other way
    (era_ceiling, manual_quit) gave no way to tell "this tribe never tried"
    from "tried and lost." tribe.combat_record already tracks both an
    attacker's own "Conquest" tally and a defender's "Conquest Defense" one
    (actions.py._record_combat) -- this just reads it back. Returns None (no
    trailing sentence at all) when neither ever happened AND the tribe never
    even reached the era DECLARE_CONQUEST requires -- a hollow "0 attempts"
    line would be noise for the vastly more common case of a run that stopped
    long before then. A tribe that DID reach War and World Domination but
    still shows no activity here gets an explicit "reached it and never
    attempted conquest" instead of the same silence, since that's exactly the
    "did War even happen" ambiguity this was written to resolve."""
    attack = tribe.combat_record.get("Conquest", {})
    defense = tribe.combat_record.get("Conquest Defense", {})
    won, lost = attack.get("won", 0), attack.get("lost", 0)
    held, fell = defense.get("won", 0), defense.get("lost", 0)
    if not (won or lost or held or fell):
        if tribe.era == "war_and_world_domination_era":
            return "Reached War and World Domination but never attempted or faced DECLARE_CONQUEST"
        return None
    bits = []
    if won or lost:
        bits.append(f"declared conquest {won + lost} time(s) ({won} won, {lost} lost)")
    if held or fell:
        bits.append(f"was the target of conquest {held + fell} time(s) ({held} held, {fell} fell)")
    return "; ".join(bits).capitalize()


def _append_expedition_path_point(exp: dict, x: int, y: int) -> None:
    """See config.EXPEDITION_PATH_MAX_POINTS's own comment -- a pure
    visualization breadcrumb (backend logic never reads this), capped so a
    months-long push doesn't grow the websocket payload/board_history.db row
    forever. Simply stops recording past the cap rather than a sliding
    window, so the frontend's incremental Path2D cache (path.length grew by
    exactly one) never has a reason to fall back to a full rebuild."""
    if len(exp["path"]) < config.EXPEDITION_PATH_MAX_POINTS:
        exp["path"].append([x, y])


def _livestock_surplus_threshold(tribe: "Tribe") -> int:
    """How large tribe.eggs/tribe.flock can grow before Simulation.
    _advance_livestock_feast starts auto-eating the surplus -- see config.
    LIVESTOCK_SURPLUS_THRESHOLD's own comment. Scales with population the same
    max(floor, population-scaled) shape actions.expedition_capacity already
    uses, so a large tribe can sustain a genuinely larger flock instead of one
    permanently capped at the same dozen regardless of size. Sent to the
    frontend via Tribe.to_dict() rather than recomputed there, so the
    "surplus feasted on" display can never drift from what actually happened
    server-side."""
    return max(config.LIVESTOCK_SURPLUS_THRESHOLD, tribe.population // config.LIVESTOCK_SURPLUS_POPULATION_DIVISOR)


def _scaled_population_loss(tribe: "Tribe") -> int:
    """Starvation/dehydration's per-event death toll -- see config.
    POPULATION_LOSS_DIVISOR's own comment. Same max(floor, population-scaled)
    shape _livestock_surplus_threshold/expedition_capacity already use: still
    exactly 1 for a small tribe (unchanged from the old flat constant), a real
    deterrent for a large one."""
    return max(1, tribe.population // config.POPULATION_LOSS_DIVISOR)


def _era_resource_amount(tribe: "Tribe", resource: str) -> int:
    """Reads one of an Era's requires_resources/advancement_cost entries off a
    tribe. wood/stone/water/food are real Tribe attributes (getattr handles
    them); a named unique resource like "Fur" or "Orosite Ore" (mines/
    tannery -- see world.UNIQUE_RESOURCE_BY_BIOME, actions.py._build_tannery)
    only ever lives in tribe.unique_resources, a plain dict, so getattr alone
    would silently always read 0 for those and no era could ever actually gate
    on them. hasattr is the dispatch: every core resource is a real attribute
    Tribe.__init__ always sets, so it's never confused with a unique_resources
    key even though some of those (\"Serpent's Gold\") aren't valid Python
    identifiers -- getattr/setattr both accept arbitrary strings regardless."""
    if hasattr(tribe, resource):
        return getattr(tribe, resource)
    return tribe.unique_resources.get(resource, 0)


def _spend_era_resource(tribe: "Tribe", resource: str, amount: int) -> None:
    """The spending half of _era_resource_amount's dispatch -- same
    core-attribute-vs-unique_resources split, floored at 0 either way (a core
    resource attribute is never allowed negative elsewhere in this codebase,
    and a unique_resources entry shouldn't become the sole exception)."""
    if hasattr(tribe, resource):
        setattr(tribe, resource, max(0, getattr(tribe, resource) - amount))
    else:
        tribe.unique_resources[resource] = max(0, tribe.unique_resources.get(resource, 0) - amount)


class Simulation:
    def __init__(
        self, tribe_configs: list[dict], ollama_url: str = config.OLLAMA_URL,
        immortality_cycles: int = 0,
    ):
        if not tribe_configs:
            raise ValueError("Simulation needs at least one tribe")
        # Opt-in, off (0) by default -- a spectator-facing mode for watching what
        # happens *after* the survival crisis (scouting maturing, trade, breeding, era
        # advancement) instead of every run getting cut off by extinction at 30-60
        # cycles before any of that plays out. See Simulation._lose_population: this
        # suppresses the actual population-loss consequence only, while self.cycle <=
        # immortality_cycles -- it never touches what a tribe's own live prompt is
        # told. The same "Your people are starving" facts, the same crisis framing,
        # the same reasoning test -- a tribe that would have gone extinct just keeps
        # facing the same real pressure with the stakes quietly held back, not a tribe
        # that's been let off the hook and knows it.
        self.immortality_cycles = immortality_cycles
        self.client = OllamaClient(ollama_url)
        self.scheduler = ModelBatchScheduler(self.client)
        self.world = Landscape(config.GRID_SIZE)
        self.trauma = AncestralTraumaMatrix(config.GRID_SIZE)
        self.translation = TranslationConfidenceMatrix()
        self.event_log = RunEventLog()
        # Shared identifier with the event log's own filename (e.g. "run_20260830_
        # 113014") so a full per-cycle board-state row (backend/board_history.py) and
        # this run's narrative chronicle can be cross-referenced by the same id.
        self.run_id = self.event_log.path.stem
        self.tribes: dict[str, Tribe] = {}
        for i, cfg in enumerate(tribe_configs[: config.MAX_TRIBES]):
            # An explicit x/y (e.g. to set up two tribes starting near each other) is an
            # initial condition, same category as SPAWN_POINTS itself -- it says nothing
            # about what either tribe then chooses to do about being close.
            if "x" in cfg and "y" in cfg:
                x, y = cfg["x"], cfg["y"]
            else:
                x, y = SPAWN_POINTS[i % len(SPAWN_POINTS)]
            tid = f"tribe_{i}"
            self.tribes[tid] = Tribe(tid, cfg["name"], cfg["model"], x, y, COLORS[i % len(COLORS)], self.event_log)
            if i < len(SPAWN_WATER_TARGET_KINDS):
                self.tribes[tid].seed_scout_heading_toward_water(self.world, kinds=SPAWN_WATER_TARGET_KINDS[i])
            else:
                self.tribes[tid].seed_scout_heading_toward_water(self.world)
        # Neutral, non-AI raid/trade targets (backend/actions.py._raid/_trade) --
        # see config.MINOR_SETTLEMENT_COUNT's own comment for the full design note.
        self.minor_settlements: list[dict] = []
        self._spawn_minor_settlements()
        self.cycle = 0
        self.paused = False
        self.status = "OPERATIONAL"
        self.game_over = False
        # Explicit request: "we are missing 'the end'" -- a run that reached
        # the era ceiling used to just keep stepping forever with nothing left
        # to progress toward (confirmed live: 400+ cycles, over half a real
        # run, spent this way). Set by _trigger_game_over alongside
        # self.game_over/self.status; game_over_summary is the Overseer-voice
        # retrospective the frontend's end-of-run splash actually displays.
        self.game_over_reason: str | None = None
        self.game_over_summary: str = ""
        # A wandering storm cloud (see Simulation._advance_weather) -- world weather,
        # independent of any tribe. None when no storm is active; otherwise
        # {"x", "y", "heading", "cycles_left"}. lightning_strike is only ever set for
        # the exact cycle a strike happens (None otherwise), read by both
        # _build_visible_entities (a nearby tribe's live fact) and snapshot() (the
        # frontend's one-cycle flash).
        self.storm_cloud: dict | None = None
        self.lightning_strike: tuple[int, int] | None = None
        # A raider attack or tribe-vs-tribe RAID/STRIKE_RAIDER_CAMP resolution this
        # cycle -- list-shaped (unlike lightning_strike) since more than one could
        # resolve in the same cycle across different tribes. Cleared at the start of
        # every step() and repopulated fresh, the same one-cycle-lifetime pattern as
        # lightning_strike, so the frontend naturally sees each entry as a brief flash.
        self.recent_encounters: list[dict] = []
        self.self_mod = (
            SelfModEngine(self.client, tribe_configs[0]["model"], config.SELF_MOD_COOLDOWN_CYCLES)
            if config.ENABLE_SELF_MODIFICATION
            else None
        )

    @classmethod
    async def create(
        cls, tribe_configs: list[dict], ollama_url: str = config.OLLAMA_URL,
        immortality_cycles: int = 0,
    ) -> "Simulation":
        """Preferred constructor: runs a one-time VRAM sanity check per model before
        building the simulation, and drops a warning into a tribe's chronicle (rather
        than blocking it) if its model looks too large for the configured budget."""
        guard = HardwareVRAMBoundaryGuard(ollama_url, config.VRAM_LIMIT_GB)
        warnings: dict[str, str] = {}
        for cfg in tribe_configs[: config.MAX_TRIBES]:
            ok, warning = await guard.verify_vram_safety_margin(cfg["model"])
            if not ok:
                warnings[cfg["name"]] = warning

        sim = cls(tribe_configs, ollama_url, immortality_cycles)
        for tribe in sim.tribes.values():
            if tribe.name in warnings:
                tribe.history.append(f"VRAM WARNING: {warnings[tribe.name]}")
        await asyncio.gather(*(sim._install_chief(tribe) for tribe in sim.tribes.values()))
        return sim

    async def _install_chief(self, tribe: "Tribe") -> None:
        """One-time in-fiction leadership contest (see leadership.py) -- the resulting
        philosophy becomes standing context in every future turn, generated by the tribe
        itself rather than scripted from outside it. If the tribe isn't already on fresh
        water, the election is told only that water hasn't been confirmed nearby (no
        coordinates -- see leadership.py's docstring for why) and may decide -- its own
        call, not ours -- to decree that finding some is a priority. Specifically fresh
        water, not "water" generally: a tribe standing on the coast still has no drinking
        water, so ocean doesn't count as already solved. What else the ocean might be
        good for is left entirely open."""
        water_needed = self.world.biome(tribe.x, tribe.y) not in ("river", "lake")
        context = tribe.pending_chief_context
        tribe.pending_chief_context = ""  # consumed once, whether or not this election uses it

        result = await elect_chief(self.client, tribe.model, tribe.name, water_needed, context)
        tribe.chief_name = result.get("chief_name", "")
        tribe.chief_philosophy = result.get("guiding_philosophy", "")
        victory = result.get("victory_method", "")
        tribe.chief_victory = victory
        if tribe.chief_name:
            tribe.chiefs_elected += 1
            note = f"{tribe.chief_name} has become chief"
            tribe.history.append(f"{note} ({victory})." if victory else f"{note}.")

        # A weak/small model can put a bare bool or string where a nested object was
        # asked for (seen live: {"water_decision": true} from llama3.2:1b) -- valid
        # JSON, wrong shape. `... or {}` alone doesn't catch a truthy non-dict (True or
        # {} is still True), so the isinstance check is load-bearing, not decorative.
        water_decision = result.get("water_decision")
        if not isinstance(water_decision, dict):
            water_decision = {}
        if water_needed and water_decision.get("decreed"):
            tribe.chief_decree = self._WATER_DECREE_TEXT
            reason = water_decision.get("reason", "")
            entry = f"Chief {tribe.chief_name} decrees: {tribe.chief_decree}"
            tribe.history.append(f"{entry} ({reason})." if reason else f"{entry}.")

    _WATER_DECREE_TEXT = "prioritize dispatching scouts to find reliable water"

    def _clear_resolved_water_decree(self, tribe: "Tribe") -> None:
        """The water-finding decree above used to never expire on its own -- even long
        after a scout actually confirmed water, the exact same decree kept getting fed
        into every future turn's prompt, continuously pointing every cycle's reasoning
        back at scouting for water specifically regardless of what the tribe actually
        needed by then. Real data confirmed this: tribes kept scouting for water they
        already had. Cleared the instant its own stated condition (confirmed_water_
        sites is non-empty) is objectively met -- a resolved fact, not a fresh
        decision to declare it done."""
        if tribe.chief_decree == self._WATER_DECREE_TEXT and tribe.confirmed_water_sites:
            tribe.chief_decree = ""

    async def _resolve_birth(self, tribe: "Tribe") -> None:
        """Resolves a BREED action's pending_birth (see actions.py._breed) with a
        real, non-scripted LLM call (backend/breeding.py) -- same pattern as
        pending_chief_context/_install_chief: the mechanical decision (two named
        people are starting a family) happens synchronously in the action handler,
        the actual outcome is generated here, in the same cycle."""
        parent_a = tribe.pending_birth["parent_a"]
        parent_b = tribe.pending_birth["parent_b"]
        tribe.pending_birth = None

        result = await breed_individuals(self.client, tribe.model, tribe.name, parent_a, parent_b)
        child_name = result.get("child_name") or f"child of {parent_a} and {parent_b}"
        note = result.get("note", "")

        tribe.population += 1
        tribe.max_population = max(tribe.max_population, tribe.population)
        tribe.lineage.append({
            "child_name": child_name, "parents": [parent_a, parent_b], "cycle": self.cycle,
        })
        entry = f"{parent_a} and {parent_b} welcome a child, {child_name}"
        tribe.history.append(f"{entry} -- {note}" if note else f"{entry}.")

    async def _resolve_hatch(self, tribe: "Tribe") -> None:
        """Resolves a GATHER_EGGS action's pending_hatch (see actions.py._gather_eggs)
        with a real, non-scripted LLM call (backend/genetics.py's hatch()) -- same
        pattern as _resolve_birth, applied to the flock instead of the tribe's own
        population. A founding egg (no existing pair to cross) just hatches with a
        plain trait; once two flock members exist, hatch() crosses their traits with
        one mutation, the same spirit as genetics.py's dormant breed()."""
        parents = tribe.pending_hatch["parents"]
        tribe.pending_hatch = None

        if parents:
            result = await hatch(self.client, tribe.model, parents[0], parents[1], tribe.era)
        else:
            result = {"trait": "unremarkable but hardy", "note": "the first of the flock hatches"}
        trait = result.get("trait") or "unremarkable but hardy"
        note = result.get("note", "")

        if tribe.flock == 0 and tribe.territory_center is not None:
            # One-time flock pen, placed the moment the first egg actually hatches --
            # matches every other building's "the real thing exists now" placement
            # trigger, not the earlier GATHER_EGGS action that only started the
            # (possibly multi-cycle) hatch.
            w, h = config.BUILDING_FOOTPRINTS["flock_pen"]
            slot = architect.find_free_slot(self.world, tribe, "flock_pen")
            if slot is not None:
                architect.record_building(tribe, "flock_pen", slot[0], slot[1], w, h, self.cycle)
        tribe.flock += 1
        tribe.flock_lineage.append({
            "trait": trait,
            "parents": [p["trait"] for p in parents] if parents else [],
            "cycle": self.cycle,
            "note": note,
        })
        entry = "an egg hatches -- the flock grows"
        tribe.history.append(f"{entry} ({note})." if note else f"{entry}.")
        self._award_trophy(tribe, "Flock Keeper")

    async def _resolve_settlement_naming(self, tribe: "Tribe") -> None:
        """Resolves pending_settlement_naming (set in _check_for_celebration, once
        the tribe has genuinely settled next to real water) with a real, non-scripted
        LLM call (backend/leadership.py's name_settlement) -- same pattern as
        _resolve_birth/_resolve_hatch."""
        tribe.pending_settlement_naming = False
        biome = self.world.biome(tribe.x, tribe.y)
        result = await name_settlement(self.client, tribe.model, tribe.name, tribe.chief_name, biome)
        tribe.settlement_name = result.get("settlement_name") or f"{tribe.name}'s Settlement"
        note = result.get("note", "")
        entry = f"Chief {tribe.chief_name} names the settlement {tribe.settlement_name}" if tribe.chief_name else f"the settlement is named {tribe.settlement_name}"
        tribe.history.append(f"{entry} -- {note}" if note else f"{entry}.")

    async def _resolve_cultural_crossover(self, tribe: "Tribe") -> None:
        """Resolves DECLARE_ALLIANCE's pending_cultural_crossover (see actions.py.
        _declare_alliance) with a real, non-scripted LLM call (backend/genetics.py's
        breed()) -- same pending_X/resolve pattern as _resolve_birth/_resolve_hatch,
        applied to two whole tribes' cultures instead of two people or two flock
        members.

        Adapted to what's actually real on Tribe, rather than the 'ideology'/
        'lexicon' dict shape breed() was originally written against (there's no
        separate per-tribe vocabulary field): chief_philosophy stands in for
        'ideology', and TranslationConfidenceMatrix.stabilized_tokens -- the real,
        empirically-converged shared vocabulary between this exact pair -- stands
        in for 'lexicon'.

        Deliberately doesn't overwrite chief_philosophy directly -- that's the
        night-cycle reflection's own authority (backend/reflection.py), not
        something a single alliance should silently override. This only records
        the moment as real tribal history for both sides, an emergent event
        either tribe's own future reasoning can reference on its own, not a
        scripted rewrite of who they are."""
        rival_id = tribe.pending_cultural_crossover
        tribe.pending_cultural_crossover = None
        rival = self.tribes.get(rival_id)
        if rival is None or rival.extinct:
            return

        shared_tokens = self.translation.stabilized_tokens(tribe.id, rival.id)
        lexicon = {token: "a word both tribes have converged on" for token in shared_tokens}
        tribe_a = {"ideology": tribe.chief_philosophy or "no fixed creed yet", "lexicon": lexicon}
        tribe_b = {"ideology": rival.chief_philosophy or "no fixed creed yet", "lexicon": lexicon}
        result = await breed(self.client, tribe.model, tribe_a, tribe_b, tribe.era)
        note = result.get("note") or "two cultures briefly touch, then go their own way"
        entry = f"{tribe.name} and {rival.name} exchange ideas as new allies -- {note}"
        tribe.history.append(entry)
        rival.history.append(entry)

    def _hold_tribal_gathering(self, tribe: "Tribe") -> None:
        """An innate tradition, not a chief's choice: every tribe, whatever its
        philosophy or model, gathers once per in-game day (config.DAY_LENGTH_CYCLES,
        mirroring frontend/index.html's own sun/moon cycle -- this really does land at
        the in-game dawn a spectator sees onscreen) to take stock together. Unlike the
        night cycle (backend/reflection.py -- an occasional, chief-specific
        reconsideration of philosophy that costs a real LLM call), this is a cheap,
        deterministic recap of real facts: nothing here is interpreted, so it fires
        reliably for every tribe regardless of model quality.

        Named individuals' recent achievements and any standing unclaimed honor are
        real facts Tribe.to_dict already exposes to the UI, but that the live turn
        prompt never carried back into the tribe's own reasoning -- gathering_brief,
        read by _build_visible_entities the same way a taboo is, is what closes that
        gap."""
        new_trophies = [t for t in tribe.trophies if t["cycle"] > tribe.last_gathering_cycle]
        unclaimed = [
            a for a in tribe.custom_awards
            if not any(t["name"] == a["name"] for t in tribe.trophies)
        ]
        pop_delta = tribe.population - tribe.population_at_last_gathering

        parts = []
        if new_trophies:
            parts.append("since the last gathering, " + "; ".join(
                f"{t['chief']} earned the '{t['name']}' honor" for t in new_trophies
            ))
        if unclaimed:
            parts.append("still unclaimed: " + ", ".join(f"the '{a['name']}' ({a['category']})" for a in unclaimed))
        if pop_delta > 0:
            parts.append(f"the tribe has grown by {pop_delta} since the last gathering")
        elif pop_delta < 0:
            parts.append(f"the tribe has lost {-pop_delta} since the last gathering")
        if tribe.chief_name:
            parts.append(f"Chief {tribe.chief_name}'s guiding philosophy still stands: {tribe.chief_philosophy}")

        tribe.gathering_brief = "; ".join(parts) if parts else "a quiet gathering -- nothing new to report"
        tribe.history.append(f"The tribe gathers as the sun rises. {tribe.gathering_brief}.")
        tribe.last_gathering_cycle = self.cycle
        tribe.population_at_last_gathering = tribe.population

    def _hold_evening_recap(self, tribe: "Tribe") -> None:
        """Explicit request: "at the beginning of the night they should sort of
        recap the accomplishments of the day." Fires at the actual dusk boundary
        (cycle % DAY_LENGTH_CYCLES == DAY_LENGTH_CYCLES // 2, the same day/night split
        frontend/index.html's own sun/moon arc uses) -- distinct from
        _run_night_cycle's occasional, costly philosophy reconsideration
        (NIGHT_CYCLE_EVERY_N_CYCLES drifts against the visual day/night boundary, so
        it isn't reliably "at dusk" at all) and from _hold_tribal_gathering's own
        dawn recap of the *previous* day. Same cheap, deterministic, real-facts-only
        shape as the dawn gathering -- reads (never resets) its last_gathering_cycle/
        population_at_last_gathering baseline, so "today" always means "since this
        morning's gathering" without needing a second set of tracking fields."""
        new_trophies = [t for t in tribe.trophies if t["cycle"] > tribe.last_gathering_cycle]
        pop_delta = tribe.population - tribe.population_at_last_gathering

        parts = []
        if new_trophies:
            parts.append("today " + "; ".join(
                f"{t['chief']} earned the '{t['name']}' honor" for t in new_trophies
            ))
        if pop_delta > 0:
            parts.append(f"the tribe grew by {pop_delta} today")
        elif pop_delta < 0:
            parts.append(f"the tribe lost {-pop_delta} today")
        parts.append(f"{tribe.wood} wood, {tribe.stone} stone, {tribe.food} food, and {tribe.water} water on hand")

        recap = "; ".join(parts)
        tribe.history.append(f"As the sun sets, the tribe takes stock of the day's work: {recap}.")

    def _build_night_inventory(self, tribe: "Tribe") -> str:
        """A structured "state of affairs" snapshot for the night-cycle reviewer,
        alongside the raw chronicle -- what the chief actually takes stock of after the
        day's council, before retiring to sleep and dream on it. The raw chronicle
        alone tends to just echo whatever the tribe has been doing turn after turn (its
        own recent phrasing), which made a real mismatch -- surplus water, zero food,
        still settling scouts out for water long after it's secured -- easy for the
        reviewer to miss entirely. Facts only; the reviewer still decides for itself
        what, if anything, should change."""
        lines = [
            f"Population: {tribe.population}.",
            f"Resources on hand: {tribe.wood} wood, {tribe.stone} stone, {tribe.food} food, {tribe.water} water.",
        ]
        survival_bias, _critical = survival_bias_string(
            tribe.food, tribe.water, tribe.population, tribe.fishing_learned, tribe.cooking_learned,
            water_secure=_is_water_secure(tribe), food_secure=_is_food_secure(tribe),
        )
        if survival_bias:
            lines.append(survival_bias)
        if self._is_camped(tribe):
            lines.append("The tribe is camped on farmable ground.")
        else:
            lines.append(
                f"The tribe has not made camp anywhere farmable yet "
                f"({tribe.cycles_since_relocate}/{config.SETTLEMENT_STABILITY_CYCLES} cycles without relocating)."
            )
        nxt = next_era(tribe.era)
        if nxt is not None:
            gaps = []
            if tribe.population < nxt.requires_population:
                gaps.append(f"population {tribe.population}/{nxt.requires_population}")
            for resource, minimum in nxt.requires_resources.items():
                have = _era_resource_amount(tribe, resource)
                if have < minimum:
                    gaps.append(f"{resource} {have}/{minimum}")
            if gaps:
                lines.append(f"To reach {nxt.label}, still short on: {', '.join(gaps)}.")
        return " ".join(lines)

    async def _run_night_cycle(self, tribe: "Tribe") -> None:
        """The "night cycle" (backend/reflection.py): a larger reviewing model looks
        back at this tribe's own recent history and decides for itself whether its
        guiding philosophy should change. Runs far less often than a live turn (see
        config.NIGHT_CYCLE_EVERY_N_CYCLES) and with a different, larger model than
        whatever the tribe plays live with -- the piece from the original design
        transcript that gives a tribe's own accumulated experience a chance to
        compound into wisdom over time, distinct from breed()/breed_individuals'
        cross-tribe/cross-individual crossover."""
        recent_events = list(tribe.history)[-config.NIGHT_CYCLE_HISTORY_WINDOW:]
        inventory = self._build_night_inventory(tribe)
        result = await reflect_on_history(
            self.client, config.NIGHT_CYCLE_REVIEWER_MODEL, tribe.name,
            tribe.chief_philosophy, recent_events, inventory,
        )
        # The chief's own reasoning for this reflection -- kept even when the
        # philosophy didn't change, so the frontend has something real to show as a
        # night-time thought bubble (see index.html's drawThoughtBubble) beyond just
        # "nothing changed."
        if result.get("reasoning"):
            tribe.last_reflection = result["reasoning"]
            tribe.last_reflection_cycle = self.cycle
        if result.get("changed"):
            old_philosophy = tribe.chief_philosophy
            tribe.chief_philosophy = result.get("revised_philosophy", old_philosophy)
            reasoning = result.get("reasoning", "")
            entry = f"Reflecting on recent events, Chief {tribe.chief_name} reconsiders the tribe's philosophy: {tribe.chief_philosophy}"
            tribe.history.append(f"{entry} ({reasoning})." if reasoning else f"{entry}.")

        # Captures the chief's own proposed honor; Simulation._check_custom_awards is
        # what actually hands it out once someone earns it (see reflection.py's
        # AWARD_CATEGORIES docstring).
        proposed = result.get("proposed_award")
        if isinstance(proposed, dict) and proposed.get("name") and proposed.get("category") in AWARD_CATEGORIES:
            if not any(a["name"] == proposed["name"] for a in tribe.custom_awards):
                tribe.custom_awards.append({
                    "name": proposed["name"], "category": proposed["category"], "cycle": self.cycle,
                })
                tribe.history.append(
                    f"Chief {tribe.chief_name} establishes a new honor, the '{proposed['name']}', "
                    f"for excellence in {proposed['category']} -- not yet awarded to anyone."
                )

        # See config.NIGHT_CYCLE_RANDOM_BREED_CHANCE -- a chance encounter independent
        # of any specific celebration milestone, using the exact same eligibility rule
        # and $0 cost every other breeding path already uses (_eligible_breeding_pair,
        # BREED_FOOD_COST/WATER_COST).
        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            if random.random() < config.NIGHT_CYCLE_RANDOM_BREED_CHANCE:
                pair = _eligible_breeding_pair(tribe)
                if pair is not None:
                    parent_a, parent_b = pair
                    tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                    tribe.history.append(
                        f"in the quiet of the night, {parent_a} and {parent_b} decide to start a family together"
                    )

    async def add_tribe(self, name: str, model: str, x: int | None = None, y: int | None = None) -> str | None:
        """Injects a new tribe into an already-running simulation. Returns an error
        message on failure (max tribes reached), or None on success. Runs the same
        one-time VRAM check and chief election as Simulation.create()."""
        if len(self.tribes) >= config.MAX_TRIBES:
            return f"Cannot add tribe: maximum of {config.MAX_TRIBES} tribes reached."

        index = len(self.tribes)
        tid = f"tribe_{index}"
        if x is None or y is None:
            x, y = SPAWN_POINTS[index % len(SPAWN_POINTS)]
        color = COLORS[index % len(COLORS)]
        tribe = Tribe(tid, name, model, x, y, color, self.event_log)
        if index < len(SPAWN_WATER_TARGET_KINDS):
            tribe.seed_scout_heading_toward_water(self.world, kinds=SPAWN_WATER_TARGET_KINDS[index])
        else:
            tribe.seed_scout_heading_toward_water(self.world)

        guard = HardwareVRAMBoundaryGuard(self.client.base_url, config.VRAM_LIMIT_GB)
        ok, warning = await guard.verify_vram_safety_margin(model)
        if not ok:
            tribe.history.append(f"VRAM WARNING: {warning}")
        await self._install_chief(tribe)

        self.tribes[tid] = tribe
        # A fresh tribe means the game isn't over anymore, even if every previous tribe
        # died -- undoes _trigger_game_over's stop/unload so stepping resumes.
        self.game_over = False
        self.game_over_reason = None
        self.game_over_summary = ""
        if self.status == "GAME OVER":
            self.status = "OPERATIONAL"
        return None

    def snapshot(self) -> dict:
        tribe_ids = list(self.tribes.keys())
        consensus = []
        for i in range(len(tribe_ids)):
            for j in range(i + 1, len(tribe_ids)):
                a_id, b_id = tribe_ids[i], tribe_ids[j]
                summary = self.translation.pair_summary(a_id, b_id)
                # Explicit request: "I think they might have an Alliance or even
                # Trading with each other now but there is little to no readout
                # of that status and condition." tribe.stance_toward already
                # exists (DECLARE_ALLIANCE/DECLARE_WAR, actions.py) and is
                # symmetric -- was just never surfaced anywhere in the UI.
                stance = self.tribes[a_id].stance_toward.get(b_id, "NEUTRAL")
                consensus.append({
                    "a": self.tribes[a_id].name, "b": self.tribes[b_id].name,
                    "stance": stance,
                    **summary,
                })

        return {
            "cycle": self.cycle,
            "status": self.status,
            "game_over_reason": self.game_over_reason,
            "game_over_summary": self.game_over_summary,
            "paused": self.paused,
            "immortality_cycles": self.immortality_cycles,
            "storm_cloud": {"x": self.storm_cloud["x"], "y": self.storm_cloud["y"]} if self.storm_cloud else None,
            "lightning_strike": list(self.lightning_strike) if self.lightning_strike else None,
            "recent_encounters": self.recent_encounters,
            "tribes": {tid: t.to_dict() for tid, t in self.tribes.items()},
            "minor_settlements": self.minor_settlements,
            "structures": [{"x": x, "y": y, **info} for (x, y), info in self.world.constructions.items()],
            "trails": [
                {
                    "x": x, "y": y, "wear": t["wear"], "color": t["color"],
                    "crossings": t.get("crossings", 0), "owner": t.get("owner"),
                    "is_toll_road": self.world.is_toll_road(x, y),
                }
                for (x, y), t in self.world.trails.items()
            ],
            "linguistic_consensus": consensus,
        }

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    async def step(self) -> None:
        if self.paused or self.game_over:
            return
        # See the per-tribe unload check near the end of this method -- a tribe can
        # go extinct mid-cycle from several different sources (upkeep starvation, a
        # raider attack, a lost raid...), so this snapshot is how "newly extinct
        # this cycle" gets detected without hooking every one of those call sites
        # individually.
        previously_extinct = {tid for tid, tribe in self.tribes.items() if tribe.extinct}
        self.cycle += 1
        self.event_log.current_cycle = self.cycle
        self.translation.decay()
        self.world.regenerate(config.DEPLETION_REGEN_PER_CYCLE)
        self.world.decay_trails(config.TRAIL_DECAY_PER_CYCLE)
        self._advance_weather()
        self._advance_minor_settlements()
        # One-cycle-lifetime, same as lightning_strike -- repopulated fresh below by
        # _check_raider_attack/_raid/_strike_raider_camp, whichever fire this cycle.
        self.recent_encounters = []

        # Bug report: "scouts fired before the day started." These used to run
        # *after* the per-tribe turn loop below, so a SCOUT (or any other action)
        # dispatched on the exact cycle a new day/dusk boundary lands appended its
        # own chronicle line before "the tribe gathers as the sun rises"/the evening
        # recap for that same cycle -- reading, top to bottom, as if the day's first
        # action happened before the day itself began. Moved here, before any of
        # today's actions are resolved, so the day/dusk announcement always leads.
        if self.cycle % config.DAY_LENGTH_CYCLES == 0:
            for tribe in self.tribes.values():
                if not tribe.extinct:
                    self._hold_tribal_gathering(tribe)

        if self.cycle % config.DAY_LENGTH_CYCLES == config.DAY_LENGTH_CYCLES // 2:
            for tribe in self.tribes.values():
                if not tribe.extinct:
                    self._hold_evening_recap(tribe)

        requests = []
        contexts = {}
        for tid, tribe in self.tribes.items():
            if tribe.extinct:
                continue
            request, ctx = self._prepare_turn(tribe)
            requests.append(request)
            contexts[tid] = ctx

        results = await self.scheduler.run_batch(requests)

        for tid, tribe in self.tribes.items():
            if tribe.extinct:
                continue
            outcome = results.get(tid, {"intent": {}, "latency_ms": 0.0})
            self._apply_turn(tribe, outcome["intent"], outcome["latency_ms"], contexts[tid])
            self._advance_automatic_fire(tribe)
            self._advance_automatic_boat(tribe)
            self._advance_wall_security(tribe)
            # Live bug ("water is a problem and it should never be after they
            # settle"): these three post a cycle's passive food/water income --
            # _apply_upkeep used to run first and could drain a thin carried-over
            # buffer negative before this same cycle's own settled-water/fishing/
            # farming income had landed, triggering a real thirst/hunger death the
            # end-of-turn display never showed (it looked stable because the income
            # arrived a moment later, same cycle, refilling it back up). Posting
            # income before the drain lets a cycle's own income actually cover that
            # same cycle's own upkeep instead of only the next one's.
            self._advance_water_supply(tribe)
            self._advance_food_supply(tribe)
            self._advance_wood_supply(tribe)
            self._advance_stone_supply(tribe)
            self._advance_fish_supply(tribe)
            self._advance_farming(tribe)
            self._apply_upkeep(tribe)
            self._check_raider_attack(tribe)
            self._advance_raider_approach(tribe)
            self._advance_battalion_patrol(tribe)
            self._advance_battalion_readiness_upkeep(tribe)
            self._grow_population(tribe)
            self._advance_era_if_ready(tribe)
            if not tribe.settlement_name and not tribe.pending_settlement_naming and self._is_settled_near_water(tribe):
                self._celebrate_settling(tribe)
            self._advance_mine_yield(tribe)
            self._advance_in_territory_site_yields(tribe)
            self._advance_tannery_yield(tribe)
            self._advance_resource_trails(tribe)
            self._advance_flock(tribe)
            self._advance_flock_eggs(tribe)
            self._advance_livestock_feast(tribe)
            self._advance_city_founding(tribe)
            self._check_chief_trophies(tribe)
            self._check_for_celebration(tribe)

        # History: this used to be gated here (dawn-only once settled, every cycle
        # for a still-searching tribe) to fix "Exploration time is like 6 now, but
        # this is being counted as 6 cycles, not full days" -- EXPEDITION_MAX_DAYS/
        # HUNTING_PARTY_MAX_DAYS/EXPLORATION_PARTY_MAX_DAYS were always meant as
        # real days, and a flat every-cycle call made a "day 6" scout give up after
        # 0.3 of a real day. That fix then caused its own regression (a tribe died
        # of thirst before ever founding, since a scout's round trip taking real
        # days didn't come with a bigger starting buffer to match) -- fixed by
        # exempting a still-searching tribe from the dawn gate.
        #
        # Explicit request ("everyone moving on the board moves at the pace of 1
        # sky tick"): a settled tribe's expedition used to sit still for an entire
        # day then jump its whole day's distance at once on the dawn boundary --
        # visibly a teleport, not movement. The gate now lives inside
        # _advance_one_expedition itself (is_new_day) instead of here: movement
        # happens every cycle for every tribe regardless of settlement, only the
        # per-cycle distance and the once-a-day bookkeeping (day count, "daily"
        # resource gains, hunting rolls) still differ by has_ever_settled -- so the
        # exact real-day pacing/regression-fix above is unchanged, just no longer
        # tied to whether movement itself happens this cycle.
        for tribe in self.tribes.values():
            if tribe.extinct or not tribe.expeditions:
                continue
            self._advance_expeditions(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct:
                self._clear_resolved_water_decree(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.pending_birth:
                await self._resolve_birth(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.pending_hatch:
                await self._resolve_hatch(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.pending_settlement_naming:
                await self._resolve_settlement_naming(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.pending_cultural_crossover:
                await self._resolve_cultural_crossover(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and not tribe.chief_name:
                await self._install_chief(tribe)

        if self.cycle % config.NIGHT_CYCLE_EVERY_N_CYCLES == 0:
            for tribe in self.tribes.values():
                if not tribe.extinct and tribe.chief_name:
                    await self._run_night_cycle(tribe)

        # Explicit request: "when a tribe dies off are we unloading the model" --
        # previously only the ALL-tribes-extinct game-over case (_trigger_game_over)
        # ever unloaded anything; a single tribe going extinct while others played on
        # left its model sitting resident in Ollama's VRAM for no reason (nothing will
        # ever call it again unless a fresh ADD_TRIBE reuses the same model choice).
        # Only unloads a model no other still-living tribe is also using.
        newly_extinct = {tid for tid, tribe in self.tribes.items() if tribe.extinct and tid not in previously_extinct}
        if newly_extinct:
            still_used = {t.model for t in self.tribes.values() if not t.extinct}
            for tid in newly_extinct:
                model = self.tribes[tid].model
                if model not in still_used:
                    await self.client.unload_model(model)

        living_tribes = [t for t in self.tribes.values() if not t.extinct]
        if self.tribes and not living_tribes:
            await self._trigger_game_over("extinction")
        # War and World Domination era's real victory condition.
        # Simulation._merge_tribes physically removes a conquered rival from
        # self.tribes (unlike ordinary hazard/starvation extinction, which only
        # flags tribe.extinct and leaves them in the dict) -- "exactly one
        # tribe remains, and it got there via at least one real conquest" is
        # the clean, unambiguous signal that this is a win, not just an empty
        # board left behind by unrelated hazard deaths.
        elif len(self.tribes) == 1 and next(iter(self.tribes.values())).conquests_won > 0:
            await self._trigger_game_over("world_domination")
        # Explicit request: "we are missing 'the end'" -- every still-living
        # tribe reaching the era ceiling (next_era returns None) is just as
        # real an ending as total extinction; a real run kept stepping 400+
        # cycles past this point with nothing left to progress toward.
        elif living_tribes and all(next_era(t.era) is None for t in living_tribes):
            await self._trigger_game_over("era_ceiling")

        if self.self_mod:
            self.self_mod.tick()
            max_latency = max((r["latency_ms"] for r in results.values()), default=0.0)
            if not self.self_mod.on_cooldown and max_latency > config.SELF_MOD_LATENCY_THRESHOLD_MS:
                self.status = "REFACTORING"
                await self.self_mod.attempt_patch(f"turn batch latency {max_latency:.0f}ms")
                importlib.reload(physics)
                self.status = "OPERATIONAL"

        if self.cycle % config.MEMORY_CONSOLIDATE_EVERY_N_CYCLES == 0:
            for tribe in self.tribes.values():
                tribe.memory.consolidate()

    def _advance_weather(self) -> None:
        """A wandering storm cloud, entirely independent of any tribe's actions -- the
        world has weather whether or not anyone's watching. Spawns rarely (only
        checked while no storm is already active), wanders with a heading that jitters
        a little each cycle rather than flying a dead-straight line, and expires after
        STORM_LIFESPAN_CYCLES either way. A strike is rolled once per cycle while a
        storm is active; self.lightning_strike is set only for that one cycle (cleared
        at the top of every call) so it reads as a flash, not a standing hazard.

        A tribe caught directly under a strike takes a real, small hit through the same
        _lose_population channel every other hazard uses -- immortality (see
        Simulation.__init__) protects it exactly the same way. A tribe merely within
        LIGHTNING_STRIKE_RADIUS doesn't get hurt, just a fact about what happened (see
        _build_visible_entities) -- on a forest tile, that fact echoes the same "how
        did anyone first learn fire" question raised this session. Nothing here awards
        fire automatically; it's a real, unscripted event a tribe's own reasoning could
        in principle notice and act on, the same honest test as any other fact."""
        self.lightning_strike = None

        if self.storm_cloud is None:
            if random.random() < config.STORM_SPAWN_CHANCE:
                self.storm_cloud = {
                    "x": random.randint(0, self.world.grid_size - 1),
                    "y": random.randint(0, self.world.grid_size - 1),
                    "heading": random.uniform(0, 2 * math.pi),
                    "cycles_left": config.STORM_LIFESPAN_CYCLES,
                }
            return

        cloud = self.storm_cloud
        cloud["heading"] += random.uniform(-config.STORM_HEADING_JITTER, config.STORM_HEADING_JITTER)
        cloud["x"] = max(0, min(self.world.grid_size - 1, round(cloud["x"] + math.cos(cloud["heading"]) * config.STORM_SPEED)))
        cloud["y"] = max(0, min(self.world.grid_size - 1, round(cloud["y"] + math.sin(cloud["heading"]) * config.STORM_SPEED)))
        cloud["cycles_left"] -= 1

        if random.random() < config.LIGHTNING_STRIKE_CHANCE:
            sx, sy = cloud["x"], cloud["y"]
            self.lightning_strike = (sx, sy)
            for tribe in self.tribes.values():
                if tribe.extinct or (tribe.x, tribe.y) != (sx, sy):
                    continue
                self.trauma.radiate_event_wave(sx, sy, config.LIGHTNING_TRAUMA_MAGNITUDE, config.LIGHTNING_TRAUMA_RADIUS)
                self._lose_population(tribe, config.LIGHTNING_HAZARD_POPULATION_LOSS, cause="lightning")
                tribe.history.append("lightning struck the heart of camp")

        if cloud["cycles_left"] <= 0:
            self.storm_cloud = None

    def _build_visible_entities(self, tribe: Tribe, biome: str, nearby: list[dict],
                                 memories: list[dict], available_actions: list[str]) -> tuple[list[str], str]:
        visible_entities = [f"structure:{s['type']}@({s['x']},{s['y']})" for s in nearby]
        visible_entities += [f"memory(cycle {m['cycle']}): {m['text']}" for m in memories]

        # A lightning strike only lasts one cycle (see _advance_weather) -- a real,
        # unscripted event, not a directive about what it means or what to do next.
        if self.lightning_strike:
            lx, ly = self.lightning_strike
            distance = ((tribe.x - lx) ** 2 + (tribe.y - ly) ** 2) ** 0.5
            if distance < 1.5:
                visible_entities.append("lightning just struck directly at your camp")
            elif distance <= config.LIGHTNING_STRIKE_RADIUS:
                if self.world.biome(lx, ly) == "forest":
                    visible_entities.append(f"lightning struck a tree near ({lx},{ly}) -- it looks like it's burning")
                else:
                    visible_entities.append(f"lightning struck nearby, at ({lx},{ly})")
        # taboos accumulates for a tribe's whole lifetime (see TribeMemory.consolidate,
        # which can add up to 3 more every MEMORY_CONSOLIDATE_EVERY_N_CYCLES) -- slicing
        # the first 3 meant that once any 3 existed, nothing learned later ever surfaced
        # again, however important (e.g. a hard-won confirmed water location, discovered
        # after an early wolf-attack warning already claimed those 3 slots). The most
        # recently learned facts are shown instead, so new knowledge isn't permanently
        # buried by old.
        visible_entities += [f"taboo: {t}" for t in tribe.memory.taboos[-3:]]
        # Explicit request: "make sure they remember all the important discover
        # sites when they are making decisions. those locations are important
        # to the progress of the Tribe and civilization." These six lists used
        # to be sliced to the 3 most recent, same "new knowledge buries old"
        # trap the taboo slice above was already fixed for once -- a tribe that
        # scouted 5 quarry sites over a long run would silently lose the first
        # 2 from its own facts, however valuable. Every confirmed site is a
        # one-time, deduplicated discovery (see _advance_one_expedition), so
        # these lists only grow as large as genuinely distinct real finds --
        # not the unbounded, ever-repeating kind of list slicing exists to cap.
        visible_entities += [f"confirmed water source at ({x},{y})" for x, y in tribe.confirmed_water_sites]
        # Explicit request: an in-territory site already producing its passive
        # freebie (_advance_in_territory_site_yields) shouldn't still be named as
        # somewhere worth traveling to gather -- it's already covered.
        visible_entities += [
            f"confirmed lumber-rich area at ({x},{y})" for x, y in tribe.lumber_sites
            if not self._site_in_own_territory(tribe, x, y)
        ]
        visible_entities += [
            f"a {site['type']} was found at ({site['x']},{site['y']})" for site in tribe.wildlife_sites
            if not self._site_in_own_territory(tribe, site["x"], site["y"])
        ]
        visible_entities += [
            f"confirmed stone-rich area at ({x},{y})" for x, y in tribe.quarry_sites
            if not self._site_in_own_territory(tribe, x, y)
        ]
        visible_entities += [
            f"a vein of {site['resource']} was found at ({site['x']},{site['y']})" for site in tribe.mine_sites
        ]
        visible_entities += [f"raiders reported near ({x},{y})" for x, y in tribe.raider_sightings]

        # Bug report: "one is not exploring and one only explores in one
        # direction." A real, honest fact about how lopsided (or absent) a
        # tribe's own scouting coverage has actually been -- not a nudge
        # toward SCOUT specifically or any particular heading, just naming
        # what the tribe's own confirmed discoveries (which now all persist,
        # see the site-list facts above) actually show about where it's
        # already looked versus never looked at all.
        all_known_sites = (
            tribe.confirmed_water_sites + tribe.lumber_sites
            + [(s["x"], s["y"]) for s in tribe.wildlife_sites]
            + tribe.quarry_sites + [(s["x"], s["y"]) for s in tribe.mine_sites] + tribe.raider_sightings
        )
        if not all_known_sites:
            visible_entities.append(
                "No scouting has turned up anything yet -- the wider world beyond home remains "
                "completely unknown in every direction."
            )
        elif len(all_known_sites) >= 3:
            directions_seen = {_compass_direction(x - tribe.x, y - tribe.y) for x, y in all_known_sites}
            if len(directions_seen) == 1:
                visible_entities.append(
                    f"Every confirmed discovery so far lies to the {next(iter(directions_seen))} -- every "
                    "other direction remains completely unexplored."
                )

        if tribe.raiders_approaching:
            ax, ay = tribe.raiders_approaching["x"], tribe.raiders_approaching["y"]
            cycles_left = tribe.raiders_approaching["cycles_left"]
            visible_entities.append(
                f"RAIDERS ARE RIDING IN, currently near ({ax},{ay}) -- {cycles_left} cycles until they "
                "reach camp. This is real time to prepare, not a surprise."
            )
        if tribe.gathering_brief:
            visible_entities.append(f"this morning's gathering: {tribe.gathering_brief}")
        # Factual telemetry about this exact tile, not a suggestion to move -- what the
        # tribe does with the information is entirely its own reasoning.
        for resource in ("wood", "stone", "water", "game", "forage"):
            level = self.world.scarcity(resource, tribe.x, tribe.y)
            if level > 0:
                visible_entities.append(f"local {resource} scarcity here: {level:.0%}")

        # Resource scarcity above only ever reports *past* depletion -- a tribe standing
        # on a pristine tile gets no signal that game is even present before it's already
        # hunted some. This is a real, occasional sighting instead: scan a small radius
        # (game can be heard/spotted nearby, not just underfoot) and roll a chance scaled
        # by the richest nearby tile's own game yield -- a mountain or ocean tile is
        # essentially silent, a forest is the likeliest place to hear something. Named for
        # whichever hunting action is actually unlocked (GAME_SPECIES_LABEL) as a
        # fallback, but GAME_SPECIES_BY_BIOME gives the richest nearby tile's own biome
        # the final say when it has an entry -- deer in a forest, rabbits on the plains,
        # not the same species word regardless of where the sighting actually is.
        hunting_action = next((a for a in available_actions if a in GAME_SPECIES_LABEL), None)
        if hunting_action:
            best_multiplier = 0.0
            best_biome = None
            for dx in range(-config.GAME_SIGHTING_RADIUS, config.GAME_SIGHTING_RADIUS + 1):
                for dy in range(-config.GAME_SIGHTING_RADIUS, config.GAME_SIGHTING_RADIUS + 1):
                    nearby_biome = self.world.biome(tribe.x + dx, tribe.y + dy)
                    multiplier = BIOME_YIELD_MULTIPLIER["game"].get(nearby_biome, 0.0)
                    if multiplier > best_multiplier:
                        best_multiplier, best_biome = multiplier, nearby_biome
            if best_multiplier > 0 and random.random() < config.GAME_SIGHTING_CHANCE_BASE * best_multiplier:
                pool = GAME_SPECIES_BY_BIOME.get(best_biome, (GAME_SPECIES_LABEL[hunting_action],))
                species = random.choice(pool)
                visible_entities.append(f"wildlife sighting: signs of {species} nearby")

        # Cross-tribe proximity awareness, independent of whether the other tribe has
        # ever broadcast anything (see config.RIVAL_PRECISE_AWARENESS_RADIUS/
        # RIVAL_DISTANT_SIGHTING_RADIUS) -- without this, the default ~62-tile spawn
        # distance meant tribes had no way to ever notice each other at all, which is
        # the real reason TRADE/RAID never fired in a single run this session.
        for other in self.tribes.values():
            if other.id == tribe.id or other.extinct:
                continue
            dx, dy = other.x - tribe.x, other.y - tribe.y
            distance = math.hypot(dx, dy)
            if distance <= config.RIVAL_PRECISE_AWARENESS_RADIUS:
                visible_entities.append(
                    f"{other.name} is nearby at ({other.x},{other.y}), about {distance:.0f} tiles away"
                )
                # Military branch, step 7 (plan file valiant-forging-falcon.md): "The
                # Chief has to actually be able to reach DECLARE_CONQUEST -- not just
                # mechanically possible, but visible: a real Might fact in the
                # prompt." Only shown once a Battalion actually exists on this side --
                # a tribe with no Battalion yet has nothing meaningful to compare, and
                # RAID/DECLARE_CONQUEST's own win chance is unaffected by Might until
                # then anyway (see actions._might_adjusted_win_chance).
                if tribe.battalion_size > 0:
                    visible_entities.append(
                        f"Military strength (Might) compared to {other.name}: "
                        f"{compute_might(tribe)} vs. their estimated {compute_might(other)}."
                    )
            elif distance <= config.RIVAL_DISTANT_SIGHTING_RADIUS:
                visible_entities.append(f"distant signs of {other.name} somewhere to the {_compass_direction(dx, dy)}")

        # A broadcast is only overheard within BROADCAST_HEARING_RADIUS -- previously
        # audible map-wide regardless of distance, which gave away free information and
        # removed any incentive to actually travel toward another tribe.
        for other in self.tribes.values():
            if other.id == tribe.id or other.extinct or not other.last_broadcast:
                continue
            distance = ((other.x - tribe.x) ** 2 + (other.y - tribe.y) ** 2) ** 0.5
            if distance <= config.BROADCAST_HEARING_RADIUS:
                visible_entities.append(
                    f"overheard: {other.name} broadcasted '{other.last_broadcast}' while performing {other.last_action}"
                )

        # Explicit follow-up from the Agentic Evolution spec reconciliation (Age 4's
        # Declare_Geopolitical_Posture): a declared stance is known policy, not
        # something that needs proximity to remember -- surfaced regardless of
        # distance, unlike the broadcast/sighting facts above.
        for other_id, stance in tribe.stance_toward.items():
            other = self.tribes.get(other_id)
            if other is not None and not other.extinct:
                visible_entities.append(f"Currently {stance.lower()} with {other.name}.")
        # The spectator UI's own "Path to the Next Era" panel already computes exactly
        # this (Tribe.to_dict's next_era block) -- it just never made it back into the
        # tribe's own reasoning. Naming the *specific* still-short resource(s) is the
        # fact; which one (if any) to prioritize is still the tribe's own call.
        #
        # Explicit hypothesis (2026-08-31): this used to be one more line buried in
        # visible_entities, the same flattened list as every generic terrain/landmark
        # fact -- real data showed two tribes settle, survive, and sit at population
        # 6-11 and near-zero wood/stone for 70+ cycles despite this fact stating
        # exactly what they were short on the whole time. That's the same salience
        # problem the survival-crisis fact had before it was moved to be the last
        # thing before the JSON slot (see get_prime_consciousness_prompt's docstring)
        # -- not duplicated here, *moved*, so the prompt doesn't get wordier for it.
        # Not returned via visible_entities at all now; compile_live_state_prompt
        # renders it as its own section, same tier as the survival instinct layer.
        era_gap_note = ""
        nxt = next_era(tribe.era)
        if nxt is not None:
            gaps = []
            if tribe.population < nxt.requires_population:
                gaps.append(f"population {tribe.population}/{nxt.requires_population}")
            for resource, minimum in nxt.requires_resources.items():
                have = _era_resource_amount(tribe, resource)
                if have < minimum:
                    gaps.append(f"{resource} {have}/{minimum}")
            if gaps:
                era_gap_note = f"To reach {nxt.label}, still short on: {', '.join(gaps)}."

        if not visible_entities:
            visible_entities = ["none"]
        return visible_entities, era_gap_note

    def _near_confirmed_water(self, tribe: Tribe, x: int | None = None, y: int | None = None) -> bool:
        """Explicit request: "make this an initial territory with a bounding area
        around it that is larger than the Discovery." A single confirmed water
        tile was too fragile a RELOCATE target -- landing one tile off onto
        non-qualifying ground meant never actually settling despite being right
        next to real water. Used by both settlement checks below as an
        additional way to qualify, on top of the exact-biome-match check, not a
        replacement for it. x/y default to the tribe's current position, but can
        be passed explicitly to check a hypothetical/past position instead (see
        _settlement_ground_ok)."""
        px, py = (tribe.x, tribe.y) if x is None else (x, y)
        return any(
            max(abs(px - wx), abs(py - wy)) <= config.SETTLEMENT_WATER_TERRITORY_RADIUS
            for wx, wy in tribe.confirmed_water_sites
        )

    def _settlement_ground_ok(self, tribe: Tribe, x: int | None = None, y: int | None = None) -> bool:
        """The ground-qualification half of _is_camped (biome match or near confirmed
        water), without the stability-cycle clock. Factored out so _apply_turn's
        cycles_since_relocate reset (below) can tell a RELOCATE that hops between two
        tiles that both already qualify -- e.g. jittering among several confirmed water
        sites a tile or two apart -- from one that actually leaves settled-worthy
        ground, instead of restarting the settlement clock on every such hop."""
        px, py = (tribe.x, tribe.y) if x is None else (x, y)
        return self.world.biome(px, py) in config.FARMABLE_BIOMES or self._near_confirmed_water(tribe, px, py)

    def _is_camped(self, tribe: Tribe) -> bool:
        """Whether this tribe has held still on decent (farmable or near-water)
        ground long enough to work the land here -- see config.
        SETTLEMENT_STABILITY_CYCLES/FARMABLE_BIOMES. GATHER_WOOD/GATHER_STONE are
        gated on this: a nomadic band stockpiling timber and quarried stone before
        it's even made camp never made sense, but it took no real fact to notice
        that until now.

        Renamed from _is_settled (explicit request, 2026-09-04): this only ever
        tested camp viability, not the real, permanent-commitment meaning
        "settled" now carries elsewhere (tribe.has_ever_settled/_found_territory,
        gated on real arrival -- see _is_settled_near_water below, and
        Simulation._prepare_turn's still_journeying check). A tribe can camp,
        gather, and even farm at a good spot long before ever committing to
        found its city there."""
        if tribe.cycles_since_relocate < config.SETTLEMENT_STABILITY_CYCLES:
            return False
        return self._settlement_ground_ok(tribe)

    def _is_settled_near_water(self, tribe: Tribe) -> bool:
        """Stricter than _is_camped: PLANT_CROP/GATHER_EGGS need a tribe that actually
        resettled somewhere with real, easily accessible water -- "plains" alone (which
        counts for the general _is_camped/GATHER_WOOD gate) doesn't mean that, per the
        original design spec for farming. Keeps the "settled" name -- unlike
        _is_camped, this is the real "putting down roots near water" signal
        (drives the "Settling" celebration/naming and the RELOCATE lockout).
        Note has_ever_settled/_found_territory deliberately still key off the
        looser _is_camped (plus not still_journeying, see _prepare_turn) rather
        than this stricter check -- requiring real water there would permanently
        strand a tribe that genuinely never finds any."""
        if tribe.cycles_since_relocate < config.SETTLEMENT_STABILITY_CYCLES:
            return False
        return (
            self.world.biome(tribe.x, tribe.y) in config.FARMING_REQUIRES_ADJACENT_WATER
            or self._near_confirmed_water(tribe)
        )

    def _site_in_own_territory(self, tribe: Tribe, x: int, y: int) -> bool:
        """True if (x, y) falls inside this tribe's own territory_radius -- used by
        _advance_in_territory_site_yields to know which discovered sites qualify
        for the passive freebie, and to stop naming an already-passive site as
        somewhere still worth traveling to gather."""
        if tribe.territory_center is None:
            return False
        cx, cy = tribe.territory_center
        return ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 <= tribe.territory_radius

    def _resolve_toll(self, tribe: Tribe, cx: int, cy: int, nx: int, ny: int) -> tuple[int, int]:
        """Explicit request: "trails that have been traversed more than 5 times
        by anyone will automatically evolve into visible and owned roads that
        others may travel for a fee... The first trailblazer gets the
        ownership and tolls (automatically collected when used or crossed).
        can't pay, can't cross." Called by every movement call site (RELOCATE,
        SCOUT/HUNTING_PARTY/EXPLORATION_PARTY outbound and return) right after
        physics.terrain_aware_step computes a candidate step, before it's
        actually committed. `cx, cy` is wherever the mover (the tribe itself
        for RELOCATE, or an expedition's own field position for everything
        else -- never assumed to be tribe.x/y) actually stands right now, so a
        blocked party stalls in place rather than snapping back to the home
        camp. Returns (nx, ny) unchanged if no toll applies or it's paid;
        returns (cx, cy) if the toll can't be afforded -- blocked from
        crossing this cycle, same as physically hitting impassable terrain.
        No pathfinding-level rerouting around a blocked tile -- a still-
        blocked party just doesn't advance until it can pay or the model
        picks a different target."""
        if not self.world.is_toll_road(nx, ny):
            return nx, ny
        owner_id = self.world.road_owner(nx, ny)
        if owner_id is None or owner_id == tribe.id:
            return nx, ny
        owner = self.tribes.get(owner_id)
        if owner is None or owner.extinct:
            return nx, ny  # no one left to collect -- free passage
        if tribe.wood < config.TOLL_FEE_WOOD:
            return cx, cy  # can't pay -- blocked, stay put
        tribe.wood -= config.TOLL_FEE_WOOD
        # Code-quality pass: same uncapped-mutation bug class as the expedition-
        # homecoming fix above, just a much smaller per-crossing amount -- still
        # worth routing through the cap for a heavily-trafficked toll road.
        self._capped_add(owner, "wood", config.TOLL_FEE_WOOD)
        return nx, ny

    def _prepare_turn(self, tribe: Tribe) -> tuple[dict, dict]:
        """Builds this tribe's prompt with no network calls; returns (request, context)."""
        biome = self.world.biome(tribe.x, tribe.y)
        nearby = self.world.nearby_structures(tribe.x, tribe.y)
        ghost_bias = self.trauma.bias_string(tribe.x, tribe.y)
        survival_bias, survival_critical = survival_bias_string(
            tribe.food, tribe.water, tribe.population, tribe.fishing_learned, tribe.cooking_learned,
            water_secure=_is_water_secure(tribe), food_secure=_is_food_secure(tribe),
        )
        # NUDGE (2026-08-31, explicit request: "the warnings do not mention settling
        # as an alternative to low water"). A tribe already sitting on a chronic water
        # shortage may well already know exactly where real water is (a confirmed
        # site) without ever having relocated there -- the survival warning itself
        # used to only ever say "gather more here" or "scout for more," never that
        # settling at an already-known site would actually fix this for good.
        # Appended directly onto the same warning line the model already reads
        # closely, not left as a separate, easier-to-miss fact.
        #
        # Live bug: this used to gate on `not _is_settled_near_water(tribe)`, which
        # stays True for the tribe's entire SETTLEMENT_STABILITY_CYCLES wait *even
        # after it has physically arrived* at the confirmed site -- confirmed live,
        # a starving tribe already parked on its own water tile kept getting told
        # "relocate there to fix this" every cycle, and RELOCATE-ing in place instead
        # of switching to GATHER_FOOD is very likely why it stayed at 0 food for 6+
        # cycles and lost over half its population. Gating on `_near_confirmed_water`
        # (proximity only, no stability wait) means the suggestion stops the moment
        # arrival actually happens, not once settling officially finishes.
        if survival_bias and "water" in survival_bias.lower() and tribe.confirmed_water_sites and not self._near_confirmed_water(tribe):
            wx, wy = tribe.confirmed_water_sites[-1]
            survival_bias += f" Settling at the confirmed water source ({wx},{wy}) would fix this for good, not just this cycle."

        # NUDGE (2026-09-08, live-run finding): confirmed via board_history.db that
        # the exact same "starving" warning fired for 600+ consecutive cycles on a
        # tribe that grew from a handful of people to a population of 21,000+ --
        # the model's own reasoning kept choosing HUNT_DEER (a single deer) in
        # direct response, every single time, because that's the only remedy this
        # warning ever names. It never once mentioned BUILD_KITCHEN, the actual
        # permanent fix once a real food source already exists (_is_food_secure --
        # Kitchen plus a proven Fishery or harvest) -- the warning stayed frozen at
        # primitive-era advice regardless of how far the tribe had actually grown.
        # Same shape as the water-settling addendum just above: appended onto the
        # warning line itself, only once the real gate (a proven food source) is
        # already met and the one missing building would end this for good.
        # Gated on BUILD_KITCHEN's own real prerequisites (cooking_learned and a
        # long house standing -- see actions._build_kitchen), not just food_secure's
        # other half, so this never promises an action the tribe can't actually take
        # yet -- same "facts must be real" discipline every other nudge here follows.
        if (
            survival_bias and "food" in survival_bias.lower() and not tribe.kitchen_built
            and tribe.cooking_learned and tribe.long_houses_built > 0
            and (tribe.fishery_built or tribe.last_harvest_cycle > 0)
        ):
            survival_bias += " Building a Kitchen would make this food security permanent, not just this cycle."
        memories = tribe.memory.recall(f"{biome} at {tribe.x},{tribe.y}")
        # Renamed from `settled` (explicit request, 2026-09-04): _is_camped only
        # ever tested camp viability, not real settlement -- see _is_camped's own
        # docstring. `camping_near_water` below keeps its old name/meaning; it's
        # the real settlement signal.
        camped = self._is_camped(tribe)
        settled_near_water = self._is_settled_near_water(tribe)
        # Explicit correction: has_ever_settled used to require settled_near_water
        # specifically -- a tribe that settled on plains away from water would stay
        # permanently stuck in the pre-settlement action set even after reaching
        # Bronze Age, since PLANT_CROP/GATHER_EGGS/CATCH_FISH's own water-adjacency
        # requirement was already dropped ("this is a Settled gate," not a real-water
        # one) but the outer gate wrapping the whole unlocked_actions_through(era)
        # branch still checked the stricter condition. FARMING_REQUIRES_ADJACENT_WATER
        # is a strict subset of FARMABLE_BIOMES, so this is a pure expansion -- never
        # fires later than before, only possibly sooner.
        # Explicit bug report, confirmed live: both tribes in a real run showed
        # has_ever_settled=True (territory founded, walls/Town Hall placed)
        # while their own most recent history line was still "RELOCATE: Moving
        # towards confirmed water access" -- they founded the whole city on a
        # stopover, then kept walking toward the water they were actually
        # heading for. Root cause: cycles_since_relocate counts cycles spent on
        # settlement-qualifying ground (see its own field comment), not cycles
        # spent motionless -- a multi-turn march across farmable plains toward
        # a still-distant target accrues the same "stability" credit as
        # actually staying put, so _is_camped could turn True mid-journey.
        # tribe.last_target != tribe.x/y (the same signal journey_note, below,
        # already uses) means a chosen destination hasn't been reached yet --
        # settling waits for that arrival instead of firing out from under it.
        still_journeying = tribe.last_target is not None and tribe.last_target != [tribe.x, tribe.y]
        # Explicit bug report: Tribe 2 settled on dry plains at cycle ~10 (the
        # stability clock alone, camped on FARMABLE_BIOMES, doesn't require water)
        # with no water ever confirmed -- that permanently lifted every
        # pre-settlement restriction, including RELOCATE being hidden entirely
        # until confirmed_water_sites is non-empty (see the PRE_SETTLEMENT_ACTIONS
        # branch below, "RELOCATE should not show until they find water"). Once
        # settled, that protection is gone for good, so the tribe was free to
        # RELOCATE toward a confirmed stone vein instead -- and did, repeatedly,
        # spending scarce water it didn't have to chase a resource it already had
        # plenty of, while genuinely dying of thirst.
        #
        # Requiring real confirmed water here (not just "a scouting trip
        # concluded, water or not") closes that loop completely: RELOCATE stays
        # hidden and the tribe stays uncommitted until water is genuinely found.
        # This map always has a reachable river and lake, and SCOUT/GATHER_WATER/
        # GATHER_FOOD stay available indefinitely while searching (unlimited
        # attempts, not just one) -- so this doesn't reintroduce the "permanently
        # stranded tribe" risk the looser _is_camped gate was originally written
        # to avoid.
        #
        # Second bug, found the very next run: `or tribe.confirmed_water_sites`
        # here checked whether a site had EVER been confirmed anywhere on the
        # map, not whether the tribe was actually there -- the instant a distant
        # scout came home with news, has_ever_settled fired wherever the tribe
        # happened to be standing that same cycle (often mid-GATHER, nowhere
        # near the real site), founding territory there. Every later RELOCATE
        # toward the real water then got clamped to that wrong center's
        # territory_radius, unable to ever actually arrive -- "got stuck at its
        # own territory boundary... did not move in a direct line to the water
        # found." settled_near_water alone already means "within
        # SETTLEMENT_WATER_TERRITORY_RADIUS of a confirmed site right now" (see
        # _near_confirmed_water) -- the correct, proximity-based check; the
        # existence-based OR added nothing but this bug.
        if camped and not still_journeying and not tribe.has_ever_settled and settled_near_water:
            tribe.has_ever_settled = True
            tribe.settled_at_cycle = self.cycle
            self._found_territory(tribe)

        if not tribe.has_ever_settled:
            # Explicit request: narrow the choice set before a tribe has ever proven
            # it can settle properly -- see config.PRE_SETTLEMENT_ACTIONS. A one-way
            # unlock (has_ever_settled never clears again) once it does.
            available_actions = sorted(set(unlocked_actions_through(tribe.era)) & set(config.PRE_SETTLEMENT_ACTIONS))
            if not tribe.confirmed_water_sites:
                # Explicit request: "RELOCATE should not show until they find water
                # and the place to settle" -- relocating without a known destination
                # wasn't meaningfully different from wandering at random. Forces
                # SCOUT first: RELOCATE only becomes a real, informed choice once a
                # scout has actually confirmed somewhere worth moving toward.
                available_actions = [a for a in available_actions if a != "RELOCATE"]
            elif still_journeying:
                # Explicit request: "if then start RELOCATING to a place to settle
                # with Water, then can not issue new commands until they Settle."
                # Pre-settlement RELOCATE's target is already mechanically forced to
                # the nearest confirmed water site (see the RELOCATE branch below) --
                # once that march is actually under way (still_journeying), it's a
                # real, informed choice with nothing left to reconsider until arrival;
                # letting the model pick GATHER_WATER/SCOUT/etc. instead mid-march
                # just delays reaching water it already knows is there.
                available_actions = ["RELOCATE"]
        else:
            available_actions = sorted(unlocked_actions_through(tribe.era))
        if not camped:
            available_actions = [a for a in available_actions if a not in ("GATHER_WOOD", "GATHER_STONE")]
        # Regression: RELOCATE used to lock out on the same looser `settled` check
        # GATHER_WOOD uses (any farmable ground, long enough) -- but farming/eggs need
        # the *stricter* settled_near_water condition, and a tribe that settles on
        # merely-farmable, non-water ground (plains) would hit the general threshold
        # first, lose RELOCATE forever, and be permanently unable to ever reach real
        # water and actually farm. RELOCATE only locks in once a tribe has settled
        # somewhere that's actually good enough for that -- next to real water.
        #
        # Second regression: settled_near_water alone goes True up to
        # SETTLEMENT_WATER_TERRITORY_RADIUS tiles before actual arrival -- a tribe
        # still mid-march (still_journeying, not yet tribe.has_ever_settled) could
        # already be within that radius, which stripped RELOCATE here the same
        # cycle the pre-settlement branch above was forcing RELOCATE as the ONLY
        # choice (see "can not issue new commands until they Settle") -- an empty
        # available_actions, crashing _resolve_action's own fallback. This lockout
        # is only meant for a tribe that has actually settled and shouldn't
        # casually uproot -- gate it on that, not just current proximity.
        if settled_near_water and tribe.has_ever_settled:
            # A tribe that has genuinely put down roots next to real water --
            # invested in long enough to be gathering wood and stone and farming here
            # -- shouldn't be one bad turn away from uprooting the whole settlement on
            # a whim. A real constraint on the choice set, not a scripted override of
            # whatever the tribe would otherwise decide.
            available_actions = [a for a in available_actions if a != "RELOCATE"]

            # Live-run finding: the fact-only nudge ("manually gathering more here is
            # no longer necessary," below) didn't stop models from still picking
            # GATHER_WATER after settling near real water -- the same reflexive-default
            # failure mode GATHER_FOOD showed before it got the same one-way
            # retirement treatment. Explicit follow-up request confirmed doing the
            # same thing here. One-way, like foraging_retired -- see Tribe.__init__.
            if not tribe.watering_retired:
                tribe.watering_retired = True
                tribe.history.append(
                    f"\U0001f4dc {tribe.name} no longer needs to manually gather water -- GATHER_WATER "
                    "is retired now that settling here keeps it flowing in on its own"
                )
            available_actions = [a for a in available_actions if a != "GATHER_WATER"]

        # Explicit correction: PLANT_CROP/GATHER_EGGS/CATCH_FISH used to require the
        # stricter settled_near_water check (a real adjacent water tile) -- "the
        # requirement of 'real' water is bogus, this is a Settled gate," same general
        # settling condition as GATHER_WOOD/STONE, not a stricter one layered on top.
        if not camped:
            available_actions = [
                a for a in available_actions if a not in ("PLANT_CROP", "GATHER_EGGS", "CATCH_FISH", "BUILD_DOCK")
            ]

        # Explicit correction ("they shouldn't build a dock until they learn to
        # fish"): BUILD_DOCK used to be reachable the moment a tribe settled, a bet
        # that building it would nudge the tribe toward fishing -- live data showed
        # wood spent on it (among other buildings) while genuinely starving with
        # fishing still unlearned. CATCH_FISH itself never required a dock, so
        # gating the dock on fishing_learned instead (matching how Sawmill/Quarry/
        # Tannery already gate on it) doesn't create a deadlock -- it just reorders
        # which comes first.
        if not tribe.fishing_learned:
            available_actions = [a for a in available_actions if a != "BUILD_DOCK"]

        # Explicit request: "they do not have to build_fire after they have it
        # once. it should leave the action list after discovered and be known
        # ubiquitously." Fire used to stay in available_actions forever, so a
        # tribe that already knew how to make fire kept getting asked to
        # re-decide it (and re-spend wood on it) at every new settlement. Same
        # one-way "generalist narrows once proven" shape as COOK_FOOD just below:
        # fire itself is retired from the choice set the moment it's ever been
        # built, the real prerequisite (tribe.fire_ever_built) COOK_FOOD already
        # reads for its own gate.
        if tribe.fire_ever_built:
            available_actions = [a for a in available_actions if a != "BUILD_FIRE"]

        # Explicit request: "if you learn to hunt successfully and you learn to
        # build fire successfully, you should get the chance to learn cooking...
        # this can happen early." COOK_FOOD is gated on real, proven
        # prerequisites rather than era progression. Retires once learned, the
        # same one-way "generalist narrows/task is done" shape foraging_retired
        # and watering_retired use -- there's nothing left to decide once
        # cooking is known forever.
        #
        # Loosened (live bug: "never landed a clean hunt? that's very
        # intolerant" -- a real run went 346 cycles without a single
        # HUNTING_PARTY catch, permanently blocking cooking, and by extension
        # the Kitchen, on hunting luck alone). Matches BUILD_FIRE's own gate
        # just above exactly: hunt_ever_succeeded or foraged_ever_succeeded --
        # foraging succeeds almost immediately for any tribe, so a run of bad
        # hunting luck no longer locks cooking out entirely.
        if tribe.cooking_learned:
            available_actions = [a for a in available_actions if a != "COOK_FOOD"]
        elif not ((tribe.hunt_ever_succeeded or tribe.foraged_ever_succeeded) and tribe.fire_ever_built):
            available_actions = [a for a in available_actions if a != "COOK_FOOD"]

        # Explicit request: "they don't need to CATCH_FISH once they know how."
        # _advance_fish_supply's passive daily catch already covers it from the
        # moment fishing_learned is set -- same one-way "generalist narrows once
        # a proven passive replacement exists" shape BUILD_FIRE/COOK_FOOD use,
        # retired silently since the fishing_learned milestone is already its
        # own celebrated moment elsewhere.
        if tribe.fishing_learned:
            available_actions = [a for a in available_actions if a != "CATCH_FISH"]

        # Explicit request: GATHER_FOOD is too generic once a tribe has real
        # experience -- it kept acting as a catch-all "satisfy hunger" default even
        # after fishing was rebalanced to strictly outperform it (confirmed live:
        # CATCH_FISH chosen once across 6 tribe-runs despite that). A generalist
        # narrows into a specialist once it has a genuinely proven, passive
        # replacement -- fishing_learned or a farm that has actually completed a
        # harvest (not just been planted -- crop_growth isn't food yet). One-way,
        # like has_ever_settled, and archived as a real chronicle event rather than
        # silently vanishing from the list -- old capability becomes tribal history,
        # not an unexplained gap.
        if not tribe.foraging_retired and (tribe.fishing_learned or tribe.last_harvest_cycle > 0):
            tribe.foraging_retired = True
            reason = "fishing" if tribe.fishing_learned else "farming"
            tribe.history.append(
                f"\U0001f4dc {tribe.name} has grown beyond simple foraging -- GATHER_FOOD is retired "
                f"now that {reason} reliably sustains them"
            )
        if tribe.foraging_retired:
            available_actions = [a for a in available_actions if a != "GATHER_FOOD"]

        # Explicit request ("2 rings is enough. they will have to build outside
        # the walls once they hit that point"): the wall pipeline retires the
        # same one-way way GATHER_FOOD/GATHER_WATER do above, once config.
        # MAX_WALL_RINGS is reached and the outermost ring is fully reinforced
        # -- a real ceiling instead of an unbounded ratchet on ring count.
        # Named CONSTRUCT_WALL below (not EXPAND_TERRITORY) since 2026-09-08 --
        # see actions._construct_wall's own docstring for the merge.
        if (
            not tribe.walls_complete
            and len(tribe.wall_rings) >= config.MAX_WALL_RINGS
            and city_layout.ring_fully_reinforced(tribe.wall_rings[-1])
        ):
            tribe.walls_complete = True
            tribe.history.append(
                f"\U0001f4dc {tribe.name}'s walls are complete at {len(tribe.wall_rings)} rings -- "
                "CONSTRUCT_WALL is retired now that the settlement has reached its full defensive size; "
                "future building spreads out beyond the walls"
            )
        if tribe.walls_complete:
            available_actions = [a for a in available_actions if a != "CONSTRUCT_WALL"]

        # Explicit request ("i know you can see they kept try to build a dock when
        # they already had one"): every other one-time structure with a single
        # boolean "already built" flag gets the same fire/cooking/foraging
        # treatment above -- retired from the choice set the moment its flag is
        # set, instead of sitting in the menu as a permanent no-op a model can
        # keep reflexively re-choosing forever (a live run wasted 77 turns
        # re-"building" an already-standing Dock this exact way). Long House/
        # Warehouse/farm plots are deliberately excluded -- genuinely repeatable,
        # gated by their own real capacity check, not a single flag.
        available_actions = [
            a for a in available_actions
            if not (a in ONE_TIME_BUILD_FLAGS and getattr(tribe, ONE_TIME_BUILD_FLAGS[a]))
        ]

        # See AFFORDABILITY_CHECKS's own comment -- never dangle an action the
        # tribe cannot possibly afford right now; each action's own guard clause
        # would just silently no-op it anyway.
        available_actions = [
            a for a in available_actions
            if a not in AFFORDABILITY_CHECKS or AFFORDABILITY_CHECKS[a](tribe, self.world)
        ]

        # See config.ACTION_REPETITION_THROTTLE_THRESHOLD/COOLDOWN and
        # Simulation._track_action_repetition -- once an action has been thrown out
        # of the menu for fixating, it stays out until its cooldown cycle passes.
        tribe.throttled_actions = {a: until for a, until in tribe.throttled_actions.items() if until > self.cycle}
        available_actions = [a for a in available_actions if a not in tribe.throttled_actions]

        # See SURVIVAL_CRISIS_ACTIONS's comment -- the same threshold instincts.py's
        # survival_bias_string already uses (kept in sync deliberately, not
        # duplicated as a new number), but enforced as a menu cut instead of only a
        # fact. Hysteresis via tribe.food_crisis_active/water_crisis_active: entering
        # needs the critical line, clearing needs the warning line.
        #
        # Live report ("'hunger warning suppressed' ... they shouldn't fire at all
        # under these conditions"): _is_food_secure/_is_water_secure short-circuit
        # this explicitly now, the same way survival_bias_string's own text already
        # does -- this used to compute crisis state from raw numbers alone, correct
        # only by the incidental fact that a secure tribe's stock happens to already
        # be huge by the time this runs (Simulation._advance_food_supply/
        # _advance_water_supply top up earlier in the same step()), not by an actual
        # guarantee. A real, explicit guarantee now, matching the text.
        upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
        if _is_food_secure(tribe):
            tribe.food_crisis_active = False
        elif tribe.food <= upkeep * config.HUNGER_CRITICAL_CYCLES_LEFT:
            tribe.food_crisis_active = True
        elif tribe.food > upkeep * config.HUNGER_WARNING_CYCLES_LEFT:
            tribe.food_crisis_active = False
        if _is_water_secure(tribe):
            tribe.water_crisis_active = False
        elif tribe.water <= upkeep * config.THIRST_CRITICAL_CYCLES_LEFT:
            tribe.water_crisis_active = True
        elif tribe.water > upkeep * config.THIRST_WARNING_CYCLES_LEFT:
            tribe.water_crisis_active = False

        # See WALL_LOCK_ACTIONS's own comment. Applied before the survival-crisis
        # cut below so a genuine food/water crisis can still narrow further on top
        # of this -- a wall commitment never overrides real starvation.
        if tribe.wall_commitment_active:
            allowed = WALL_LOCK_ACTIONS | ({"BUILD_LONG_HOUSE"} if tribe.wall_lock_long_house_credits > 0 else set())
            wall_locked_only = [a for a in available_actions if a in allowed]
            if wall_locked_only:
                available_actions = wall_locked_only
            else:
                # Fail-open guard, same shape as survival_crisis just below -- an
                # unusual gating combination shouldn't be able to soft-lock a tribe
                # with zero real choices.
                tribe.wall_commitment_active = False

        survival_crisis = tribe.food_crisis_active or tribe.water_crisis_active
        # Explicit request: "suspend crisis for 10 cycles beyond Territory lock."
        # The march to reach confirmed water is itself expensive (RELOCATE's own
        # food/water cost, several cycles running) -- a tribe often arrives and
        # founds its city already right at the crisis threshold, immediately
        # cutting its menu down to SURVIVAL_CRISIS_ACTIONS the instant it's
        # finally able to build/farm/settle in. food_crisis_active/
        # water_crisis_active themselves stay real (still shown as facts
        # elsewhere) -- only the menu-narrowing below is suspended, and only for
        # this one grace window right after founding.
        if survival_crisis and tribe.settled_at_cycle is not None:
            cycles_since_settling = self.cycle - tribe.settled_at_cycle
            if cycles_since_settling < config.SETTLEMENT_CRISIS_GRACE_CYCLES:
                survival_crisis = False
        if survival_crisis:
            # Fail-open guard: never cut the menu down to nothing (e.g. an
            # unusual pre-settlement gating combination) -- a soft-lock is worse
            # than an occasional bad choice getting through.
            survival_only = [a for a in available_actions if a in SURVIVAL_CRISIS_ACTIONS]
            if survival_only:
                available_actions = survival_only
            else:
                survival_crisis = False

        visible_entities, era_gap_note = self._build_visible_entities(tribe, biome, nearby, memories, available_actions)
        if tribe.wall_commitment_active:
            credit_note = (
                " A Long House can still be built early, spending a banked credit."
                if tribe.wall_lock_long_house_credits > 0 else ""
            )
            visible_entities.append(
                "The wall section already under construction has to be finished before anything else -- "
                f"only gathering what it needs and continuing CONSTRUCT_WALL are being offered.{credit_note}"
            )
        if survival_crisis:
            # Same "don't keep them in the dark" reasoning as the repetition
            # throttle's own fact just below -- instincts.py's survival_bias
            # already explains WHY, this explains why the menu itself just got
            # shorter, so a suddenly narrower list doesn't read as an unexplained
            # gap the way BUILD_DOCK re-appearing forever once did.
            visible_entities.append(
                "The crisis is severe enough that only actions which could directly help right now "
                "are being offered -- building, expansion, trade, and family plans can wait until the "
                "tribe is safely fed and watered again."
            )
        if tribe.throttled_actions:
            # See "should we always keep them in the dark like this?" -- unlike
            # tribe.history (spectator/chronicle-only, never reaches the model's own
            # prompt), this must land in visible_entities to actually explain the
            # throttle to the tribe itself, in the same in-fiction "Historian" voice
            # the user proposed.
            names = ", ".join(sorted(tribe.throttled_actions))
            plural = "s" if len(tribe.throttled_actions) > 1 else ""
            visible_entities.append(
                f"The Historian has counseled against repeating {names} for now, after it was chosen too "
                f"many cycles in a row -- that action{plural} will return to consideration again soon. "
                "Choose something genuinely different in the meantime."
            )
        # NUDGE (2026-08-31, explicit request: an "Instant Enlightenment" for a chief
        # whose last answer didn't match any real action -- see _resolve_action/
        # _apply_turn). Names exactly what was said, exactly what a valid answer looks
        # like, and a best-effort guess at what was probably meant, so a genuine
        # parse miss becomes a one-time teachable fact instead of a silent, invisible
        # no-op the tribe never gets a chance to correct. Shown once, then cleared --
        # this is a fact about what just happened, not a standing rule.
        if tribe.last_confusion:
            raw = tribe.last_confusion["raw"]
            guess = tribe.last_confusion["guess"]
            fallback = tribe.last_confusion["fallback"]
            guess_clause = f" You most likely meant {guess} -- consider it strongly now." if guess else ""
            visible_entities.append(
                f"Last cycle's answer ('{raw}') did not match any valid action, so {fallback} was taken "
                f"instead. Your visual_action must be copied exactly from the list below, nothing "
                f"else.{guess_clause}"
            )
            tribe.last_confusion = None
        if not tribe.has_ever_settled:
            visible_entities.append(
                "The tribe has not yet settled anywhere for good, so only survival and exploration "
                "actions are available right now. Settling properly, next to real water, will open up "
                "building, hunting parties, trade, raiding, and starting families."
            )
            if not tribe.confirmed_water_sites:
                visible_entities.append(
                    "RELOCATE is not available yet -- no real water source has been confirmed to move "
                    "toward. Sending scouts out is how a real destination gets found."
                )
        if not camped:
            # Bug report: "look at the Mountain Tribe and tell me why they
            # aren't Settled." This used to always say "on farmable ground"
            # regardless of whether the tribe's current tile actually
            # qualifies -- true once already there, but easy to misread as
            # still needing to relocate somewhere else. Now states plainly
            # when the ground already qualifies and the only thing left is
            # time, versus the tribe genuinely standing somewhere that
            # doesn't count at all.
            already_good_ground = self._settlement_ground_ok(tribe)
            if already_good_ground:
                visible_entities.append(
                    f"This ground already qualifies for settling -- {tribe.cycles_since_relocate}/"
                    f"{config.SETTLEMENT_STABILITY_CYCLES} cycles without relocating so far. Staying here "
                    "without choosing RELOCATE again will finish settling; relocating somewhere that no "
                    "longer qualifies resets this progress back to 0."
                )
            else:
                visible_entities.append(
                    "Wood and stone are not yet being gathered here -- this ground doesn't qualify for "
                    "settling at all, and the tribe would need to relocate somewhere farmable "
                    f"({tribe.cycles_since_relocate}/{config.SETTLEMENT_STABILITY_CYCLES} cycles without "
                    "relocating so far, but that alone won't be enough here)."
                )
        if settled_near_water:
            visible_entities.append(
                "The tribe has settled here, next to real water, and is no longer considering relocating. "
                "Water now flows in on its own each cycle -- manually gathering more here is no longer necessary."
            )
        elif camped:
            # A real, non-final state: good enough ground to gather wood/stone on, but
            # not good enough for farming/eggs -- RELOCATE stays on the table
            # specifically so the tribe can still choose to move on toward real water.
            visible_entities.append(
                "This ground supports gathering wood and stone, but has no real water access for "
                "farming or a flock -- relocating toward confirmed water is still an option."
            )

        # NUDGE (2026-09-05, live bug: a tribe with cooking learned still starved two
        # members). Foraging the same settled tile over and over wears that ground
        # down (see _harvest's scarcity/depletion), so a tribe leaning on GATHER_FOOD
        # cycle after cycle sees its own yield keep shrinking even with cooking's
        # multiplier applied on top. PLANT_CROP's yield never touches that scarcity
        # mechanic at all -- named directly, not left implicit, since a bare "you
        # could farm" fact already proved too easy to never act on (94 cycles,
        # zero attempts, in the run that surfaced this). Gated on real food pressure
        # (so a comfortably-fed tribe with zero plots isn't nagged for no reason) and
        # on PLANT_CROP actually being in available_actions right now -- same "never
        # dangle" reasoning as AFFORDABILITY_CHECKS: no point naming a fix the tribe
        # can't actually reach this cycle (no wood, no free plot, no territory yet).
        food_pressure = tribe.food_crisis_active or tribe.food <= upkeep * config.HUNGER_WARNING_CYCLES_LEFT
        if "PLANT_CROP" in available_actions and tribe.farm_plots == 0 and food_pressure:
            visible_entities.append(
                "Food gathered from this same ground keeps yielding less the more it's foraged -- "
                "planting a farm plot (PLANT_CROP) grows food here without that same wear-down, and "
                "pays out automatically, again and again, once it matures. Up to a few plots can be "
                "tended at once."
            )

        # NUDGE (2026-09-06, explicit report: "Kitchen and Cooking are not coming
        # 'easy'... we have not made these obvious or appealing offers"). Confirmed
        # across multiple real long runs: BUILD_KITCHEN was never chosen even once,
        # despite being reachable, in a 756-cycle run where cooking itself was
        # learned and a Kitchen would have stacked to 9x food from every future
        # forage/hunt/catch. Named directly, the same "don't leave a real payoff
        # implicit" treatment PLANT_CROP's own nudge above already proved out.
        if "BUILD_KITCHEN" in available_actions:
            visible_entities.append(
                "A Kitchen would stack with cooking for nine times as much food from every future "
                "forage, hunt, or catch, instead of only three -- affordable right now."
            )

        if not settled_near_water and tribe.confirmed_water_sites:
            # NUDGE (2026-08-30, explicit request: "the Water Bringer must lead the
            # whole tribe to the settlement location"). Scouts confirming water used
            # to only ever surface as a coordinate in the landmark list -- live runs
            # showed a confirmed water site sitting unused for hundreds of cycles
            # while the tribe kept scouting or gathering elsewhere instead of actually
            # relocating there. Names the real target directly rather than leaving
            # the connection to "confirmed water source at (x,y)" implicit; RELOCATE
            # is still the tribe's own choice to make, this just states plainly what
            # settling there would unlock.
            #
            # Strengthened (bug report: "they still search for water even after
            # they found it") -- naming the benefit alone wasn't stopping the
            # tribe from sending yet another scout after water it already knows
            # about. Now states outright that searching for water specifically is
            # done; SCOUT still has real, separate uses (lumber/wildlife/quarry/
            # mine/raiders), so this doesn't retire it, just closes off the one
            # already-answered reason to use it.
            wx, wy = tribe.confirmed_water_sites[-1]
            if self._near_confirmed_water(tribe):
                # Bug report: "it looks like they want to consider relocating
                # when they are on top of the water discovery site." The old
                # fact always said "RELOCATE there" regardless of whether the
                # tribe's current position already qualified -- if
                # settled_near_water was still False for some other reason
                # (not enough cycles yet, or standing within the territory
                # radius but not on an exact river/lake tile), the tribe kept
                # getting told to travel to a place it was already standing.
                visible_entities.append(
                    f"The tribe is already at or near the confirmed water site ({wx},{wy}) -- "
                    "relocating again would accomplish nothing; simply remaining here without "
                    "choosing RELOCATE again is what finishes settling."
                )
            else:
                visible_entities.append(
                    f"Water has already been found at ({wx},{wy}) -- no further scouting is needed to "
                    "search for it. RELOCATE there to finally settle and begin farming and raising a flock."
                )

        if "HUNTING_PARTY" in available_actions and tribe.wildlife_sites:
            # NUDGE (2026-08-31, explicit request: "scouts have to evolve so they can
            # inform the hunters and gatherers"). A confirmed wildlife-rich area used
            # to only ever surface as a bare coordinate in the landmark list, same gap
            # water had before the relocate nudge above -- the scouting work already
            # happened, but nothing connected it to the hunting action that could
            # actually use it. HUNTING_PARTY only appears in available_actions once
            # settled (see PRE_SETTLEMENT_ACTIONS), so this can't suggest an action a
            # still-nomadic tribe couldn't take anyway.
            site = tribe.wildlife_sites[-1]
            gx, gy, site_type = site["x"], site["y"], site["type"]
            if site_type == "Wolf Den":
                # Explicit request: naming the real site type is what makes a
                # Wolf Den mean something different from a Deer Stand -- this
                # is the same wolf-pack hazard HUNT_DEER/HUNTING_PARTY already
                # carry in forest (config.HUNT_HAZARD_CHANCE), just named as a
                # known location instead of a blind, biome-wide risk.
                visible_entities.append(
                    f"A wolf den was confirmed at ({gx},{gy}) -- a real hunting risk known to be "
                    "there, not just a blind chance."
                )
            else:
                visible_entities.append(
                    f"A {site_type.lower()} was confirmed at ({gx},{gy}) -- a hunting party sent "
                    "there would likely fare better than hunting blind."
                )

        if "COOK_FOOD" in available_actions:
            # Explicit request: cooking's eligibility (a proven hunt + a proven
            # fire, both real facts on the tribe) is otherwise silent -- same
            # "nudge harder once the gate is actually met" category as the
            # farm-plot/flock/fishing eligibility nudges just below.
            #
            # Live report ("Cooking is a good one to see"): this used to say
            # "much further" while the BUILD_KITCHEN nudge right below it names
            # its own real multiplier outright ("nine times... instead of only
            # three") -- same inconsistency as BUILD_SAWMILL's stale action
            # description this session already found elsewhere. Matches
            # COOKING_FOOD_MULTIPLIER now, the same real number COOK_FOOD's own
            # action description already states.
            visible_entities.append(
                "The tribe has both hunted successfully and built a fire before -- learning to cook "
                "would make every future forage, hunt, or catch worth three times as much food from "
                "then on."
            )

        # Real wall ring, not one progress-bar tile (2026-09-02 redesign) -- ring 0
        # is what BUILD_LONG_HOUSE/Moat/Torches gate on; further rings are purely
        # additional defense-in-depth, not a further gate on anything else.
        ring0 = tribe.wall_rings[0] if tribe.wall_rings else None
        ring0_reinforced = bool(ring0) and city_layout.ring_fully_reinforced(ring0)

        # Checks era-unlock, not "CONSTRUCT_WALL in available_actions" -- the
        # membership test still goes False whenever nothing is currently
        # affordable (see _can_afford_construct_wall), which is exactly the
        # state these nudges most need to explain.
        if "CONSTRUCT_WALL" in unlocked_actions_through(tribe.era):
            # NUDGE (2026-09-01, explicit request: live logs showed the chief
            # repeatedly choosing BUILD_LONG_HOUSE against a wall that wasn't
            # finished yet, over and over, each attempt silently rejected inside
            # _build_long_house -- CONSTRUCT_WALL and BUILD_LONG_HOUSE both unlock at
            # the same era, so the chief had no way to know the wall wasn't done
            # without this being stated as a fact. Same category as the COOK_FOOD
            # eligibility nudge above: a real gate the tribe couldn't otherwise see.
            if ring0 is None:
                # Bug report: "wall building is not coming up for them" -- live
                # runs showed a tribe sitting at Tribal Synapse for many cycles
                # with a wall never even started, buried among a dozen other
                # newly-unlocked actions at the same era with nothing calling it
                # out specifically. Sawmill/Quarry/Kitchen/Keep/Moat all build on
                # a wall existing first, so this is the one foundational nudge
                # missing relative to every other eligibility nudge here.
                visible_entities.append(
                    "No wall has been started here yet -- CONSTRUCT_WALL is available now, and a long "
                    "house, sawmill, quarry, kitchen, and further defenses all build on the first wall "
                    "ring existing first."
                )
            elif not city_layout.ring_fully_built(ring0):
                # Explicit correction: "we have to separate the natural barriers
                # out so they are not in the count and the build need is exact.
                # the way it is makes it look like they already built a section
                # which they may be confusing with a completed Wall." A natural
                # barrier needs nothing from the tribe -- folding it into the same
                # count as real, built sections understated exactly how much
                # CONSTRUCT_WALL still has left to do.
                real_sections = [s for s in ring0["sections"] if not s["natural_barrier"]]
                built = sum(1 for s in real_sections if s["progress"] >= 100)
                unlocked_count = sum(1 for s in real_sections if s["unlocked"])
                real_total = len(real_sections)
                natural_count = len(ring0["sections"]) - real_total
                natural_note = (
                    f" ({natural_count} more section{'s' if natural_count != 1 else ''} already stand for "
                    "free, thanks to natural terrain)" if natural_count else ""
                )
                # Explicit request: "keep the Long House blocked note hidden
                # until [the wall is done and] they build one" -- BUILD_LONG_HOUSE
                # is already absent from available_actions for this entire
                # stretch (_can_afford_build_long_house), so naming it here every
                # single cycle just nagged about an option the tribe can't even
                # see yet. The "now worth building" callout below already fires
                # at the one moment it's actually true.
                #
                # Live bug ("Walls didn't unlock for some reason and they wasted
                # cycles"): confirmed via board_history -- a tribe called
                # CONSTRUCT_WALL well over 100 times against a ring with zero
                # sections ever unlocked, and a separate EXPAND_TERRITORY action
                # that would have unlocked one never once got picked instead,
                # across two independent live traces -- the same "a fact doesn't
                # reliably redirect a small model to a DIFFERENT action" pattern
                # documented elsewhere in this file. 2026-09-08: rather than a
                # third nudge attempt, CONSTRUCT_WALL and EXPAND_TERRITORY were
                # merged into one action (see actions._construct_wall) -- the
                # tribe no longer needs to be told which action unlocks what,
                # since there's only one action left to pick. These lines are now
                # purely informational (how much of the ring is real progress).
                if unlocked_count < real_total and built >= unlocked_count:
                    if unlocked_count == 0:
                        visible_entities.append(
                            f"The settlement's first wall ring has no section unlocked yet -- CONSTRUCT_WALL "
                            f"unlocks the first one automatically{natural_note}."
                        )
                    else:
                        visible_entities.append(
                            f"The settlement's first wall ring has {built}/{real_total} real sections built, and "
                            f"every unlocked section is complete -- CONSTRUCT_WALL unlocks the next "
                            f"one automatically{natural_note}."
                        )
                else:
                    visible_entities.append(
                        f"The settlement's first wall ring has {built}/{real_total} real sections built{natural_note}."
                    )
            elif not ring0_reinforced:
                if tribe.long_houses_built == 0:
                    visible_entities.append(
                        "The first wall ring is complete -- a long house is now worth building for real, "
                        "lasting shelter."
                    )
                else:
                    visible_entities.append(
                        "The first wall ring stands complete -- sections can still be reinforced for a "
                        "stronger defense."
                    )

        if "BUILD_MOAT" in available_actions and not tribe.moat_built and ring0_reinforced:
            visible_entities.append(
                "The first wall ring has been fully reinforced -- a moat is now available, a cheaper "
                "alternative defense investment."
            )
        if tribe.fire_ever_built and ring0_reinforced:
            visible_entities.append(
                "Fire is known and the first wall ring stands fully reinforced -- torches now line it "
                "for free, a further defense bonus."
            )

        if tribe.long_houses_built > 0:
            if not tribe.keep_built and tribe.long_houses_built >= config.KEEP_LONG_HOUSES_REQUIRED:
                visible_entities.append(
                    f"{tribe.long_houses_built} long houses stand -- a keep is now worth building for a "
                    "further defense bonus."
                )
            elif tribe.keep_built and not tribe.fortress_built and tribe.long_houses_built >= config.FORTRESS_LONG_HOUSES_REQUIRED:
                visible_entities.append(
                    f"{tribe.long_houses_built} long houses stand and the keep is complete -- a fortress "
                    "is now worth building for a further defense bonus."
                )
            elif tribe.fortress_built and not tribe.castle_built and tribe.long_houses_built >= config.CASTLE_LONG_HOUSES_REQUIRED:
                visible_entities.append(
                    f"{tribe.long_houses_built} long houses stand and the fortress is complete -- a "
                    "castle is now worth building for a further defense bonus."
                )

        if "BUILD_SAWMILL" in available_actions and not tribe.sawmill_built:
            if tribe.wood_ever_gathered:
                if tribe.lumber_sites:
                    lx, ly = tribe.lumber_sites[-1]
                    visible_entities.append(
                        f"A stand of trees is known at ({lx},{ly}) -- and wood has been gathered here "
                        "before, so a sawmill built at the settlement would triple every future load "
                        "of gathered wood."
                    )
                else:
                    visible_entities.append(
                        "Wood has been gathered here before -- a sawmill built at the settlement would "
                        "triple every future load of gathered wood."
                    )
        if "BUILD_TANNERY" in available_actions and not tribe.tannery_built:
            if tribe.hunt_ever_succeeded:
                warren_sites = [s for s in tribe.wildlife_sites if s["type"] == "Rabbit Warren"]
                if warren_sites:
                    wx, wy = warren_sites[-1]["x"], warren_sites[-1]["y"]
                    visible_entities.append(
                        f"A rabbit warren is known at ({wx},{wy}) -- and a hunt has already succeeded, "
                        "so a tannery built at the settlement would bring in a steady supply of Fur."
                    )
                else:
                    visible_entities.append(
                        "A hunt has already succeeded -- a tannery built at the settlement would bring "
                        "in a steady supply of Fur, and extra meat from every future hunt."
                    )
        if "BUILD_KITCHEN" in available_actions and not tribe.kitchen_built:
            if tribe.cooking_learned and tribe.long_houses_built > 0:
                visible_entities.append(
                    "Cooking is known and real shelter stands -- a kitchen would turn cooked meals into "
                    "excellent food, stretching stores even further."
                )
        if "BUILD_QUARRY" in available_actions and not tribe.quarry_built:
            if tribe.stone_ever_gathered:
                if tribe.quarry_sites:
                    qx, qy = tribe.quarry_sites[-1]
                    visible_entities.append(
                        f"A stone-rich site is known at ({qx},{qy}) -- and stone has been gathered here "
                        "before, so a quarry built at the settlement would triple the value of every "
                        "future load of harvested stone."
                    )
                else:
                    visible_entities.append(
                        "Stone has been gathered here before -- a quarry built at the settlement would "
                        "triple the value of every future load of harvested stone."
                    )
        if "BUILD_MINE" in available_actions and not tribe.mine_built:
            if tribe.quarry_built and tribe.mine_sites:
                site = tribe.mine_sites[-1]
                visible_entities.append(
                    f"A vein of {site['resource']} is known at ({site['x']},{site['y']}) -- excavating a "
                    "mine would bring in a steady supply of it, a resource no other tribe's own land "
                    "necessarily shares."
                )
            elif tribe.quarry_built:
                visible_entities.append(
                    "Quarrying is mastered, but no vein of a unique resource has been found yet -- "
                    "scouting may turn one up."
                )
        if "BUILD_FORGE" in available_actions and not tribe.forge_built and tribe.mine_built:
            ore_in_stock = tribe.unique_resources.get(tribe.mine_resource_name, 0)
            if ore_in_stock >= config.FORGE_ITEM_ORE_COST:
                visible_entities.append(
                    f"The mine has produced {tribe.mine_resource_name} -- a forge would let it be worked "
                    "into real tools, weapons, and inventions instead of just sitting in storage."
                )
        if "FORGE_ITEM" in available_actions and tribe.forge_built and tribe.items:
            visible_entities.append(
                f"{len(tribe.items)} crafted item(s) are on hand -- each can be redeemed for its stored "
                "value (USE_ITEM) or handed over in a future trade."
            )

        # Military branch, step 1 (plan file valiant-forging-falcon.md) -- explicit
        # eligibility nudge, added 2026-09-08 after a live run showed the exact
        # "isolated gamble, nothing feeding into it" pattern the original
        # DECLARE_CONQUEST TODO already named, one step earlier than expected:
        # confirmed via board_history.db that an individual had 9 personally-
        # credited trophies (WARRIOR_TROPHY_THRESHOLD is 3) for many days straight
        # across a 38-day run, and NAME_WARRIOR was never chosen even once -- the
        # entire Military branch stayed permanently unreachable behind it. Same
        # "nudge harder once a real gate is met" shape COOK_FOOD/CONSTRUCT_WALL's
        # own nudges above already use.
        if "NAME_WARRIOR" in available_actions:
            candidate = _eligible_warrior_candidate(tribe)
            if candidate is not None:
                visible_entities.append(
                    f"{candidate} has earned {config.WARRIOR_TROPHY_THRESHOLD} or more personal trophies -- "
                    "NAME_WARRIOR would appoint them Warrior, letting them lead a Battalion once a Barracks "
                    "is built and soldiers are trained."
                )

        # Military branch, step 7 (plan file valiant-forging-falcon.md): "The Chief
        # has to actually be able to reach DECLARE_CONQUEST -- not just mechanically
        # possible, but visible... an eligibility nudge once it's genuinely a good
        # bet." Original root-cause theory (2026-09-07) for why this action had
        # never fired even once: "an isolated, all-or-nothing gamble with nothing
        # feeding into it." Now there's something real to feed into it -- only
        # fires once this tribe has an actual Battalion AND a known, nearby rival
        # whose own Might is meaningfully behind (config.
        # DECLARE_CONQUEST_NUDGE_MIGHT_RATIO), the same "nudge harder once a real
        # gate is met" shape the COOK_FOOD/CONSTRUCT_WALL nudges above already use.
        if "DECLARE_CONQUEST" in available_actions and tribe.battalion_size > 0:
            nearby_rivals = [
                other for other in self.tribes.values()
                if other.id != tribe.id and not other.extinct
                and math.hypot(other.x - tribe.x, other.y - tribe.y) <= config.RIVAL_PRECISE_AWARENESS_RADIUS
            ]
            if nearby_rivals:
                rival = min(nearby_rivals, key=lambda o: math.hypot(o.x - tribe.x, o.y - tribe.y))
                tribe_might, rival_might = compute_might(tribe), compute_might(rival)
                if tribe_might > rival_might * config.DECLARE_CONQUEST_NUDGE_MIGHT_RATIO:
                    visible_entities.append(
                        f"{rival.name}'s Battalion is meaningfully weaker (Might {rival_might} vs. this "
                        f"tribe's {tribe_might}) -- DECLARE_CONQUEST at ({rival.x},{rival.y}) is a real, "
                        "favorable bet now, not just a blind gamble."
                    )

        settlement_actions = ("PLANT_CROP", "GATHER_EGGS", "CATCH_FISH")
        if any(a in unlocked_actions_through(tribe.era) for a in settlement_actions):
            if not camped:
                visible_entities.append(
                    "Crops, eggs, and fishing all need the tribe to have settled here -- it hasn't put "
                    f"down roots yet ({tribe.cycles_since_relocate}/{config.SETTLEMENT_STABILITY_CYCLES} "
                    "cycles without relocating, on farmable ground)."
                )
            else:
                # NUDGE (2026-08-30/31, explicit "nudge harder" request): a plain,
                # concrete suggestion once the gate is actually met, not just silent
                # availability -- same category as the survival-critical nudge in
                # instincts.py. These are still ordinary entries in available_actions
                # the model chooses or ignores; this doesn't force any of them.
                if tribe.farm_plots == 0:
                    visible_entities.append(
                        "The tribe has settled here -- this ground could support a farm plot."
                    )
                if tribe.flock == 0:
                    visible_entities.append(
                        "No flock has been started yet -- wild fowl nest near settlements like this, so "
                        "gathering their eggs here could begin one."
                    )
                if not tribe.fishing_learned:
                    visible_entities.append(
                        "No one has fished here yet -- a single successful catch would make fishing a "
                        "permanent, daily source of food from then on."
                    )
                elif "HUNTING_PARTY" in available_actions:
                    # Explicit observation: "it should be an easy choice, fish
                    # locally, no travel time, or send a hunting party taking
                    # an indefinite amount of time depending on if they find a
                    # Stand of Deer to hunt... still travel vs. home." Both
                    # ACTION_DESCRIPTIONS already say this on their own
                    # (CATCH_FISH pays out immediately; HUNTING_PARTY food does
                    # nothing until the party walks all the way home), but nothing
                    # ever put the two side by side -- small models don't
                    # reliably synthesize a comparison across two separate
                    # glossary entries on their own.
                    visible_entities.append(
                        "Fishing here pays out food immediately with no travel time, once caught -- a "
                        "hunting party takes several days round trip and isn't guaranteed to find anything."
                    )

        # Explicit finding: "I wonder why one figured out fishing and the other
        # farming but only one figured out both." The eligibility nudges above
        # already suggest each food method unconditionally and independently -- this
        # isn't a missing fact, it's the same salience problem the era-progress fact
        # had before that got moved to the dedicated GROWTH IMPERATIVE LAYER. Once a
        # tribe has proven ONE food method works, the felt pressure to try the other
        # disappears even though the suggestion was there the whole time. Elevated
        # to the same top-tier slot as era_gap_note rather than duplicating the
        # already-existing eligibility nudge.
        diversification_note = ""
        has_fish = tribe.fishing_learned
        has_farm = tribe.farm_plots > 0 or tribe.last_harvest_cycle > 0
        if settled_near_water and has_fish and not has_farm:
            diversification_note = (
                "Fishing sustains the tribe daily, but no crop has ever been planted -- relying on a "
                "single food source is its own risk that planting a farm would reduce."
            )
        elif settled_near_water and has_farm and not has_fish:
            diversification_note = (
                "Farming has proven itself, but fishing has never been tried -- relying on a single "
                "food source is its own risk that fishing would reduce."
            )

        # See _warehouse_capacity_note's own docstring (top of this file) for the
        # live trace evidence behind it. Its sibling _wall_expansion_note (same
        # trace, the EXPAND_TERRITORY side) was retired 2026-09-08 once
        # CONSTRUCT_WALL/EXPAND_TERRITORY were merged into one action -- there's no
        # longer a second action to nudge the model toward picking.
        warehouse_note = _warehouse_capacity_note(tribe)

        if tribe.fishing_learned:
            visible_entities.append(
                "Fishing has been mastered here -- food now flows in on its own each cycle, on top of "
                "anything caught by hand."
            )

        if tribe.farm_plots > 0:
            visible_entities.append(
                f"{tribe.farm_plots} farm plot(s) growing ({tribe.crop_growth}/100 toward the next harvest)."
            )
        if tribe.flock > 0:
            last_note = tribe.flock_lineage[-1]["note"] if tribe.flock_lineage else ""
            flock_line = f"A flock of {tribe.flock} is being kept."
            if last_note:
                flock_line += f" Most recent hatchling: {last_note}"
            visible_entities.append(flock_line)

        # A tribe starving/dehydrating while sitting on 100+ wood or stone had a real
        # information gap: the stockpile itself never said "this is already more than
        # enough." Only surfaced alongside an actual food/water warning (survival_bias
        # non-empty) -- a fact about the mismatch, not a standing nudge to stop
        # gathering wood every other cycle too.
        if survival_bias:
            surplus = []
            if tribe.wood >= config.MATERIAL_SURPLUS_THRESHOLD:
                surplus.append(f"{tribe.wood} wood")
            if tribe.stone >= config.MATERIAL_SURPLUS_THRESHOLD:
                surplus.append(f"{tribe.stone} stone")
            if surplus:
                visible_entities.append(
                    f"Already stockpiled well beyond any near-term building need: {', '.join(surplus)}."
                )

        # NUDGE (2026-08-31, explicit request: "when it comes to inventory management
        # they always want to resupply the lowest item first, in order"). States the
        # tribe's own four core stockpiles ranked lowest-to-highest as a plain fact --
        # the raw numbers already appear in METABOLIC STOCKPILES above, but nothing
        # previously said which one is actually the most urgent to address. Still just
        # a ranking, not a command; the model picks whether and how to act on it.
        stockpile_order = sorted(
            (("wood", tribe.wood), ("stone", tribe.stone), ("food", tribe.food), ("water", tribe.water)),
            key=lambda pair: pair[1],
        )
        visible_entities.append(
            "Resource priority, lowest to highest: "
            + ", ".join(f"{name} ({amount})" for name, amount in stockpile_order)
            + " -- resupplying the lowest one first is usually the most efficient use of this turn."
        )

        # Stated as fact (what you previously chose), not as an instruction to continue --
        # whether to keep going or change course is left entirely to the model.
        journey_note = ""
        if tribe.last_target and tribe.last_target != [tribe.x, tribe.y]:
            journey_note = (
                f"Last cycle you set target_vector to ({tribe.last_target[0]}, {tribe.last_target[1]}); "
                "you have not yet arrived there."
            )
        if tribe.expeditions:
            # Bug report: "2 scouts going same direction still" -- right out of
            # the gate, before any real discovery data exists to compare
            # compass bearings against (see the lopsided-coverage fact
            # elsewhere, which only ever looks at confirmed sites). This named
            # who was out and what day/phase they were on, but never where
            # they were actually headed -- a second SCOUT call had no way to
            # tell it would just be covering the same ground again.
            party_word = {"scout": "scouts", "hunt": "a hunting party", "explore": "an exploration party"}
            slots_left = expedition_capacity(tribe) - len(tribe.expeditions)
            # Explicit request: "I am concerned about excess chatter... a lot
            # of players on the board." Naming every single party in one
            # unbounded run-on sentence got real (a live prompt with 7 parties
            # named individually, close to MAX_CONCURRENT_EXPEDITIONS_CEILING)
            # once removing the per-kind dispatch cap made reaching that
            # ceiling routine, not rare. Below config.
            # FIELD_REPORT_DETAIL_THRESHOLD, full per-party detail is cheap and
            # still worth showing whole; at or above it, a kind+count summary
            # instead -- except right at the capacity ceiling (slots_left <=
            # 0), where knowing exactly who's about to come home is a real
            # decision input (is waiting one more cycle worth it), so full
            # detail always shows there regardless of the count.
            if len(tribe.expeditions) < config.FIELD_REPORT_DETAIL_THRESHOLD or slots_left <= 0:
                reports = "; ".join(
                    f"{party_word.get(exp.get('kind'), 'a party')} led by {exp['lead_scout']} "
                    f"(day {exp['day']}, {exp['phase']}, headed toward "
                    f"({exp['target'][0]},{exp['target'][1]}))"
                    for exp in tribe.expeditions
                )
            else:
                kind_counts: dict[str, int] = {}
                for exp in tribe.expeditions:
                    kind_counts[exp.get("kind", "party")] = kind_counts.get(exp.get("kind", "party"), 0) + 1
                kind_word = {"scout": "scouting", "hunt": "hunting", "explore": "exploring"}
                kind_summary = ", ".join(
                    f"{count} {kind_word.get(kind, kind)}" for kind, count in kind_counts.items()
                )
                returning = sum(1 for exp in tribe.expeditions if exp["phase"] == "returning")
                returning_note = f" -- {returning} already heading home" if returning else ""
                reports = f"{len(tribe.expeditions)} parties in the field ({kind_summary}){returning_note}"
            capacity_note = (
                " No one left to send out until one returns."
                if slots_left <= 0
                else f" You could send out {slots_left} more at once."
            )
            journey_note += f" Still in the field: {reports}.{capacity_note}"

        world_state = {
            "cycle": self.cycle,
            "x": tribe.x,
            "y": tribe.y,
            "biome": biome,
            "biome_label": BIOME_LABELS.get(biome, biome),
            "population": tribe.population,
            "wood": tribe.wood,
            "stone": tribe.stone,
            "food": tribe.food,
            "water": tribe.water,
            "era": tribe.era,
            "available_actions": available_actions,
            "visible_entities": visible_entities,
            "journey_note": journey_note,
            # Combined into one growth-tier slot (see prompts.py's GROWTH IMPERATIVE
            # LAYER): all three are the same category of "not urgent, but real" pressure,
            # and all need the same salience fix era_gap_note already proved out -- a
            # fact buried in the generic list gets ignored even when it's true.
            "growth_note": " ".join(n for n in (era_gap_note, diversification_note, warehouse_note) if n),
        }
        # See wellbeing.compute_wellbeing -- a slower-moving, five-tier read on the
        # tribe's overall condition, distinct from the moment-to-moment survival_bias
        # above. Cached on the tribe (not just injected into this turn's prompt) so
        # the frontend can render the same numbers the chief itself is reasoning
        # from -- one source of truth, not a UI-only recomputation.
        tribe.wellbeing = compute_wellbeing(tribe, city_layout.wall_defense_fraction(tribe))
        lineage_note = ""
        if tribe.lineage:
            latest = tribe.lineage[-1]
            parents = latest.get("parents") or []
            parent_clause = f", child of {parents[0]} and {parents[1]}" if len(parents) == 2 else ""
            lineage_note = f"{latest['child_name']}{parent_clause}, born cycle {latest['cycle']}"
        base_prompt = get_prime_consciousness_prompt(
            tribe.name, tribe.model, tribe.chief_name, tribe.chief_philosophy, tribe.chief_decree,
            tribe.chief_victory, lineage_note, tuple(available_actions),
        )
        rival_tribes = [t for t in self.tribes.values() if t.id != tribe.id and not t.extinct]
        threat_assessment = threat_assessment_string(tribe, rival_tribes)
        prompt = compile_live_state_prompt(
            base_prompt, world_state, ghost_bias, survival_bias, tribe.wellbeing.get("summary", ""),
            threat_assessment,
        )
        panicked = "DREAD" in ghost_bias or survival_critical
        temperature = config.ANCESTRAL_DREAD_TEMPERATURE if panicked else config.DEFAULT_TEMPERATURE

        request = {"id": tribe.id, "model": tribe.model, "prompt": prompt, "temperature": temperature}
        return request, {"biome": biome, "available_actions": available_actions}

    def _track_action_repetition(self, tribe: Tribe, action: str) -> None:
        """Explicit request, after a live run showed one tribe choose GATHER_STONE on
        49% of all 728 turns (and a different run's tribe choose BREED on 63.8%) while
        other real needs went untouched -- see config.ACTION_REPETITION_THROTTLE_*.
        RELOCATE is exempt: a real, sustained multi-cycle journey is documented,
        desired behavior (see README), not fixation."""
        if action == tribe.action_streak_name:
            tribe.action_streak_count += 1
        else:
            tribe.action_streak_name = action
            tribe.action_streak_count = 1
        if action == "RELOCATE":
            return
        if tribe.action_streak_count >= config.ACTION_REPETITION_THROTTLE_THRESHOLD:
            tribe.throttled_actions[action] = self.cycle + config.ACTION_REPETITION_THROTTLE_COOLDOWN
            tribe.action_streak_count = 0
            tribe.history.append(
                f"{tribe.name} has repeated {action} too many times in a row -- "
                "the Historian insists on a different approach for a while"
            )

    def _apply_turn(self, tribe: Tribe, intent: dict, latency_ms: float, ctx: dict) -> None:
        raw_action = intent.get("visual_action", "(no action provided)")
        action, unresolved_raw = _resolve_action(raw_action, ctx["available_actions"])
        if unresolved_raw is not None:
            guess = _guess_intended_action(unresolved_raw, ctx["available_actions"])
            # Records the actual fallback taken (action) alongside raw/guess so next
            # cycle's correction fact can say what really happened, since IDLE's
            # removal means it's never accurate to say "nothing happened" anymore.
            tribe.last_confusion = {"raw": unresolved_raw[:80], "guess": guess, "fallback": action}
        else:
            tribe.last_confusion = None
        broadcast = intent.get("synthetic_language_broadcast") or ""
        target = intent.get("target_vector", [tribe.x, tribe.y])
        if not (isinstance(target, list) and len(target) == 2):
            target = [tribe.x, tribe.y]
        try:
            target = (int(target[0]), int(target[1]))
        except (TypeError, ValueError):
            target = (tribe.x, tribe.y)

        # last_decision_target records the model's raw submission (for decision_log.py's
        # offline analysis, before any correction below) -- last_target below may end up
        # holding a mechanically-corrected value instead.
        tribe.last_decision_target = [target[0], target[1]]

        # Explicit request: "the bounds-safe function is too loose at the edges of
        # our board." Everything from here down used `target` completely unclamped
        # -- a hallucinated out-of-range coordinate from a small model rode straight
        # through into tribe.last_target and every action handler. last_decision_
        # target (above) deliberately keeps the untouched raw value for offline
        # analysis; `target` itself now bounces back inward the same way SCOUT/
        # EXPLORATION_PARTY's own compass targets already do, instead of nothing at
        # all.
        target = (
            physics.reflect_into_grid(target[0], self.world.grid_size),
            physics.reflect_into_grid(target[1], self.world.grid_size),
        )

        # Only RELOCATE actually moves the tribe -- everything else happens wherever it
        # currently stands. last_target/journey_note (see _prepare_turn) specifically
        # track an in-progress relocation, not just whatever coordinate a GATHER_WOOD
        # turn happened to carry.
        if action == "RELOCATE":
            # Live bug, confirmed via last_decision_target on a real run: a small model
            # (llama3.2:1b) asked to RELOCATE toward a confirmed water site named
            # explicitly in its own prompt instead submitted its own current position as
            # target_vector, every single time, for 15+ consecutive cycles while
            # starving -- a guaranteed no-op that left it standing still until it died.
            # Same category of failure as SCOUT's unreliable target_vector (see
            # evolution2civ-facts-vs-mechanics-pattern.md) -- the fix there was to stop
            # trusting the model's coordinate and compute the real one mechanically.
            #
            # Explicit design (2026-09-04): "Closest one wins, keep it simple for
            # now, plus, unless they have other reports and data informing them,
            # they won't have any judgment otherwise." Before a tribe has ever
            # settled, RELOCATE's target_vector is now ignored entirely, the same
            # way SCOUT's already is -- mechanically picks whichever confirmed
            # water site is nearest to where the tribe stands right now, rather
            # than trusting the model to reason about which of several reported
            # sites is actually closer. A tribe already standing at the nearest
            # site computes that same site again (distance 0), so this is a
            # harmless no-op once arrived, same as the narrower check it replaces.
            if not tribe.has_ever_settled and tribe.confirmed_water_sites:
                target = min(
                    (tuple(site) for site in tribe.confirmed_water_sites),
                    key=lambda site: (site[0] - tribe.x) ** 2 + (site[1] - tribe.y) ** 2,
                )
            tribe.last_target = [target[0], target[1]]
        else:
            # Live bug report: a tribe camped for 229 cycles on good ground,
            # confirmed water in hand, never settled -- has_ever_settled's own
            # still_journeying check (below) stayed permanently True because
            # tribe.last_target was left over from a RELOCATE chosen many
            # cycles ago and never cleared once the model moved on to
            # GATHER_FOOD/other actions instead of continuing that march. A
            # journey that isn't being actively continued isn't "still"
            # anything -- clearing it here means only an actual, ongoing
            # RELOCATE (chosen last cycle, not just at some point in history)
            # can ever hold settlement back. Also keeps journey_note (see
            # _prepare_turn) from citing a target the tribe already abandoned.
            tribe.last_target = None
        pos_before = (tribe.x, tribe.y)

        hazard_note = self._apply_action(tribe, action, ctx["biome"], target)

        # Regression: this used to reset cycles_since_relocate to 0 purely because
        # RELOCATE was the *chosen action*, even when the tribe had already arrived
        # and target_vector pointed at its own current tile -- terrain_aware_step is a
        # genuine no-op there. A model that keeps re-issuing RELOCATE toward an
        # already-reached confirmed water site (the fact recommending it never stops
        # being true just because they arrived) could never accumulate any settlement
        # progress at all, standing right on the water forever. Only a real change in
        # position should restart the clock.
        #
        # Second regression, found live: a tribe with several confirmed water sites a
        # tile or two apart (common once a scout has walked the shoreline) kept
        # jittering RELOCATE between them -- (19,62) -> (20,61) -> (20,62) -- each hop
        # a genuine position change, so the clock kept restarting to 0 forever even
        # though every one of those tiles already satisfied _is_camped's ground
        # check. A small model can't be talked out of this with a fact (see
        # evolution2civ-facts-vs-mechanics-pattern.md) -- the fix is mechanical: only
        # reset the clock when the move actually leaves settlement-qualifying ground,
        # not just when the coordinates change.
        if action == "RELOCATE" and (tribe.x, tribe.y) != pos_before:
            if self._settlement_ground_ok(tribe, *pos_before) and self._settlement_ground_ok(tribe):
                tribe.cycles_since_relocate += 1
            else:
                tribe.cycles_since_relocate = 0
        else:
            tribe.cycles_since_relocate += 1
        tribe.last_broadcast = broadcast
        tribe.last_action = action
        self._track_action_repetition(tribe, action)
        self.translation.record_broadcast(tribe.id, broadcast, action)

        # Regression: this used to hard-cut at 60 chars with no ellipsis, silently
        # chopping the model's reasoning off mid-word most of the time -- the prompt
        # already asks for "brief" reasoning (see prompts.py's MANDATORY REACTION
        # SCHEMA), so this was working against, not with, that instruction. A much
        # higher cap here is just a guard against a model ignoring "brief" entirely.
        rationale = str(intent.get("metacognitive_rationale", ""))
        if len(rationale) > 240:
            rationale = rationale[:240].rstrip() + "…"
        entry = f"[{latency_ms:.0f}ms] {action}: {rationale}"
        if unresolved_raw is not None:
            # Marks the chronicle entry as a fallback substitution (action was
            # chosen by _resolve_action, not the tribe) rather than a real decision.
            entry += f" (unrecognized decision text: '{unresolved_raw[:60]}')"
        if hazard_note:
            entry += f" | {hazard_note}"
        tribe.history.append(entry)

        weight = 0.85 if hazard_note else (0.75 if action in ("BUILD_FIRE", "CONSTRUCT_WALL") else 0.3)
        memory_text = f"At ({tribe.x},{tribe.y}) in {ctx['biome']}, chose {action}."
        if hazard_note:
            memory_text += f" {hazard_note}."
        tribe.memory.remember(memory_text, self.cycle, weight)

    async def _trigger_game_over(self, reason: str) -> None:
        """Ends the run for real -- there will be no more turns, ever, for any
        model this session used, until a fresh ADD_TRIBE clears this back to
        normal (see add_tribe). Four ways to get here: every tribe has gone
        extinct (reason="extinction"), one tribe conquered every rival
        outright (reason="world_domination" -- War and World Domination
        era's real victory condition, see step()'s own check and
        Tribe.conquests_won), every still-living tribe has reached the era
        ceiling with nowhere further to progress (reason="era_ceiling" --
        explicit request: "we are missing 'the end'", after a real run spent
        400+ cycles, over half its total length, stepping with nothing left
        to reach), or the user hit QUIT (reason="manual_quit" -- explicit
        follow-up: "since I can click Quit anytime, it should come up when I
        quit", so a manually-ended run gets the same real summary instead of
        silently reloading with nothing shown). Rather than let
        step() keep getting called every tick forever (harmless but pointless
        once there's truly nothing left to change) and leave every model
        sitting loaded in Ollama until its keep_alive window expires on its
        own, stop stepping, generate the Overseer-voice retrospective the
        frontend's end-of-run splash displays, and unload every model
        immediately."""
        self.game_over = True
        self.game_over_reason = reason
        self.status = "GAME OVER"
        self.game_over_summary = self._generate_game_over_summary(reason)
        await self.shutdown()

    def _generate_game_over_summary(self, reason: str) -> str:
        """A detached, analytical retrospective on this run -- explicit
        request: "an intelligent summary of the game (Overseer/Scientist
        perspective)." Built entirely from data already on hand (final tribe
        stats, trophies, chief lineage) -- no extra model call, this reads as
        an observer's report on what happened, not another in-fiction voice
        the tribes themselves might use.

        Live report, after a run reached War and World Domination without
        either tribe conquering the other: the card said which era each tribe
        reached and nothing else about what actually happened there -- no
        final build (so "did they ever build a Castle/Object Creator" was
        unanswerable after the fact) and no record of whether DECLARE_CONQUEST
        (that era's one real action -- see actions.py._declare_conquest) was
        ever attempted, only whether it *succeeded* (a separate reason,
        "world_domination", triggered only by an actual merge). A reader
        watching the final standing had no way to tell "nobody ever tried"
        from "tried and lost." Both gaps are filled from data already tracked
        (tribe.buildings' own boolean flags, tribe.combat_record's "Conquest"/
        "Conquest Defense" entries -- see actions.py._record_combat) rather
        than anything new being computed here."""
        lines = []
        if reason == "extinction":
            lines.append("OVERSEER LOG: Every observed population has ceased to exist.")
        elif reason == "world_domination":
            lines.append(
                "OVERSEER LOG: A single population now accounts for the entire observed civilization. "
                "Every rival has been absorbed by conquest."
            )
        elif reason == "era_ceiling":
            lines.append(
                "OVERSEER LOG: Every surviving population has exhausted the known stages of "
                "civilizational development. No further advancement remains observable."
            )
        else:
            lines.append("OVERSEER LOG: Observation ended by operator request. Final standing recorded below.")
        for tribe in self.tribes.values():
            status = "extinct" if tribe.extinct else "surviving"
            cause_note = f", cause of collapse: {tribe.extinction_cause or 'unknown'}" if tribe.extinct else ""
            era_label = next((e.label for e in ERAS if e.key == tribe.era), tribe.era)
            trophy_names = ", ".join(t["name"] for t in tribe.trophies) or "none recorded"
            lines.append(
                f"-- {tribe.name} ({tribe.model}): {status}{cause_note}. Reached {era_label}. "
                f"Peak population {tribe.max_population}, final population {tribe.population}. "
                f"Chiefs elected: {tribe.chiefs_elected}. Distinctions: {trophy_names}. "
                f"Final build: {_final_build_summary(tribe)}."
            )
            conquest_note = _conquest_record_summary(tribe)
            if conquest_note:
                lines[-1] += f" {conquest_note}."
        living = [t for t in self.tribes.values() if not t.extinct]
        if reason == "world_domination" and living:
            victor = living[0]
            conquered = ", ".join(victor.conquered_tribe_names) or "unknown rivals"
            lines.append(
                f"Analysis: {victor.name} achieved total domination, conquering {conquered}. "
                f"Session concluded at cycle {self.cycle}."
            )
        elif reason in ("era_ceiling", "manual_quit") and living:
            leader = max(living, key=lambda t: t.max_population)
            lines.append(
                f"Analysis: {leader.name} attained the highest peak population ({leader.max_population}) "
                f"among surviving populations. Session concluded at cycle {self.cycle}."
            )
        else:
            lines.append(f"Analysis: session concluded at cycle {self.cycle}.")
        return "\n".join(lines)

    async def shutdown(self) -> None:
        """Best-effort cleanup when this session ends for any reason -- an explicit
        STOP, or a browser tab just closing/reloading mid-game (see app.py's
        ws_handler, which calls this in its finally block). PAUSE only stops
        stepping; nothing before this ever actually released the models a run had
        loaded unless every tribe happened to go fully extinct first, so closing or
        reloading the tab mid-game left them resident in Ollama's VRAM until their
        keep_alive window expired on its own -- real contention on repeated
        restarts. Safe to call even if _trigger_game_over already did this
        (unloading an already-unloaded model is a no-op).

        Regression: unloading concurrently via asyncio.gather (a longer timeout was
        tried first, see unload_model's own docstring) still intermittently left one
        of two models resident -- confirmed live, one QUIT unloaded mistral:7b but not
        phi4-mini, with unload_model's own broad except swallowing whatever actually
        went wrong. Ollama already appears to serialize the real VRAM eviction work
        regardless of how the requests arrive, so concurrency here was buying nothing
        but a chance for the second request to collide with the first mid-eviction.
        Sequential now -- shutdown only ever runs once, as a session ends, so a few
        extra seconds costs nothing that matters."""
        models = {tribe.model for tribe in self.tribes.values()}
        for model in models:
            await self.client.unload_model(model)

    def _biggest_tribe_snapshot(self) -> dict[str, int]:
        """The stockpile a minor settlement spawns/respawns with -- a snapshot of
        whichever real tribe currently has the highest population, not a flat
        invented number, so loot scales with how developed the world actually is.
        Falls back to zero if every tribe is extinct (a settlement just sits there
        with nothing until the next respawn check finds a living tribe again)."""
        living = [t for t in self.tribes.values() if not t.extinct]
        if not living:
            return {"wood": 0, "stone": 0, "food": 0, "water": 0}
        biggest = max(living, key=lambda t: t.population)
        return {resource: getattr(biggest, resource) for resource in ("wood", "stone", "food", "water")}

    def _spawn_minor_settlements(self) -> None:
        """Neutral, non-AI raid/trade targets -- see config.MINOR_SETTLEMENT_COUNT's
        own comment. Placed on real buildable ground, kept clear of every tribe's own
        starting camp so a settlement doesn't spawn right on top of one."""
        occupied = [(t.x, t.y) for t in self.tribes.values()]
        for _ in range(config.MINOR_SETTLEMENT_COUNT):
            x, y = self._find_minor_settlement_site(occupied)
            occupied.append((x, y))
            self.minor_settlements.append({
                "x": x, "y": y, "raids_remaining": config.MINOR_SETTLEMENT_MAX_RAIDS,
                "depleted_at_cycle": None, **self._biggest_tribe_snapshot(),
            })

    def _inside_any_territory(self, x: int, y: int) -> bool:
        """True if (x, y) falls within any settled tribe's own territory_radius
        (a real circle around territory_center -- same Euclidean measure
        Simulation._resolve_raider_attack/actions._relocate already use for
        territory, not the Chebyshev spacing check below) plus
        config.MINOR_SETTLEMENT_TERRITORY_BUFFER of margin past it."""
        for t in self.tribes.values():
            if t.territory_center is None:
                continue
            tcx, tcy = t.territory_center
            buffer = t.territory_radius + config.MINOR_SETTLEMENT_TERRITORY_BUFFER
            if ((x - tcx) ** 2 + (y - tcy) ** 2) ** 0.5 <= buffer:
                return True
        return False

    def _find_minor_settlement_site(self, occupied: list[tuple[int, int]]) -> tuple[int, int]:
        min_spacing = config.GRID_SIZE // (config.MINOR_SETTLEMENT_COUNT + len(self.tribes) + 1)
        for _ in range(200):
            x = random.randint(0, self.world.grid_size - 1)
            y = random.randint(0, self.world.grid_size - 1)
            if biome_at(x, y) in config.UNBUILDABLE_BIOMES:
                continue
            if self._inside_any_territory(x, y):
                continue
            if all(max(abs(x - ox), abs(y - oy)) >= min_spacing for ox, oy in occupied):
                return x, y
        return x, y  # rare fallback on a very crowded/small map -- accept the last try

    def _advance_minor_settlements(self) -> None:
        """Respawns any settlement that's sat depleted (raided out 3 times) for
        MINOR_SETTLEMENT_RESPAWN_CYCLES -- fresh stock re-snapshotted from whichever
        tribe is currently biggest, raid count reset. Explicit request: "Defeated
        Raider bubbles should... leave to a new location if they win after the
        Raid" -- respawns at a freshly picked site (the same territory-aware
        _find_minor_settlement_site placement uses) instead of sitting at the
        exact same coordinate it was raided out at."""
        for ms in self.minor_settlements:
            if ms["depleted_at_cycle"] is None:
                continue
            if self.cycle - ms["depleted_at_cycle"] >= config.MINOR_SETTLEMENT_RESPAWN_CYCLES:
                occupied = [(t.x, t.y) for t in self.tribes.values()] + [
                    (other["x"], other["y"]) for other in self.minor_settlements if other is not ms
                ]
                ms["x"], ms["y"] = self._find_minor_settlement_site(occupied)
                ms.update(self._biggest_tribe_snapshot())
                ms["raids_remaining"] = config.MINOR_SETTLEMENT_MAX_RAIDS
                ms["depleted_at_cycle"] = None

    def _discover_sites_along_route(self, tribe: Tribe, x: int, y: int, scout: str) -> None:
        """Checks one point a scout actually walked through for a real, pre-seeded
        lumber/wildlife/quarry/mine site (world.site_seed_points) -- extracted so a
        multi-leg pushed-onward trip (see _advance_one_expedition's outbound arrival
        branch) can call this once per leg instead of only for the trip's final
        stopping point. 2026-09-02 rework ("a twisted sparse matrix assignment based
        on the existing map"): these are real, fixed locations a scout discovers by
        landing within world.SITE_DISCOVERY_RADIUS of one, not an independent chance
        roll on their exact tile -- each site type has its own independent seed set,
        so two types can't stack on the same coordinate by construction."""
        grid_size = self.world.grid_size
        lumber_found = find_nearby_site("lumber", x, y, grid_size, set(tribe.lumber_sites))
        if lumber_found is not None:
            tribe.lumber_sites.append(lumber_found)
        known_wildlife = {(s["x"], s["y"]) for s in tribe.wildlife_sites}
        wildlife_found = find_nearby_site("wildlife", x, y, grid_size, known_wildlife)
        if wildlife_found is not None:
            wx, wy = wildlife_found
            site_type = random.choice(WILDLIFE_SITE_TYPES)
            tribe.wildlife_sites.append({"x": wx, "y": wy, "type": site_type})
            if tribe.last_celebration_cycle != self.cycle:
                self._celebrate_game_discovery(tribe, wx, wy)
        quarry_found = find_nearby_site("quarry", x, y, grid_size, set(tribe.quarry_sites))
        if quarry_found is not None:
            tribe.quarry_sites.append(quarry_found)
        # Explicit request: "Mines can [also] contain the Unique Resource of the
        # Biome (these locations are scattered about the map)." Same pre-seeded
        # discovery as above; the one deliberate exception stays -- a mine's
        # resource name is read off whatever real biome the pre-seeded point itself
        # sits on (world.UNIQUE_RESOURCE_BY_BIOME), not the scout's own tile.
        known_mines = {(site["x"], site["y"]) for site in tribe.mine_sites}
        mine_found = find_nearby_site("mine", x, y, grid_size, known_mines)
        if mine_found is not None:
            mx, my = mine_found
            mine_biome = biome_at(mx, my)
            resource_name = UNIQUE_RESOURCE_BY_BIOME.get(mine_biome, "Unknown Ore")
            tribe.mine_sites.append({"x": mx, "y": my, "biome": mine_biome, "resource": resource_name})
            tribe.history.append(
                f"{scout} also reports something rarer at ({mx},{my}) -- a vein of "
                f"{resource_name}, waiting to be excavated"
            )

    def _advance_expeditions(self, tribe: Tribe) -> None:
        """Advances every one of a tribe's in-field parties by one day (see
        actions.py._scout/_hunting_party) -- a tribe can have up to
        actions.expedition_capacity(tribe) out at once. Iterates a snapshot of the list
        since a party can complete (and remove itself) mid-loop."""
        for exp in list(tribe.expeditions):
            if self._advance_one_expedition(tribe, exp):
                tribe.expeditions.remove(exp)

    def _advance_one_expedition(self, tribe: Tribe, exp: dict) -> bool:
        """Advances an in-progress expedition. Runs every cycle regardless of what
        action the tribe chose that turn -- the party is out in the field on its
        own, not waiting for the tribe's attention each cycle. Movement (and every
        position-dependent check: hazards, water-sensing, arrival) happens every
        single cycle; each branch's own is_new_day flag additionally gates the
        once-a-day bookkeeping -- the day count, "daily" resource gains, and
        hunting/exploration's own rolls -- to real days, not raw cycles (see
        config.SETTLED_EXPEDITION_SPEED's own comment for why). Outbound: walk
        toward target, succeeding immediately on real fresh water or on reaching
        the destination, or giving up after EXPEDITION_MAX_DAYS. Returning: walk
        back toward camp; arrival is the only moment a finding becomes real,
        actionable knowledge (memory + chronicle) -- a party that hasn't made it
        home yet knows something the tribe as a whole does not. Returns True once
        this expedition is over and should be removed from tribe.expeditions.

        Wears (and benefits from) the same worn-trail mechanic as RELOCATE: a route
        used by enough expeditions gets faster over time, so a destination just out of
        one expedition's EXPEDITION_MAX_DAYS reach can become reachable a few attempts
        later purely by repeatedly trying the same path -- effort compounding into
        infrastructure, not a scripted distance override."""
        if exp["phase"] == "outbound":
            # Explicit request ("everyone moving on the board moves at the pace of
            # 1 sky tick"): movement itself happens every cycle for a hunting or
            # exploration party regardless of settlement -- is_new_day gates only
            # the once-a-day bookkeeping (the day count, "daily" resource gains,
            # and hazard/hunting/exploration rolls below), matching each kind's
            # own tuned per-day odds/totals exactly as before. Position-dependent
            # checks that aren't a repeated hazard roll (water-sensing, arrival)
            # still run every single cycle, since those must never risk skipping
            # a tile a party actually crosses.
            #
            # Explicit follow-up ("Scouts in particular should get their speed
            # bonus back, stealthing past observations, and moving quick to
            # report findings"): speed IS a plain SCOUT's whole role -- a single
            # fast jump through danger is inherently lower-exposure than
            # dawdling through it slowly, so a settled tribe's scout reverts
            # fully to the original once-a-day batch advance (full EXPEDITION_
            # SPEED, only actually running this whole function on is_new_day)
            # instead of the smoother-but-slower per-cycle pace hunting/
            # exploration parties now use. A still-searching tribe's scout was
            # already exempt from this and keeps advancing (and moving fast)
            # every cycle regardless.
            #
            # Explicit follow-up: "the push-on can be at normal pace, not
            # slower/shorter. It's painful to watch scouts sit at the borders
            # and not move much if at all visibly." A settled tribe's scout
            # normally only advances once every DAY_LENGTH_CYCLES (20) cycles --
            # fine for one quick patrol-and-report, but a pushed-onward search
            # (see the outbound arrival branch below) needs several such legs
            # back to back, stacking multiple 20-cycle waits into one agonizing
            # watch. Once a party has ever pushed onward (exp["pushing_onward"]),
            # it moves every cycle for the rest of this trip, outbound and
            # returning both -- the same full-speed pace a not-yet-settled
            # tribe's scout already gets, not a new slower mode.
            is_new_day = (
                not tribe.has_ever_settled or exp.get("pushing_onward") or self.cycle % config.DAY_LENGTH_CYCLES == 0
            )
            is_scout = exp.get("kind") == "scout"
            if tribe.has_ever_settled and is_scout and not is_new_day:
                return False
            if is_new_day:
                exp["day"] += 1
            px, py = exp["pos"]
            tx, ty = exp["target"]
            bonus = self.world.trail_speed_bonus(px, py, config.MAX_TRAIL_BONUS_SPEED)
            # See actions.py._build_road -- a flat, always-on version of the same
            # trail bonus above, since a deliberately-built road doesn't need to
            # wear in from repeated travel the way a trail does. See config.
            # SETTLED_EXPEDITION_SPEED's own comment for why settlement status
            # (and kind, for scouts) picks the per-cycle distance here.
            if is_scout or not tribe.has_ever_settled:
                speed_base = config.EXPEDITION_SPEED
            else:
                speed_base = config.SETTLED_EXPEDITION_SPEED
            # Object Creator era's expedition_boost effect: a flat extra
            # tiles/cycle per created object of that category (see config.
            # CREATED_OBJECT_EXPEDITION_SPEED_BONUS), not a percentage --
            # measured the same way ROAD_SPEED_BONUS already is.
            expedition_boost_count = sum(1 for obj in tribe.created_objects if obj["category"] == "expedition_boost")
            base_speed = (
                speed_base + bonus + (config.ROAD_SPEED_BONUS if tribe.road_built else 0)
                + expedition_boost_count * config.CREATED_OBJECT_EXPEDITION_SPEED_BONUS
            )
            # Explicit request: "travel speed is 5x on toll roads."
            if self.world.is_toll_road(px, py):
                base_speed *= config.TOLL_ROAD_SPEED_MULTIPLIER
            nx, ny = physics.terrain_aware_step(px, py, tx, ty, base_speed=base_speed, has_boat=tribe.boat_built)
            nx, ny = self._resolve_toll(tribe, px, py, nx, ny)
            self._wear_trail_for_expedition(exp, tribe, nx, ny)
            mark_visited_sector(tribe, nx, ny)
            exp["pos"] = [nx, ny]
            _append_expedition_path_point(exp, nx, ny)
            # Explicit correction: "the volcano is a Hazard they will die if they
            # go there." Unlike the river's drowning risk (only ever checked on
            # the outbound leg inside the water-sensing branch below, since a
            # volcano tile can never register as "sensed water"), this needs a
            # check here so an outbound leg crossing volcano ground carries the
            # same real risk the returning leg already does (below).
            #
            # Live bug ("Tribe 1's Scouts all went to the Volcano and died, then
            # the whole Tribe died"): gated to is_new_day, same as the hazard
            # rolls below. Before this, a settled tribe's party moving the new
            # slow SETTLED_EXPEDITION_SPEED (1/cycle) could spend many consecutive
            # cycles lingering on or near the volcano tile, rolling this same
            # per-day-tuned chance independently every single cycle instead of
            # once per real day -- a fast pre-fix party would have crossed the
            # danger zone in 1-2 cycles total, so cumulative risk barely mattered;
            # a slow one spending 10+ cycles there faced 10+ independent rolls.
            # This restores the original once-per-day exposure the chance
            # constant was actually tuned against; only the raw position update
            # above still happens every cycle.
            #
            # Explicit design correction: "volcano, cliff, beach, ocean should
            # all have the same treatment... any party goes out, they discover
            # a hazard, 1 person in the party dies, they run home to report
            # the death and the hazard area." These four used to fire (losing
            # someone) without ever ending the trip -- a party could take a
            # hit and just keep walking deeper into danger unless some
            # unrelated later condition happened to also turn it back. Each
            # now returns True only on an actual death (a mere discovery with
            # no death still landmarks the ground -- see _landmark_hazard --
            # but doesn't end the day), checked the same way
            # _expedition_raider_ambush already is just below: whichever one
            # actually kills someone (biomes are mutually exclusive, so at
            # most one ever does) ends the day immediately and heads the party
            # home. River/lake are the deliberate exception -- see
            # _expedition_river_hazard's own docstring for why a death there
            # never ends the trip; it's only reached via the water-sensing
            # branch further down, which already heads home on its own terms
            # (finding water, not the hazard, is what ends that leg).
            if is_new_day:
                if (
                    self._volcano_hazard(tribe, nx, ny)
                    or self._cliffs_hazard(tribe, nx, ny)
                    or self._ocean_hazard(tribe, nx, ny)
                    or self._shoals_hazard(tribe, nx, ny)
                ):
                    exp["phase"] = "returning"
                    return False
                exp["food_gathered"] += config.EXPEDITION_OUTBOUND_DAILY_FOOD
                # Explicit correction: "foragers do not need to bring water back
                # once they are settled, they should start bringing back
                # everything else though" -- a settled-near-water tribe's passive
                # daily supply (_advance_water_supply) already covers this; a
                # trickle of foraged water on top just clutters the report.
                if not self._is_settled_near_water(tribe):
                    exp["water_gathered"] += config.EXPEDITION_OUTBOUND_DAILY_WATER
            reached_biome = biome_at(nx, ny)
            scout = exp["lead_scout"]

            # Same is_new_day gating as the volcano check above -- an ambush
            # chance rolled every cycle instead of once a day would make a slow-
            # moving settled tribe's party far more likely to be ambushed than
            # one that used to cross the same ground in a single fast jump.
            if is_new_day and self._expedition_raider_ambush(tribe, exp, nx, ny):
                exp["phase"] = "returning"
                return False

            if [nx, ny] == [px, py] and [px, py] != [tx, ty]:
                # Regression: physics.terrain_aware_step falls back to "stay put" when
                # every candidate step toward the target is ocean (boxed in on every
                # axis -- see its own docstring). A pushed-onward target
                # (extend_ray_to_grid_edge) can land past the actual coastline into
                # open water, which made this fallback fire every single day forever --
                # a live run caught a party stuck at the same tile for 400+ days,
                # hoarding phantom food/water in its own counters the whole time.
                # Physically unable to advance at all is just as much "nowhere left to
                # search" as reaching the grid's literal edge -- give up the same way.
                # Checked before the hunt/water split below since it's kind-agnostic --
                # a hunting party can push its own target into the ocean exactly the
                # same way (see _advance_hunting_party_outbound's own push-onward). The
                # target-equals-position exclusion matters for a hunt/scout that's
                # deliberately working right where it already stands (target == origin)
                # -- that's arrival, not being boxed in, and has its own handling below.
                #
                # Live report ("hard time with expeditions 'looking' stuck... turn them
                # around right away, so we don't see this hanging on the edge of the
                # board for so long"): turning the phase to "returning" here already
                # happens the instant the game notices, but a settled scout otherwise
                # only re-checks its position once every DAY_LENGTH_CYCLES (20) cycles
                # (see is_new_day above) -- without pushing_onward, the newly-returning
                # party would sit at that same boxed-in tile doing nothing for up to 20
                # more cycles before its first step home. Reusing the same flag the
                # patrol-push mechanic already uses for exactly this "resume full pace
                # for the rest of the trip" purpose (see this function's own docstring
                # comment above), so the walk home actually starts moving next cycle.
                exp["pushing_onward"] = True
                exp["phase"] = "returning"
                tribe.history.append(f"{scout}'s party can go no further this way and turns back after {exp['day']} days")
                return False
            # Hunting/exploration's own daily rolls and gains (catch chance, wolf
            # hazard, wood/stone foraged) are gated to is_new_day too -- calling
            # them every movement cycle instead of once a day would inflate their
            # tuned odds/totals purely from the new per-cycle movement, not from
            # anything actually different happening. On a non-new-day cycle the
            # party still just walks (already done above); hunt returns early
            # either way (no generic discovery for a hunting party), explore falls
            # through to the same generic checks every other kind shares below.
            if exp.get("kind") == "hunt":
                if is_new_day:
                    self._advance_hunting_party_outbound(tribe, exp, reached_biome, scout)
                return False
            if exp.get("kind") == "explore" and is_new_day and self._advance_exploration_party_outbound(tribe, exp, reached_biome, scout):
                return False  # forced home early (carry capacity or day limit) -- already flipped to returning
            # Explicit request: "the find water scouting needs to be removed
            # from available actions after they Settle. The scouts can still
            # explore and report sightings and discoveries." Water-sensing used
            # to unconditionally cut every expedition short the moment it
            # passed near any water, even for an already-settled tribe with
            # nothing left to gain from finding more -- crowding out the
            # lumber/wildlife/quarry/mine/raider discoveries a scout could
            # otherwise report from the same trip. Once genuinely settled near
            # water, a party now passes water by and keeps searching, the same
            # as if there were none nearby at all.
            #
            # Bug report: "clearly they see water, the scout walked right
            # through it." A single EXPEDITION_SPEED step can cover more
            # ground (up to 10 tiles) than WATER_SENSING_RADIUS (6) -- sensing
            # only at the step's final landing tile let a party leap clean
            # over a river narrower than the step itself without ever
            # registering it. Now checks every whole tile actually crossed
            # this step (_interpolated_path, the same helper the resource-
            # trail mechanic uses) -- in reverse, destination first, so an
            # arrival directly on/right next to water is still what's
            # reported (preserving the existing on-tile drowning-risk
            # mechanic below), falling back to an earlier point along the
            # same step only if the destination itself didn't sense anything.
            sensed = None
            if not self._is_settled_near_water(tribe):
                for ix, iy in reversed(_interpolated_path(px, py, nx, ny)):
                    sensed = self._sense_nearby_water(ix, iy, config.WATER_SENSING_RADIUS)
                    if sensed:
                        break
            if sensed:
                wx, wy = sensed
                on_water_now = (wx, wy) == (nx, ny)
                if on_water_now:
                    self._expedition_river_hazard(tribe, nx, ny)  # covers river and lake -- see its own docstring
                exp["found"] = [wx, wy]
                exp["phase"] = "returning"
                if on_water_now:
                    tribe.history.append(f"{scout}'s party has found fresh water and is heading home to report it")
                else:
                    tribe.history.append(f"{scout}'s party hears water nearby and marks ({wx},{wy}) before heading home to report it")
            elif [nx, ny] == [tx, ty]:
                # Reached the assigned patrol point (config.SCOUT_PATROL_DISTANCE tiles
                # along this dispatch's rotating heading -- see actions.py._scout) with
                # no water found. Bug report: "they go big long lines like they are
                # flying." This used to push the party onward toward the map's true
                # edge whenever days remained, on the theory that "running out of world
                # to search" was the only honest reason to stop -- that produced
                # exactly the ruler-straight, cross-map dashes being complained about.
                # Shortening the leash: reaching the assigned patrol point is real
                # completion now, the same as a party that happens to land right on the
                # grid's literal edge -- more, shorter local patrols over many
                # dispatches (still covering new ground each time via
                # scout_rotation_index) instead of one long committed sprint. The
                # terrain actually reached is still worth reporting home (see how
                # terrain_report drives lumber/quarry/mine/wildlife discovery below).
                #
                # Explicit follow-up, after watching a live scout turn back at day
                # 3 of an available 6 with days to spare: "they should have
                # continued." A plain SCOUT (not exploration party, which has its
                # own separate day-limit-aware logic) now pushes onward to a fresh
                # patrol leg along the SAME heading when real days remain, instead
                # of always stopping here -- still short, bounded hops
                # (_push_past_visited_ground, the identical "keep walking, skip
                # already-covered ground" idea used at dispatch), not the old
                # unbounded dash to the grid's true edge this branch was
                # originally narrowed away from. Every leg's ground still gets
                # checked for a real site once the party finally comes home (see
                # _discover_sites_along_route and terrain_checkpoints below), not
                # just the leg that happens to end the trip.
                if exp.get("kind") == "scout" and exp["day"] < exp["max_days"]:
                    exp.setdefault("terrain_checkpoints", []).append((nx, ny))
                    ox, oy = exp["origin"]
                    heading = math.atan2(ty - oy, tx - ox)
                    new_tx, new_ty = _push_past_visited_ground(
                        tribe, nx, ny, heading, config.SCOUT_PATROL_DISTANCE, self.world.grid_size
                    )
                    # Explicit request, after finding a live party bounce forever
                    # near a coastline: "so they aren't turning back like they
                    # should. maybe we should 'kick' them back home automatically
                    # for simplification." _push_past_visited_ground's own
                    # grid-edge reflection can send the next leg's target
                    # BACKWARD -- once a heading is close enough to an edge that
                    # the full patrol distance would overshoot off the map, the
                    # reflected point lands closer to home than out. Confirmed
                    # live: a party heading due west from (60,10) settled into a
                    # permanent (10,10)<->(15,10) bounce, never actually turning
                    # back. Only push onward if the new leg is genuinely farther
                    # from where this trip started than the ground just covered --
                    # otherwise there's no real ground left this heading, and it's
                    # simpler and more honest to send them home now.
                    made_progress = math.hypot(new_tx - ox, new_ty - oy) > math.hypot(nx - ox, ny - oy)
                    if (new_tx, new_ty) != (nx, ny) and made_progress:
                        exp["target"] = [new_tx, new_ty]
                        exp["pushing_onward"] = True
                        return False
                exp["terrain_report"] = reached_biome
                exp["phase"] = "returning"
                label = BIOME_LABELS.get(reached_biome, reached_biome)
                tribe.history.append(f"{scout}'s party surveys ({nx},{ny}), {label}, after {exp['day']} days and heads home to report")
            return False
        else:  # returning
            # See the matching outbound-leg comment above -- same is_new_day/
            # scout-speed split, including the pushing_onward full-pace
            # exemption for the whole rest of a trip that ever pushed onward.
            is_new_day = (
                not tribe.has_ever_settled or exp.get("pushing_onward") or self.cycle % config.DAY_LENGTH_CYCLES == 0
            )
            is_scout = exp.get("kind") == "scout"
            if tribe.has_ever_settled and is_scout and not is_new_day:
                return False
            px, py = exp["pos"]
            ox, oy = exp["origin"]
            bonus = self.world.trail_speed_bonus(px, py, config.MAX_TRAIL_BONUS_SPEED)
            if is_scout or not tribe.has_ever_settled:
                speed_base = config.EXPEDITION_SPEED
            else:
                speed_base = config.SETTLED_EXPEDITION_SPEED
            # Object Creator era's expedition_boost effect: a flat extra
            # tiles/cycle per created object of that category (see config.
            # CREATED_OBJECT_EXPEDITION_SPEED_BONUS), not a percentage --
            # measured the same way ROAD_SPEED_BONUS already is.
            expedition_boost_count = sum(1 for obj in tribe.created_objects if obj["category"] == "expedition_boost")
            base_speed = (
                speed_base + bonus + (config.ROAD_SPEED_BONUS if tribe.road_built else 0)
                + expedition_boost_count * config.CREATED_OBJECT_EXPEDITION_SPEED_BONUS
            )
            # Explicit request: "travel speed is 5x on toll roads."
            if self.world.is_toll_road(px, py):
                base_speed *= config.TOLL_ROAD_SPEED_MULTIPLIER
            nx, ny = physics.terrain_aware_step(px, py, ox, oy, base_speed=base_speed, has_boat=tribe.boat_built)
            nx, ny = self._resolve_toll(tribe, px, py, nx, ny)
            self._wear_trail_for_expedition(exp, tribe, nx, ny)
            mark_visited_sector(tribe, nx, ny)
            exp["pos"] = [nx, ny]
            _append_expedition_path_point(exp, nx, ny)
            if is_new_day:
                exp["food_gathered"] += config.EXPEDITION_RETURN_DAILY_FOOD
                if not self._is_settled_near_water(tribe):  # see the matching outbound-leg comment above
                    exp["water_gathered"] += config.EXPEDITION_RETURN_DAILY_WATER
                # Live bug ("Tribe 1's Scouts all went to the Volcano and died,
                # then the whole Tribe died") -- see the matching outbound-leg
                # comment above. These three used to roll every single cycle; a
                # slow-moving settled tribe's party lingering near danger for
                # many cycles faced that many independent rolls instead of the
                # one per real day these chances were actually tuned against.
                self._expedition_river_hazard(tribe, nx, ny)
                self._volcano_hazard(tribe, nx, ny)
                self._cliffs_hazard(tribe, nx, ny)
                self._ocean_hazard(tribe, nx, ny)
                self._shoals_hazard(tribe, nx, ny)
                self._expedition_raider_ambush(tribe, exp, nx, ny)
            # Live bug ("Scouts, after settlement, are doing weird things"):
            # the outbound leg already gives up when physics.terrain_aware_step
            # reports boxed in by ocean on every axis (see that check above --
            # a live run once caught a party frozen at the same tile for 400+
            # days), but the returning leg never got the same treatment.
            # Confirmed live: a settled tribe's scout heading home got stuck at
            # a single tile 42 tiles from camp for 30+ cycles straight, silently
            # re-rolling the exact same "stay put" fallback once a day forever
            # with no way to ever actually arrive. A boxed-in party can't
            # sensibly retarget (it's already given up searching), so treat
            # getting stuck as arrival: whatever it found/gathered becomes real
            # right here instead of never being reported at all.
            boxed_in = [nx, ny] == [px, py] and [px, py] != [ox, oy]
            if boxed_in:
                tribe.history.append(
                    f"{exp['lead_scout']}'s party can go no further on the way home and reports in from "
                    f"here after {exp['day']} days"
                )
            if [nx, ny] == [ox, oy] or boxed_in:
                # Whatever was foraged along the way comes home regardless of whether the
                # expedition succeeded -- the trip cost real time either way, so it isn't
                # a total loss on a failed search. The findings themselves only become
                # real, actionable knowledge for the tribe at this exact moment.
                #
                # Live report ("Water warnings shouldn't be firing -- never run out once
                # settled"): traced to these four lines adding straight to tribe.food/
                # water/wood/stone with no _capped_add, the one uncapped hole in an
                # otherwise fully-capped storage system. _capped_add's own math (cap -
                # current, floored at 0) means once ANY of these pushes a resource even
                # slightly over its storage cap, every later *capped* addition -- the
                # passive settled-water/fish/farm income this whole safety net exists for
                # -- silently adds zero forever, with nothing telling the tribe why, until
                # consumption alone drags the stockpile back under the cap. Confirmed live
                # (run_20260908_103409): a 3400+ population tribe's water sat flat-to-
                # declining for 25+ cycles despite _advance_water_supply's own formula
                # computing a large enough passive grant every cycle that it should have
                # been climbing fast.
                food_gained = round(exp["food_gathered"] * _food_multiplier(tribe))
                food_home = self._capped_add(tribe, "food", food_gained)
                water_home = self._capped_add(tribe, "water", exp["water_gathered"])
                scout = exp["lead_scout"]
                forage_note = f"bringing back {food_home} food"
                if food_home < food_gained:
                    forage_note += f" ({food_gained - food_home} more spoiled -- stores already full)"
                # water_gathered never accrues once _is_settled_near_water is true (see
                # the matching outbound/returning-leg gate above) -- omit it from the
                # note entirely rather than always reporting a flat "0 water".
                if water_home:
                    forage_note += f" and {water_home} water foraged along the way"
                elif exp["water_gathered"]:
                    forage_note += " foraged along the way (water stores already full)"
                else:
                    forage_note += " foraged along the way"
                # Only EXPLORATION_PARTY ever populates these -- see actions.py.
                # _exploration_party. .get(..., 0) leaves scout/hunt/trade untouched.
                wood_gained, stone_gained = exp.get("wood_gathered", 0), exp.get("stone_gathered", 0)
                if wood_gained or stone_gained:
                    wood_home = self._capped_add(tribe, "wood", wood_gained)
                    stone_home = self._capped_add(tribe, "stone", stone_gained)
                    forage_note += f", {wood_home} wood and {stone_home} stone"
                recipient = f"Chief {tribe.chief_name}" if tribe.chief_name else "the tribe"

                if exp.get("kind") == "hunt":
                    self._report_hunting_party_home(tribe, exp, scout, forage_note, recipient)
                    return True

                # Independent of whatever terrain/water was found this trip -- a
                # scout could plausibly spot both a resource site and raider sign on
                # the same journey (see config.RAIDER_SIGHTING_CHANCE's own comment
                # for why this isn't a biome-matched roll like the terrain_report
                # branches below). Radiates dread AT THE SIGHTING COORDINATE, not the
                # tribe's own camp -- a place now known to be dangerous, not
                # something that happened at home.
                if random.random() < config.RAIDER_SIGHTING_CHANCE:
                    # Bug report: "we have a lot of Raider camps right on top of a
                    # resource." A resource-site discovery (terrain_report, below)
                    # is recorded at this exact exp["target"] tile -- reporting the
                    # raider sighting there too meant any trip where both
                    # independent rolls succeeded stacked them on the identical
                    # tile. Nudged to a nearby point instead, same "spotted on the
                    # same journey" idea, no longer literally the same spot.
                    off = config.RAIDER_SIGHTING_OFFSET
                    rx = max(0, min(self.world.grid_size - 1, exp["target"][0] + random.randint(-off, off)))
                    ry = max(0, min(self.world.grid_size - 1, exp["target"][1] + random.randint(-off, off)))
                    # Bug report: "there is a raider in the ocean." A raw target/
                    # offset point can land in open water (deflected around, per
                    # physics.terrain_aware_step, or just never reached) with no
                    # land check at all. "Not biome-matched" (see this constant's
                    # own comment) means no specific biome is required, not that
                    # literally underwater counts.
                    if biome_at(rx, ry) not in config.UNBUILDABLE_BIOMES:
                        self._record_raider_sighting(tribe, rx, ry)
                        self.trauma.radiate_event_wave(
                            rx, ry, config.RAIDER_SIGHTING_TRAUMA_MAGNITUDE, config.RAIDER_SIGHTING_TRAUMA_RADIUS
                        )
                        tribe.memory.remember(f"Scouts spotted signs of raiders near ({rx},{ry}).", self.cycle, weight=0.7)
                        tribe.history.append(f"{scout} reports signs of raiders near ({rx},{ry}) on the way home -- best be cautious")

                if exp["found"]:
                    fx, fy = exp["found"]
                    tribe.expeditions_succeeded += 1
                    tribe.scout_successes += 1
                    # Explicit request: credit the scout who actually found the water,
                    # not the chief -- _check_chief_trophies' river/lake-standing case
                    # still credits the chief (a real, different circumstance: the
                    # chief personally leading the tribe onto water with no scout
                    # involved), but this is the path that fires in practice, and
                    # crediting only ever the chief meant a young tribe could go a very
                    # long time with just one named individual (the chief), leaving
                    # _eligible_breeding_pair permanently empty until a much
                    # higher-threshold trophy (Master Pathfinder/Master Hunter) came in.
                    self._award_trophy(tribe, "Water Bringer", individual=scout)
                    if tribe.scout_successes == config.MILESTONE_SCOUT_SUCCESSES:
                        self._award_trophy(tribe, "Master Pathfinder", individual=scout)
                    self._check_custom_awards(tribe, "scouting", individual=scout)
                    tribe.memory.remember(f"Scouts confirmed fresh water at ({fx},{fy}).", self.cycle, weight=0.9)
                    is_new_site = (fx, fy) not in tribe.confirmed_water_sites
                    if is_new_site:
                        tribe.confirmed_water_sites.append((fx, fy))
                    tribe.history.append(
                        f"{scout} is home and gives {recipient} a full report: "
                        f"fresh water confirmed at ({fx},{fy}), {forage_note}"
                    )
                    # `!= self.cycle`, not the full CELEBRATION_COOLDOWN_CYCLES gate --
                    # this is meant to fire on every genuine new find, just not twice
                    # in the exact same cycle (two expeditions can both arrive home
                    # with news this same tick -- see _advance_expeditions' loop over
                    # every in-field party) or on top of an unrelated celebration
                    # (settling, harvest) that already happened this same cycle.
                    if is_new_site and not self._is_settled_near_water(tribe) and tribe.last_celebration_cycle != self.cycle:
                        self._celebrate_water_discovery(tribe, fx, fy)
                elif exp["terrain_report"]:
                    label = BIOME_LABELS.get(exp["terrain_report"], exp["terrain_report"])
                    tx, ty = exp["target"]
                    tribe.memory.remember(f"Scouts explored toward ({tx},{ty}) and found {label} terrain.", self.cycle, weight=0.6)
                    self._discover_sites_along_route(tribe, tx, ty, scout)
                    # Explicit request: "the Scout returned before they found water
                    # on the first outbound run. They should have continued and
                    # reported all the sightings at once on returning." A scout
                    # that pushed onward through several patrol legs (see the
                    # outbound arrival branch above) checks every earlier leg's
                    # ground for a real site too, not just the final one -- one
                    # combined report for the whole trip instead of only the last
                    # stretch of it.
                    for cx, cy in exp.get("terrain_checkpoints", []):
                        self._discover_sites_along_route(tribe, cx, cy, scout)
                    tribe.history.append(
                        f"{scout} is home and gives {recipient} a full report: "
                        f"{label} terrain at ({tx},{ty}), {forage_note}"
                    )
                else:
                    tribe.history.append(
                        f"{scout} is home and gives {recipient} a full report: "
                        f"nothing new found, though not empty-handed -- {forage_note}"
                    )
                return True
            return False

    def _sense_nearby_water(self, x: int, y: int, radius: int) -> tuple[int, int] | None:
        """Scans a radius around (x, y) for the nearest river or lake tile. A scout
        doesn't need to physically wade in to know water is close -- running water
        carries, and a lake is visible well before its shore. Ocean is deliberately
        excluded; that's the map's edge, not a "water source" worth reporting home
        about. Returns the closest qualifying tile, or None if the radius is dry."""
        best: tuple[int, int] | None = None
        best_dist = None
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist = dx * dx + dy * dy
                if dist > radius * radius:
                    continue
                wx, wy = x + dx, y + dy
                if not (0 <= wx < self.world.grid_size and 0 <= wy < self.world.grid_size):
                    continue
                if biome_at(wx, wy) not in ("river", "lake"):
                    continue
                if best_dist is None or dist < best_dist:
                    best, best_dist = (wx, wy), dist
        return best

    def _landmark_hazard(self, tribe: Tribe, x: int, y: int, hazard_label: str) -> None:
        """Explicit design spec: "if anyone discovers a hazard, even if no one
        dies, they landmark it... it needs a 'dangerous' sounding name." Called
        unconditionally by every hazard function below the moment a party is
        physically on that hazard's biome, independent of the separate,
        chance-based death roll each one still makes on its own -- merely
        knowing the ground is dangerous doesn't require losing someone first.

        Dedup'd against tribe.hazard_landmarks (same {"x","y",name-field} shape
        and same linear dedup check config.LANDMARK_NAMES's own reward
        Landmarks already use against tribe.landmarks) so a lingering party
        doesn't re-report the same known-dangerous tile every single day --
        "no party should ever linger.\""""
        if any(lm["x"] == x and lm["y"] == y for lm in tribe.hazard_landmarks):
            return
        name = random.choice(config.HAZARD_LANDMARK_NAMES)
        tribe.hazard_landmarks.append({"x": x, "y": y, "name": name})
        tribe.history.append(
            f"the party marks {hazard_label} near ({x},{y}) as {name} -- a known danger for anyone who comes after"
        )
        tribe.memory.remember(
            f"{name} near ({x},{y}) is known dangerous ground -- {hazard_label}.", self.cycle, weight=0.5,
        )
        self.recent_encounters.append({
            "x": x, "y": y, "kind": "hazard_landmark", "label": name, "outcome": "sighted",
        })

    def _expedition_river_hazard(self, tribe: Tribe, x: int, y: int) -> bool:
        """The same drowning risk GATHER_WATER already carries on a river tile
        (config.DROWNING_HAZARD_CHANCE) -- a traveling party crossing or camped on a
        river isn't any safer than a tribe standing on one to fill jugs. Returns True
        if it claimed someone (population loss and trauma already applied).

        Explicit design correction: "other water like the River and Lake [is]
        hazardous, yes, someone dies if they don't swim -- scouts can get over
        pretty easily to form a path... if they still have time and people in
        the party alive, they can continue on." Extended from river-only to
        also cover lake -- lake tiles used to be an explicit no-op ("no
        current to drown in"), but the user's own framing is that any open
        water carries the same real risk, not just a river's current
        specifically. Unlike volcano/cliffs/ocean/shoals (which force the
        party home on an actual death), this one never does -- callers don't
        check this return value to decide whether to keep going.

        Explicit exception to the grave-marker every other hazard death here
        leaves: "the only exception to the grave-marker/landmark is the River
        and Lake. We imagine these grave markers as being washed away over
        time, fleeting -- the water is not covered in death." A death here
        still gets the landmark (the danger itself is real and worth knowing),
        the history line, and the memory entry, but not the recent_encounters
        "hazard_death" skull marker every land hazard leaves.

        Losing someone this way also becomes a real, remembered lesson, not just a
        chronicle line that scrolls away: a high-weight memory entry (see
        TribeMemory.consolidate) is what actually promotes into a standing taboo the
        tribe's own reasoning sees every future turn -- the survivors telling the rest
        of the tribe what happened, not the simulation warning them directly."""
        biome = biome_at(x, y)
        if biome not in ("river", "lake"):
            return False
        self._landmark_hazard(tribe, x, y, "a treacherous crossing")
        if random.random() >= config.DROWNING_HAZARD_CHANCE:
            return False
        self.trauma.radiate_event_wave(x, y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS)
        self._lose_population(tribe, config.DROWNING_HAZARD_POPULATION_LOSS, cause="drowning")
        water_word = "current" if biome == "river" else "cold water"
        tribe.history.append(
            f"the {water_word} pulled someone under while the party was crossing near ({x},{y}) -- "
            "the rest press on"
        )
        tribe.memory.remember(
            f"A water crossing near ({x},{y}) drowned one of our own -- real danger there.",
            self.cycle, weight=0.85,
        )
        return True

    def _volcano_hazard(self, tribe: Tribe, x: int, y: int) -> bool:
        """Explicit correction: "the volcano is a Hazard they will die if they go
        there." Same shape as _expedition_river_hazard just above, far more
        likely to strike (config.VOLCANO_HAZARD_CHANCE 0.75 vs. drowning's own
        DROWNING_HAZARD_CHANCE 0.08) -- this needs to read as a serious,
        well-known danger, not a mild river crossing. Population cost is the
        same single life every environmental hazard in this file costs (see
        VOLCANO_HAZARD_POPULATION_LOSS's own comment -- it used to be 5, an
        outlier that could extinction a young tribe in 2-3 hits; the chance,
        not the death toll, is what makes this one worse). Called from every
        real way a tribe's people could end up on that tile: expedition movement
        (the same two call sites _expedition_river_hazard already hooks) and
        RELOCATE (actions._relocate) -- broader coverage than the river hazard
        gets, since "if they go there" has to mean any of them, not just an
        expedition passing through.

        Explicit design split: a discovery here (see _landmark_hazard) always
        happens on arrival, but an actual death is what forces the party home
        -- "1 person dies, they run home to report the death"; surviving the
        ground unscathed, they mark it and move on."""
        if biome_at(x, y) != "volcano":
            return False
        self._landmark_hazard(tribe, x, y, "a volcano")
        if random.random() >= config.VOLCANO_HAZARD_CHANCE:
            return False
        self.trauma.radiate_event_wave(x, y, config.VOLCANO_TRAUMA_MAGNITUDE, config.VOLCANO_TRAUMA_RADIUS)
        self._lose_population(tribe, config.VOLCANO_HAZARD_POPULATION_LOSS, cause="volcano")
        tribe.history.append(f"the volcano's toxic fumes and scalding ground claimed lives near ({x},{y}) -- the survivors flee home to report it")
        tribe.memory.remember(
            f"The volcano near ({x},{y}) is deadly -- real danger there, stay away.",
            self.cycle, weight=0.9,
        )
        self.recent_encounters.append({
            "x": x, "y": y, "kind": "hazard_death", "label": "Lost to the volcano", "outcome": "struck",
        })
        return True

    def _cliffs_hazard(self, tribe: Tribe, x: int, y: int) -> bool:
        """Explicit request: "we need to add back the Cliffs hazard or those
        Scouts stay there." Same shape as _expedition_river_hazard/
        _volcano_hazard -- see config.CLIFFS_HAZARD_CHANCE's own comment for
        why this is moderate rather than volcano-severe. Called from every
        real way a tribe's people could end up on a cliffs tile: expedition
        movement and RELOCATE (actions._relocate), same broad coverage
        _volcano_hazard already gets. Same discovery/death split as
        _volcano_hazard's own comment."""
        if biome_at(x, y) != "cliffs":
            return False
        self._landmark_hazard(tribe, x, y, "loose cliffside rock")
        if random.random() >= config.CLIFFS_HAZARD_CHANCE:
            return False
        self.trauma.radiate_event_wave(x, y, config.CLIFFS_TRAUMA_MAGNITUDE, config.CLIFFS_TRAUMA_RADIUS)
        self._lose_population(tribe, config.CLIFFS_HAZARD_POPULATION_LOSS, cause="cliffs")
        tribe.history.append(f"the cliffs near ({x},{y}) claimed a life on the loose rock -- the survivors head home to report it")
        tribe.memory.remember(
            f"The cliffs near ({x},{y}) are dangerous underfoot -- real risk lingering there.",
            self.cycle, weight=0.8,
        )
        self.recent_encounters.append({
            "x": x, "y": y, "kind": "hazard_death", "label": "Lost to the cliffs", "outcome": "struck",
        })
        return True

    def _shoals_hazard(self, tribe: Tribe, x: int, y: int) -> bool:
        """Explicit design correction: "volcano, cliff, beach, ocean should
        all have the same treatment." Same shape as _cliffs_hazard exactly --
        "beach" is this map's shoals biome, the sandy counterpart to cliffs
        along the same coastline (world.biome_at picks one or the other per
        coastal tile), and had no hazard of its own at all before this. Same
        discovery/death split as _volcano_hazard's own comment."""
        if biome_at(x, y) != "shoals":
            return False
        self._landmark_hazard(tribe, x, y, "a rip current off the shoals")
        if random.random() >= config.SHOALS_HAZARD_CHANCE:
            return False
        self.trauma.radiate_event_wave(x, y, config.SHOALS_TRAUMA_MAGNITUDE, config.SHOALS_TRAUMA_RADIUS)
        self._lose_population(tribe, config.SHOALS_HAZARD_POPULATION_LOSS, cause="shoals")
        tribe.history.append(f"a rip current off the shoals near ({x},{y}) claimed a life -- the survivors head home to report it")
        tribe.memory.remember(
            f"The shoals near ({x},{y}) are dangerous underfoot -- real risk lingering there.",
            self.cycle, weight=0.8,
        )
        self.recent_encounters.append({
            "x": x, "y": y, "kind": "hazard_death", "label": "Lost to the shoals", "outcome": "struck",
        })
        return True

    def _ocean_hazard(self, tribe: Tribe, x: int, y: int) -> bool:
        """Explicit spec: "Ocean is instant kill 1, report, gravemarker." A
        safety net, not a routine check -- physics.terrain_aware_step already
        deflects ordinary movement around ocean, so this only ever fires
        through the known reflected/overshot-target edge case (see physics.
        reflect_into_grid's own docstring). Certain rather than a rolled
        chance, matching "instant" -- so unlike every sibling hazard here,
        there's no separate "discovered but survived" case to landmark on its
        own; reaching this tile at all is the death."""
        if biome_at(x, y) != "ocean":
            return False
        self.trauma.radiate_event_wave(x, y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS)
        self._lose_population(tribe, config.OCEAN_HAZARD_POPULATION_LOSS, cause="ocean")
        tribe.history.append(f"the open sea claimed a life near ({x},{y}) -- the survivors head home to report it")
        tribe.memory.remember(
            f"The open sea off ({x},{y}) is fatal -- real danger, never go there.",
            self.cycle, weight=0.9,
        )
        self.recent_encounters.append({
            "x": x, "y": y, "kind": "hazard_death", "label": "Lost to the sea", "outcome": "struck",
        })
        return True

    def _record_raider_sighting(self, tribe: Tribe, x: int, y: int) -> None:
        """Shared by both places a raider sighting gets logged (a scout's report and
        an in-field ambush). Caps the list at config.RAIDER_SIGHTING_MAX_REMEMBERED,
        most-recent-first -- see that constant's own comment for why this list needs
        trimming at all when the other LANDMARK_TYPES lists don't."""
        if (x, y) in tribe.raider_sightings:
            return
        tribe.raider_sightings.append((x, y))
        overflow = len(tribe.raider_sightings) - config.RAIDER_SIGHTING_MAX_REMEMBERED
        if overflow > 0:
            del tribe.raider_sightings[:overflow]

    def _relocate_raider_sighting_after_ambush(self, tribe: Tribe, x: int, y: int) -> None:
        """Explicit request: "after an ambush the Raiders move from that
        location." Clears any existing sighting record for this exact spot and
        records a new one nearby instead (the same RAIDER_SIGHTING_OFFSET nudge
        a scout's own raider-sighting report already uses) -- so the same
        location doesn't keep generating identical repeat encounters once its
        raiders have actually been engaged, win or lose.

        Explicit follow-up, after a real test flake exposed the gap: this used
        to pick independent x/y offsets, which could land back on the exact
        tile just cleared (both offsets rolling 0 -- "cast elsewhere" ending
        up nowhere), or on unbuildable terrain, which silently dropped the
        relocation entirely (raiders just vanish, contradicting "cast
        elsewhere... not just gone"). Angle + a distance floored at
        RAIDER_SIGHTING_MIN_OFFSET guarantees real displacement every time; a
        bounded retry loop (same shape _find_minor_settlement_site's own
        placement search already uses) guarantees a real landing spot instead
        of silently giving up on the first unlucky roll."""
        if (x, y) in tribe.raider_sightings:
            tribe.raider_sightings.remove((x, y))
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.randint(config.RAIDER_SIGHTING_MIN_OFFSET, config.RAIDER_SIGHTING_OFFSET)
            rx = max(0, min(self.world.grid_size - 1, x + round(dist * math.cos(angle))))
            ry = max(0, min(self.world.grid_size - 1, y + round(dist * math.sin(angle))))
            if (rx, ry) != (x, y) and biome_at(rx, ry) not in config.UNBUILDABLE_BIOMES:
                self._record_raider_sighting(tribe, rx, ry)
                return
        # Rare fallback on a very crowded/small map -- same "accept defeat
        # after enough tries" shape _find_minor_settlement_site's own search
        # uses. Raiders simply stay dispersed this time rather than force a
        # placement onto invalid ground.

    def _expedition_raider_ambush(self, tribe: Tribe, exp: dict, x: int, y: int) -> bool:
        """Explicit request: "It would be interesting to see a Scout encounter a
        RAIDER group" -- a real, in-the-field ambush during travel, distinct from the
        settlement-level attack (_check_raider_attack) and from a report-based
        sighting (RAIDER_SIGHTING_CHANCE) -- a party physically running into raiders,
        not a rumor or a distant attack on the camp. Gated the same as the
        settlement attack: raiders being active against a tribe at all is tied to
        that tribe having something worth raiding. Returns True if it happened
        (population loss and trauma already applied); the caller ends the trip
        immediately, the same way the wolf-pack hazard ends a hunt outright."""
        if not tribe.has_ever_settled or random.random() >= config.EXPEDITION_RAIDER_AMBUSH_CHANCE:
            return False
        # Explicit request: "When a Boat encounters a Raider on the Water, the
        # Boat wins automatically and goes home to deliver the counter-raid
        # loot." Same loot shape _resolve_raider_attack's own defended branch
        # already uses, but always wins here -- a boat on the water it was
        # actually built for beats raiders on foot outright, no roll needed.
        if tribe.boat_built and biome_at(x, y) in config.BOAT_WATER_BIOMES:
            # Live bug, same shape as Simulation._resolve_raider_attack's own
            # fix (a previous commit, discovered from a different tribe's live
            # run than the one that caught this one): this used to setattr the
            # requested amount directly, bypassing _capped_add -- since the
            # requested amount is a fraction of the tribe's OWN current
            # stockpile, that's an uncapped compound multiplier every time this
            # fires, not loot recovered from the raiders. Confirmed live: three
            # of these landed in a single cycle (762+914+1097=2773 wood at
            # once), pushing a tribe's wood well past its own _storage_cap.
            looted = {
                resource: self._capped_add(tribe, resource, round(getattr(tribe, resource) * config.RAIDER_DEFEAT_LOOT_FRACTION))
                for resource in ("wood", "stone", "food")
            }
            self.trauma.radiate_event_wave(x, y, config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
            tribe.history.append(
                f"{exp['lead_scout']}'s boat runs down a raider band on the water near ({x},{y}) and heads "
                f"home with the counter-raid loot -- {looted['food']} food, {looted['wood']} wood, and "
                f"{looted['stone']} stone recovered"
            )
            self.recent_encounters.append({
                "x": x, "y": y, "kind": "raider_attack", "label": "Raiders routed by boat", "outcome": "repelled",
            })
            _record_combat(tribe, "Ambush", "won")
            self._relocate_raider_sighting_after_ambush(tribe, x, y)
            exp["phase"] = "returning"
            return True

        # Explicit finding: "how many times did they defend and get loot? in an
        # ambush scenario" -- turned out to be zero, a structural guarantee (no
        # defend branch existed at all off a boat). Real chance added, in the
        # spirit of _resolve_raider_attack's own population-scaled defense, but
        # deliberately lower/capped -- a traveling party carries no wall, keep,
        # moat, or torches with it, just its own numbers.
        defense_chance = min(
            config.EXPEDITION_AMBUSH_DEFENSE_MAX_CHANCE,
            config.EXPEDITION_AMBUSH_DEFENSE_BASE_CHANCE
            + (tribe.population // 10) * config.EXPEDITION_AMBUSH_DEFENSE_POPULATION_BONUS_PER_10,
        )
        if random.random() < defense_chance:
            # Same storage-cap fix as the boat branch just above.
            looted = {
                resource: self._capped_add(tribe, resource, round(getattr(tribe, resource) * config.RAIDER_DEFEAT_LOOT_FRACTION))
                for resource in ("wood", "stone", "food")
            }
            self.trauma.radiate_event_wave(x, y, config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
            tribe.history.append(
                f"{exp['lead_scout']}'s party fought off an ambush near ({x},{y}) and heads home with the "
                f"counter-raid loot -- {looted['food']} food, {looted['wood']} wood, and {looted['stone']} stone recovered"
            )
            self.recent_encounters.append({
                "x": x, "y": y, "kind": "raider_attack", "label": "Ambush repelled", "outcome": "repelled",
            })
            _record_combat(tribe, "Ambush", "won")
            self._relocate_raider_sighting_after_ambush(tribe, x, y)
            exp["phase"] = "returning"
            return True

        self.trauma.radiate_event_wave(x, y, config.RAIDER_SIGHTING_TRAUMA_MAGNITUDE, config.RAIDER_SIGHTING_TRAUMA_RADIUS)
        self._lose_population(tribe, config.EXPEDITION_RAIDER_AMBUSH_POPULATION_LOSS, cause="raider_ambush")
        # Explicit spec: "wandering Raids on the board [take] 75% of their
        # collected holdings and the 1 life, if they lose." A wandering
        # party's real "holdings" are whatever it's actually carrying in the
        # field -- cuts whichever of these the expedition kind happens to
        # have (food_caught only exists for a hunting party, wood/stone only
        # for an exploration party), not the tribe's home stockpile.
        for field in ("food_gathered", "water_gathered", "wood_gathered", "stone_gathered", "food_caught"):
            if field in exp:
                exp[field] = round(exp[field] * (1 - config.EXPEDITION_RAIDER_AMBUSH_LOOT_FRACTION))
        self._relocate_raider_sighting_after_ambush(tribe, x, y)
        tribe.history.append(
            f"{exp['lead_scout']}'s party was ambushed by raiders near ({x},{y}), losing most of what "
            "they'd gathered, and flees for home"
        )
        tribe.memory.remember(
            f"Raiders ambushed our party near ({x},{y}) -- real danger there.", self.cycle, weight=0.85,
        )
        # Explicit spec: "report, gravemarker" -- reuses the same "hazard_death"
        # kind (☠️) every other wandering-hazard death (volcano/river/cliffs/
        # ocean) already reports with, rather than the "raider_attack" (⚔️)
        # kind a won/repelled encounter uses -- this is a loss with a real
        # life lost, not a clash the tribe can be proud of.
        self.recent_encounters.append({
            "x": x, "y": y, "kind": "hazard_death", "label": "Lost to raiders", "outcome": "struck",
        })
        _record_combat(tribe, "Ambush", "lost")
        return True

    def _advance_hunting_party_outbound(self, tribe: Tribe, exp: dict, current_biome: str, scout: str) -> None:
        """One outbound day for a HUNTING_PARTY expedition (see actions.py._hunting_party).
        Every day out is its own roll of the same wolf-pack hazard an instant hunt
        carries, and its own chance of a catch scaled by wherever the party currently
        stands' real game yield -- a party camped on a mountain or ocean tile is no
        likelier to succeed there than an instant hunt would be. A hazard or a catch
        both end the search immediately; the catch itself still isn't real food until
        the party makes it all the way home (see _report_hunting_party_home)."""
        px, py = exp["pos"]
        # Explicit design split: volcano/cliffs/ocean/shoals are always-fatal
        # terrain -- one hit and the party heads straight home (matches the
        # general SCOUT/EXPLORATION_PARTY outbound leg's own coverage; this
        # used to only check river/volcano, missing cliffs/ocean/shoals
        # entirely). A river/lake crossing is different -- "someone dies if
        # they don't swim... if they still have time and people in the party
        # alive, they can continue on" -- so that one doesn't end the trip
        # here; its own return value is ignored, same as every other caller.
        self._expedition_river_hazard(tribe, px, py)
        if (
            self._volcano_hazard(tribe, px, py)
            or self._cliffs_hazard(tribe, px, py)
            or self._ocean_hazard(tribe, px, py)
            or self._shoals_hazard(tribe, px, py)
        ):
            exp["phase"] = "returning"
            return

        if random.random() < config.HUNT_HAZARD_CHANCE:
            self._lose_population(tribe, config.HUNT_HAZARD_POPULATION_LOSS, cause="wolf_attack")
            exp["phase"] = "returning"
            tribe.history.append(f"a wolf pack struck {scout}'s hunting party -- the survivors turn back")
            self.recent_encounters.append({
                "x": px, "y": py, "kind": "wolf_attack", "label": "Wolf pack!", "outcome": "struck",
            })
            return

        game_multiplier = BIOME_YIELD_MULTIPLIER["game"].get(current_biome, 0.0)
        # Live bug: "never landed a clean hunt? that's very intolerant." A
        # nonzero-but-tiny multiplier (cliffs/desert at 0.05) made a catch
        # nearly impossible in practice, not just harder -- floored up to a
        # real minimum. True zero-game biomes (ocean, volcano) are untouched.
        if game_multiplier > 0:
            game_multiplier = max(game_multiplier, config.HUNTING_PARTY_MIN_GAME_MULTIPLIER)
        if game_multiplier > 0 and random.random() < config.HUNTING_PARTY_CATCH_CHANCE_BASE * game_multiplier:
            exp["food_caught"] = random.randint(config.HUNTING_PARTY_CATCH_FOOD_MIN, config.HUNTING_PARTY_CATCH_FOOD_MAX)
            exp["phase"] = "returning"
            tribe.history.append(f"{scout}'s hunting party made a catch and is heading home")
            return

        # An arbitrary day-count cutoff used to end an unsuccessful hunt regardless of
        # whether there was still ground worth covering. Same fix as SCOUT: push
        # onward toward the edge of the grid the first time the party reaches its
        # declared spot with nothing caught, and only give up once it's actually run
        # out of world in that direction -- a real stopping point, not a countdown.
        px, py = exp["pos"]
        tx, ty = exp["target"]
        if [px, py] == [tx, ty]:
            if not exp.get("pushed_onward"):
                exp["pushed_onward"] = True
                ex, ey = physics.extend_ray_to_grid_edge(exp["origin"][0], exp["origin"][1], tx, ty, self.world.grid_size)
                exp["target"] = [ex, ey]
                tribe.history.append(f"{scout}'s hunting party found nothing at ({px},{py}) and pushes onward")
            else:
                exp["phase"] = "returning"
                tribe.history.append(
                    f"{scout}'s hunting party reaches the edge of the hunting grounds after {exp['day']} days "
                    "with nothing caught -- they turn back"
                )

    def _advance_exploration_party_outbound(self, tribe: Tribe, exp: dict, current_biome: str, scout: str) -> bool:
        """One outbound day for an EXPLORATION_PARTY (see actions.py.
        _exploration_party). Adds everything SCOUT's own outbound/return
        doesn't already do: real wood/stone gathered along the way (on top of
        the food/water every expedition already forages), a real
        carrying-capacity limit (not just a day count), a chance to spot a
        rival tribe's settlement, and a chance to discover a Landmark.
        Returns True if the day's outbound processing should stop here
        (already decided to head home -- carry capacity or day limit hit),
        False to fall through to the same water/terrain discovery every other
        kind shares (see the caller, _advance_one_expedition)."""
        px, py = exp["pos"]
        exp["wood_gathered"] += round(
            config.EXPLORATION_PARTY_DAILY_WOOD * BIOME_YIELD_MULTIPLIER["wood"].get(current_biome, 1.0)
        )
        exp["stone_gathered"] += round(
            config.EXPLORATION_PARTY_DAILY_STONE * BIOME_YIELD_MULTIPLIER["stone"].get(current_biome, 1.0)
        )

        carried = exp["wood_gathered"] + exp["stone_gathered"] + exp["food_gathered"] + exp["water_gathered"]
        if carried >= config.EXPLORATION_PARTY_CARRY_CAPACITY or exp["day"] >= exp["max_days"]:
            exp["phase"] = "returning"
            reason = (
                "laden with all they can carry" if carried >= config.EXPLORATION_PARTY_CARRY_CAPACITY
                else f"after {exp['day']} days out"
            )
            tribe.history.append(f"{scout}'s exploration party turns back {reason}")
            return True

        # Explicit request: "anything out there can be discovered including
        # settlements... whatever they find." A rival's settlement, spotted
        # from a real distance -- doesn't require the tight physical contact
        # DECLARE_ALLIANCE/RAID/TRADE need, just line of sight while passing by.
        for other in self.tribes.values():
            if other.id == tribe.id or other.extinct:
                continue
            if (other.x - px) ** 2 + (other.y - py) ** 2 <= config.SETTLEMENT_SIGHTING_RADIUS ** 2:
                note = f"{scout}'s exploration party spots {other.name}'s settlement in the distance"
                if not tribe.history or tribe.history[-1] != note:
                    tribe.history.append(note)
                break

        # A Landmark -- a real, persistent discovery with its own one-time
        # reward, distinct from Mine's ore (explicit request).
        if random.random() < config.LANDMARK_DISCOVERY_CHANCE and not any(
            lm["x"] == px and lm["y"] == py for lm in tribe.landmarks
        ):
            name = random.choice(config.LANDMARK_NAMES)
            resource = random.choice(config.LANDMARK_RESOURCE_NAMES)
            reward = random.randint(config.LANDMARK_REWARD_MIN, config.LANDMARK_REWARD_MAX)
            tribe.landmarks.append({"x": px, "y": py, "resource": name})
            self._capped_unique_add(tribe, resource, reward)
            self.trauma.radiate_event_wave(px, py, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
            # Explicit request: "Finding and marking Landmarks increases Fame...
            # A Landmark in your Territory gives you even more Fame."
            in_territory = self._site_in_own_territory(tribe, px, py)
            tribe.fame += config.FAME_PER_LANDMARK_IN_TERRITORY if in_territory else config.FAME_PER_LANDMARK
            # Explicit request: "When a Boat encounters a Landmark, it has a Boat
            # Party (same look-effect as Settlement Celebration) celebrating the
            # Landmark." Same shared cooldown/breeding-opportunity shape
            # _celebrate_water_discovery/_celebrate_game_discovery already use
            # for their own one-time discoveries.
            if tribe.boat_built and current_biome in config.BOAT_WATER_BIOMES:
                tribe.last_celebration_cycle = self.cycle
                tribe.history.append(
                    f"\U0001f389 {scout}'s crew holds a boat party on the water at ({px},{py}), celebrating "
                    f"the discovery of {name} -- {reward} {resource} claimed"
                )
                if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
                    pair = _eligible_breeding_pair(tribe)
                    if pair is not None:
                        parent_a, parent_b = pair
                        tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                        tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")
            else:
                tribe.history.append(
                    f"{scout}'s exploration party discovers {name} at ({px},{py}) -- {reward} {resource} claimed"
                )
        return False

    def _report_hunting_party_home(self, tribe: Tribe, exp: dict, scout: str, forage_note: str, recipient: str) -> None:
        # See actions.py._hunt_deer's own matching Tannery meat bonus comment --
        # same flat pre-multiplier addition, applied here too so a multi-day
        # hunting party's catch benefits the same way an instant hunt does.
        base_caught = exp.get("food_caught", 0)
        if base_caught and tribe.tannery_built:
            base_caught += config.TANNERY_MEAT_BONUS_PER_HUNT
        caught_gained = round(base_caught * _food_multiplier(tribe))
        if caught_gained:
            # See the matching fix/comment on the general homecoming branch above --
            # this was the same uncapped tribe.food += pattern, just in the hunt-
            # specific report path.
            caught = self._capped_add(tribe, "food", caught_gained)
            tribe.expeditions_succeeded += 1
            tribe.hunt_successes += 1
            tribe.hunt_ever_succeeded = True  # see actions.py._cook_food's own prerequisite
            if tribe.hunt_successes == config.MILESTONE_HUNT_SUCCESSES:
                self._award_trophy(tribe, "Master Hunter", individual=scout)
            self._check_custom_awards(tribe, "hunting", individual=scout)
            caught_note = f"{caught} food caught"
            if caught < caught_gained:
                caught_note += f" ({caught_gained - caught} more spoiled -- stores already full)"
            tribe.history.append(
                f"{scout}'s hunting party is home and gives {recipient} a full report: "
                f"{caught_note}, {forage_note}"
            )
        else:
            tribe.history.append(
                f"{scout}'s hunting party is home and gives {recipient} a full report: "
                f"nothing caught, though not empty-handed -- {forage_note}"
            )

    def _apply_action(self, tribe: Tribe, action: str, biome: str, target: tuple[int, int]) -> str | None:
        # action always comes from _resolve_action, which now guarantees a real,
        # currently-available action -- no IDLE fallback needed here anymore.
        return ACTION_REGISTRY[action](self, tribe, biome, target)

    def _apply_upkeep(self, tribe: Tribe) -> None:
        """Larger tribes cost more to sustain each tick. Left unpaid, someone dies --
        this is what makes hunger and thirst actual stakes rather than numbers that
        only ever go up. Cooking no longer reduces this drain -- see config.
        COOKING_FOOD_MULTIPLIER's own comment: it multiplies food production at the
        harvest point instead (actions._food_multiplier), the same shape every other
        resource-mastery building already uses.

        Explicit request: "bath house bolsters Well-Being upkeep once built" --
        a real reduction to this same per-cycle drain (config.
        BATH_HOUSE_UPKEEP_MULTIPLIER), which also directly raises wellbeing.py's
        physiological tier score since that's computed from this exact buffer."""
        upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
        if tribe.bath_house_built:
            upkeep = max(1, round(upkeep * config.BATH_HOUSE_UPKEEP_MULTIPLIER))
        tribe.food -= upkeep
        tribe.water -= upkeep

        if tribe.food < 0:
            tribe.food = 0
            self._starve(tribe)
        if tribe.water < 0:
            tribe.water = 0
            self._dehydrate(tribe)

    def _check_raider_attack(self, tribe: Tribe) -> None:
        """Trigger only -- see _resolve_raider_attack for the actual outcome.
        Explicit request: "I do want to see RAIDERs ride in over time," so a
        triggered attack no longer resolves in the same invisible instant it's
        rolled -- it starts a real, visible, multi-cycle approach instead
        (_advance_raider_approach), giving the tribe actual advance warning it can
        act on (finishing a wall) before the attack lands.

        A real, population-scaled hazard -- see config.RAIDER_HAZARD_* for the full
        rationale (deliberately not a scripted "your people are not safe" fact with
        nothing behind it -- a hardcoded HUNT_DEER directive was already reverted once
        on that exact principle). Gated behind tribe.has_ever_settled (a nomadic band
        has nothing worth raiding) and a cooldown (mirrors CELEBRATION_COOLDOWN_CYCLES)
        so this reads as discrete events, not background noise. Runs once per tribe
        per cycle regardless of the tribe's own chosen action -- a system-level event,
        the same category as _apply_upkeep."""
        if not tribe.has_ever_settled or tribe.extinct or tribe.raiders_approaching:
            return
        if tribe.raiders_repelled_by_wall:
            return
        if self.cycle - tribe.last_raider_attack_cycle < config.RAIDER_HAZARD_COOLDOWN_CYCLES:
            return

        attack_chance = min(
            config.RAIDER_HAZARD_MAX_CHANCE,
            config.RAIDER_HAZARD_MAX_CHANCE * tribe.population / config.RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE,
        )
        if random.random() >= attack_chance:
            return

        tribe.last_raider_attack_cycle = self.cycle
        angle = random.uniform(0, 2 * math.pi)
        sx = round(tribe.x + config.RAIDER_APPROACH_START_DISTANCE * math.cos(angle))
        sy = round(tribe.y + config.RAIDER_APPROACH_START_DISTANCE * math.sin(angle))
        tribe.raiders_approaching = {
            "start_x": sx, "start_y": sy, "x": sx, "y": sy,
            "cycles_left": config.RAIDER_APPROACH_CYCLES, "total_cycles": config.RAIDER_APPROACH_CYCLES,
        }
        tribe.history.append(
            f"raiders have been spotted riding in from ({sx},{sy}) -- "
            f"{config.RAIDER_APPROACH_CYCLES} cycles until they arrive"
        )

    def _advance_raider_approach(self, tribe: Tribe) -> None:
        """One cycle of an in-progress raider approach (see _check_raider_attack) --
        runs every cycle regardless of the tribe's own chosen action, the same
        category as _apply_upkeep. A real, visible countdown, not an instant
        off-screen resolve.

        Explicit request: a jagged, explorational path rather than a straight line
        -- echoes drawCelestialLighting's own arc-across-the-sky shape (an
        established visual language this project already uses for "movement over
        cycles"), but weaving instead of smooth. A perpendicular wobble tapers to
        zero as they arrive, so they still land exactly on the settlement."""
        approach = tribe.raiders_approaching
        if approach is None or tribe.extinct:
            return
        approach["cycles_left"] -= 1
        if approach["cycles_left"] <= 0:
            tribe.raiders_approaching = None
            self._resolve_raider_attack(tribe)
            return
        t = 1 - approach["cycles_left"] / approach["total_cycles"]
        dx = tribe.x - approach["start_x"]
        dy = tribe.y - approach["start_y"]
        dist = math.hypot(dx, dy) or 1.0
        perp_x, perp_y = -dy / dist, dx / dist
        wobble = math.sin(t * math.pi * 3) * (1 - t) * config.RAIDER_APPROACH_START_DISTANCE * 0.3
        approach["x"] = round(approach["start_x"] + dx * t + perp_x * wobble)
        approach["y"] = round(approach["start_y"] + dy * t + perp_y * wobble)

    def _resolve_raider_attack(self, tribe: Tribe) -> None:
        """The actual outcome, once an approach (see _check_raider_attack/
        _advance_raider_approach) finishes counting down.

        Defense is additive, not binary: population alone gives some chance to fight
        back (the same "more hands" logic actions.py._raid's own population-ratio win
        chance already uses), a wall at the tribe's own tile adds more on top, scaled
        continuously by its own construction progress (actions.py._construct_wall) --
        a half-built wall gives roughly half the bonus, not zero and not full -- and a
        river/lake tile is a natural partial barrier of its own (RAIDER_DEFENSE_WATER_
        BONUS), so a settled-near-water tribe needs less constructed wall for the same
        real protection. No wall never means automatic loss; a wall never means
        automatic safety.

        Explicit finding: raiders were being repelled too consistently -- the raiding
        force itself never scaled with what it was actually attacking, so population/
        wall bonuses alone could reliably clear the defense cap for any moderately
        developed tribe. raider_strength scales with the same population signal that
        already drives whether an attack happens at all: a bigger, wealthier tribe
        draws a genuinely stronger raiding force, which is what makes a wall (and
        water) actually matter rather than population alone being enough."""
        wall_fraction = city_layout.wall_defense_fraction(tribe)
        ring0_reinforced = bool(tribe.wall_rings) and city_layout.ring_fully_reinforced(tribe.wall_rings[0])
        raider_strength = min(1.0, tribe.population / config.RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE)

        defense_chance = max(0.0, min(
            config.RAIDER_DEFENSE_MAX_CHANCE,
            config.RAIDER_DEFENSE_BASE_CHANCE
            + (tribe.population // 10) * config.RAIDER_DEFENSE_POPULATION_BONUS_PER_10
            + config.RAIDER_DEFENSE_WALL_BONUS_AT_FULL_PROGRESS * wall_fraction
            # Defense-in-depth: each fully-reinforced ring behind the outermost one.
            + city_layout.inner_ring_defense_bonus(tribe)
            + (config.RAIDER_DEFENSE_WATER_BONUS if self._is_settled_near_water(tribe) else 0.0)
            # See actions.py._build_keep/_build_fortress/_build_castle -- each a
            # real bonus stacked on top of the wall's own, not another way to
            # reach the same ceiling faster.
            + (config.KEEP_DEFENSE_BONUS if tribe.keep_built else 0.0)
            + (config.FORTRESS_DEFENSE_BONUS if tribe.fortress_built else 0.0)
            + (config.CASTLE_DEFENSE_BONUS if tribe.castle_built else 0.0)
            # See actions.py._build_moat -- a cheaper alternative to a second
            # wall layer, not a replacement for the wall itself.
            + (config.MOAT_DEFENSE_BONUS if tribe.moat_built else 0.0)
            # Explicit request: "Torches can be a freebie for building walls 2
            # levels" -- free once a tribe has both fire and a fully reinforced
            # first wall ring, no action or cost of its own.
            + (config.TORCHES_DEFENSE_BONUS if tribe.fire_ever_built and ring0_reinforced else 0.0)
            # Object Creator era's defense_boost effect -- see actions.py.
            # _created_object_bonus.
            + _created_object_bonus(tribe, "defense_boost")
            - config.RAIDER_STRENGTH_DEFENSE_PENALTY_AT_MAX * raider_strength
        ))
        if random.random() < defense_chance:
            tribe.raids_defended += 1
            _record_combat(tribe, "Home Defense", "won")
            self._award_trophy(tribe, "Raid Breaker")
            # Live bug, confirmed against a real run: this used to setattr the
            # requested amount directly, bypassing _capped_add entirely -- since
            # the requested amount is itself a fraction of the tribe's OWN
            # current stockpile, that's not "loot recovered from the raiders" at
            # all, it's a flat compound multiplier on the tribe's own resources
            # with no ceiling. Confirmed live: a tribe's wood went 14,948 ->
            # 159,935 over ~270 cycles in exact x1.2 steps (RAIDER_DEFEAT_LOOT_
            # FRACTION=0.2 at raider_strength=1.0, i.e. every successful defense
            # once population passed RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE),
            # entirely passive -- that tribe never issued a single RAID action.
            # Routing through _capped_add closes it the same way every other
            # resource gain in this project already respects _storage_cap.
            looted = {
                resource: self._capped_add(
                    tribe, resource, round(getattr(tribe, resource) * config.RAIDER_DEFEAT_LOOT_FRACTION * raider_strength)
                )
                for resource in ("wood", "stone", "food")
            }
            self.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
            loot_note = f" -- {looted['food']} food, {looted['wood']} wood, and {looted['stone']} stone recovered from what they left behind"
            note = f"raiders were spotted approaching camp and repelled{loot_note}"
            tribe.history.append(f"{note} -- the walls held" if wall_fraction > 0 else note)
            self.recent_encounters.append({
                "x": tribe.x, "y": tribe.y, "kind": "raider_attack",
                "label": "Raiders repelled", "outcome": "repelled",
            })
            return

        loss = round(
            config.RAIDER_ATTACK_POPULATION_LOSS_UNDEFENDED
            - (config.RAIDER_ATTACK_POPULATION_LOSS_UNDEFENDED - config.RAIDER_ATTACK_POPULATION_LOSS_AT_FULL_WALL) * wall_fraction
        )
        for resource in ("wood", "stone", "food", "water"):
            stolen = round(getattr(tribe, resource) * config.RAIDER_STEAL_FRACTION)
            setattr(tribe, resource, getattr(tribe, resource) - stolen)
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_TRAUMA_MAGNITUDE, config.RAID_TRAUMA_RADIUS)
        self._lose_population(tribe, loss, cause="raider_attack")
        # Explicit request: a wall that fails to stop a raid doesn't stay standing at
        # whatever progress it had -- the tribe has to do some rebuilding, the same as
        # any other real defensive structure that gets breached. Only removed on an
        # actual failed defense, not on a successful repel (wall_fraction > 0 branch
        # above returns before reaching here).
        #
        # 2026-09-02 redesign: a breach reset the outermost ring's every real
        # (non-natural) section to 0. Live feedback: with only one ring built (the
        # common early-game case), that read as "the whole wall falls" -- overly
        # harsh. city_layout.breach_outer_ring now knocks down exactly one layer of
        # the single weakest real section in the outer ring, leaving the rest of that
        # ring and every inner ring standing.
        if wall_fraction > 0:
            city_layout.breach_outer_ring(tribe)
        tribe.history.append(
            "raiders struck the camp -- the wall blunted the worst of it, but a section was breached" if wall_fraction > 0.3 else
            "raiders struck the camp -- defenses failed, supplies stolen"
        )
        self.recent_encounters.append({
            "x": tribe.x, "y": tribe.y, "kind": "raider_attack",
            "label": "Raiders struck", "outcome": "struck",
        })
        _record_combat(tribe, "Home Defense", "lost")

    def _lose_population(self, tribe: Tribe, amount: int, cause: str = "unknown") -> None:
        """The single place population ever decreases. A tribe can now actually go
        extinct (population 0) rather than being propped up at a permanent
        population-1 floor -- extinction is marked, announced, and radiates a much
        larger trauma event than an ordinary death. `cause` feeds the scoreboard
        record (backend/scoreboard.py) so a benchmark can distinguish "starved" from
        "lost a raid," not just "died."

        Any loss also carries a chance of claiming the chief specifically, once a
        tribe survives it -- a chief was previously permanent flavor text no matter
        what happened to the people underneath them. A tribe that survives a chief's
        death gets a genuine leadership vacuum until Simulation.step() runs a fresh
        succession contest, not a name that just silently stays put forever.

        While self.cycle <= self.immortality_cycles (see Simulation.__init__), the
        actual population change and extinction are suppressed -- the hazard/
        starvation/raid event that called this still happened (its history line,
        trauma wave, and chief-death roll below all still fire normally), only the
        body count is held back. Chief succession keeps happening during immunity on
        purpose: that's real texture (lineage, trophies, a fresh philosophy), not the
        extinction this mode exists to defer."""
        if tribe.extinct:
            return
        immune = self.cycle <= self.immortality_cycles
        if not immune:
            tribe.population = max(0, tribe.population - amount)
            if tribe.population == 0:
                tribe.extinct = True
                tribe.extinction_cause = cause
                tribe.history.append(f"{tribe.name} has gone extinct.")
                self.trauma.radiate_event_wave(
                    tribe.x, tribe.y, config.EXTINCTION_TRAUMA_MAGNITUDE, config.EXTINCTION_TRAUMA_RADIUS
                )
                record_tribe_result(tribe, cause=cause, cycles_survived=self.cycle)
                return
        if tribe.chief_name and random.random() < config.CHIEF_DEATH_CHANCE_ON_LOSS:
            fallen = tribe.chief_name
            tribe.chief_deaths += 1
            tribe.chief_name = ""
            # Regression: the fallen chief's philosophy and decree used to just vanish
            # here, with nothing carried into the next election -- a live-run complaint
            # ("new chiefs aren't inheriting the old chief's knowledge"). _install_chief
            # already has a real mechanism for exactly this (pending_chief_context, see
            # _merge_tribes' conquest case) -- it was just never wired up for an
            # ordinary chief death. This doesn't force continuity: the next election is
            # simply told what the predecessor believed and decreed, and decides for
            # itself whether to keep it, adapt it, or break from it entirely.
            legacy = f'governed by this guiding philosophy: "{tribe.chief_philosophy}"' if tribe.chief_philosophy else "left no clear guiding philosophy behind"
            decree_note = f' Their standing decree was: "{tribe.chief_decree}".' if tribe.chief_decree else ""
            tribe.pending_chief_context = (
                f"The previous chief, {fallen}, has just died, having {legacy}.{decree_note} "
                "The new chief inherits this legacy and may choose to continue it, adapt it, "
                "or break from it entirely -- that judgment is theirs to make."
            )
            tribe.chief_philosophy = ""
            tribe.chief_decree = ""
            tribe.chief_victory = ""
            tribe.history.append(f"Chief {fallen} has died. {tribe.name} is left without a leader.")

    def _starve(self, tribe: Tribe) -> None:
        tribe.history.append("starvation claimed lives")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.STARVATION_TRAUMA_MAGNITUDE, config.STARVATION_TRAUMA_RADIUS
        )
        self._lose_population(tribe, _scaled_population_loss(tribe), cause="starvation")

    def _dehydrate(self, tribe: Tribe) -> None:
        tribe.history.append("thirst claimed lives")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DEHYDRATION_TRAUMA_MAGNITUDE, config.DEHYDRATION_TRAUMA_RADIUS
        )
        self._lose_population(tribe, _scaled_population_loss(tribe), cause="thirst")

    def _grow_population(self, tribe: Tribe) -> None:
        """Explicit request: "much bigger populations going to war" without the
        real-time cost a flat +1/cycle would impose at that scale (confirmed
        against real run data: every model tested grew at ~1/cycle, no
        exceptions -- see config.POPULATION_GROWTH_SCALE_DIVISOR's own comment).
        Growth is population-scaled (a real demographic curve, not a raised
        flat rate) times a Well-Being factor -- a thriving tribe grows faster
        than a merely large one, not just a bigger one.

        Live bug, confirmed against a real run: an earlier version averaged all
        five Maslow tiers (wellbeing.TIER_LABELS) into that factor, floored so
        growth could slow but never truly stop. esteem/self_actualization
        (trophies, era progress) have nothing to do with whether a tribe can
        feed more mouths, and kept propping the average up even while
        physiological (the actual food/water tier) sat at 0.0 -- a tribe in
        total, sustained famine still grew at ~90% of full speed, because 1.0
        esteem + 0.24 self_actualization masked a dead physiological score in
        the average. Compounding on a population-proportional base that can
        only ever slow down, never reach zero, is unbounded by construction --
        confirmed live: population 129 -> 10,835 in ~110 cycles while food
        never recovered. Keyed on physiological alone now (the tier that
        actually measures food/water security -- wellbeing.compute_wellbeing's
        own buffer_cycles formula), with no floor: a real, sustained famine can
        and must bring growth to an honest zero, the same way it already can
        for every other food-gated system here. A well-fed tribe still grows
        faster than the old flat rate ever did (POPULATION_GROWTH_WELLBEING_
        MAX_MULTIPLIER > 1), it just isn't propped up by unrelated achievements
        anymore. tribe.wellbeing is whatever _prepare_turn last computed
        (compute_wellbeing runs every turn already); an empty dict (the very
        first cycle, before any turn has run yet) reads as a neutral 0.5
        rather than crashing or silently zeroing growth out before the
        simulation has even really started."""
        if tribe.food > config.POPULATION_GROWTH_FOOD_THRESHOLD and tribe.population < config.POPULATION_GROWTH_CAP:
            base_growth = max(1, tribe.population // config.POPULATION_GROWTH_SCALE_DIVISOR)
            physiological = tribe.wellbeing.get("tiers", {}).get("physiological", 0.5)
            growth = round(base_growth * physiological * config.POPULATION_GROWTH_WELLBEING_MAX_MULTIPLIER)
            if growth > 0:
                tribe.population += growth
                tribe.food -= min(tribe.food, config.POPULATION_GROWTH_FOOD_COST * growth)
        tribe.max_population = max(tribe.max_population, tribe.population)

    def _advance_era_if_ready(self, tribe: Tribe) -> None:
        nxt = next_era(tribe.era)
        if nxt is None:
            return
        # RESEARCH's real payoff (actions.py._research/config.
        # INNOVATION_ERA_DISCOUNT_PER_RESEARCH): every completed research permanently
        # shaves a little off the next era's own thresholds and cost, capped so
        # advancement is never free. Recomputed fresh against next_era() each check --
        # not baked into eras.py's own numbers -- so it always reflects research done
        # since the tribe's last advancement, not just at the moment of this one.
        discount = min(
            config.INNOVATION_ERA_DISCOUNT_CAP,
            tribe.research_completed * config.INNOVATION_ERA_DISCOUNT_PER_RESEARCH,
        )
        if tribe.population < round(nxt.requires_population * (1 - discount)):
            return
        for resource, minimum in nxt.requires_resources.items():
            if _era_resource_amount(tribe, resource) < round(minimum * (1 - discount)):
                return

        for resource, amount in nxt.advancement_cost.items():
            discounted = round(amount * (1 - discount))
            _spend_era_resource(tribe, resource, discounted)
        tribe.era = nxt.key
        tribe.history.append(nxt.announcement.format(tribe=tribe.name))
        if nxt.founds_city:
            # Live bug report: a tribe reached Monolithic Era and grew a full,
            # maxed-out city (6 buildings) having never built a single Long House --
            # city founding and the housing ladder were two completely disconnected
            # progression tracks. _advance_city_founding (below) does the real gating
            # now; this just marks the milestone reached, same as every other
            # Sawmill/Quarry/Mine/Tannery gate already requires long_houses_built > 0.
            tribe.city_founding_eligible = True
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.ERA_ADVANCE_PRIDE_MAGNITUDE, config.ERA_ADVANCE_PRIDE_RADIUS
        )

    def _capped_add(self, tribe: Tribe, resource: str, amount: int) -> int:
        """Live-run correction (2026-09-02): the storage cap actions.py._add_capped
        enforces on manual gathering was silently not applied to any of the
        *passive* per-cycle income below (settled water, fish, farm harvest) --
        exactly the source of a tribe's real runaway stockpile (540 water on a
        live run), since passive income usually dwarfs anything a manual GATHER_*
        action adds. Same generous per-resource ceiling (_storage_cap), just
        enforced here too. Returns the amount actually added, since a caller may
        need to report the real number, not the nominal one."""
        cap = _storage_cap(tribe)
        current = getattr(tribe, resource)
        added = max(0, min(amount, cap - current))
        setattr(tribe, resource, current + added)
        return added

    def _capped_unique_add(self, tribe: Tribe, resource_name: str, amount: int) -> int:
        """Same as _capped_add, for the tribe.unique_resources dict (Mine ore,
        Tannery Fur) -- each named resource gets its own cap ceiling, same as
        every other resource. Returns the amount actually added, same reason
        _capped_add does -- a caller reporting what a trade/action actually
        delivered needs the real number, not the nominal one."""
        cap = _storage_cap(tribe)
        current = tribe.unique_resources.get(resource_name, 0)
        added = max(0, min(amount, cap - current))
        tribe.unique_resources[resource_name] = current + added
        return added

    def _advance_automatic_fire(self, tribe: Tribe) -> None:
        """Explicit request, after a live run showed a tribe sitting on 900 idle
        wood that never once chose BUILD_FIRE in 728 cycles: 'at least make hunting
        or gathering the gate to the automatic fire.' Fire has essentially no real
        downside and unlocks cooking (a real food multiplier) -- guaranteed now,
        the same way settled water/fish supply are already passive systems rather
        than something a model has to remember to keep choosing. The only thing
        that still requires a real, proven achievement is which door unlocks it (a
        successful hunt or a successful forage, either one), not the ignition
        itself -- free, no wood cost, since the whole point is removing this as a
        point of failure. BUILD_FIRE itself is unaffected -- still there as a
        faster manual path, and already retired from available_actions the moment
        fire_ever_built is set, by either route."""
        if tribe.fire_ever_built or not (tribe.hunt_ever_succeeded or tribe.foraged_ever_succeeded):
            return
        self.world.add_construction(tribe.x, tribe.y, "fire", self.cycle)
        tribe.fire_ever_built = True
        if tribe.territory_center is not None:
            slot = architect.find_free_slot(self.world, tribe, "fire")
            if slot is not None:
                architect.record_building(tribe, "fire", slot[0], slot[1], 1, 1, self.cycle)
        tribe.history.append(f"{tribe.name} discovers fire -- cooking is within reach now")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.BUILD_FIRE_PRIDE_MAGNITUDE, config.BUILD_FIRE_PRIDE_RADIUS
        )

    def _advance_automatic_boat(self, tribe: Tribe) -> None:
        """Explicit follow-up: "if they build a Dock, they can get a Boat" --
        automatic once real, the same shape _advance_automatic_fire already
        uses (a Dock plus mastered fishing is proof enough a Boat makes sense,
        no separate action/cost needed). "Give the boat mobility in the clean
        water, not the sea" -- grants real speed through river/lake tiles
        (config.BOAT_WATER_BIOMES, backend/physics.py.terrain_aware_step) for
        every future RELOCATE/expedition; ocean stays exactly as impassable as
        ever, deliberately not an ocean-crossing mechanic."""
        if tribe.boat_built or not (tribe.dock_built and tribe.fishing_learned):
            return
        tribe.boat_built = True
        if tribe.territory_center is not None:
            w, h = config.BUILDING_FOOTPRINTS["boat"]
            slot = architect.find_free_slot(self.world, tribe, "boat")
            if slot is not None:
                architect.record_building(tribe, "boat", slot[0], slot[1], w, h, self.cycle)
        tribe.history.append(f"{tribe.name} builds a boat -- the river is a highway now, not an obstacle")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS
        )

    def _advance_wall_security(self, tribe: Tribe) -> None:
        """Explicit request: "once the Wall is complete all Raiders are kicked
        out of the area or absorbed. either is fine by me." A one-time
        transition the first cycle the first wall ring is genuinely finished
        (city_layout.ring_fully_built) -- any raider camps already known
        nearby are cleared (driven off/absorbed either reading is consistent
        with wiping the list), and _check_raider_attack's own matching guard
        means this tribe is never raided again from here on."""
        if tribe.raiders_repelled_by_wall or not tribe.wall_rings:
            return
        if not city_layout.ring_fully_built(tribe.wall_rings[0]):
            return
        tribe.raiders_repelled_by_wall = True
        tribe.raider_sightings = []
        tribe.raiders_approaching = None
        tribe.history.append(f"{tribe.name}'s completed wall drives every raider from the area for good")
        self._celebrate_wall_complete(tribe)

    def _celebrate_wall_complete(self, tribe: Tribe) -> None:
        """Live report: "one wall should be 100% and celebrate... I don't think it
        fell together" -- finishing the first wall ring only ever got a plain
        history line (the raiders-driven-out one above), never the real "🎉 X is
        celebrating" banner every other named milestone gets (settling, a
        harvest, learning to fish). Same shape as _celebrate_settling/
        _celebrate_harvest -- called from _advance_wall_security, the one place
        that already fires exactly once when the ring first finishes."""
        tribe.last_celebration_cycle = self.cycle
        spent = _celebration_cost(tribe)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        tribe.fame += config.FAME_PER_CELEBRATION
        tribe.history.append(
            f"\U0001f389 {tribe.name} celebrates the wall's completion, spending {spent} food on a {_feast_word(tribe)}{_celebration_shout(tribe)}"
        )
        self._award_trophy(tribe, "Wall Warden")
        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

    def _wear_trail_for_expedition(self, exp: dict, tribe: Tribe, x: int, y: int) -> None:
        """World.wear_trail's own `crossings` counter (config.ROAD_EVOLVE_
        CROSSINGS/_check_road_evolution) increments once per call with no
        dedup at all -- fine for "a different traveler passed through," wrong
        for "the same expedition already wore this exact tile earlier in this
        same trip." Explicit correction: "they do not get more than 1 wear
        per move on their way, if they double back, it does not count as
        another wearing down." A party that bounces off a boxed-in edge, or a
        pushed-onward leg that re-crosses its own earlier ground, shouldn't
        rack up crossings faster than genuinely distinct travelers would --
        that's what let a route no other tribe had used yet as heavily as
        the crossing count implied evolve into a toll road on the strength of
        one party's own back-and-forth. exp["worn_tiles"] is this one
        expedition's own scoped memory (kept off exp["path"], which is capped
        for wire/storage size and would silently stop tracking on a very long
        trip) -- a fresh expedition starts with none, and a genuinely later,
        separate trip (even by the same scout) wears normally again."""
        worn = exp.setdefault("worn_tiles", set())
        key = (x, y)
        if key in worn:
            return
        worn.add(key)
        self.world.wear_trail(x, y, config.TRAIL_WEAR_PER_PASS, tribe.color, tribe.id)
        self._check_road_evolution(x, y)

    def _check_road_evolution(self, x: int, y: int) -> None:
        """Explicit request: "a road will be similar [to the wall] but a big
        episode of major development making trade far more likely and travel
        easier." A trail silently flips World.is_toll_road the instant its
        lifetime crossings pass config.ROAD_EVOLVE_CROSSINGS -- this catches the
        exact cycle that happens (crossings == threshold + 1, i.e. the crossing
        that just tipped it over) and turns it into a real, one-time celebration
        for whichever tribe actually owns this tile (World.road_owner -- the
        first tribe to ever walk it, not necessarily whoever's walking it right
        now that pushed it over). Called after every World.wear_trail site.

        Live report: a heavily-used, long route can have many tiles' crossing
        counts cross ROAD_EVOLVE_CROSSINGS within the same cycle (a full
        expedition leg wears every tile it passes through in one go -- see
        World.wear_trail's own callers) -- confirmed against real run data,
        ~40 separate tiles evolving in a single cycle, each its own full
        _celebrate_road_complete call. _celebrate_road_complete's cost is a
        FRACTION of current food (_celebration_cost), so any one of those looks
        harmless alone, but ~40 in the same cycle compounds into spending
        nearly the whole stockpile in one tick regardless of how much passive
        income (fishing, farming, whatever) the tribe actually has -- the same
        "uncapped feast on every single one would drain food faster than
        [income] produces it" risk _advance_farming's own harvest-celebration
        call already guards against with this identical cooldown check, that
        this one was simply never given. The road still evolves (World.trails'
        own state, above) and every OTHER tile that crossed this same cycle
        still becomes a real toll road -- only the repeated narrated feast is
        throttled to one per cooldown window, not the underlying mechanic."""
        entry = self.world.trails.get((x, y))
        if entry is None or entry.get("crossings", 0) != config.ROAD_EVOLVE_CROSSINGS + 1:
            return
        owner_id = entry.get("owner")
        owner = self.tribes.get(owner_id) if owner_id else None
        if owner is None or owner.extinct:
            return
        if self.cycle - owner.last_celebration_cycle < config.CELEBRATION_COOLDOWN_CYCLES:
            return
        self._celebrate_road_complete(owner, x, y)

    def _celebrate_road_complete(self, tribe: Tribe, x: int, y: int) -> None:
        """Same real "🎉 celebrates" treatment _celebrate_wall_complete gets --
        a road tribes actually built through repeated use, not a stat flip
        no one notices. The road's own real effects (World.trail_speed_bonus/
        TOLL_ROAD_SPEED_MULTIPLIER for travel, toll fees in _resolve_toll for
        trade) are unchanged; this is the moment they become real, not new
        mechanics on top of them."""
        tribe.last_celebration_cycle = self.cycle
        tribe.toll_roads_completed += 1
        spent = _celebration_cost(tribe)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        tribe.fame += config.FAME_PER_CELEBRATION
        tribe.history.append(
            f"\U0001f389 {tribe.name} celebrates a real road taking shape at ({x},{y}) -- trade and travel "
            f"flow easier from here on, spending {spent} food on a {_feast_word(tribe)}{_celebration_shout(tribe)}"
        )
        self._award_trophy(tribe, "Road Warden")
        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

    def _advance_water_supply(self, tribe: Tribe) -> None:
        """Explicit request: "like relocate, gather water becomes irrelevant once they
        have settled." A tribe genuinely settled next to real water shouldn't need to
        keep manually choosing GATHER_WATER every cycle just to stand still -- the
        same "passive consequence, not a discrete action" category as crop growth.
        GATHER_WATER still works and still adds more on top of this.

        Live bug ("water is a problem and it should never be after they settle"):
        a settled tribe with one active farm plot showed 32 straight "thirst
        claimed lives" events while its own displayed water sat at a flat, stable-
        looking 2 the whole time. Two compounding causes -- this formula only ever
        margined against upkeep, blind to _advance_farming's own real water draw
        (config.CROP_WATER_PER_PLOT_PER_CYCLE per plot), so a farmed settlement's
        true total draw quietly exceeded the "safe" 1.5x supply once a plot was
        planted; and _apply_upkeep ran before this in Simulation.step, draining a
        thin carried-over buffer negative before this cycle's own income had even
        landed, triggering a real population-loss event the same cycle's end-of-
        turn number never showed. Folding the farm draw into the margin here fixes
        the sustained deficit; Simulation.step now runs this (and fish supply and
        farming) before upkeep so a cycle's own income can actually cover that same
        cycle's own drain instead of only the next one's.

        Explicit request (2026-09-08): "every water source a Tribe finds, adds to
        the passive Water income, so they should just get an Infinity sign for
        water once they find 2 or 3." confirmed_water_sites was never actually
        wired into this formula -- a second or third confirmed source genuinely
        changed nothing, confirmed against a real day-12 live run where water
        declined for 100+ straight cycles despite several confirmed sources on
        record. Now it does: config.WATER_SECURITY_SITE_THRESHOLD distinct
        confirmed sources takes water off the management board for good, the
        same permanent-mastery shape fishing_learned/cooking_learned already
        give their own resource -- unconditional on current position (a tribe
        that's proven it knows where the water is doesn't lose that knowledge
        by walking away from any one source), and naturally one-way since
        confirmed_water_sites only ever grows."""
        if _is_water_secure(tribe):
            tribe.water = _storage_cap(tribe)
            return
        if self._is_settled_near_water(tribe):
            upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
            farm_draw = config.CROP_WATER_PER_PLOT_PER_CYCLE * tribe.farm_plots
            well_bonus = config.WELL_SUPPLY_BONUS_MULTIPLIER if tribe.well_built else 1.0
            self._capped_add(
                tribe, "water", round((upkeep + farm_draw) * config.SETTLED_WATER_SUPPLY_MULTIPLIER * well_bonus)
            )

    def _advance_food_supply(self, tribe: Tribe) -> None:
        """Food's counterpart to _advance_water_supply's water-security branch above
        -- see that one's own docstring for the live report this pattern is built
        from. Explicit request: "let them have Infinity if they Build a Kitchen and
        have either a Fishery or a Farm." A Kitchen alone already multiplies every
        future forage/hunt/catch nine-fold (see BUILD_KITCHEN's own description) --
        paired with a genuinely proven, passive food source (a Fishery's steady
        daily catch, or at least one real harvest ever actually brought in), that's
        real, permanent food mastery, not just a lucky stockpile. tribe.
        last_harvest_cycle (not the live farm_plots count) is the proof: a plot can
        wither and later regrow, but "this tribe has successfully farmed before"
        never un-happens -- the same permanent-proof shape _prepare_turn's own
        diversification_note already uses for has_farm. kitchen_built/fishery_built
        are themselves already permanent (nothing in this project ever un-builds a
        structure), so the whole condition only ever turns on, never off."""
        if _is_food_secure(tribe):
            tribe.food = _storage_cap(tribe)

    def _advance_wood_supply(self, tribe: Tribe) -> None:
        """Wood's own version of the "real achievement, real security" idea
        food/water just got -- explicit request: "if they Build a Sawmill...
        and have discovered and are using a Timber Grove to get wood, they can
        have the treatment." Deliberately softer than food/water's own "always
        topped to the storage cap," by explicit design choice: unlike food/
        water, wood has no automatic per-cycle drain to guard against -- nothing
        dies of a wood shortage -- so topping it to the cap every cycle
        regardless of spending would make every future building free forever, a
        much bigger change than "never starve." A real, generous passive income
        instead, the same shape config.MINE_YIELD_PER_CYCLE/
        TANNERY_YIELD_PER_CYCLE already use for their own resource, just at
        wood's own larger scale -- genuinely solves the "wood starved at scale"
        problem this whole fix is a response to, without erasing the building
        economy outright.

        tribe.lumber_site (singular, set the moment BUILD_SAWMILL succeeds with
        at least one lumber_sites entry already known -- see
        actions._build_sawmill) is exactly "discovered and using a Timber
        Grove": the real site Simulation._advance_resource_trails already wears
        a path to."""
        if _is_wood_secure(tribe):
            self._capped_add(tribe, "wood", config.WOOD_SECURITY_DAILY_INCOME)

    def _advance_stone_supply(self, tribe: Tribe) -> None:
        """Stone's own version of _advance_wood_supply above -- same reasoning,
        same shape, but a higher bar by explicit request: real stone mastery
        needs a Quarry AND a Mine, not a Quarry alone. See _is_stone_secure's
        own docstring for why that's still one real, earned achievement and
        not an arbitrary extra hurdle."""
        if _is_stone_secure(tribe):
            self._capped_add(tribe, "stone", config.STONE_SECURITY_DAILY_INCOME)

    def _advance_battalion_patrol(self, tribe: Tribe) -> None:
        """Military branch step 4 (plan file valiant-forging-falcon.md) -- explicit
        request: "start implementing the autonomous patrol behavior... I do want to
        add it as a visual player on the board not just a passthru I won't see
        immediately." A trained Battalion (tribe.battalion_size > 0) that isn't
        currently out and isn't on cooldown launches a patrol on its own -- no chief
        action required, matching the original brainstorm ("they will also actively
        patrol and take on Raiders"). tribe.battalion_patrol is a real, moving dict
        (not tribe.expeditions -- a Battalion is never chief-dispatched and must
        never compete with SCOUT/HUNTING_PARTY for expedition_capacity's own limited
        slots), so the frontend can render it like any other on-map entity.

        Explicit follow-up: "they can patrol for a set number of cycles, then go
        back to training. Cooldown for 3 whole days" -- BATTALION_PATROL_DURATION_DAYS
        out, then home, then BATTALION_PATROL_COOLDOWN_DAYS resting before the next
        patrol can launch. "cooldown on patrol, training has its own controls" --
        this cooldown only ever blocks a new patrol from *starting*; TRAIN_BATTALION
        is untouched by it.

        Scoped to raiders only ("Other Threats is just raiders, keep it simple for
        now") -- minor settlements have no per-tribe discovery tracking to check
        against (see actions.py._find_minor_settlement), so autonomous raiding of
        those is explicitly out of this step. Reuses ACTION_REGISTRY[
        "STRIKE_RAIDER_CAMP"] directly once the patrol reaches a known
        raider_sightings location rather than duplicating that action's own
        win-chance/loot logic -- unlike a chief-issued action it doesn't append its
        own tribe.history line, so this method appends one either way."""
        if tribe.battalion_size <= 0 or tribe.territory_center is None:
            return

        patrol = tribe.battalion_patrol
        if patrol is None:
            if self.cycle < tribe.battalion_cooldown_until_cycle:
                return
            cx, cy = tribe.territory_center
            patrol = {"pos": [cx, cy], "phase": "patrolling", "started_cycle": self.cycle, "target": None}
            tribe.battalion_patrol = patrol
            tribe.history.append(f"{tribe.name}'s Battalion of {tribe.battalion_size} marches out on patrol")

        duration_cycles = config.BATTALION_PATROL_DURATION_DAYS * config.DAY_LENGTH_CYCLES
        if patrol["phase"] == "patrolling" and self.cycle - patrol["started_cycle"] >= duration_cycles:
            patrol["phase"] = "returning"
            patrol["target"] = None

        px, py = patrol["pos"]
        cx, cy = tribe.territory_center

        if patrol["phase"] == "returning":
            nx, ny = physics.terrain_aware_step(px, py, cx, cy, base_speed=config.BATTALION_PATROL_SPEED)
            patrol["pos"] = [nx, ny]
            if (nx, ny) == (cx, cy):
                tribe.battalion_patrol = None
                tribe.battalion_cooldown_until_cycle = (
                    self.cycle + config.BATTALION_PATROL_COOLDOWN_DAYS * config.DAY_LENGTH_CYCLES
                )
                tribe.history.append(f"{tribe.name}'s Battalion returns from patrol to train again")
            return

        target = patrol.get("target")
        if target is None and tribe.raider_sightings:
            target = min(tribe.raider_sightings, key=lambda s: (s[0] - px) ** 2 + (s[1] - py) ** 2)
            patrol["target"] = list(target)

        if target is None:
            return

        tx, ty = target
        if (px, py) == (tx, ty):
            if tuple(target) in tribe.raider_sightings:
                result = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](self, tribe, biome_at(px, py), target)
                if result:
                    tribe.history.append(f"{tribe.name}'s Battalion patrol {result}")
            patrol["target"] = None
        else:
            nx, ny = physics.terrain_aware_step(px, py, tx, ty, base_speed=config.BATTALION_PATROL_SPEED)
            patrol["pos"] = [nx, ny]

    def _advance_battalion_readiness_upkeep(self, tribe: Tribe) -> None:
        """Military branch, step 5 (Might's Training factor, compute_might) --
        explicit request: "not overpowered, more like bolster and upkeep."
        The bolster half lives in actions._train_battalion (called by the
        Chief); this is the upkeep half -- a small, steady drain every
        cycle regardless of whether the Battalion is out on patrol or at
        home, the same "you have to keep paying into this" shape
        BATTALION_PATROL's own cooldown already gives the patrol side of
        this branch. Only drains once there's an actual Battalion to
        neglect -- readiness starts and stays at 0.0 otherwise (see Tribe.
        __init__), so this would be a no-op either way, but the check keeps
        the intent explicit."""
        if tribe.battalion_size <= 0 or tribe.battalion_readiness <= 0.0:
            return
        tribe.battalion_readiness = max(0.0, tribe.battalion_readiness - config.BATTALION_READINESS_DECAY_PER_CYCLE)

    def _advance_fish_supply(self, tribe: Tribe) -> None:
        """Once fishing is learned (the first successful CATCH_FISH), food flows in
        daily the same way water already does once settled -- not a second knowledge
        subsystem, just the same "action unlocks a passive system" shape applied to a
        different resource. Explicit request: "they don't need to CATCH_FISH once
        they know how" -- unlike GATHER_WATER (which stays available as a manual
        top-up on top of the passive water supply), CATCH_FISH itself retires from
        available_actions the moment fishing_learned is set (see _prepare_turn) --
        this passive flow is the only source of fish food from then on. Gated on
        the same general settled check CATCH_FISH's own availability used, not the
        stricter settled_near_water -- explicit correction that the extra
        water-adjacency distinction was bogus."""
        if tribe.fishing_learned and self._is_camped(tribe):
            upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
            fishery_bonus = config.FISHERY_SUPPLY_BONUS_MULTIPLIER if tribe.fishery_built else 1.0
            dock_bonus = (1 + config.DOCK_FISH_CATCH_BONUS_FRACTION) if tribe.dock_built else 1.0
            amount = round(
                upkeep * config.FISHING_SUPPLY_MULTIPLIER * fishery_bonus * dock_bonus * _food_multiplier(tribe)
            )
            self._capped_add(tribe, "food", amount)

    def _advance_mine_yield(self, tribe: Tribe) -> None:
        """Once a mine is excavated (actions.py._build_mine) AND its ore has
        actually been fetched at least once (actions.py._gather_ore), its named
        unique resource flows in daily -- same passive "action unlocks a
        system" shape _advance_fish_supply already uses, gated on the same
        general settled check. Explicit correction: excavating the mine alone
        used to start this immediately -- "they do not harvest on a Discovery,
        so they have to fetch it once" first."""
        if tribe.mine_built and tribe.ore_ever_gathered and tribe.mine_resource_name and self._is_camped(tribe):
            self._capped_unique_add(tribe, tribe.mine_resource_name, config.MINE_YIELD_PER_CYCLE)

    def _advance_in_territory_site_yields(self, tribe: Tribe) -> None:
        """Explicit request: "if they are lucky enough to have a resource or
        anything in the territory they settle in, it's a daily allotted freebie
        they never have to gather from." Once a discovered lumber/wildlife/quarry
        site falls inside the tribe's own territory_radius, it starts
        contributing a small passive daily trickle -- same "action unlocks a
        passive system" shape _advance_water_supply/_advance_fish_supply/
        _advance_mine_yield already use for water/fish/ore. mine_sites isn't
        repeated here -- a tribe's own excavated Mine (_advance_mine_yield, just
        above) already covers that resource once built and fetched.

        Live follow-up ("GATHER_FOOD is the 'territory collector'... they go to
        each site in the territory and collect"): this used to pay the same
        flat per_site amount whether one site or five fell inside the
        territory -- a boolean "any," not a real running total. Now sums per_site
        across every in-territory site of each kind, so a tribe's own growing,
        elastic list of discoveries actually shows up in what comes home each
        day, not just whether the list is non-empty."""
        if tribe.territory_center is None:
            return
        upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
        per_site = max(1, round(upkeep * config.IN_TERRITORY_SITE_YIELD_MULTIPLIER))
        lumber_count = sum(1 for x, y in tribe.lumber_sites if self._site_in_own_territory(tribe, x, y))
        wildlife_count = sum(1 for s in tribe.wildlife_sites if self._site_in_own_territory(tribe, s["x"], s["y"]))
        quarry_count = sum(1 for x, y in tribe.quarry_sites if self._site_in_own_territory(tribe, x, y))
        if lumber_count:
            self._capped_add(tribe, "wood", per_site * lumber_count)
        if wildlife_count:
            self._capped_add(tribe, "food", per_site * wildlife_count)
        if quarry_count:
            self._capped_add(tribe, "stone", per_site * quarry_count)

    def _advance_tannery_yield(self, tribe: Tribe) -> None:
        """Once a tannery is built (actions.py._build_tannery), Fur flows in
        daily -- mirrors _advance_mine_yield exactly, into the same
        unique_resources dict."""
        if tribe.tannery_built and self._is_camped(tribe):
            self._capped_unique_add(tribe, "Fur", config.TANNERY_YIELD_PER_CYCLE)

    def _advance_resource_trails(self, tribe: Tribe) -> None:
        """Explicit request: "if they have found a Quarry, Mine, Stand of Trees
        to Harvest, these are collectables that must be fetched and so
        trails/roads to them should be established naturally." None of
        Sawmill/Quarry/Mine involve a discrete fetch action the model chooses
        (see their own docstrings -- built at the settlement, working
        passively from then on), so nothing else would ever wear a path to the
        real site each one actually draws from. This does that automatically,
        the same wear_trail mechanic RELOCATE/SCOUT already use along the
        straight line between the settlement and each site -- heavily-used
        routes eventually evolve into real toll roads themselves (see world.
        is_toll_road), exactly like any other trail would.

        Live bug, confirmed against a real run: this used to wear every tile
        on the route every single cycle, unconditionally -- since the route
        never changes, every one of its tiles got incremented in perfect
        lockstep forever, so they all crossed ROAD_EVOLVE_CROSSINGS on the
        exact same cycle (~40 tiles evolving, and celebrating, at once).
        "Once per real day is good enough" -- gated the same way a settled
        scout's own movement already is (self.cycle % DAY_LENGTH_CYCLES), so
        a route still wears in gradually over a site's operational life (the
        original intent) without every tile marching in identical lockstep
        every single cycle."""
        if self.cycle % config.DAY_LENGTH_CYCLES != 0:
            return
        for site in (tribe.lumber_site, tribe.quarry_site, tribe.mine_site, tribe.tannery_site):
            if site is None:
                continue
            for px, py in _interpolated_path(tribe.x, tribe.y, site[0], site[1]):
                self.world.wear_trail(px, py, config.TRAIL_WEAR_PER_PASS, tribe.color, tribe.id)
                mark_visited_sector(tribe, px, py)
                self._check_road_evolution(px, py)

    def _advance_farming(self, tribe: Tribe) -> None:
        """A planted crop plot (actions.py._plant_crop) grows on its own every cycle,
        the same "passive consequence, not a discrete action" category as upkeep and
        population growth -- once planted, tending it isn't something the model has to
        keep choosing to do. Harvest fires automatically on maturity.

        Real stakes, not a free one-way counter: growth needs water the same way a
        person does. Enough on hand and the plot grows and drinks its share; too little
        and the plot withers on the vine (lost outright) instead of quietly stalling --
        a real cost for neglecting a farm during a water crisis, mirroring the flock's
        own feed-or-shrink stakes in _advance_flock."""
        if tribe.farm_plots <= 0:
            return
        water_needed = config.CROP_WATER_PER_PLOT_PER_CYCLE * tribe.farm_plots
        if tribe.water < water_needed:
            tribe.farm_plots -= 1
            tribe.crop_growth = 0
            tribe.history.append("a farm plot withers for lack of water")
            return
        tribe.water -= water_needed
        # Explicit request: fish fertilizer -- once fishing is learned, whatever a
        # tribe already does with its catch (guts, scraps, the parts that aren't
        # eaten) is assumed to go back into the soil, roughly halving the time a plot
        # takes to mature. Tied to fishing_learned rather than a separate fertilizer
        # resource/action -- the same "one flag, no new subsystem" shape the rest of
        # fishing already uses.
        growth = config.CROP_GROWTH_PER_CYCLE
        if tribe.fishing_learned:
            growth *= config.FISH_FERTILIZER_GROWTH_MULTIPLIER
        tribe.crop_growth += growth
        if tribe.crop_growth >= 100:
            tribe.crop_growth = 0
            # Explicit finding: a harvest used to be a flat CROP_HARVEST_YIELD per
            # plot regardless of population -- every other resource-producing
            # mechanic (_harvest, staged wall construction) already scales with
            # _labor_multiplier ("more hands get more done"), but farming never did.
            # For any tribe past starting size, a single GATHER_FOOD action could
            # already out-yield an entire ~10-cycle farming cycle, which is a real
            # economic reason to never bother planting, independent of any framing
            # bias. More hands to bring in a harvest should mean more harvested.
            #
            # Explicit correction: "we have scaled the Farm a bit hard" -- unlike a
            # gather action (throttled by the model actually choosing it), a
            # harvest fires automatically and unconditionally, so the same
            # uncapped _labor_multiplier that's fine for GATHER_FOOD let a large,
            # late-game population's harvest run into the thousands against a
            # few-hundred-food storage cap. See config.FARM_LABOR_MULTIPLIER_CAP's
            # own comment for the real numbers this was caught against.
            labor_multiplier = min(_labor_multiplier(tribe.population), config.FARM_LABOR_MULTIPLIER_CAP)
            harvested = round(config.CROP_HARVEST_YIELD * tribe.farm_plots * labor_multiplier * _food_multiplier(tribe))
            added = self._capped_add(tribe, "food", harvested)
            tribe.last_harvest_cycle = self.cycle
            if added < harvested:
                tribe.history.append(
                    f"the farm plots yield a harvest -- {added} food gathered in "
                    f"(stores nearly full, {harvested - added} wasted)"
                )
            else:
                tribe.history.append(f"the farm plots yield a harvest -- {added} food gathered in")
            self._award_trophy(tribe, "Harvester")
            # Explicit request: "a grand harvest is a real celebration." Cooldown-
            # gated (unlike _celebrate_water_discovery/_celebrate_settling, which are
            # each essentially one-time) since a harvest recurs every ~10 cycles per
            # plot -- an uncapped feast on every single one would drain food faster
            # than farming produces it.
            if self.cycle - tribe.last_celebration_cycle >= config.CELEBRATION_COOLDOWN_CYCLES:
                self._celebrate_harvest(tribe)

    def _advance_flock(self, tribe: Tribe) -> None:
        """A flock isn't a one-way counter -- it eats, and once established it can
        also breed on its own (the same "passive consequence" category as crop
        growth), without another GATHER_EGGS action. Real stakes both ways: undersized
        on feed and it shrinks; big enough and fed, and it can grow by itself."""
        if tribe.flock <= 0:
            return
        feed_needed = config.FLOCK_UPKEEP_FOOD_PER_MEMBER * tribe.flock
        if tribe.food < feed_needed:
            tribe.flock -= 1
            tribe.history.append("part of the flock is lost for lack of feed")
            return
        tribe.food -= feed_needed
        hatch_chance = config.FLOCK_NATURAL_HATCH_CHANCE
        if tribe.hatchery_built:
            hatch_chance = min(1.0, hatch_chance * config.HATCHERY_HATCH_CHANCE_MULTIPLIER)
        if (
            tribe.flock >= config.FLOCK_MIN_SIZE_TO_BREED
            and tribe.pending_hatch is None
            and random.random() < hatch_chance
        ):
            parents = tribe.flock_lineage[-2:] if len(tribe.flock_lineage) >= 2 else None
            tribe.pending_hatch = {"parents": parents}

    def _advance_flock_eggs(self, tribe: Tribe) -> None:
        """A living flock lays eggs passively each cycle into tribe.eggs -- see
        config.EGGS_LAID_PER_FLOCK_PER_CYCLE_DIVISOR's own comment. Entirely
        separate from GATHER_EGGS/_advance_flock's natural-hatch chance, both of
        which grow tribe.flock directly and never touch this stockpile."""
        if tribe.flock <= 0:
            return
        laid = tribe.flock // config.EGGS_LAID_PER_FLOCK_PER_CYCLE_DIVISOR
        if laid:
            tribe.eggs += laid

    def _advance_livestock_feast(self, tribe: Tribe) -> None:
        """See config.LIVESTOCK_SURPLUS_THRESHOLD's own comment -- once eggs or
        flock grow past the tribe's own scaled threshold, the surplus is
        automatically eaten as food each cycle rather than piling up forever
        with no payoff."""
        threshold = _livestock_surplus_threshold(tribe)
        if tribe.eggs > threshold:
            surplus = tribe.eggs - threshold
            tribe.eggs = threshold
            self._capped_add(tribe, "food", surplus * config.EGG_FEAST_FOOD_VALUE)
        if tribe.flock > threshold:
            surplus = tribe.flock - threshold
            tribe.flock = threshold
            self._capped_add(tribe, "food", surplus * config.FLOCK_FEAST_FOOD_VALUE)

    def _advance_city_founding(self, tribe: Tribe) -> None:
        """Real gate on Era.founds_city eligibility (see _advance_era_if_ready): a
        city doesn't formally get founded until at least one Long House stands, the
        same real-housing dependency Sawmill/Quarry/Mine/Tannery already require.
        Rechecked every cycle rather than only at the instant the era advances, so
        a tribe that reaches the milestone before building any housing still founds
        its city the moment a Long House finally goes up, instead of being
        permanently denied for having built things in the "wrong" order."""
        if tribe.founded_city or not tribe.city_founding_eligible:
            return
        if tribe.long_houses_built > 0:
            tribe.founded_city = True
            tribe.history.append(f"the first Long House stands -- {tribe.name} formally founds a city")

    def _choose_territory_center(self, tribe: Tribe) -> tuple[int, int]:
        """Picks where the Hut/Town Hall and wall ring 0 actually get centered --
        not necessarily the tribe's exact settling tile. Explicit correction after
        a live run: "the territories contain a lot of water and so, the need to
        back away from the Hut so that only 1 natural Wall at most exists." A
        tribe settled deep in a river bend, or against a concave lake shore, can
        have the water cross ring 0's circumference (radius WALL_RING_RADIUS_STEP)
        in two or more places -- opposite banks of the same river, say -- turning
        most of the wall into freebie natural_barrier sections (city_layout.
        _is_natural_barrier) instead of a real perimeter worth building.

        Searches outward from the settling tile in a spiral, checking each
        candidate's real ring-0 footprint (city_layout.build_ring, the same
        function that actually builds it -- not a separate estimate that could
        drift from it), and returns the closest one at or under config.
        TERRITORY_MAX_ACCEPTABLE_NATURAL_BARRIERS. Backing away like this
        doesn't undo the water access that qualified the tribe to settle here
        in the first place (see _is_settled_near_water/
        SETTLEMENT_WATER_TERRITORY_RADIUS) -- that was already earned at the
        tribe's actual position; this only decides where the walls and the
        building sit. Falls back to whichever candidate found the fewest
        natural barriers if none hits the target within the search radius, and
        never proposes a center sitting on unbuildable ground.

        Live bug report: "Tribe 1 found a perfect spot, built a Hut then
        mysteriously transferred the Territory and Hut to the 2nd found Water
        site." The tribe never actually moved (confirmed via board_history --
        tribe.x/y stayed put the whole run) -- this search used to range out to
        4*WALL_RING_RADIUS_STEP (48 tiles) and, at the old >1-barrier
        threshold, moved 12 tiles away from a settling tile that only had 3 of
        8 sections on the river -- exactly far enough to read as "the city
        teleported" and to leave the actual population outside their own wall
        ring's real coverage (the ring itself only reaches
        WALL_RING_RADIUS_STEP out from its center). Direct visual confirmation
        (replaying the exact captured state) plus explicit follow-up
        feedback -- "the first place Tribe 1 landed, including Territory, was
        perfect" -- established that 3 of 8 is good, defensible, river-framed
        ground, not a defect worth relocating over. Search distance capped
        (2*WALL_RING_RADIUS_STEP, keeping the settling tile well inside the
        eventual ring even in the worst case) AND the acceptance threshold
        raised to TERRITORY_MAX_ACCEPTABLE_NATURAL_BARRIERS, so a spot only
        gets abandoned once water genuinely dominates the ring, not just
        touches a few sections of it."""
        origin = (tribe.x, tribe.y)
        best_center, best_count = origin, None
        max_search_radius = 2 * config.WALL_RING_RADIUS_STEP
        for radius in range(0, max_search_radius + 1, 3):
            angles = [0.0] if radius == 0 else [i * math.pi / 8 for i in range(16)]
            for angle in angles:
                cx = tribe.x + round(radius * math.cos(angle))
                cy = tribe.y + round(radius * math.sin(angle))
                if self.world.biome(cx, cy) in config.UNBUILDABLE_BIOMES:
                    continue
                ring = city_layout.build_ring(self.world, (cx, cy), ring_index=0)
                count = sum(1 for sec in ring["sections"] if sec["natural_barrier"])
                if best_count is None or count < best_count:
                    best_center, best_count = (cx, cy), count
                if count <= config.TERRITORY_MAX_ACCEPTABLE_NATURAL_BARRIERS:
                    return (cx, cy)
        return best_center

    def _found_territory(self, tribe: Tribe) -> None:
        """Grants a real, owned territory the instant a tribe first qualifies as
        settled (has_ever_settled) -- explicit request: "when a Tribe becomes
        Settled, they automatically get a region of territory they own." Anchors
        territory_center at a chosen founding coordinate (deliberately not always
        tribe.x/y -- see _choose_territory_center's own comment, and Tribe.
        __init__'s comment on why RELOCATE can't be allowed to drag a city's
        buildings/walls around once chosen), builds the first wall ring, and
        places the Town Hall centered on that same coordinate."""
        tribe.territory_center = self._choose_territory_center(tribe)
        tribe.territory_radius = config.WALL_RING_RADIUS_STEP
        ring0 = city_layout.build_ring(self.world, tribe.territory_center, ring_index=0)
        # Safety net for the rare case _choose_territory_center's search never found a
        # candidate under the cap and fell back to its best (fewest-barrier) attempt --
        # see cap_natural_barriers' own comment for why this can't be applied inside
        # the search itself.
        city_layout.cap_natural_barriers(ring0["sections"])
        tribe.wall_rings = [ring0]
        w, h = config.BUILDING_FOOTPRINTS["town_hall"]
        cx, cy = tribe.territory_center
        architect.record_building(tribe, "town_hall", cx - w // 2, cy - h // 2, w, h, self.cycle)

        # Live bug report: "built a Hut then mysteriously transferred the
        # Territory and Hut to the 2nd found Water site." The tribe never
        # actually moved -- _choose_territory_center backed the center away
        # from the settling tile (see its own docstring) and the resulting
        # distance happened to read as a teleport with nothing explaining it.
        # A silent, un-narrated placement is the real bug here, not the
        # distance itself -- this states plainly, at the moment it happens,
        # why the buildings aren't sitting exactly where the tribe stands.
        if (cx, cy) != (tribe.x, tribe.y):
            dist = round(((cx - tribe.x) ** 2 + (cy - tribe.y) ** 2) ** 0.5)
            tribe.history.append(
                f"the Hut and Town Hall are raised {dist} tiles from where the tribe actually stands, at "
                f"({cx},{cy}) -- building a wall right on the settling spot would have left too much of it "
                "as open water, so the builders backed off to solid, defensible ground instead"
            )

        # Explicit request: "when they start to build a Wall we need to force
        # existing Raider sites out of the Territory and for some distance away
        # from the Territory boundary." Anything now caught inside this fresh
        # territory (+ buffer) gets relocated to a freshly picked, territory-aware
        # site -- checked against every tribe's territory, not just this one, so
        # this also cleans up anything a rival's own territory already covers.
        occupied = [(t.x, t.y) for t in self.tribes.values()] + [(ms["x"], ms["y"]) for ms in self.minor_settlements]
        for ms in self.minor_settlements:
            if self._inside_any_territory(ms["x"], ms["y"]):
                occupied.remove((ms["x"], ms["y"]))
                ms["x"], ms["y"] = self._find_minor_settlement_site(occupied)
                occupied.append((ms["x"], ms["y"]))

    def _award_trophy(self, tribe: Tribe, name: str, individual: str | None = None) -> None:
        """`individual`, when given, credits a specific named person (e.g. the scout or
        hunter who actually earned a milestone trophy) instead of the chief -- the
        dict's "chief" key is kept for backward compatibility (existing tests/scoreboard
        data read it) even though it isn't always literally the chief anymore."""
        if any(t["name"] == name for t in tribe.trophies):
            return  # once per tribe's lifetime
        credited = individual or tribe.chief_name or "an unknown chief"
        tribe.trophies.append({"name": name, "chief": credited, "cycle": self.cycle})
        tribe.history.append(f"\U0001f3c6 {credited} earns the '{name}' trophy for {tribe.name}!")

    def _check_custom_awards(self, tribe: Tribe, category: str, individual: str | None = None) -> None:
        """The other half of the night-cycle award stub (see reflection.py's
        AWARD_CATEGORIES docstring): a chief can propose an honor of their own during
        the night cycle, but until now nothing checked a real counter and actually
        handed it out. Called from the same real-event sites that already check the
        built-in milestone trophies (a scout's confirmed water, a hunting party's
        catch, a completed trade, a won raid) -- the first genuine act of excellence
        in the proposed category after the chief establishes it becomes its first (and,
        since _award_trophy pays out once per tribe lifetime, only) recipient. An
        honest milestone tied to a specific real achievement, not an arbitrary round
        number invented just for this."""
        for award in tribe.custom_awards:
            if award["category"] == category:
                self._award_trophy(tribe, award["name"], individual=individual)

    def _check_chief_trophies(self, tribe: Tribe) -> None:
        """A lightweight legacy system credited to whichever chief is in power the
        moment each is first earned -- 'Water Bringer' is deliberately the standout,
        since reliable water access is the single hardest survival problem this
        simulation poses."""
        if self.world.biome(tribe.x, tribe.y) in ("river", "lake"):
            self._award_trophy(tribe, "Water Bringer")
        if tribe.food >= config.FOOD_TROPHY_THRESHOLD:
            self._award_trophy(tribe, "Well Fed")
        if tribe.population > 8:
            self._award_trophy(tribe, "Growing Legacy")

    def _celebrate_water_discovery(self, tribe: Tribe, fx: int, fy: int) -> None:
        """Explicit request: a scout confirming fresh water for the first time (not
        just re-confirming an already-known site, and only while the tribe hasn't
        settled somewhere with real water access yet) should read as its own event --
        "you found water! now we party!" -- not wait on the unrelated food-surplus/
        discovery-weight gate _check_for_celebration normally requires. Shares that
        method's cooldown bookkeeping and breeding side-effect (the same "lots of time
        for breeding and mating" a party creates) so the two don't both fire the same
        cycle, but this one always fires on a genuine new find."""
        tribe.last_celebration_cycle = self.cycle
        spent = _celebration_cost(tribe)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        tribe.fame += config.FAME_PER_CELEBRATION
        tribe.history.append(
            f"\U0001f389 {tribe.name} celebrates the discovery of water at ({fx},{fy}), spending {spent} "
            f"food on a {_feast_word(tribe)} -- the tribe will move to settle there soon{_celebration_shout(tribe)}"
        )
        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

    def _celebrate_game_discovery(self, tribe: Tribe, tx: int, ty: int) -> None:
        """A scout reporting back a genuinely new game-rich site is its own real find --
        same "you found something! now we party!" treatment as _celebrate_water_
        discovery, just for a small-game site instead of water. Terrain reports only
        ever carry memory weight 0.6, below CELEBRATION_DISCOVERY_WEIGHT, so this would
        otherwise never trigger the generic _check_for_celebration path at all."""
        tribe.last_celebration_cycle = self.cycle
        spent = _celebration_cost(tribe)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        tribe.fame += config.FAME_PER_CELEBRATION
        tribe.history.append(
            f"\U0001f389 {tribe.name} celebrates the discovery of a game-rich site at ({tx},{ty}), "
            f"spending {spent} food on a {_feast_word(tribe)}{_celebration_shout(tribe)}"
        )
        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

    def _check_for_celebration(self, tribe: Tribe) -> None:
        """Automatic and threshold-based, same pattern as era advancement/trophies/
        population growth -- not a discrete action the model has to remember to pick.
        This session's own data (BREED sat free and genuinely eligible for 20+ live
        cycles without ever being chosen) suggests these models rarely reach for a new
        discrete choice at all, so a reward gated behind choosing one more action would
        likely suffer the same fate.

        Fires on a real resource surplus (reusing FOOD_TROPHY_THRESHOLD, the same
        "Well Fed" bar) OR a genuine new discovery -- any memory entry just recorded
        this exact cycle at or above CELEBRATION_DISCOVERY_WEIGHT, the same weight that
        already promotes a memory into a permanent taboo/lesson (see TribeMemory.
        consolidate) -- i.e. this tribe's own definition of "something worth
        remembering forever," not a threshold invented just for this. Spends a real
        fraction of the surplus (the mass gathering effort), radiates real pride
        through the area same as any other proud event, and -- if two distinct named
        individuals are already eligible -- is what naturally brings them together,
        without needing the model to separately choose BREED."""
        if self.cycle - tribe.last_celebration_cycle < config.CELEBRATION_COOLDOWN_CYCLES:
            return

        # Explicit finding: this used to fire on surplus alone every single cooldown
        # window forever -- real the first few times, but not a fresh reason to spend
        # food indefinitely once a tribe has proven it can reliably sustain a
        # surplus. Retires after CELEBRATION_SURPLUS_RETIREMENT_COUNT, the same
        # "generalist narrows to specialist" shape GATHER_FOOD's own retirement uses.
        # The discovery branch below never retires -- each one is a genuinely new,
        # distinct thing, not a repeat of the same "yay, food" flavor.
        surplus = (
            tribe.food >= config.FOOD_TROPHY_THRESHOLD
            and tribe.surplus_celebrations < config.CELEBRATION_SURPLUS_RETIREMENT_COUNT
        )
        # Explicit request: "they celebrated a 'fresh discovery' but they need to
        # name it" -- this used to only check WHETHER a qualifying memory existed,
        # never which one, so the chronicle line could never say what was actually
        # discovered. Keeps the actual entry (most recent if more than one landed
        # this cycle) so the celebration can name the real thing being celebrated.
        discovery_entries = [
            e for e in tribe.memory.entries
            if e["cycle"] == self.cycle and e["weight"] >= config.CELEBRATION_DISCOVERY_WEIGHT
        ]
        if not surplus and not discovery_entries:
            return

        tribe.last_celebration_cycle = self.cycle
        spent = _celebration_cost(tribe)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        tribe.fame += config.FAME_PER_CELEBRATION
        reason = f"a fresh discovery: {discovery_entries[-1]['text']}" if discovery_entries else "a season of plenty"
        if not discovery_entries:
            tribe.surplus_celebrations += 1
            if tribe.surplus_celebrations == config.CELEBRATION_SURPLUS_RETIREMENT_COUNT:
                tribe.history.append(
                    f"\U0001f4dc {tribe.name} no longer celebrates mere plenty -- a comfortable "
                    "surplus has become the norm, not a special occasion"
                )
        tribe.history.append(
            f"\U0001f389 {tribe.name} holds a celebration for {reason}, spending {spent} food on a {_feast_word(tribe)}{_celebration_shout(tribe)}"
        )

        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

    def _celebrate_settling(self, tribe: Tribe) -> None:
        """Explicit request: settling somewhere for good is worth its own celebration,
        not just whatever unrelated surplus/discovery celebration happens to fire
        next. Fires once, the first time _is_settled_near_water becomes true (guarded
        by tribe.settlement_name being unset) -- names the settlement "during a party"
        (backend/leadership.py's name_settlement, resolved in Simulation.step()), same
        pattern as _celebrate_water_discovery."""
        tribe.last_celebration_cycle = self.cycle
        spent = _celebration_cost(tribe)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        tribe.fame += config.FAME_PER_CELEBRATION
        tribe.history.append(
            f"\U0001f389 {tribe.name} celebrates settling here for good, spending {spent} food on a {_feast_word(tribe)}{_celebration_shout(tribe)}"
        )
        tribe.pending_settlement_naming = True
        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

    def _celebrate_harvest(self, tribe: Tribe) -> None:
        """Explicit request: "a grand harvest is a real celebration." Caller
        (_advance_farming) already checked the cooldown before calling this, since a
        harvest recurs every ~10 cycles per plot -- unlike _celebrate_water_discovery/
        _celebrate_settling (each essentially one-time), this one is cooldown-gated
        the same way the generic _check_for_celebration is."""
        tribe.last_celebration_cycle = self.cycle
        spent = _celebration_cost(tribe)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        tribe.fame += config.FAME_PER_CELEBRATION
        tribe.history.append(
            f"\U0001f389 {tribe.name} holds a harvest festival, spending {spent} food on a {_feast_word(tribe)}{_celebration_shout(tribe)}"
        )
        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

    def _celebrate_fishing_learned(self, tribe: Tribe) -> None:
        """The first successful CATCH_FISH is its own real milestone -- same "you
        learned something! now we party!" treatment as _celebrate_water_discovery/
        _celebrate_game_discovery, just for fishing instead of a scouted site."""
        tribe.last_celebration_cycle = self.cycle
        spent = _celebration_cost(tribe)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        tribe.fame += config.FAME_PER_CELEBRATION
        tribe.history.append(
            f"\U0001f389 {tribe.name} celebrates learning to fish, spending {spent} food on a {_feast_word(tribe)}{_celebration_shout(tribe)}"
        )
        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

    def _merge_tribes(self, attacker: Tribe, defender: Tribe) -> str:
        """A defender's population has been driven to zero by accumulated raid
        losses (actions.py._raid transfers population rather than just destroying
        it) -- their survivors, resources, and remaining history become a new, more
        advanced entity instead of simply disappearing into extinction. Mutates
        `attacker` in place (rather than constructing a fresh Tribe and swapping it
        into self.tribes) so every reference the calling turn already holds --
        Simulation._apply_turn keeps mutating `tribe` after this action handler
        returns -- keeps pointing at the right object. Chief-less on completion; the
        same per-cycle succession check that already handles a fallen chief's
        replacement (see step()) picks this up automatically next cycle, exactly like
        a founding election -- no special-casing needed, and it still runs the
        model's own reasoning about where reliable water actually is."""
        old_name = attacker.name
        attacker.name = f"{old_name} (Advanced)"
        # See Tribe.conquests_won's own comment and Simulation.step's
        # world_domination victory check -- the real signal that this tribe
        # became the last one standing by actually conquering rivals, not by
        # outlasting others through unrelated hazard deaths.
        attacker.conquests_won += 1
        attacker.conquered_tribe_names.append(defender.name)
        # Same uncapped-mutation bug as the expedition-homecoming fix above, just
        # for the rarer whole-tribe-absorption path -- routed through _capped_add
        # so a big merge can't silently blow past the attacker's own storage cap
        # and disable its passive income (settled water/fish/farm) from then on.
        self._capped_add(attacker, "wood", defender.wood)
        self._capped_add(attacker, "stone", defender.stone)
        self._capped_add(attacker, "food", defender.food)
        self._capped_add(attacker, "water", defender.water)
        attacker.population += defender.population
        defender.population = 0
        attacker.max_population = max(attacker.max_population, attacker.population)
        if era_index(defender.era) > era_index(attacker.era):
            attacker.era = defender.era
        attacker.chief_name = ""
        attacker.chief_philosophy = ""
        attacker.chief_decree = ""
        attacker.chief_victory = ""
        attacker.pending_chief_context = (
            f"This tribe was just formed when {old_name} triumphed in battle over {defender.name} "
            f"and absorbed their surviving population. The new chief inherits a people freshly "
            f"unified by conquest, not a tribe with a long shared history."
        )
        defender.extinct = True
        record_tribe_result(defender, cause="absorbed", cycles_survived=self.cycle)
        attacker.history.append(
            f"{old_name} has fully absorbed {defender.name}'s survivors and become {attacker.name}!"
        )
        del self.tribes[defender.id]
        return attacker.name
