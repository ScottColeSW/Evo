import asyncio
import difflib
import importlib
import math
import random

from . import architect, city_layout, config, physics
from .actions import (
    ACTION_REGISTRY, BIOME_YIELD_MULTIPLIER, GAME_SPECIES_BY_BIOME, GAME_SPECIES_LABEL,
    _eligible_breeding_pair, _execute_trade, _find_trade_partner, _food_multiplier, _item_storage_cap,
    _labor_multiplier, _storage_cap, expedition_capacity,
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
# Explicit placement request (2026-09-03) replaced the first two slots with a direct,
# landmark-relative brief instead of "pick a point in the right-named biome": the first
# slot southeast of world.LAKE_CENTER, the second in the north-center, above the river --
# both still far enough from that water to need a real scouting trip (not a freebie), and
# both still clearing SCOUT_PATROL_DISTANCE=25 from the grid edge (or as close to it as
# the requested position allows -- 22-24 here, same real ceiling the old Mountains slot
# hit, not a regression). Slots 2/3 are unchanged fallbacks for a 3rd/4th tribe.
SPAWN_POINTS = [(37, 75), (50, 22), (50, 55), (40, 37)]
COLORS = ["#c084fc", "#fb923c", "#34d399", "#60a5fa"]

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
    "BUILD_WELL": "well_built",
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
SURVIVAL_CRISIS_ACTIONS = {
    "GATHER_FOOD", "HUNT_DEER", "CATCH_FISH", "HUNTING_PARTY", "GATHER_EGGS", "COOK_FOOD",
    "GATHER_WATER", "SCOUT",
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


def _can_afford_construct_wall(tribe, world) -> bool:
    cost = _wall_next_afford_cost(tribe)
    if cost is None:
        return True  # let the action's own "nothing to build" message surface instead
    wood_cost, stone_cost = cost
    return tribe.wood >= wood_cost and tribe.stone >= stone_cost


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
    guaranteed no-op never dangles in the menu."""
    ring0_done = bool(tribe.wall_rings) and city_layout.ring_fully_built(tribe.wall_rings[0])
    if not ring0_done and tribe.wall_lock_long_house_credits <= 0:
        return False
    houses_needed = max(1, -(-tribe.population // config.HOUSING_POPULATION_PER_LONG_HOUSE))
    if tribe.long_houses_built >= houses_needed:
        return False
    return (
        tribe.wood >= config.LONG_HOUSE_WOOD_COST and tribe.stone >= config.LONG_HOUSE_STONE_COST
        and _can_place(tribe, world, "long_house")
    )


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
    "EXPAND_TERRITORY": lambda t, w: t.wood >= config.TERRITORY_EXPANSION_WOOD_COST and t.stone >= config.TERRITORY_EXPANSION_STONE_COST,
    "PLANT_CROP": lambda t, w: (
        t.farm_plots < config.MAX_FARM_PLOTS and t.wood >= config.PLANT_CROP_WOOD_COST
        and _can_place(t, w, "farm_plot")
    ),
    "BREED": lambda t, w: t.food >= config.BREED_FOOD_COST and t.water >= config.BREED_WATER_COST,
    # Explicit request: "it's unwise to Trade before we have a full Wall" --
    # see actions.py._send_trade_emissary's matching real prerequisite. Instant
    # TRADE is left alone (a chance encounter, not a deliberate choice to
    # expose the tribe) -- only the deliberate, multi-day search is gated.
    "SEND_TRADE_EMISSARY": lambda t, w: bool(t.wall_rings) and city_layout.ring_fully_built(t.wall_rings[0]),
    # Both a real resource cost AND config.ITEM_STORAGE_CAP_BASE's own ceiling --
    # see _forge_item's matching "item stores are already full" no-op message.
    "FORGE_ITEM": lambda t, w: (
        len(t.items) < _item_storage_cap(t)
        and t.wood >= config.FORGE_ITEM_WOOD_COST
        and t.unique_resources.get(t.mine_resource_name, 0) >= config.FORGE_ITEM_ORE_COST
    ),
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
        self.chief_name = ""
        self.chief_philosophy = ""
        self.chief_decree = ""
        self.chief_victory = ""
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
        self.scout_rotation_index = stagger
        # See actions.py._exploration_party -- its own separate rotating heading
        # (offset from SCOUT's own sweep) so the two don't retrace each other's
        # ground. landmarks: {"x", "y", "resource"} entries, one per Landmark
        # actually found (Simulation._advance_exploration_party_outbound).
        self.explore_rotation_index = stagger
        self.landmarks: list[dict] = []
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
        # See actions.py._build_road -- one-way. Adds a flat speed bonus to every
        # future expedition (Simulation._advance_one_expedition), the same shape a
        # well-worn trail already grants.
        self.road_built = False
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

    def to_dict(self) -> dict:
        era_label = next((e.label for e in ERAS if e.key == self.era), self.era)
        survival_warning, _ = survival_bias_string(
            self.food, self.water, self.population, self.fishing_learned, self.cooking_learned
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
            "road_built": self.road_built,
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
            "kitchen_built": self.kitchen_built,
            "tannery_built": self.tannery_built,
            "forge_built": self.forge_built,
            "items": self.items,
            "warehouses_built": self.warehouses_built,
            "foraging_retired": self.foraging_retired,
            "watering_retired": self.watering_retired,
            "last_harvest_cycle": self.last_harvest_cycle,
            "flock": self.flock,
            "eggs": self.eggs,
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
        # Neutral, non-AI raid/trade targets (backend/actions.py._raid/_trade) --
        # see config.MINOR_SETTLEMENT_COUNT's own comment for the full design note.
        self.minor_settlements: list[dict] = []
        self._spawn_minor_settlements()
        self.cycle = 0
        self.paused = False
        self.status = "OPERATIONAL"
        self.game_over = False
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
            tribe.food, tribe.water, tribe.population, tribe.fishing_learned, tribe.cooking_learned
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
                have = getattr(tribe, resource, 0)
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

        guard = HardwareVRAMBoundaryGuard(self.client.base_url, config.VRAM_LIMIT_GB)
        ok, warning = await guard.verify_vram_safety_margin(model)
        if not ok:
            tribe.history.append(f"VRAM WARNING: {warning}")
        await self._install_chief(tribe)

        self.tribes[tid] = tribe
        # A fresh tribe means the game isn't over anymore, even if every previous tribe
        # died -- undoes _trigger_game_over's stop/unload so stepping resumes.
        self.game_over = False
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
            self._apply_upkeep(tribe)
            self._check_raider_attack(tribe)
            self._advance_raider_approach(tribe)
            self._grow_population(tribe)
            self._advance_era_if_ready(tribe)
            if not tribe.settlement_name and not tribe.pending_settlement_naming and self._is_settled_near_water(tribe):
                self._celebrate_settling(tribe)
            self._advance_water_supply(tribe)
            self._advance_fish_supply(tribe)
            self._advance_mine_yield(tribe)
            self._advance_in_territory_site_yields(tribe)
            self._advance_tannery_yield(tribe)
            self._advance_resource_trails(tribe)
            self._advance_farming(tribe)
            self._advance_flock(tribe)
            self._advance_flock_eggs(tribe)
            self._advance_livestock_feast(tribe)
            self._advance_city_founding(tribe)
            self._check_chief_trophies(tribe)
            self._check_for_celebration(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.expeditions:
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

        if self.tribes and all(tribe.extinct for tribe in self.tribes.values()):
            await self._trigger_game_over()

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
                have = getattr(tribe, resource, 0)
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
        SCOUT/HUNTING_PARTY/trade emissary outbound and return) right after
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
        owner.wood += config.TOLL_FEE_WOOD
        return nx, ny

    def _prepare_turn(self, tribe: Tribe) -> tuple[dict, dict]:
        """Builds this tribe's prompt with no network calls; returns (request, context)."""
        biome = self.world.biome(tribe.x, tribe.y)
        nearby = self.world.nearby_structures(tribe.x, tribe.y)
        ghost_bias = self.trauma.bias_string(tribe.x, tribe.y)
        survival_bias, survival_critical = survival_bias_string(
            tribe.food, tribe.water, tribe.population, tribe.fishing_learned, tribe.cooking_learned
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
        if camped and not still_journeying and not tribe.has_ever_settled:
            tribe.has_ever_settled = True
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
        if settled_near_water:
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
        # this can happen early." COOK_FOOD is gated on these two real, proven
        # prerequisites rather than era progression -- see Tribe.hunt_ever_succeeded/
        # fire_ever_built. Retires once learned, the same one-way "generalist
        # narrows/task is done" shape foraging_retired and watering_retired use --
        # there's nothing left to decide once cooking is known forever.
        if tribe.cooking_learned:
            available_actions = [a for a in available_actions if a != "COOK_FOOD"]
        elif not (tribe.hunt_ever_succeeded and tribe.fire_ever_built):
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
        upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
        if tribe.food <= upkeep * config.HUNGER_CRITICAL_CYCLES_LEFT:
            tribe.food_crisis_active = True
        elif tribe.food > upkeep * config.HUNGER_WARNING_CYCLES_LEFT:
            tribe.food_crisis_active = False
        if tribe.water <= upkeep * config.THIRST_CRITICAL_CYCLES_LEFT:
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
            visible_entities.append(
                "The tribe has both hunted successfully and built a fire before -- learning to cook "
                "would make stored food go much further from then on."
            )

        # Real wall ring, not one progress-bar tile (2026-09-02 redesign) -- ring 0
        # is what BUILD_LONG_HOUSE/Moat/Torches gate on; further rings are purely
        # additional defense-in-depth, not a further gate on anything else.
        ring0 = tribe.wall_rings[0] if tribe.wall_rings else None
        ring0_reinforced = bool(ring0) and city_layout.ring_fully_reinforced(ring0)

        if "CONSTRUCT_WALL" in available_actions:
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
            party_word = {"scout": "scouts", "hunt": "a hunting party"}
            reports = "; ".join(
                f"{party_word.get(exp.get('kind'), 'a party')} led by {exp['lead_scout']} "
                f"(day {exp['day']}, {exp['phase']}, headed toward "
                f"({exp['target'][0]},{exp['target'][1]}))"
                for exp in tribe.expeditions
            )
            slots_left = expedition_capacity(tribe) - len(tribe.expeditions)
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
            # LAYER): both are the same category of "not urgent, but real" pressure,
            # and both need the same salience fix era_gap_note already proved out --
            # a fact buried in the generic list gets ignored even when it's true.
            "growth_note": " ".join(n for n in (era_gap_note, diversification_note) if n),
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

    async def _trigger_game_over(self) -> None:
        """Every tribe in this session has gone extinct -- there will be no more turns,
        ever, for any model this session used. Rather than let step() keep getting
        called every tick forever (harmless but pointless once requests is always
        empty) and leave every model sitting loaded in Ollama until its keep_alive
        window expires on its own, stop stepping and unload them immediately. A fresh
        ADD_TRIBE clears this back to normal (see add_tribe)."""
        self.game_over = True
        self.status = "GAME OVER"
        await self.shutdown()

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

    def _advance_expeditions(self, tribe: Tribe) -> None:
        """Advances every one of a tribe's in-field parties by one day (see
        actions.py._scout/_hunting_party) -- a tribe can have up to
        actions.expedition_capacity(tribe) out at once. Iterates a snapshot of the list
        since a party can complete (and remove itself) mid-loop."""
        for exp in list(tribe.expeditions):
            if self._advance_one_expedition(tribe, exp):
                tribe.expeditions.remove(exp)

    def _advance_one_expedition(self, tribe: Tribe, exp: dict) -> bool:
        """One day of an in-progress expedition. Runs every cycle regardless of what
        action the tribe chose that turn -- the party is out in the field on its own,
        not waiting for the tribe's attention each cycle. Outbound: walk toward target,
        succeeding immediately on real fresh water or on reaching the destination, or
        giving up after EXPEDITION_MAX_DAYS. Returning: walk back toward camp; arrival
        is the only moment a finding becomes real, actionable knowledge (memory +
        chronicle) -- a party that hasn't made it home yet knows something the tribe as
        a whole does not. Returns True once this expedition is over and should be
        removed from tribe.expeditions.

        Wears (and benefits from) the same worn-trail mechanic as RELOCATE: a route
        used by enough expeditions gets faster over time, so a destination just out of
        one expedition's EXPEDITION_MAX_DAYS reach can become reachable a few attempts
        later purely by repeatedly trying the same path -- effort compounding into
        infrastructure, not a scripted distance override."""
        if exp["phase"] == "outbound":
            exp["day"] += 1
            px, py = exp["pos"]
            tx, ty = exp["target"]
            bonus = self.world.trail_speed_bonus(px, py, config.MAX_TRAIL_BONUS_SPEED)
            # See actions.py._build_road -- a flat, always-on version of the same
            # trail bonus above, since a deliberately-built road doesn't need to
            # wear in from repeated travel the way a trail does.
            base_speed = config.EXPEDITION_SPEED + bonus + (config.ROAD_SPEED_BONUS if tribe.road_built else 0)
            # Explicit request: "travel speed is 5x on toll roads."
            if self.world.is_toll_road(px, py):
                base_speed *= config.TOLL_ROAD_SPEED_MULTIPLIER
            nx, ny = physics.terrain_aware_step(px, py, tx, ty, base_speed=base_speed, has_boat=tribe.boat_built)
            nx, ny = self._resolve_toll(tribe, px, py, nx, ny)
            self.world.wear_trail(nx, ny, config.TRAIL_WEAR_PER_PASS, tribe.color, tribe.id)
            mark_visited_sector(tribe, nx, ny)
            self._check_road_evolution(nx, ny)
            exp["pos"] = [nx, ny]
            exp["path"].append([nx, ny])
            # Explicit correction: "the volcano is a Hazard they will die if they
            # go there." Unlike the river's drowning risk (only ever checked on
            # the outbound leg inside the water-sensing branch below, since a
            # volcano tile can never register as "sensed water"), this needs an
            # unconditional check here so an outbound leg crossing volcano ground
            # carries the same real risk the returning leg already does (below).
            self._volcano_hazard(tribe, nx, ny)
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

            if self._expedition_raider_ambush(tribe, exp, nx, ny):
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
                exp["phase"] = "returning"
                tribe.history.append(f"{scout}'s party can go no further this way and turns back after {exp['day']} days")
                return False
            if exp.get("kind") == "hunt":
                self._advance_hunting_party_outbound(tribe, exp, reached_biome, scout)
                return False
            if exp.get("kind") == "trade":
                self._advance_trade_emissary_outbound(tribe, exp, scout)
                return False
            if exp.get("kind") == "explore" and self._advance_exploration_party_outbound(tribe, exp, reached_biome, scout):
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
                    self._expedition_river_hazard(tribe, nx, ny)  # a no-op on a lake tile -- no current to drown in
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
                exp["terrain_report"] = reached_biome
                exp["phase"] = "returning"
                label = BIOME_LABELS.get(reached_biome, reached_biome)
                tribe.history.append(f"{scout}'s party surveys ({nx},{ny}), {label}, after {exp['day']} days and heads home to report")
            return False
        else:  # returning
            px, py = exp["pos"]
            ox, oy = exp["origin"]
            bonus = self.world.trail_speed_bonus(px, py, config.MAX_TRAIL_BONUS_SPEED)
            base_speed = config.EXPEDITION_SPEED + bonus + (config.ROAD_SPEED_BONUS if tribe.road_built else 0)
            # Explicit request: "travel speed is 5x on toll roads."
            if self.world.is_toll_road(px, py):
                base_speed *= config.TOLL_ROAD_SPEED_MULTIPLIER
            nx, ny = physics.terrain_aware_step(px, py, ox, oy, base_speed=base_speed, has_boat=tribe.boat_built)
            nx, ny = self._resolve_toll(tribe, px, py, nx, ny)
            self.world.wear_trail(nx, ny, config.TRAIL_WEAR_PER_PASS, tribe.color, tribe.id)
            mark_visited_sector(tribe, nx, ny)
            self._check_road_evolution(nx, ny)
            exp["pos"] = [nx, ny]
            exp["path"].append([nx, ny])
            exp["food_gathered"] += config.EXPEDITION_RETURN_DAILY_FOOD
            if not self._is_settled_near_water(tribe):  # see the matching outbound-leg comment above
                exp["water_gathered"] += config.EXPEDITION_RETURN_DAILY_WATER
            self._expedition_river_hazard(tribe, nx, ny)
            self._volcano_hazard(tribe, nx, ny)
            self._expedition_raider_ambush(tribe, exp, nx, ny)
            if [nx, ny] == [ox, oy]:
                # Whatever was foraged along the way comes home regardless of whether the
                # expedition succeeded -- the trip cost real time either way, so it isn't
                # a total loss on a failed search. The findings themselves only become
                # real, actionable knowledge for the tribe at this exact moment.
                food_home = round(exp["food_gathered"] * _food_multiplier(tribe))
                tribe.food += food_home
                tribe.water += exp["water_gathered"]
                scout = exp["lead_scout"]
                forage_note = f"bringing back {food_home} food"
                # water_gathered never accrues once _is_settled_near_water is true (see
                # the matching outbound/returning-leg gate above) -- omit it from the
                # note entirely rather than always reporting a flat "0 water".
                if exp["water_gathered"]:
                    forage_note += f" and {exp['water_gathered']} water foraged along the way"
                else:
                    forage_note += " foraged along the way"
                # Only EXPLORATION_PARTY ever populates these -- see actions.py.
                # _exploration_party. .get(..., 0) leaves scout/hunt/trade untouched.
                wood_home, stone_home = exp.get("wood_gathered", 0), exp.get("stone_gathered", 0)
                if wood_home or stone_home:
                    tribe.wood += wood_home
                    tribe.stone += stone_home
                    forage_note += f", {wood_home} wood and {stone_home} stone"
                recipient = f"Chief {tribe.chief_name}" if tribe.chief_name else "the tribe"

                if exp.get("kind") == "hunt":
                    self._report_hunting_party_home(tribe, exp, scout, forage_note, recipient)
                    return True

                if exp.get("kind") == "trade":
                    self._report_trade_emissary_home(tribe, exp, scout, forage_note, recipient)
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
                    # 2026-09-02 rework ("a twisted sparse matrix assignment based on
                    # the existing map"): lumber/wildlife/quarry/mine sites are no
                    # longer decided fresh on every report -- they're real, pre-seeded
                    # locations (world.site_seed_points) a scout discovers by landing
                    # within world.SITE_DISCOVERY_RADIUS of one, not an independent
                    # chance roll on their exact tile. Solves the earlier fairness
                    # fix's own remaining gap for free: each site type has its own
                    # independent seed set, so two types can no longer stack on the
                    # same coordinate by construction, no nudging needed.
                    grid_size = self.world.grid_size
                    lumber_found = find_nearby_site("lumber", tx, ty, grid_size, set(tribe.lumber_sites))
                    if lumber_found is not None:
                        tribe.lumber_sites.append(lumber_found)
                    known_wildlife = {(s["x"], s["y"]) for s in tribe.wildlife_sites}
                    wildlife_found = find_nearby_site("wildlife", tx, ty, grid_size, known_wildlife)
                    if wildlife_found is not None:
                        wx, wy = wildlife_found
                        site_type = random.choice(WILDLIFE_SITE_TYPES)
                        tribe.wildlife_sites.append({"x": wx, "y": wy, "type": site_type})
                        if tribe.last_celebration_cycle != self.cycle:
                            self._celebrate_game_discovery(tribe, wx, wy)
                    quarry_found = find_nearby_site("quarry", tx, ty, grid_size, set(tribe.quarry_sites))
                    if quarry_found is not None:
                        tribe.quarry_sites.append(quarry_found)
                    # Explicit request: "Mines can [also] contain the Unique
                    # Resource of the Biome (these locations are scattered about
                    # the map)." Same pre-seeded discovery as above; the one
                    # deliberate exception stays -- a mine's resource name is read
                    # off whatever real biome the pre-seeded point itself sits on
                    # (world.UNIQUE_RESOURCE_BY_BIOME), not the scout's own tile.
                    known_mines = {(site["x"], site["y"]) for site in tribe.mine_sites}
                    mine_found = find_nearby_site("mine", tx, ty, grid_size, known_mines)
                    if mine_found is not None:
                        mx, my = mine_found
                        mine_biome = biome_at(mx, my)
                        resource_name = UNIQUE_RESOURCE_BY_BIOME.get(mine_biome, "Unknown Ore")
                        tribe.mine_sites.append({"x": mx, "y": my, "biome": mine_biome, "resource": resource_name})
                        tribe.history.append(
                            f"{scout} also reports something rarer at ({mx},{my}) -- a vein of "
                            f"{resource_name}, waiting to be excavated"
                        )
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

    def _expedition_river_hazard(self, tribe: Tribe, x: int, y: int) -> bool:
        """The same drowning risk GATHER_WATER already carries on a river tile
        (config.DROWNING_HAZARD_CHANCE) -- a traveling party crossing or camped on a
        river isn't any safer than a tribe standing on one to fill jugs. Returns True
        if it claimed someone (population loss and trauma already applied).

        Losing someone this way also becomes a real, remembered lesson, not just a
        chronicle line that scrolls away: a high-weight memory entry (see
        TribeMemory.consolidate) is what actually promotes into a standing taboo the
        tribe's own reasoning sees every future turn -- the survivors telling the rest
        of the tribe what happened, not the simulation warning them directly."""
        if biome_at(x, y) != "river" or random.random() >= config.DROWNING_HAZARD_CHANCE:
            return False
        self.trauma.radiate_event_wave(x, y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS)
        self._lose_population(tribe, config.DROWNING_HAZARD_POPULATION_LOSS, cause="drowning")
        tribe.history.append("the river's current pulled someone under while the party was crossing -- the survivors turn back to warn the others")
        tribe.memory.remember(
            f"A river crossing near ({x},{y}) drowned one of our own -- the current there is a real danger.",
            self.cycle, weight=0.85,
        )
        return True

    def _volcano_hazard(self, tribe: Tribe, x: int, y: int) -> bool:
        """Explicit correction: "the volcano is a Hazard they will die if they go
        there." Same shape as _expedition_river_hazard just above, far more
        severe (config.VOLCANO_HAZARD_CHANCE/_POPULATION_LOSS vs. drowning's own
        DROWNING_HAZARD_CHANCE/_POPULATION_LOSS) -- this needs to read as a
        serious, well-known danger, not a mild river crossing. Called from every
        real way a tribe's people could end up on that tile: expedition movement
        (the same two call sites _expedition_river_hazard already hooks) and
        RELOCATE (actions._relocate) -- broader coverage than the river hazard
        gets, since "if they go there" has to mean any of them, not just an
        expedition passing through."""
        if biome_at(x, y) != "volcano" or random.random() >= config.VOLCANO_HAZARD_CHANCE:
            return False
        self.trauma.radiate_event_wave(x, y, config.VOLCANO_TRAUMA_MAGNITUDE, config.VOLCANO_TRAUMA_RADIUS)
        self._lose_population(tribe, config.VOLCANO_HAZARD_POPULATION_LOSS, cause="volcano")
        tribe.history.append(f"the volcano's toxic fumes and scalding ground claimed lives near ({x},{y}) -- the survivors flee, warning the rest to stay away")
        tribe.memory.remember(
            f"The volcano near ({x},{y}) is deadly -- real danger there, stay away.",
            self.cycle, weight=0.9,
        )
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
            looted = {
                resource: round(getattr(tribe, resource) * config.RAIDER_DEFEAT_LOOT_FRACTION)
                for resource in ("wood", "stone", "food")
            }
            for resource, amount in looted.items():
                setattr(tribe, resource, getattr(tribe, resource) + amount)
            self.trauma.radiate_event_wave(x, y, config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
            tribe.history.append(
                f"{exp['lead_scout']}'s boat runs down a raider band on the water near ({x},{y}) and heads "
                f"home with the counter-raid loot -- {looted['food']} food, {looted['wood']} wood, and "
                f"{looted['stone']} stone recovered"
            )
            self.recent_encounters.append({
                "x": x, "y": y, "kind": "raider_attack", "label": "Raiders routed by boat", "outcome": "repelled",
            })
            exp["phase"] = "returning"
            return True
        self.trauma.radiate_event_wave(x, y, config.RAIDER_SIGHTING_TRAUMA_MAGNITUDE, config.RAIDER_SIGHTING_TRAUMA_RADIUS)
        self._lose_population(tribe, config.EXPEDITION_RAIDER_AMBUSH_POPULATION_LOSS, cause="raider_ambush")
        self._record_raider_sighting(tribe, x, y)
        tribe.history.append(f"{exp['lead_scout']}'s party was ambushed by raiders near ({x},{y}) and flees for home")
        tribe.memory.remember(
            f"Raiders ambushed our party near ({x},{y}) -- real danger there.", self.cycle, weight=0.85,
        )
        self.recent_encounters.append({
            "x": x, "y": y, "kind": "raider_attack", "label": "Scouts ambushed", "outcome": "struck",
        })
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
        if self._expedition_river_hazard(tribe, px, py):
            exp["phase"] = "returning"
            return
        if self._volcano_hazard(tribe, px, py):
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

    def _advance_trade_emissary_outbound(self, tribe: Tribe, exp: dict, scout: str) -> None:
        """One outbound day for a SEND_TRADE_EMISSARY expedition (see
        actions.py._send_trade_emissary) -- nearly the same mechanic as
        HUNTING_PARTY's own outbound advance, per explicit confirmation: same
        day-by-day travel, same push-onward-then-give-up ending. Every day checks
        for a rival tribe within the same proximity instant TRADE already requires;
        finding one executes the exchange immediately, at the point of contact --
        the emissary still has to walk home to report it, but the goods have
        already moved for both sides."""
        px, py = exp["pos"]
        partner = _find_trade_partner(self, tribe, px, py)
        if partner is not None:
            _execute_trade(self, tribe, partner)
            exp["trade_partner"] = partner.name
            exp["phase"] = "returning"
            tribe.history.append(f"{scout}'s emissary finds {partner.name} and opens trade -- heading home to report")
            return

        tx, ty = exp["target"]
        if [px, py] == [tx, ty]:
            if not exp.get("pushed_onward"):
                exp["pushed_onward"] = True
                ex, ey = physics.extend_ray_to_grid_edge(exp["origin"][0], exp["origin"][1], tx, ty, self.world.grid_size)
                exp["target"] = [ex, ey]
                tribe.history.append(f"{scout}'s emissary finds no one at ({px},{py}) and pushes onward")
            else:
                exp["phase"] = "returning"
                tribe.history.append(
                    f"{scout}'s emissary reaches the edge of explored land after {exp['day']} days "
                    "with no one to trade with -- they turn back"
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

    def _report_trade_emissary_home(self, tribe: Tribe, exp: dict, scout: str, forage_note: str, recipient: str) -> None:
        """Arrival-home report for a SEND_TRADE_EMISSARY expedition -- the trade
        itself (if any) already happened the moment the emissary found a partner
        (see _advance_trade_emissary_outbound); this just tells the tribe what
        happened, the same "not real until you're home" shape _report_hunting_
        party_home uses for a catch."""
        partner_name = exp.get("trade_partner")
        if partner_name:
            tribe.history.append(
                f"{scout} is home and gives {recipient} a full report: traded with {partner_name}, {forage_note}"
            )
        else:
            tribe.history.append(
                f"{scout} is home and gives {recipient} a full report: found no one to trade with, {forage_note}"
            )

    def _report_hunting_party_home(self, tribe: Tribe, exp: dict, scout: str, forage_note: str, recipient: str) -> None:
        # See actions.py._hunt_deer's own matching Tannery meat bonus comment --
        # same flat pre-multiplier addition, applied here too so a multi-day
        # hunting party's catch benefits the same way an instant hunt does.
        base_caught = exp.get("food_caught", 0)
        if base_caught and tribe.tannery_built:
            base_caught += config.TANNERY_MEAT_BONUS_PER_HUNT
        caught = round(base_caught * _food_multiplier(tribe))
        if caught:
            tribe.food += caught
            tribe.expeditions_succeeded += 1
            tribe.hunt_successes += 1
            tribe.hunt_ever_succeeded = True  # see actions.py._cook_food's own prerequisite
            if tribe.hunt_successes == config.MILESTONE_HUNT_SUCCESSES:
                self._award_trophy(tribe, "Master Hunter", individual=scout)
            self._check_custom_awards(tribe, "hunting", individual=scout)
            tribe.history.append(
                f"{scout}'s hunting party is home and gives {recipient} a full report: "
                f"{caught} food caught, {forage_note}"
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
            - config.RAIDER_STRENGTH_DEFENSE_PENALTY_AT_MAX * raider_strength
        ))
        if random.random() < defense_chance:
            tribe.raids_defended += 1
            self._award_trophy(tribe, "Raid Breaker")
            looted = {
                resource: round(getattr(tribe, resource) * config.RAIDER_DEFEAT_LOOT_FRACTION * raider_strength)
                for resource in ("wood", "stone", "food")
            }
            for resource, amount in looted.items():
                setattr(tribe, resource, getattr(tribe, resource) + amount)
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
        self._lose_population(tribe, config.STARVATION_POPULATION_LOSS, cause="starvation")

    def _dehydrate(self, tribe: Tribe) -> None:
        tribe.history.append("thirst claimed lives")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DEHYDRATION_TRAUMA_MAGNITUDE, config.DEHYDRATION_TRAUMA_RADIUS
        )
        self._lose_population(tribe, config.DEHYDRATION_POPULATION_LOSS, cause="thirst")

    def _grow_population(self, tribe: Tribe) -> None:
        if tribe.food > config.POPULATION_GROWTH_FOOD_THRESHOLD and tribe.population < config.POPULATION_GROWTH_CAP:
            tribe.population += 1
            tribe.food -= config.POPULATION_GROWTH_FOOD_COST
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
            if getattr(tribe, resource, 0) < round(minimum * (1 - discount)):
                return

        for resource, amount in nxt.advancement_cost.items():
            discounted = round(amount * (1 - discount))
            setattr(tribe, resource, max(0, getattr(tribe, resource) - discounted))
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

    def _capped_unique_add(self, tribe: Tribe, resource_name: str, amount: int) -> None:
        """Same as _capped_add, for the tribe.unique_resources dict (Mine ore,
        Tannery Fur) -- each named resource gets its own cap ceiling, same as
        every other resource."""
        cap = _storage_cap(tribe)
        current = tribe.unique_resources.get(resource_name, 0)
        added = max(0, min(amount, cap - current))
        tribe.unique_resources[resource_name] = current + added

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

    def _check_road_evolution(self, x: int, y: int) -> None:
        """Explicit request: "a road will be similar [to the wall] but a big
        episode of major development making trade far more likely and travel
        easier." A trail silently flips World.is_toll_road the instant its
        lifetime crossings pass config.ROAD_EVOLVE_CROSSINGS -- this catches the
        exact cycle that happens (crossings == threshold + 1, i.e. the crossing
        that just tipped it over) and turns it into a real, one-time celebration
        for whichever tribe actually owns this tile (World.road_owner -- the
        first tribe to ever walk it, not necessarily whoever's walking it right
        now that pushed it over). Called after every World.wear_trail site."""
        entry = self.world.trails.get((x, y))
        if entry is None or entry.get("crossings", 0) != config.ROAD_EVOLVE_CROSSINGS + 1:
            return
        owner_id = entry.get("owner")
        owner = self.tribes.get(owner_id) if owner_id else None
        if owner is None or owner.extinct:
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
        GATHER_WATER still works and still adds more on top of this."""
        if self._is_settled_near_water(tribe):
            upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
            well_bonus = config.WELL_SUPPLY_BONUS_MULTIPLIER if tribe.well_built else 1.0
            self._capped_add(tribe, "water", round(upkeep * config.SETTLED_WATER_SUPPLY_MULTIPLIER * well_bonus))

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
        above) already covers that resource once built and fetched."""
        if tribe.territory_center is None:
            return
        upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
        amount = max(1, round(upkeep * config.IN_TERRITORY_SITE_YIELD_MULTIPLIER))
        if any(self._site_in_own_territory(tribe, x, y) for x, y in tribe.lumber_sites):
            self._capped_add(tribe, "wood", amount)
        if any(self._site_in_own_territory(tribe, s["x"], s["y"]) for s in tribe.wildlife_sites):
            self._capped_add(tribe, "food", amount)
        if any(self._site_in_own_territory(tribe, x, y) for x, y in tribe.quarry_sites):
            self._capped_add(tribe, "stone", amount)

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
        once per cycle, the same wear_trail mechanic RELOCATE/SCOUT already
        use along the straight line between the settlement and each site --
        heavily-used routes eventually evolve into real toll roads themselves
        (see world.is_toll_road), exactly like any other trail would."""
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
            harvested = round(
                config.CROP_HARVEST_YIELD * tribe.farm_plots * _labor_multiplier(tribe.population)
                * _food_multiplier(tribe)
            )
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
        flock grow past a dozen, the surplus is automatically eaten as food each
        cycle rather than piling up forever with no payoff."""
        if tribe.eggs > config.LIVESTOCK_SURPLUS_THRESHOLD:
            surplus = tribe.eggs - config.LIVESTOCK_SURPLUS_THRESHOLD
            tribe.eggs = config.LIVESTOCK_SURPLUS_THRESHOLD
            self._capped_add(tribe, "food", surplus * config.EGG_FEAST_FOOD_VALUE)
        if tribe.flock > config.LIVESTOCK_SURPLUS_THRESHOLD:
            surplus = tribe.flock - config.LIVESTOCK_SURPLUS_THRESHOLD
            tribe.flock = config.LIVESTOCK_SURPLUS_THRESHOLD
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

    def _found_territory(self, tribe: Tribe) -> None:
        """Grants a real, owned territory the instant a tribe first qualifies as
        settled (has_ever_settled) -- explicit request: "when a Tribe becomes
        Settled, they automatically get a region of territory they own." Anchors
        territory_center at the founding coordinate (deliberately not tribe.x/y --
        see Tribe.__init__'s own comment on why RELOCATE can't be allowed to drag a
        city's buildings/walls around), builds the first wall ring, and places the
        Town Hall centered on that same coordinate."""
        tribe.territory_center = (tribe.x, tribe.y)
        tribe.territory_radius = config.WALL_RING_RADIUS_STEP
        tribe.wall_rings = [city_layout.build_ring(self.world, tribe, ring_index=0)]
        w, h = config.BUILDING_FOOTPRINTS["town_hall"]
        architect.record_building(tribe, "town_hall", tribe.x - w // 2, tribe.y - h // 2, w, h, self.cycle)

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
        attacker.wood += defender.wood
        attacker.stone += defender.stone
        attacker.food += defender.food
        attacker.water += defender.water
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
