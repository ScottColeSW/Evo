"""The action registry: each action name maps to a handler function that mutates a
tribe's state and returns an optional hazard note for the chronicle. Adding a new
action means registering a handler here, not extending an if/elif chain in
Simulation -- this is the Registry Factory referenced in the README/design notes.

Handler signature: (sim: Simulation, tribe: Tribe, biome: str, target: tuple[int, int])
-> str | None

Only RELOCATE actually moves the tribe. Everything else happens wherever the tribe
currently stands -- gathering wood doesn't require packing up camp. This split exists
because the tribe's own visible individuals are a home camp, not a single wandering
point: real settlements explore via scouts before the whole body relocates, they don't
drift a little every time someone chops wood.
"""

import math
import random

from . import architect, city_layout, config, physics
from .might import compute_might
from .world import BIOME_LABELS, biome_at, mark_visited_sector, sector_of


# Real biomes don't hand out every resource equally -- a mountain has essentially no
# game to hunt, a forest has no stone to quarry. Previously GATHER_WOOD/GATHER_STONE/
# HUNT_DEER paid the same flat yield in every biome (only local depletion scaled them),
# so a tribe standing on a bare mountain peak could "hunt deer" as effectively as one
# deep in a forest -- resources were an abstract number with no connection to what was
# actually around them. GATHER_WATER was already biome-aware (river vs. elsewhere);
# this brings the other three in line with it.
BIOME_YIELD_MULTIPLIER = {
    "wood": {"forest": 1.0, "plains": 0.4, "river": 0.3, "lake": 0.3, "mountains": 0.15,
             "cliffs": 0.0, "shoals": 0.05, "ocean": 0.0, "desert": 0.05, "volcano": 0.0},
    # Live data, 2026-09-01: a tribe settled near a lake (not mountains -- settling is
    # already pulled hard toward confirmed water, which mountains rarely coincide with)
    # accumulated wood 4712 vs. stone 11 over ~600 cycles, and never built Quarry or
    # Mine despite knowing 4 real Mine sites -- both cost 30 stone, essentially
    # unreachable at the old 0.1x off-mountain rate (a 10x gap against wood's 1.0x in
    # forest). Raised to 0.25x: mountains/cliffs stay clearly the real place to get
    # stone, but a lake- or forest-settled tribe can now actually bootstrap a Quarry
    # instead of being structurally locked out of the entire stone-building tree.
    "stone": {"mountains": 1.0, "forest": 0.25, "plains": 0.25, "river": 0.25, "lake": 0.25,
              "cliffs": 0.5, "shoals": 0.05, "ocean": 0.0, "desert": 0.15,
              # Deliberately low, not mirrored from mountains' 1.0 -- the volcano
              # is a hazard to avoid (config.VOLCANO_HAZARD_CHANCE), not a resource
              # destination; a strong stone yield there would wrongly incentivize
              # walking into it.
              "volcano": 0.1},
    "game": {"forest": 1.0, "plains": 0.6, "river": 0.3, "lake": 0.3, "mountains": 0.15,
             "cliffs": 0.05, "shoals": 0.1, "ocean": 0.0, "desert": 0.05, "volcano": 0.0},
    # Foraging (berries, fruit, wild plants) used to not exist at all -- food only ever
    # came from HUNT_DEER/HUNTING_PARTY, both carrying the same wolf-pack risk, so there
    # was no low-risk food option the way GATHER_WATER is a low-risk (if lower-yield)
    # alternative to a river tile. Deliberately profiled opposite to "game": plains is
    # the best foraging ground (open land, berries, roots), forest is only secondary --
    # real tension between forest's higher-risk/higher-yield hunting and plains' safe,
    # steady foraging, rather than one biome just being strictly best at everything.
    "forage": {"plains": 1.0, "forest": 0.6, "river": 0.4, "lake": 0.4, "mountains": 0.1,
               "cliffs": 0.0, "shoals": 0.1, "ocean": 0.0, "desert": 0.1, "volcano": 0.0},
}

# Which species word to use in a wildlife sighting (see Simulation._build_visible_entities)
# for whichever hunting action is currently unlocked -- keyed by action name, used as the
# fallback for any biome GAME_SPECIES_BY_BIOME doesn't cover.
GAME_SPECIES_LABEL = {"HUNT_DEER": "deer"}

# Species flavor by biome, keyed for the sighting's *actual* location -- deer only in a
# forest, small game the plains are really home to, wildfowl by water. Purely narrative
# (the sighting fact and the hunt yield are unaffected by which name gets used), but it
# stops every wildlife sighting reading identically ("signs of deer nearby") regardless
# of where a tribe actually stands.
GAME_SPECIES_BY_BIOME = {
    "forest": ("deer", "wild boar"),
    "plains": ("rabbits", "groundbirds"),
    "river": ("waterfowl",),
    "lake": ("waterfowl",),
    "mountains": ("mountain goats",),
    "desert": ("desert hares", "sand lizards"),
}


def _food_multiplier(tribe) -> float:
    """Cooking's real effect, redesigned 2026-09-02 to match the same "multiplier
    applied at the point of harvest" shape SAWMILL_WOOD_MULTIPLIER/
    QUARRY_STONE_MULTIPLIER/DOCK_FISH_CATCH_BONUS_FRACTION already use -- it used to
    instead divide food *consumption* in Simulation._apply_upkeep, the odd one out
    against every other resource-mastery building. Applied at every real
    food-production point (GATHER_FOOD, HUNT_DEER/HUNTING_PARTY, CATCH_FISH, passive
    fish supply, crop harvest) -- never to loot/pillage transfers, which move
    existing stockpiled food rather than producing new food. Kitchen only means
    anything once cooking is already known (_build_kitchen itself requires
    cooking_learned), so this doesn't need to guard against kitchen_built alone."""
    multiplier = 1.0
    if tribe.cooking_learned:
        multiplier *= config.COOKING_FOOD_MULTIPLIER
    if tribe.kitchen_built:
        multiplier *= config.KITCHEN_FOOD_MULTIPLIER
    # Object Creator era's gather_boost effect -- see _created_object_bonus.
    multiplier *= 1 + _created_object_bonus(tribe, "gather_boost")
    return multiplier


def _labor_multiplier(population: int) -> float:
    """More hands means more gathered per action -- upkeep (Simulation._apply_upkeep)
    already scales with population, but yield never did, so a bigger tribe was strictly
    worse off per-capita: identical output from one GATHER_WOOD regardless of whether 3
    or 30 people stood behind it, against a food/water cost that only ever grew.
    POPULATION_YIELD_BASELINE matches Tribe.__init__'s own starting population, so this
    never scales a tribe at or below starting size down -- only ever rewards growth
    past it.

    2026-09-08 rework (live report: a day-12 run had a tribe sitting on 16 wood at
    population 4767, unable to ever afford CONSTRUCT_WALL's 60-wood threshold):
    the flat ratio this used to be (population / POPULATION_YIELD_BASELINE) was
    exactly why config.LABOR_MULTIPLIER_CAP existed at all -- linear and
    unbounded, it hit 404x at the population (3232) that originally produced a
    single 1,854-food HUNT_DEER windfall large enough to spike wellbeing's
    physiological tier and reopen population growth during a real famine. Capping
    it at a flat 5.0x closed that explosion, but a flat cap can only ever pick one
    of two problems to have: reached almost immediately (population 40) and
    flatlined forever after, or reached at a believable population and still
    exploding past it. A tribe of 4767 or 16,574 (both real, this same live run)
    was strictly worse off per-capita than one of exactly 40, the identical bug
    this function was built to fix in the first place, just moved further out.

    sqrt(population / POPULATION_YIELD_BASELINE) keeps growing at every
    population instead of hitting a wall, but decelerates fast enough that it
    never reproduces the 404x-style explosion -- population 3232 now yields
    ~20x (not 5x, not 404x), population 16,574 yields ~45x. config.
    LABOR_MULTIPLIER_CAP is kept as a real backstop (never an unbounded
    ratchet, same standing principle MAX_WALL_RINGS/POPULATION_GROWTH_CAP
    already hold elsewhere), just raised far enough out that it's a genuine
    safety net again rather than the everyday ceiling every tribe past 40
    people was already slamming into."""
    return min(config.LABOR_MULTIPLIER_CAP, max(1.0, math.sqrt(population / config.POPULATION_YIELD_BASELINE)))


def _harvest(sim, tribe, resource_key, base_yield, biome):
    """Shared depletion logic: yield at this tile shrinks the more it's been harvested
    recently, and harvesting here raises that further. Capped below total depletion
    (config.MAX_SCARCITY) so staying put is costly, not a guaranteed dead end. Also
    scales by how much this biome actually supports the resource in the first place --
    see BIOME_YIELD_MULTIPLIER -- and by how many people this tribe actually has to put
    to work -- see _labor_multiplier."""
    biome_factor = BIOME_YIELD_MULTIPLIER.get(resource_key, {}).get(biome, 1.0)
    scarcity = sim.world.scarcity(resource_key, tribe.x, tribe.y)
    labor_factor = _labor_multiplier(tribe.population)
    yield_amount = round(base_yield * biome_factor * labor_factor * (1 - scarcity))
    sim.world.deplete(resource_key, tribe.x, tribe.y, config.DEPLETION_PER_HARVEST, config.MAX_SCARCITY)
    return yield_amount


def expedition_capacity(tribe) -> int:
    """How many expedition parties (scouting or hunting, any mix) this tribe can have
    out at once. config.MAX_CONCURRENT_EXPEDITIONS is the floor -- a tribe of 8
    (starting population) still gets exactly that many, and a larger tribe can spare
    more search bandwidth, the same way its upkeep cost already scales with population
    (see Simulation._apply_upkeep). Live bug report: "we need to limit the number of
    scouts and gatherers at a time... my system was throttled" -- a real run reached
    population 352, which uncapped this to 70 concurrent expeditions for one tribe
    alone. config.MAX_CONCURRENT_EXPEDITIONS_CEILING caps the per-capita growth for
    real, so a tribe's search capacity keeps scaling with population without ever
    flooding the board."""
    uncapped = max(config.MAX_CONCURRENT_EXPEDITIONS, tribe.population // config.EXPEDITION_SLOT_POPULATION_DIVISOR)
    return min(uncapped, config.MAX_CONCURRENT_EXPEDITIONS_CEILING)


def _expedition_dispatch_blocked(tribe, kind: str) -> str | None:
    """Shared dispatch gate for SCOUT/EXPLORATION_PARTY/HUNTING_PARTY -- returns
    a real in-fiction reason the party can't go out this cycle, or None if it
    can.

    Explicit correction (2026-09-07): "Each Tribe can always send a max of 3
    Orders out. They can all be the same if they want. I think we have
    stopped that incorrectly." A per-kind cap of one live party at a time
    (added for an earlier, different live bug report -- the board feeling
    "flooded" from stacking several of the same kind at once) was
    unconditionally forcing kind *diversity*: a tribe could never have, say,
    three scouts out together even with real capacity to spare, which read
    live as scouts being permanently "stuck" at one no matter what the tribe
    actually wanted. expedition_capacity(tribe) alone is the real limit
    again -- any mix, including three of the same kind, is fine as long as
    the total stays under it."""
    if len(tribe.expeditions) >= expedition_capacity(tribe):
        fields = ", ".join(
            f"{e['lead_scout']} (day {e['day']}/{e['max_days']}, {e['phase']})" for e in tribe.expeditions
        )
        return f"no one left to send -- every party is already out: {fields}"
    return None


def _storage_cap(tribe) -> int:
    """A generous ceiling, not a tight one -- STORAGE_CAP_BASE alone already clears
    every era's own resource requirement, so this is never the reason a tribe can't
    advance. It only ever catches genuinely excessive hoarding (explicit request,
    after a live run showed a tribe pile wood up to 200+ while permanently starved
    on stone). Repeatable Warehouses raise it further -- 'expansion of the tribe
    will allow that to scale storage with building needs.'

    Explicit follow-up, 2026-09-09: warehouses_built is now capped at
    config.WAREHOUSE_MAX_COUNT (real data: 47 built in one run) -- growth past
    that comes from tribe.warehouse_upgrades (UPGRADE_WAREHOUSE) instead, same
    per-tier bonus so the curve stays continuous right through the cap."""
    return (
        config.STORAGE_CAP_BASE
        + tribe.warehouses_built * config.WAREHOUSE_STORAGE_BONUS_PER_BUILDING
        + tribe.warehouse_upgrades * config.WAREHOUSE_STORAGE_BONUS_PER_BUILDING
    )


def _sustainable_population(tribe) -> int:
    """Real, resource-grounded population ceiling -- explicit request,
    2026-09-09: "so, pop. is unbounded? that's probably the reason for a
    lot of problems. We need to put a reasonable limit on this." See
    Simulation._grow_population's own docstring for the full derivation
    (the "Infinity Food/Water" security mechanics only ever top a tribe up
    to _storage_cap; population growing past storage_cap * upkeep-divisor
    means upkeep permanently outpaces even a maxed-out top-up). Every
    spontaneous-birth trigger in this codebase -- BREED itself, the
    night-cycle random breed chance, several celebration-linked
    opportunities -- already gated itself on `population <
    config.POPULATION_GROWTH_CAP`; that gate just went inert once
    POPULATION_GROWTH_CAP became infinite (an earlier explicit "no
    arbitrary cap" request). This is the one real number every one of
    those call sites should compare against instead, so there's a single
    source of truth for "does this tribe have real room to grow" rather
    than a second, parallel population-limit mechanism."""
    return _storage_cap(tribe) * config.UPKEEP_POPULATION_DIVISOR


def _has_room_to_grow(tribe) -> bool:
    return tribe.population < _sustainable_population(tribe)


def _add_capped(sim, tribe, resource: str, amount: int, label: str) -> str | None:
    """Adds `amount` of `resource` up to _storage_cap, returning a real in-fiction
    outcome (like any other action's own result) instead of silently discarding the
    overflow -- explicit design goal: the tribe should be *told*, as the actual
    result of the turn it just took, not have it vanish with no explanation.

    Explicit follow-up: "these guys need punishment for choosing the wrong
    thing... for waste when they overfill the storage." Real waste -- current
    stores already full, or this harvest partially wasted -- radiates a real
    negative trauma wave (config.WASTE_TRAUMA_MAGNITUDE), not just a narrated
    warning with no consequence."""
    cap = _storage_cap(tribe)
    current = getattr(tribe, resource)
    if current >= cap:
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.WASTE_TRAUMA_MAGNITUDE, config.WASTE_TRAUMA_RADIUS)
        return f"the {label} stores are already full -- nothing more fits"
    added = min(amount, cap - current)
    setattr(tribe, resource, current + added)
    if added < amount:
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.WASTE_TRAUMA_MAGNITUDE, config.WASTE_TRAUMA_RADIUS)
        return f"the {label} stores are nearly full -- only {added} of {amount} fits"
    return None


def _gather_wood(sim, tribe, biome, target):
    # Explicit request: "saw mill turns 1 wood into 3 wood" -- a permanent
    # multiplier on every future harvest once built (config.SAWMILL_WOOD_MULTIPLIER),
    # not a separate conversion action spent on the stockpile. Same "3x via a
    # multiplier applied once at the point of harvest" shape cooking already uses.
    amount = _harvest(sim, tribe, "wood", 10, biome)
    if tribe.sawmill_built:
        amount *= config.SAWMILL_WOOD_MULTIPLIER
    if amount > 0:
        tribe.wood_ever_gathered = True  # see actions.py._build_sawmill's own prerequisite
    return _add_capped(sim, tribe, "wood", amount, "wood")


def _gather_stone(sim, tribe, biome, target):
    # Explicit request: "quarried stone is also worth 3 times as much as a
    # harvested stone" -- mirrors _gather_wood's sawmill multiplier exactly.
    amount = _harvest(sim, tribe, "stone", 10, biome)
    if tribe.quarry_built:
        amount *= config.QUARRY_STONE_MULTIPLIER
    if amount > 0:
        tribe.stone_ever_gathered = True  # see actions.py._build_quarry's own prerequisite
    return _add_capped(sim, tribe, "stone", amount, "stone")


def _gather_water(sim, tribe, biome, target):
    # A lake is calmer than a river's current -- same drinkable status and yield, but
    # no drowning risk (see world.py's LAKE_CENTER/_is_lake).
    if biome == "river" and random.random() < config.DROWNING_HAZARD_CHANCE:
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.DROWNING_HAZARD_POPULATION_LOSS, cause="drowning")
        # Live report: "it sounds like the Hazard didn't get reported back to
        # the Tribe properly... there should be a skull-n-crossbones marker."
        # Same missing-marker gap as Simulation._expedition_river_hazard/
        # _volcano_hazard's matching fix -- a hazard death needs to show on
        # the map the same way a raider ambush/wolf attack already does.
        sim.recent_encounters.append({
            "x": tribe.x, "y": tribe.y, "kind": "hazard_death", "label": "Lost to the river", "outcome": "struck",
        })
        return "the river's current pulled someone under"
    base = config.WATER_YIELD_RIVER if biome in ("river", "lake") else config.WATER_YIELD_OFF_RIVER
    return _add_capped(sim, tribe, "water", _harvest(sim, tribe, "water", base, biome), "water")


def _hunt_deer(sim, tribe, biome, target):
    if biome == "forest" and random.random() < config.HUNT_HAZARD_CHANCE:
        tribe.food = max(0, tribe.food - config.HUNT_HAZARD_FOOD_LOSS)
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.HUNT_HAZARD_TRAUMA_MAGNITUDE, config.HUNT_HAZARD_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.HUNT_HAZARD_POPULATION_LOSS, cause="wolf_attack")
        # Explicit request: "I do want to see the Wolves encountered marked for
        # them" -- every other hazard/conflict (raids, camp strikes) already gets a
        # momentary map marker via recent_encounters; the wolf-pack hazard never did.
        sim.recent_encounters.append({
            "x": tribe.x, "y": tribe.y, "kind": "wolf_attack",
            "label": "Wolf pack!", "outcome": "struck",
        })
        return "a wolf pack struck the hunting party"
    base = _harvest(sim, tribe, "game", 15, biome)
    if tribe.tannery_built:
        # Explicit request: "it also gives the meat to the kitchen (2 meat per
        # catch) which cooks it (multiplier)" -- a flat bonus folded into the
        # same pre-multiplier harvest amount, not a separate resource, so it
        # rides the existing cook/kitchen multiplier chain like any other food.
        base += config.TANNERY_MEAT_BONUS_PER_HUNT
    amount = round(base * _food_multiplier(tribe))
    tribe.hunt_ever_succeeded = True  # see actions.py._cook_food's own prerequisite
    # See actions.py._build_deer_pen -- a real hunt-success count, distinct from
    # hunt_ever_succeeded's single flip, since the Deer Pen's own gate is "3-5
    # successful hunts," not just one.
    tribe.hunt_deer_success_count += 1
    return _add_capped(sim, tribe, "food", amount, "food")


def _forage(sim, tribe, biome, target):
    """Berries, fruit, and wild plants -- a real low-risk food option, unlike
    HUNT_DEER/HUNTING_PARTY which both carry wolf-pack risk. Lower base yield than
    hunting (10 vs. 15) since safety is the whole point: foraging trades hunting's
    higher ceiling for a guaranteed, no-hazard return."""
    amount = round(_harvest(sim, tribe, "forage", 10, biome) * _food_multiplier(tribe))
    tribe.foraged_ever_succeeded = True  # see Simulation._advance_automatic_fire
    return _add_capped(sim, tribe, "food", amount, "food")


def _already_built(sim, tribe, kind):
    existing = sim.world.constructions.get((tribe.x, tribe.y))
    return existing is not None and existing["type"] == kind and existing.get("progress", 100) >= 100


def _build_fire(sim, tribe, biome, target):
    # Without this, repeatedly choosing BUILD_FIRE at an already-built tile radiated
    # more ancestral pride every time at zero additional benefit -- a self-reinforcing
    # loop that made staying in one spot forever look increasingly attractive. A second
    # fire where one already burns accomplishes nothing.
    if _already_built(sim, tribe, "fire") or tribe.wood < config.BUILD_FIRE_WOOD_COST:
        return None
    tribe.wood -= config.BUILD_FIRE_WOOD_COST
    sim.world.add_construction(tribe.x, tribe.y, "fire", sim.cycle)
    tribe.fire_ever_built = True  # see actions.py._cook_food's own prerequisite
    # Fire is available from the very first era, long before a tribe has any real
    # territory (see Tribe.territory_center) -- placement is best-effort, never a
    # gate: a nomadic tribe's fire just isn't tracked positionally yet.
    if tribe.territory_center is not None:
        slot = architect.find_free_slot(sim.world, tribe, "fire")
        if slot is not None:
            architect.record_building(tribe, "fire", slot[0], slot[1], 1, 1, sim.cycle)
    sim.trauma.radiate_event_wave(
        tribe.x, tribe.y, config.BUILD_FIRE_PRIDE_MAGNITUDE, config.BUILD_FIRE_PRIDE_RADIUS
    )
    return None


def _cook_food(sim, tribe, biome, target):
    """Explicit request: "if you learn to hunt successfully and you learn to build
    fire successfully, you should get the chance to learn cooking... then you can
    always cook and build fire anytime." Gated on real prerequisites (Simulation.
    _prepare_turn only offers this action once tribe.hunt_ever_succeeded and
    tribe.fire_ever_built are both true) rather than needing a fire currently
    standing at this exact tile -- cooking is a skill learned once, not something
    tied to a specific structure. One-way, like fishing_learned: once learned, it
    isn't unlearned. No further effect on its own here -- Simulation._celebration_
    cost charges less, and _food_multiplier (above) makes every future food
    harvest go further from then on."""
    if tribe.cooking_learned:
        return None
    tribe.cooking_learned = True
    sim._award_trophy(tribe, "Master Chef")
    return "the tribe learns to cook -- stored food will go much further from now on"


def _construct_wall(sim, tribe, biome, target):
    """Explicit request: a wall is built in stages, like a crop, not finished in one
    action -- "30% of a wall can be built through a day with a team of 3." Reuses
    _labor_multiplier (the same "more hands get more done" concept _harvest already
    uses) instead of a separate team-size notion: at Tribe.__init__'s starting
    population (labor multiplier 1.0), one action adds ~30% progress to a section,
    reaching completion in ~4 actions; a larger tribe builds faster.

    2026-09-02 redesign: the wall is a real polygon of positioned 1x5 sections
    (backend/city_layout.py), not one progress-bar tile. city_layout.
    next_wall_work_section picks whichever section needs work next -- unfinished
    construction anywhere, across every ring, before any reinforcement -- so this
    action always has a clear, single real target without the tribe needing to
    reason about which section that is.

    2026-09-08 merge (explicit request, after two independent live traces found the
    same failure): CONSTRUCT_WALL and EXPAND_TERRITORY used to be separate actions,
    and a small model reliably never sequenced them correctly -- one tribe called
    CONSTRUCT_WALL 100+ times against a ring with nothing unlocked (see the old
    _can_afford_construct_wall comment, still on record below at
    _expand_wall_territory), and EXPAND_TERRITORY itself sat affordable and alone on
    the menu across two separate A/B tests and got picked in roughly 1 of 8 runs.
    Two different fact-based nudges toward "pick the other one" already existed and
    neither worked. CONSTRUCT_WALL now IS the whole wall pipeline: work whichever
    section needs it, and when nothing does, unlock the next one (or open a new ring)
    automatically -- the model only ever has one decision to make ("build the wall"),
    not two dependent ones."""
    target_section = city_layout.next_wall_work_section(tribe)
    if target_section is None:
        return _expand_wall_territory(sim, tribe)
    ring_i, sec_i = target_section
    section = tribe.wall_rings[ring_i]["sections"][sec_i]

    if section["progress"] >= 100:
        # Explicit request: "Torches can be a freebie for building walls 2
        # levels" / "a Moat should be available after 2 layers of walls have
        # been built." A completed section can be reinforced with more tiers, up
        # to WALL_MAX_LAYERS -- a flat cost, not another multi-action progress
        # bar the way the very first pass was.
        if tribe.wood < config.WALL_LAYER_WOOD_COST or tribe.stone < config.WALL_LAYER_STONE_COST:
            return None
        tribe.wood -= config.WALL_LAYER_WOOD_COST
        tribe.stone -= config.WALL_LAYER_STONE_COST
        section["tier"] += 1
        return f"the {section['direction']} wall section is reinforced -- tier {section['tier']} of defense now stands there"

    # Explicit request: "if they choose Wall, they have to complete it, no
    # changing orders other than to collect what is needed to complete it."
    # Engages the moment CONSTRUCT_WALL is chosen for a still-incomplete section,
    # whether or not this particular call can actually afford to make progress --
    # otherwise a tribe with nothing stockpiled yet would never actually get
    # locked into "go gather, then come back." See Simulation._prepare_turn for
    # the available_actions narrowing this drives.
    tribe.wall_commitment_active = True

    added = min(100 - section["progress"], round(config.WALL_PROGRESS_PER_ACTION_BASE * _labor_multiplier(tribe.population)))
    wood_cost = round(config.WALL_WOOD_COST_TOTAL * added / 100)
    stone_cost = round(config.WALL_STONE_COST_TOTAL * added / 100)
    if tribe.wood < wood_cost or tribe.stone < stone_cost:
        return None

    tribe.wood -= wood_cost
    tribe.stone -= stone_cost
    section["progress"] += added
    if section["progress"] >= 100:
        tribe.wall_commitment_active = False
        return f"the {section['direction']} wall section is complete"
    return f"the {section['direction']} wall section continues -- {section['progress']}% complete"


def _build_moat(sim, tribe, biome, target):
    """Explicit request: "a Moat should be available after 2 layers of walls
    have been built." A cheaper alternative investment once the first wall ring
    is fully reinforced, not a replacement for the wall already standing --
    smaller cost, smaller bonus than a reinforcement tier (Simulation.
    _resolve_raider_attack). Excluded from the real building-footprint system --
    a moat is a property of the wall ring, not a placeable rect."""
    ring0_reinforced = bool(tribe.wall_rings) and city_layout.ring_fully_reinforced(tribe.wall_rings[0])
    if tribe.moat_built or not ring0_reinforced:
        return None
    if tribe.wood < config.MOAT_WOOD_COST or tribe.stone < config.MOAT_STONE_COST:
        return None
    tribe.wood -= config.MOAT_WOOD_COST
    tribe.stone -= config.MOAT_STONE_COST
    tribe.moat_built = True
    sim._award_trophy(tribe, "Moat Digger")
    return "a moat is dug around the camp -- a further defense bonus, cheaper than another wall layer"


def _long_house_fur_discount(tribe) -> tuple[int, int, int]:
    """(wood_cost, stone_cost, furs_to_consume) for the tribe's next Long House
    -- shared by _can_afford_build_long_house (simulation.py, the menu-
    availability check) and _build_long_house below, so the two can never drift
    apart on what a Long House actually costs. Spends whichever Fur the tribe
    already has banked, up to the point where either cost hits its floor
    (config.FUR_LONG_HOUSE_MIN_*_COST) -- see those constants' own comment for
    why a floor, not a full substitute."""
    max_furs_by_wood = (config.LONG_HOUSE_WOOD_COST - config.FUR_LONG_HOUSE_MIN_WOOD_COST) // config.FUR_LONG_HOUSE_WOOD_DISCOUNT
    max_furs_by_stone = (config.LONG_HOUSE_STONE_COST - config.FUR_LONG_HOUSE_MIN_STONE_COST) // config.FUR_LONG_HOUSE_STONE_DISCOUNT
    furs = min(tribe.unique_resources.get("Fur", 0), max_furs_by_wood, max_furs_by_stone)
    wood_cost = config.LONG_HOUSE_WOOD_COST - furs * config.FUR_LONG_HOUSE_WOOD_DISCOUNT
    stone_cost = config.LONG_HOUSE_STONE_COST - furs * config.FUR_LONG_HOUSE_STONE_DISCOUNT
    return wood_cost, stone_cost, furs


def _build_long_house(sim, tribe, biome, target):
    """Explicit correction: "most structures they only need 1 of. but house
    builds are dependant on population needs" -- repeatable, not a one-time
    flag, gated each time on real population need
    (config.HOUSING_POPULATION_PER_LONG_HOUSE) so a tribe can't spam housing it
    doesn't need. tribe.long_houses_built is also the real proxy the Keep/
    Fortress/Castle tier reads for how established this settlement has become.

    Explicit correction, 2026-09-09: "I'm very tempted to remove the Wall
    restriction on it." Dropped outright -- confirmed via two independent
    live runs that CONSTRUCT_WALL simply not getting chosen (not blocked,
    just never picked) stalled a tribe's entire Long House -> Kitchen chain
    both times, regardless of what else was fixed on the wall side. Defense
    and shelter no longer share a single point of failure; the wall still
    stands on its own real defensive merit, it just no longer gates
    anything else. This retires the wall_lock_long_house_credits system
    entirely (banked "let one through early" credits have nothing left to
    apply to) -- see Simulation._prepare_turn and _construct_wall's own
    updated comments.

    Explicit request: "'furs' can make the Long Houses more comfortable and
    easier to build" -- see _long_house_fur_discount for the actual cost math;
    furs consumed here (not just checked) so a discount can only ever be used
    once per Fur, never re-applied to a later Long House."""
    houses_needed = min(config.LONG_HOUSE_MAX_COUNT, max(1, -(-tribe.population // config.HOUSING_POPULATION_PER_LONG_HOUSE)))
    if tribe.long_houses_built >= houses_needed:
        return None
    wood_cost, stone_cost, furs_used = _long_house_fur_discount(tribe)
    if tribe.wood < wood_cost or tribe.stone < stone_cost:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "long_house")
    if slot is None:
        return None
    tribe.wood -= wood_cost
    tribe.stone -= stone_cost
    if furs_used:
        tribe.unique_resources["Fur"] -= furs_used
    w, h = config.BUILDING_FOOTPRINTS["long_house"]
    architect.record_building(tribe, "long_house", slot[0], slot[1], w, h, sim.cycle)
    tribe.long_houses_built += 1
    fur_note = f" ({furs_used} Fur worked in, cheaper and cozier)" if furs_used else ""
    if tribe.long_houses_built == 1:
        sim._award_trophy(tribe, "Master Builder")
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        return f"a long house rises{fur_note} -- the tribe has real, lasting shelter for the first time"
    return f"another long house rises{fur_note} -- {tribe.long_houses_built} now stand"


def _upgrade_long_house(sim, tribe, biome, target):
    """Explicit request, 2026-09-09: "modify long houses to scale like
    warehouse." BUILD_LONG_HOUSE stops offering itself past
    config.LONG_HOUSE_MAX_COUNT (see its own AFFORDABILITY_CHECKS entry) --
    this is what a settlement reaches for past that point instead, the same
    shape actions._upgrade_warehouse already uses. Deliberately no
    footprint/placement check, unlike a fresh build -- this expands what's
    already standing, not a new structure competing for space.

    Repeatable, but cost grows by config.LONG_HOUSE_UPGRADE_COST_GROWTH per
    tier already banked (tribe.long_house_upgrades), so it doesn't just
    become the same infinite-spam problem under a new name -- necessary
    given Fortress/Castle need 40/70 long-house-equivalents, far more than
    any tribe should ever place as literal buildings."""
    tier = tribe.long_house_upgrades
    wood_cost = round(config.LONG_HOUSE_UPGRADE_WOOD_COST_BASE * (1 + tier * config.LONG_HOUSE_UPGRADE_COST_GROWTH))
    stone_cost = round(config.LONG_HOUSE_UPGRADE_STONE_COST_BASE * (1 + tier * config.LONG_HOUSE_UPGRADE_COST_GROWTH))
    if tribe.wood < wood_cost or tribe.stone < stone_cost:
        return None
    tribe.wood -= wood_cost
    tribe.stone -= stone_cost
    tribe.long_house_upgrades += 1
    total = tribe.long_houses_built + tribe.long_house_upgrades
    return f"the standing long houses are expanded -- the settlement now supports {total} household{'s' if total != 1 else ''} worth of shelter"


def _build_keep(sim, tribe, biome, target):
    """Explicit request (original): "they can have 10 houses before they
    build a Keep" (lowered to 3, 2026-09-09). First tier of the defensive
    ladder after Long House -- a real additional defense bonus stacked on
    top of the wall's own (Simulation._resolve_raider_attack). Counts
    tribe.long_house_upgrades alongside long_houses_built, same as this
    action's own AFFORDABILITY_CHECKS entry -- see LONG_HOUSE_MAX_COUNT's
    own comment for why."""
    if tribe.keep_built:
        return None
    if (tribe.long_houses_built + tribe.long_house_upgrades) < config.KEEP_LONG_HOUSES_REQUIRED:
        return f"{config.KEEP_LONG_HOUSES_REQUIRED} long houses are needed before a keep is worth building here"
    if tribe.wood < config.KEEP_WOOD_COST or tribe.stone < config.KEEP_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "keep")
    if slot is None:
        return None
    tribe.wood -= config.KEEP_WOOD_COST
    tribe.stone -= config.KEEP_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["keep"]
    architect.record_building(tribe, "keep", slot[0], slot[1], w, h, sim.cycle)
    tribe.keep_built = True
    sim._award_trophy(tribe, "Keep Warden")
    return "a keep rises -- a further defense bonus for the settlement"


def _build_fortress(sim, tribe, biome, target):
    """Explicit request: "40 [houses] until they reach a Fortress." Second tier,
    gated on the Keep already standing."""
    if tribe.fortress_built:
        return None
    if not tribe.keep_built:
        return "a keep must be built before a fortress is worth building here"
    if (tribe.long_houses_built + tribe.long_house_upgrades) < config.FORTRESS_LONG_HOUSES_REQUIRED:
        return f"{config.FORTRESS_LONG_HOUSES_REQUIRED} long houses are needed before a fortress is worth building here"
    if tribe.wood < config.FORTRESS_WOOD_COST or tribe.stone < config.FORTRESS_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "fortress")
    if slot is None:
        return None
    tribe.wood -= config.FORTRESS_WOOD_COST
    tribe.stone -= config.FORTRESS_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["fortress"]
    architect.record_building(tribe, "fortress", slot[0], slot[1], w, h, sim.cycle)
    tribe.fortress_built = True
    sim._award_trophy(tribe, "Fortress Warden")
    return "a fortress rises -- a further defense bonus for the settlement"


def _build_castle(sim, tribe, biome, target):
    """Explicit request: "70 [houses] until they can build castles." Top tier of
    the defensive ladder, gated on the Fortress already standing -- a real
    additional defense bonus stacked on top of the wall's own (Simulation.
    _resolve_raider_attack), not just a bigger cosmetic building."""
    if tribe.castle_built:
        return None
    if not tribe.fortress_built:
        return "a fortress must be built before a castle is worth building here"
    if (tribe.long_houses_built + tribe.long_house_upgrades) < config.CASTLE_LONG_HOUSES_REQUIRED:
        return f"{config.CASTLE_LONG_HOUSES_REQUIRED} long houses are needed before a castle is worth building here"
    if tribe.wood < config.CASTLE_WOOD_COST or tribe.stone < config.CASTLE_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "castle")
    if slot is None:
        return None
    tribe.wood -= config.CASTLE_WOOD_COST
    tribe.stone -= config.CASTLE_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["castle"]
    architect.record_building(tribe, "castle", slot[0], slot[1], w, h, sim.cycle)
    tribe.castle_built = True
    sim._award_trophy(tribe, "Castle Builder")
    sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.ERA_ADVANCE_PRIDE_MAGNITUDE, config.ERA_ADVANCE_PRIDE_RADIUS)
    return "a castle rises -- the tribe's defenses are stronger than any wall alone could offer"


def _build_road(sim, tribe, biome, target):
    """A permanent, tribe-built version of the same trail_speed_bonus a well-worn
    path already grants expeditions (World.trail_speed_bonus, Simulation.
    _advance_one_expedition) -- flat, not distance-decayed like a trail, since a
    road exists deliberately rather than wearing in from repeated travel."""
    if tribe.road_built or tribe.wood < config.ROAD_WOOD_COST or tribe.stone < config.ROAD_STONE_COST:
        return None
    tribe.wood -= config.ROAD_WOOD_COST
    tribe.stone -= config.ROAD_STONE_COST
    tribe.road_built = True
    return "a road is built -- every future expedition will travel faster from here on"


def _expand_wall_territory(sim, tribe):
    """The other half of CONSTRUCT_WALL's merged pipeline (see its own docstring) --
    called only as a fallback once city_layout.next_wall_work_section finds nothing
    left to build or reinforce. Unlocks exactly one new wall section per call, in
    fixed compass order -- "expansion must be done for each wall section," no
    exception for ring 0. Once every section in the outermost ring is both
    unlocked and fully reinforced, the next call opens a whole new ring
    further out instead (backend/city_layout.build_ring) -- no limit on ring
    count beyond land availability.

    Until 2026-09-08 this was its own top-level action (EXPAND_TERRITORY),
    invokable independent of wall-progress state -- the "ring must be fully
    reinforced" branch below dates from that era and is normally unreachable now
    that _construct_wall only ever calls this once next_wall_work_section has
    already confirmed every unlocked section is both built and maxed (which
    already implies full reinforcement). Left in as a defensive guard rather than
    removed -- see tests/test_actions.py's own direct test of this function for
    why it's still worth keeping.

    Live-run correction: "Wall Sections are being rendered on screen as a box
    around the settlement instead of portions of Wall being placed just inside
    the Territory dotted outline." tribe.territory_radius (what frontend/
    index.html's drawTerritory actually draws) used to grow by its own
    separately-scaled increment every single call, completely independent of
    where the wall ring geometry actually sits -- a handful of calls could
    balloon the dotted outline far past the fixed-radius ring inside it.
    territory_radius is now simply derived from how many rings exist
    (config.WALL_RING_RADIUS_STEP * ring count), so the dotted outline always
    sits exactly at the current outermost ring's own real radius, matching
    what _found_territory already sets it to at founding. The land-availability
    scaling this replaced (Simulation._local_buildable_fraction) no longer has
    anything left to scale -- natural-barrier substitution in city_layout.
    build_ring and the one-section-per-call pace already self-limit expansion
    on cramped land without it."""
    if not tribe.wall_rings:
        return None
    if tribe.wood < config.TERRITORY_EXPANSION_WOOD_COST or tribe.stone < config.TERRITORY_EXPANSION_STONE_COST:
        return None

    unlockable = city_layout.next_unlockable_section(tribe)
    if unlockable is None:
        if not city_layout.ring_fully_reinforced(tribe.wall_rings[-1]):
            return "the outermost wall ring must be fully reinforced before territory can expand further"
        # See config.MAX_WALL_RINGS's own comment -- a real ceiling, not an
        # unbounded ratchet. Simulation._prepare_turn retires CONSTRUCT_WALL
        # from the choice set once this is hit, so reaching this branch at all
        # would mean the menu itself let a stale choice through; fail closed
        # rather than open a ring past the intended cap.
        if len(tribe.wall_rings) >= config.MAX_WALL_RINGS:
            return None
        new_ring = city_layout.build_ring(sim.world, tribe.territory_center, len(tribe.wall_rings))
        # See city_layout.cap_natural_barriers' own comment: this ring sits at a
        # larger radius from the same fixed territory_center ring 0 was searched
        # for, and was never itself checked against the "at most 1 natural
        # barrier" rule -- confirmed live, one came out 5/8 free water sections.
        city_layout.cap_natural_barriers(new_ring["sections"])
        tribe.wall_rings.append(new_ring)
        unlockable = city_layout.next_unlockable_section(tribe)

    tribe.wood -= config.TERRITORY_EXPANSION_WOOD_COST
    tribe.stone -= config.TERRITORY_EXPANSION_STONE_COST
    tribe.territory_radius = config.WALL_RING_RADIUS_STEP * len(tribe.wall_rings)
    if unlockable is not None:
        ring_i, sec_i = unlockable
        tribe.wall_rings[ring_i]["sections"][sec_i]["unlocked"] = True
    sim._award_trophy(tribe, "Territory Expander")
    return f"the tribe's territory expands to a {tribe.territory_radius}-tile radius"


def _build_dock(sim, tribe, biome, target):
    """Used to be reachable the moment a tribe settled, on the theory that
    building it would be a hopeful bet that pushed the tribe toward figuring out
    fishing. Explicit correction, after live data showed models spending wood on
    it (and other buildings) while genuinely starving, with fishing still
    unlearned: gate it on tribe.fishing_learned instead, the same "real proven
    capability, not a hopeful bet" pattern Sawmill/Quarry/Tannery already use.
    CATCH_FISH itself never required a dock (see _catch_fish) so this doesn't
    create a deadlock -- fishing gets learned first, and the dock becomes a real
    reward (config.DOCK_FISH_CATCH_BONUS_FRACTION, applied in Simulation.
    _advance_fish_supply's passive daily catch since CATCH_FISH itself retires
    from available_actions the moment fishing_learned is set) rather than a bet
    placed before the tribe has ever caught anything."""
    if tribe.dock_built or not tribe.fishing_learned or tribe.wood < config.DOCK_WOOD_COST:
        return None
    tribe.wood -= config.DOCK_WOOD_COST
    tribe.dock_built = True
    # Best-effort placement, not a gate -- Dock has no long_houses_built prerequisite,
    # so it's reachable before a tribe necessarily has any real territory yet.
    if tribe.territory_center is not None:
        slot = architect.find_free_slot(sim.world, tribe, "dock")
        if slot is not None:
            w, h = config.BUILDING_FOOTPRINTS["dock"]
            architect.record_building(tribe, "dock", slot[0], slot[1], w, h, sim.cycle)
    sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.BUILD_FIRE_PRIDE_MAGNITUDE, config.BUILD_FIRE_PRIDE_RADIUS)
    return "a dock rises at the water's edge -- fishing here will pay out more from now on"


def _build_fishery(sim, tribe, biome, target):
    """A real building beyond the Dock, not just a bigger version of it -- explicit
    request: "Fishery comes after the Dock is built." Stacks
    config.FISHERY_SUPPLY_BONUS_MULTIPLIER onto the existing passive daily fish
    supply (Simulation._advance_fish_supply's own FISHING_SUPPLY_MULTIPLIER) rather
    than replacing it, a real further reason to build both."""
    if tribe.fishery_built or not tribe.dock_built:
        return None
    if tribe.wood < config.FISHERY_WOOD_COST or tribe.stone < config.FISHERY_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "fishery")
    if slot is None:
        return None
    tribe.wood -= config.FISHERY_WOOD_COST
    tribe.stone -= config.FISHERY_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["fishery"]
    architect.record_building(tribe, "fishery", slot[0], slot[1], w, h, sim.cycle)
    tribe.fishery_built = True
    sim._award_trophy(tribe, "Fishmonger")
    return "a fishery is built alongside the dock -- the daily catch flows in even more steadily now"


def _build_sawmill(sim, tribe, biome, target):
    """Explicit correction, after live data showed both tribes permanently
    blocked behind a Long House that itself needs a completed wall ring neither
    reliably finishes: "the Sawmill is... online easily if they Gather Wood
    successfully. We already have this scaling." Gated on a real proven success
    (tribe.wood_ever_gathered) instead of Long House/fishing/a scouted site --
    the multiplier payoff (config.SAWMILL_WOOD_MULTIPLIER) was always correct,
    only the gate was too far downstream. A scouted stand of trees
    (tribe.lumber_sites) is no longer required, but still used opportunistically
    for Simulation._advance_resource_trails if one happens to exist -- not
    required to exist first. One-way, like dock_built."""
    if tribe.sawmill_built or not tribe.wood_ever_gathered:
        return None
    if tribe.wood < config.SAWMILL_WOOD_COST or tribe.stone < config.SAWMILL_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "sawmill")
    if slot is None:
        return None
    tribe.wood -= config.SAWMILL_WOOD_COST
    tribe.stone -= config.SAWMILL_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["sawmill"]
    architect.record_building(tribe, "sawmill", slot[0], slot[1], w, h, sim.cycle)
    tribe.sawmill_built = True
    if tribe.lumber_sites:
        tribe.lumber_site = tribe.lumber_sites[-1]
    sim._award_trophy(tribe, "Sawyer")
    return "a sawmill rises -- every load of wood gathered from here on is worth three times as much"


def _build_quarry(sim, tribe, biome, target):
    """Mirrors _build_sawmill's own simplification exactly, for stone instead of
    wood: gated on a real proven success (tribe.stone_ever_gathered) instead of
    Long House/fishing/a scouted site. A stone-rich site (tribe.quarry_sites) is
    no longer required, but still used opportunistically for Simulation.
    _advance_resource_trails if one happens to exist."""
    if tribe.quarry_built or not tribe.stone_ever_gathered:
        return None
    if tribe.wood < config.QUARRY_WOOD_COST or tribe.stone < config.QUARRY_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "quarry")
    if slot is None:
        return None
    tribe.wood -= config.QUARRY_WOOD_COST
    tribe.stone -= config.QUARRY_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["quarry"]
    architect.record_building(tribe, "quarry", slot[0], slot[1], w, h, sim.cycle)
    tribe.quarry_built = True
    if tribe.quarry_sites:
        tribe.quarry_site = tribe.quarry_sites[-1]
    sim._award_trophy(tribe, "Quarrier")
    return "a quarry opens -- every load of stone harvested from here on is worth three times as much"


def _build_warehouse(sim, tribe, biome, target):
    """Explicit request, after a live run showed unbounded hoarding (200+ wood
    while starved on stone): repeatable, like Long House -- each one raises
    _storage_cap further. No prerequisite beyond affordability, same as Dock --
    storage is infrastructure every tribe can use from the moment it's unlocked,
    not gated behind some other milestone. Same fixed footprint every time
    (config.BUILDING_FOOTPRINTS) regardless of how much it ends up holding."""
    if tribe.wood < config.WAREHOUSE_WOOD_COST or tribe.stone < config.WAREHOUSE_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "warehouse")
    if slot is None:
        return None
    tribe.wood -= config.WAREHOUSE_WOOD_COST
    tribe.stone -= config.WAREHOUSE_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["warehouse"]
    architect.record_building(tribe, "warehouse", slot[0], slot[1], w, h, sim.cycle)
    tribe.warehouses_built += 1
    sim._award_trophy(tribe, "Quartermaster")
    return f"a warehouse rises -- storage capacity grows to {_storage_cap(tribe)} per resource"


def _upgrade_warehouse(sim, tribe, biome, target):
    """Explicit request, 2026-09-09, after real data showed one tribe building 47
    warehouses in a single run: "they shouldn't build more than 5 I think. The
    rest of the capacity comes from the build or improvement calls being made
    now for a new build." BUILD_WAREHOUSE now stops offering itself past
    config.WAREHOUSE_MAX_COUNT (see its own AFFORDABILITY_CHECKS entry) --
    this is what a mature economy reaches for past that point instead.
    Deliberately no footprint/placement check, unlike a fresh build -- this
    improves what's already standing, not a new structure competing for space.

    Repeatable, but not free to spam the same way the old flat-cost
    BUILD_WAREHOUSE was: cost grows by config.WAREHOUSE_UPGRADE_COST_GROWTH
    per tier already banked (tribe.warehouse_upgrades), so this naturally
    tapers off rather than needing its own hardcoded count cap."""
    tier = tribe.warehouse_upgrades
    wood_cost = round(config.WAREHOUSE_UPGRADE_WOOD_COST_BASE * (1 + tier * config.WAREHOUSE_UPGRADE_COST_GROWTH))
    stone_cost = round(config.WAREHOUSE_UPGRADE_STONE_COST_BASE * (1 + tier * config.WAREHOUSE_UPGRADE_COST_GROWTH))
    if tribe.wood < wood_cost or tribe.stone < stone_cost:
        return None
    tribe.wood -= wood_cost
    tribe.stone -= stone_cost
    tribe.warehouse_upgrades += 1
    return f"the standing warehouses are reinforced -- storage capacity grows to {_storage_cap(tribe)} per resource"


def _build_barracks(sim, tribe, biome, target):
    """Military branch, step 2 (plan file valiant-forging-falcon.md): real
    housing for a trained Battalion, not a symbolic building -- repeatable,
    the same "each one raises a real cap" shape _build_warehouse already uses
    for storage, rather than a population-fraction formula. How large an army
    a tribe can ever field stays visibly tied to something it actually built.
    Gated on tribe.keep_built, not just affordability -- continues the
    existing Wall -> Long House -> Keep -> Fortress/Castle defensive ladder
    rather than sitting unconnected to it (Barracks branches off Keep, the
    same way Fortress/Castle do).

    Explicit request, 2026-09-09: "I don't think they should try to have a
    Military before they have a Kitchen" -- also gated on kitchen_built, the
    real entry point into this whole branch (see this action's own
    AFFORDABILITY_CHECKS entry, which mirrors this exactly).

    Explicit request: "only allow either option [DECLARE_WAR/DECLARE_
    ALLIANCE] after both have a Barracks (it auto fills with pop)." A
    Barracks now staffs its own capacity immediately from the tribe's
    existing population -- no separate TRAIN_BATTALION action, food cost, or
    named Warrior needed just to reach a first real headcount (confirmed via
    two independent live runs that NAME_WARRIOR/BUILD_BARRACKS/
    TRAIN_BATTALION had never once fired together in practice -- NAME_WARRIOR
    itself was retired entirely on 2026-09-10, see
    _eligible_new_battalion_leader). TRAIN_BATTALION still matters afterward: growing the roster further once
    capacity rises again (another Barracks), and readiness upkeep once at
    full strength (see its own docstring)."""
    if not tribe.kitchen_built or not tribe.keep_built:
        return None
    if tribe.wood < config.BARRACKS_WOOD_COST or tribe.stone < config.BARRACKS_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "barracks")
    if slot is None:
        return None
    tribe.wood -= config.BARRACKS_WOOD_COST
    tribe.stone -= config.BARRACKS_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["barracks"]
    architect.record_building(tribe, "barracks", slot[0], slot[1], w, h, sim.cycle)
    tribe.barracks_built += 1
    capacity = _battalion_capacity(tribe)
    old_size = tribe.battalion_size
    tribe.battalion_size = max(old_size, min(capacity, tribe.population))
    _allocate_battalion_strength(tribe, tribe.battalion_size - old_size)
    sim._award_trophy(tribe, "Drillmaster")
    return f"a barracks rises, staffed at {tribe.battalion_size} strong from the tribe's own population -- a Battalion can grow to {capacity} strong in total"


def _battalion_capacity(tribe) -> int:
    """Total Battalion headcount this tribe's Barracks investment supports --
    real builds (tribe.barracks_built) plus escalating upgrades past
    config.BARRACKS_MAX_COUNT (tribe.barracks_upgrades, see _upgrade_barracks),
    summed as one total the same way Fortress/Castle already count
    long_houses_built + long_house_upgrades as one real number rather than
    two independently-drifting checks."""
    return config.BATTALION_CAPACITY_PER_BARRACKS * (tribe.barracks_built + tribe.barracks_upgrades)


def _upgrade_barracks(sim, tribe, biome, target):
    """Explicit request, 2026-09-10, after real data showed one tribe building
    57 barracks in a single run: "we have to scale Barracks like we have done
    with Warehouse, among others." BUILD_BARRACKS now stops offering itself
    past config.BARRACKS_MAX_COUNT (see its own AFFORDABILITY_CHECKS entry) --
    this is what a mature military reaches for past that point instead.
    Mirrors _upgrade_warehouse exactly: no footprint/placement check (improves
    standing capacity, not a new structure competing for space), escalating
    cost per tier already banked (tribe.barracks_upgrades) so it tapers off
    naturally rather than needing its own hardcoded count cap. Still auto-fills
    from population the same way a fresh BUILD_BARRACKS does -- the new
    headroom this unlocks is real capacity, not just a number."""
    tier = tribe.barracks_upgrades
    wood_cost = round(config.BARRACKS_UPGRADE_WOOD_COST_BASE * (1 + tier * config.BARRACKS_UPGRADE_COST_GROWTH))
    stone_cost = round(config.BARRACKS_UPGRADE_STONE_COST_BASE * (1 + tier * config.BARRACKS_UPGRADE_COST_GROWTH))
    if tribe.wood < wood_cost or tribe.stone < stone_cost:
        return None
    tribe.wood -= wood_cost
    tribe.stone -= stone_cost
    tribe.barracks_upgrades += 1
    capacity = _battalion_capacity(tribe)
    old_size = tribe.battalion_size
    tribe.battalion_size = max(old_size, min(capacity, tribe.population))
    _allocate_battalion_strength(tribe, tribe.battalion_size - old_size)
    return f"the standing barracks are reinforced -- a Battalion can grow to {capacity} strong in total"


def _train_battalion(sim, tribe, biome, target):
    """Military branch, step 3 (plan file valiant-forging-falcon.md): staged
    like CONSTRUCT_WALL -- "built up over several turns, more with more
    people" -- rather than a one-shot flip. Reuses _labor_multiplier the same
    way CONSTRUCT_WALL's own progress-per-action does, so a larger tribe
    trains faster. Costs real food per soldier (feeding real people), not
    wood/stone -- BUILD_BARRACKS already paid the building cost. No
    commitment lock the way CONSTRUCT_WALL has (see its own docstring) --
    that exists for a specific documented live bug this hasn't hit; closer in
    spirit to EXPAND_TERRITORY's own uncommitted "invest again whenever
    ready" shape.

    Military branch, step 5 (Might's Training factor): once the Battalion is
    at full headcount, this doesn't just stop mattering -- it switches to a
    cheaper maintenance drill that bolsters tribe.battalion_readiness
    (config.BATTALION_READINESS_BOLSTER_PER_ACTION) instead of growing the
    roster further. Explicit request: "not overpowered, more like bolster
    and upkeep" -- readiness has to be kept up with continued drilling, it
    isn't recruited once and forgotten (Simulation.
    _advance_battalion_readiness_upkeep drains it a little every cycle
    regardless of this action)."""
    if tribe.barracks_built <= 0:
        return None
    capacity = _battalion_capacity(tribe)

    if tribe.battalion_size < capacity:
        added = min(
            capacity - tribe.battalion_size,
            round(config.BATTALION_TRAINING_PER_ACTION_BASE * _labor_multiplier(tribe.population)),
        )
        food_cost = round(config.BATTALION_TRAINING_FOOD_COST_PER_SOLDIER * added)
        if tribe.food < food_cost:
            return None
        tribe.food -= food_cost
        tribe.battalion_size += added
        _allocate_battalion_strength(tribe, added)
        tribe.battalion_readiness = min(
            1.0, tribe.battalion_readiness + config.BATTALION_READINESS_BOLSTER_PER_ACTION
        )
        if tribe.battalion_size >= capacity:
            return f"the Battalion reaches full strength at {tribe.battalion_size}, trained and ready under {_battalion_leaders_text(tribe)}"
        return f"the Battalion trains further under {_battalion_leaders_text(tribe)} -- {tribe.battalion_size}/{capacity} strong"

    if tribe.battalion_readiness >= 1.0 or tribe.food < config.BATTALION_READINESS_UPKEEP_FOOD_COST:
        return None
    tribe.food -= config.BATTALION_READINESS_UPKEEP_FOOD_COST
    tribe.battalion_readiness = min(1.0, tribe.battalion_readiness + config.BATTALION_READINESS_BOLSTER_PER_ACTION)
    if tribe.battalion_readiness >= 1.0:
        return f"the Battalion drills to peak readiness under {_battalion_leaders_text(tribe)}"
    return f"the Battalion drills under {_battalion_leaders_text(tribe)} -- readiness at {round(tribe.battalion_readiness * 100)}%"


def _build_kitchen(sim, tribe, biome, target):
    """Explicit follow-up: "we might have to let them build a kitchen which
    improves cooked food to excellent food yielding 3 per cooked item." Only
    means anything once cooking is already known -- gated on cooking_learned +
    long_house_built (real shelter, same "building homes" signal sawmill/quarry
    use). Stacks config.KITCHEN_FOOD_MULTIPLIER on top of cooking's own
    harvest-point multiplier (see _food_multiplier above) rather than replacing
    it."""
    if tribe.kitchen_built or not (tribe.cooking_learned and tribe.long_houses_built > 0):
        return None
    if tribe.wood < config.KITCHEN_WOOD_COST or tribe.stone < config.KITCHEN_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "kitchen")
    if slot is None:
        return None
    tribe.wood -= config.KITCHEN_WOOD_COST
    tribe.stone -= config.KITCHEN_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["kitchen"]
    architect.record_building(tribe, "kitchen", slot[0], slot[1], w, h, sim.cycle)
    tribe.kitchen_built = True
    sim._award_trophy(tribe, "Gourmet")
    return "a kitchen is built -- cooked meals now count as excellent food, stretching stores even further"


def _build_mine(sim, tribe, biome, target):
    """Explicit request: "Mines can [also] contain the Unique Resource of the
    Biome... these locations are scattered about the map." Gated on quarry_built
    (excavating a named seam is a deeper extension of already knowing how to
    quarry) plus at least one site actually discovered via scouting (Simulation.
    _advance_one_expedition). Locks in the most recently discovered site's
    resource permanently -- a tribe with several discovered veins on record still
    only ever works the one it chose to excavate."""
    if tribe.mine_built or not tribe.quarry_built or not tribe.mine_sites:
        return None
    if tribe.wood < config.MINE_WOOD_COST or tribe.stone < config.MINE_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "mine")
    if slot is None:
        return None
    tribe.wood -= config.MINE_WOOD_COST
    tribe.stone -= config.MINE_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["mine"]
    architect.record_building(tribe, "mine", slot[0], slot[1], w, h, sim.cycle)
    tribe.mine_built = True
    chosen_site = tribe.mine_sites[-1]
    tribe.mine_resource_name = chosen_site["resource"]
    tribe.mine_site = (chosen_site["x"], chosen_site["y"])
    sim._award_trophy(tribe, "Prospector")
    return f"a mine is excavated -- {tribe.mine_resource_name} waits to be fetched"


def _gather_ore(sim, tribe, biome, target):
    """Explicit correction: "GATHER_ORE only comes in if they Discover a Mine.
    They do not harvest on a Discovery, so they have to fetch it once." A Mine
    produces a brand new named resource with no manual counterpart, unlike
    Sawmill/Quarry (multipliers on an existing manual gather) -- this is that
    missing manual fetch. First success flips tribe.ore_ever_gathered, the
    same "action unlocks a passive system" shape fishing_learned already uses
    for _advance_fish_supply -- see Simulation._advance_mine_yield."""
    if not tribe.mine_built:
        return None
    amount = round(config.GATHER_ORE_BASE_YIELD * _labor_multiplier(tribe.population))
    tribe.ore_ever_gathered = True
    sim._capped_unique_add(tribe, tribe.mine_resource_name, amount)
    return f"{amount} {tribe.mine_resource_name} is fetched from the mine"


def _build_tannery(sim, tribe, biome, target):
    """Explicit request: "maybe some hunters want a Tannery and they can trade
    furs too." Explicit correction, same simplification as _build_sawmill/
    _build_quarry: "the Tannery should come online easily, as they only need
    to have hunted." Gated on tribe.hunt_ever_succeeded instead of Long House/
    fishing/a scouted Rabbit Warren. A warren site is no longer required, but
    still used opportunistically for Simulation._advance_resource_trails if
    one happens to be scouted. Pays Fur into the same tribe.unique_resources
    dict mines already use, not a second parallel resource system."""
    if tribe.tannery_built or not tribe.hunt_ever_succeeded:
        return None
    if tribe.wood < config.TANNERY_WOOD_COST or tribe.stone < config.TANNERY_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "tannery")
    if slot is None:
        return None
    tribe.wood -= config.TANNERY_WOOD_COST
    tribe.stone -= config.TANNERY_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["tannery"]
    architect.record_building(tribe, "tannery", slot[0], slot[1], w, h, sim.cycle)
    tribe.tannery_built = True
    warren_sites = [s for s in tribe.wildlife_sites if s["type"] == "Rabbit Warren"]
    if warren_sites:
        chosen_site = warren_sites[-1]
        tribe.tannery_site = (chosen_site["x"], chosen_site["y"])
    sim._award_trophy(tribe, "Tanner")
    return "a tannery is built -- Fur will flow in steadily from now on"


def _build_deer_pen(sim, tribe, biome, target):
    """Explicit follow-up, 2026-09-11: "if they successfully HUNT_DEER 3-5 they
    can build a DEER_PEN that will auto-feed the Tannery 1-3 deer a day. The Deer
    can breed to recursively have the resources automagically." Same shape as the
    Fowl Coop (see actions.py._build_coop), applied to deer/Fur instead of fowl/
    flock. Gated on tribe.tannery_built (feeding an existing Tannery is the whole
    point) plus a real hunt-success count (tribe.hunt_deer_success_count, see
    _hunt_deer above), not just Tannery's own single-success hunt_ever_succeeded.

    Unlike GATHER_EGGS, HUNT_DEER has no live-capture precedent -- it's pure
    lethal harvest, so there's no existing action to bootstrap a captive herd
    from. Building the Pen is itself the founding moment (config.
    DEER_PEN_FOUNDING_COUNT): trapping a couple of deer alive during
    construction, the same way GATHER_EGGS already narrates finding a live egg."""
    if tribe.deer_pen_built or not tribe.tannery_built:
        return None
    if tribe.hunt_deer_success_count < config.DEER_PEN_HUNT_THRESHOLD:
        return None
    if tribe.wood < config.DEER_PEN_WOOD_COST or tribe.stone < config.DEER_PEN_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "deer_pen")
    if slot is None:
        return None
    tribe.wood -= config.DEER_PEN_WOOD_COST
    tribe.stone -= config.DEER_PEN_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["deer_pen"]
    architect.record_building(tribe, "deer_pen", slot[0], slot[1], w, h, sim.cycle)
    tribe.deer_pen_built = True
    tribe.deer = config.DEER_PEN_FOUNDING_COUNT
    stand_sites = [s for s in tribe.wildlife_sites if s["type"] == "Deer Stand"]
    if stand_sites:
        chosen_site = stand_sites[-1]
        tribe.deer_pen_site = (chosen_site["x"], chosen_site["y"])
    sim._award_trophy(tribe, "Deer Keeper")
    return f"a deer pen is built -- {tribe.deer} deer trapped alive to start the herd, feeding the tannery from now on"


def _build_hatchery(sim, tribe, biome, target):
    """Explicit follow-up: "the Flock and the Eggs self generate. So, maybe
    after they GATHER_EGGS in the wild, they can have a Hatchery." Gated on a
    real wild find (tribe.eggs_ever_gathered), the same "proven success, not
    flock size alone" pattern Sawmill/Quarry/Tannery use. Boosts Simulation.
    _advance_flock's own natural-hatch chance rather than the passive
    egg-laying rate (_advance_flock_eggs) -- a hatchery is where eggs get
    incubated into new flock faster, not where more eggs get laid."""
    if tribe.hatchery_built or not tribe.eggs_ever_gathered:
        return None
    if tribe.wood < config.HATCHERY_WOOD_COST or tribe.stone < config.HATCHERY_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "hatchery")
    if slot is None:
        return None
    tribe.wood -= config.HATCHERY_WOOD_COST
    tribe.stone -= config.HATCHERY_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["hatchery"]
    architect.record_building(tribe, "hatchery", slot[0], slot[1], w, h, sim.cycle)
    tribe.hatchery_built = True
    sim._award_trophy(tribe, "Hatchery Keeper")
    return "a hatchery is built -- the flock grows on its own much more reliably from now on"


def _build_coop(sim, tribe, biome, target):
    """Explicit follow-up, 2026-09-11: "eggs gathered are put into the Hatchery, the
    Hatchery incubates the eggs to hatch into the Fowl we have in the Coop, fowl
    caught are put into the Coop, fowl breed and lay eggs that go into the Hatchery."
    Promotes what used to be a free, automatic "flock_pen" placement (Simulation.
    _resolve_hatch, the instant the first egg ever hatched) into a real, chief-built
    structure -- the same "prove it, then build it for real, not a free ride" pattern
    Sawmill/Quarry/Dock already established. Gated on tribe.flock > 0 (a founding
    fowl already exists), not eggs_ever_gathered like Hatchery -- see config.
    COOP_WOOD_COST's own comment for why that avoids a bootstrap deadlock. Once both
    this and the Hatchery exist, Simulation._advance_flock switches from a
    probabilistic natural-hatch roll to actually consuming stored eggs for a
    deterministic hatch each cycle -- see that method's own docstring."""
    if tribe.coop_built or tribe.flock <= 0:
        return None
    if tribe.wood < config.COOP_WOOD_COST or tribe.stone < config.COOP_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "coop")
    if slot is None:
        return None
    tribe.wood -= config.COOP_WOOD_COST
    tribe.stone -= config.COOP_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["coop"]
    architect.record_building(tribe, "coop", slot[0], slot[1], w, h, sim.cycle)
    tribe.coop_built = True
    sim._award_trophy(tribe, "Coop Builder")
    return "a coop is built -- the flock finally has a real home, and a proper Hatchery can put it to use"


def _build_bath_house(sim, tribe, biome, target):
    """Explicit request: "bath house bolsters Well-Being upkeep once built."
    No special prerequisite beyond being settled and affordable, the same
    "infrastructure every tribe can use from the moment it's unlocked" shape
    Warehouse/Road already use -- hygiene isn't gated behind a proven success
    the way hunting/fishing/mining are. Its real effect lives in Simulation.
    _apply_upkeep (a genuine reduction to per-cycle food/water consumption,
    mirrored into wellbeing.py's physiological tier so that score reflects
    the real number being charged)."""
    if tribe.bath_house_built:
        return None
    if tribe.wood < config.BATH_HOUSE_WOOD_COST or tribe.stone < config.BATH_HOUSE_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "bath_house")
    if slot is None:
        return None
    tribe.wood -= config.BATH_HOUSE_WOOD_COST
    tribe.stone -= config.BATH_HOUSE_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["bath_house"]
    architect.record_building(tribe, "bath_house", slot[0], slot[1], w, h, sim.cycle)
    tribe.bath_house_built = True
    sim._award_trophy(tribe, "Keeper of Hygiene")
    return "a bath house is built -- the tribe's stores stretch further from now on"


def _build_library(sim, tribe, biome, target):
    """Explicit request: a Library condenses the tribe's own remembered history
    (TribeMemory) into permanent, readable entries and unlocks RESEARCH -- a
    real, repeatable path to reaching the next era sooner. Gated on
    long_houses_built > 0 (real shelter already established), the same
    "building homes" signal Kitchen/Sawmill/Quarry already use -- a Library
    only makes sense once people actually live here."""
    if tribe.library_built or tribe.long_houses_built == 0:
        return None
    if tribe.wood < config.LIBRARY_WOOD_COST or tribe.stone < config.LIBRARY_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "library")
    if slot is None:
        return None
    tribe.wood -= config.LIBRARY_WOOD_COST
    tribe.stone -= config.LIBRARY_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["library"]
    architect.record_building(tribe, "library", slot[0], slot[1], w, h, sim.cycle)
    tribe.library_built = True
    sim._award_trophy(tribe, "Keeper of Records")
    return "a library is built -- the tribe's own memory can now be studied and put to real use"


def _research(sim, tribe, biome, target):
    """The Library's real payoff: distills the tribe's highest-weight remembered
    episodes (TribeMemory.entries/taboos -- the same ranking TribeMemory.
    consolidate already uses for its own taboo cut) into one permanent Library
    entry, and permanently discounts the next era's threshold a little further
    (Simulation._advance_era_if_ready) -- a real, compounding "boosts growth and
    innovation," not a flat one-time stat bump. No-ops with nothing to study yet
    if the tribe hasn't actually remembered anything real yet."""
    if not tribe.library_built:
        return None
    ranked = sorted(tribe.memory.entries, key=lambda e: e["weight"], reverse=True)
    top = [e["text"] for e in ranked[: config.LIBRARY_ENTRY_MEMORY_COUNT]]
    top.extend(t for t in tribe.memory.taboos if t not in top)
    if not top:
        return "the library stands ready, but the tribe hasn't lived through anything worth recording yet"
    if tribe.wood < config.RESEARCH_WOOD_COST:
        return None
    tribe.wood -= config.RESEARCH_WOOD_COST
    summary = "; ".join(top[: config.LIBRARY_ENTRY_MEMORY_COUNT])
    tribe.library_entries.append({"summary": summary, "cycle": sim.cycle})
    tribe.research_completed += 1
    return f"the library records a new insight: \"{summary}\" -- the path to the next era grows a little shorter"


def _build_well(sim, tribe, biome, target):
    """Explicit request: water's passive income had no equivalent of Fishery/Dock's
    stacking bonus for food. No special prerequisite beyond being settled and
    affordable, the same "infrastructure from the moment it's unlocked" shape
    Bath House/Warehouse already use. Its real effect lives in Simulation.
    _advance_water_supply (a genuine multiplier on top of the settled-near-water
    passive supply, not a one-time top-up)."""
    if tribe.well_built:
        return None
    if tribe.wood < config.WELL_WOOD_COST or tribe.stone < config.WELL_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "well")
    if slot is None:
        return None
    tribe.wood -= config.WELL_WOOD_COST
    tribe.stone -= config.WELL_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["well"]
    architect.record_building(tribe, "well", slot[0], slot[1], w, h, sim.cycle)
    tribe.well_built = True
    sim._award_trophy(tribe, "Water Keeper")
    return "a well is dug -- the settlement's water supply flows in faster from now on"


def _build_forge(sim, tribe, biome, target):
    """Explicit request: a Mine's named ore had nowhere real to go once excavated --
    "we skipped a beat" between production and doing anything with it. Gated on
    tribe.mine_built plus at least one unit of that mine's own resource already in
    stock ("built after they get 1 Ore"), proof the mine is real and working rather
    than a second parallel discovery mechanic."""
    if tribe.forge_built or not tribe.mine_built:
        return None
    if tribe.unique_resources.get(tribe.mine_resource_name, 0) < config.FORGE_ITEM_ORE_COST:
        return "not enough ore has been mined yet to justify a forge"
    if tribe.wood < config.FORGE_WOOD_COST or tribe.stone < config.FORGE_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "forge")
    if slot is None:
        return None
    tribe.wood -= config.FORGE_WOOD_COST
    tribe.stone -= config.FORGE_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["forge"]
    architect.record_building(tribe, "forge", slot[0], slot[1], w, h, sim.cycle)
    tribe.forge_built = True
    sim._award_trophy(tribe, "Blacksmith")
    return f"a forge is built -- {tribe.mine_resource_name} can now be worked into real tools, weapons, and inventions"


def _item_storage_cap(tribe) -> int:
    """See config.ITEM_STORAGE_CAP_BASE's own comment -- a much smaller ceiling
    than _storage_cap's bulk-resource one, since each item already represents a
    real spent investment rather than something freely re-gathered. Counts
    warehouse_upgrades alongside warehouses_built, same as _storage_cap -- an
    upgraded warehouse holds more of everything, items included."""
    return config.ITEM_STORAGE_CAP_BASE + (tribe.warehouses_built + tribe.warehouse_upgrades) * config.ITEM_STORAGE_CAP_PER_WAREHOUSE


def _forge_item(sim, tribe, biome, target):
    """Turns stored ore into a real, permanent item -- a tool, a weapon, or a small
    innovation, picked at random each time (like a mine site's own resource name,
    this isn't something the tribe gets to choose directly). No durability tracked
    per explicit request; each item just carries a flat, type-based value, redeemable
    later via USE_ITEM or handed over in a TRADE."""
    if not tribe.forge_built:
        return None
    if len(tribe.items) >= _item_storage_cap(tribe):
        return "the item stores are already full -- USE_ITEM or a TRADE must free up room before another can be forged"
    if tribe.unique_resources.get(tribe.mine_resource_name, 0) < config.FORGE_ITEM_ORE_COST:
        return None
    if tribe.wood < config.FORGE_ITEM_WOOD_COST:
        return None
    tribe.unique_resources[tribe.mine_resource_name] -= config.FORGE_ITEM_ORE_COST
    tribe.wood -= config.FORGE_ITEM_WOOD_COST
    item_type = random.choice(config.ITEM_TYPES)
    item_name = random.choice(config.ITEM_NAMES_BY_TYPE[item_type])
    item = {
        "name": item_name, "type": item_type,
        "value": config.ITEM_VALUE_BY_TYPE[item_type], "cycle_made": sim.cycle,
    }
    tribe.items.append(item)
    if len(tribe.items) == 1:
        sim._award_trophy(tribe, "Artisan")
    return f"the forge produces a {item_name} ({item_type}) -- {tribe.mine_resource_name} well spent"


def _use_item(sim, tribe, biome, target):
    """Redeems the oldest crafted item for its stored value -- split across wood and
    stone, the straightforward cash-out for a value that would otherwise just sit on
    the tribe forever. No durability/degradation to model, so using an item is a
    one-shot conversion, not a repeatable wear-down."""
    if not tribe.items:
        return None
    item = tribe.items.pop(0)
    stone_gain_nominal = round(item["value"] * config.USE_ITEM_STONE_SHARE)
    wood_gain_nominal = item["value"] - stone_gain_nominal
    # Code-quality pass: this used to add straight to tribe.wood/stone with no
    # cap check -- same bug class fixed across the rest of this file/simulation.py
    # (see Simulation._capped_add's own docstring). sim._capped_add returns the
    # amount actually added, since the message below needs the real number.
    wood_gain = sim._capped_add(tribe, "wood", wood_gain_nominal)
    stone_gain = sim._capped_add(tribe, "stone", stone_gain_nominal)
    return f"the {item['name']} is put to use -- {wood_gain} wood and {stone_gain} stone recovered from its worth"


def _created_object_bonus(tribe, category: str) -> float:
    """Sums the bounded per-category magnitude across every created object of
    that category -- the single hook point every percentage-scaled Object
    Creator effect (gather_boost/combat_boost/defense_boost/
    celebration_discount) reads from. expedition_boost/population_boost use
    their own flat-amount constants instead (a tile count and a one-shot
    population grant aren't percentages), so they aren't summed here -- see
    _create_item's own population_boost branch and Simulation.
    _advance_one_expedition's expedition_boost hook."""
    return sum(config.CREATED_OBJECT_MAGNITUDE for obj in tribe.created_objects if obj["category"] == category)


def _armed_count(tribe) -> int:
    """Explicit request, 2026-09-11, for the battle popup: "an indication of
    how many of the army have weapons or magic from the forge or object
    creator." Real Forge-crafted weapons currently on hand (tribe.items,
    type == "weapon" -- tools/innovations don't count, only combat gear)
    plus every Object Creator combat_boost creation (the "magic" -- the same
    creations that already drive DECLARE_CONQUEST's effective-population
    multiplier via _created_object_bonus). Purely a display count for the
    battle popup, not a new input to the win-chance math -- combat_boost
    already feeds that separately."""
    weapons = sum(1 for item in tribe.items if item["type"] == "weapon")
    magic = sum(1 for obj in tribe.created_objects if obj["category"] == "combat_boost")
    return weapons + magic


def _might_adjusted_win_chance(tribe, rival, base_win_chance: float) -> float:
    """Military branch, step 6 (plan file valiant-forging-falcon.md): layers
    a bounded Might-based adjustment on top of the existing population-share
    win chance shared by RAID(rival)/DECLARE_CONQUEST -- never a replacement
    for it, "tribe-vs-tribe stays entirely Chief-controlled" (Might only
    ever adjusts the odds of a fight the Chief already chose to start, it
    never triggers one on its own -- see Simulation._advance_battalion_patrol's
    own docstring for the autonomous side, scoped to raiders only). Neither
    Might dominance nor Might weakness alone can make the outcome certain
    (config.MIGHT_ADJUSTED_WIN_CHANCE_FLOOR/_CEILING), the same "real
    ceiling, never an absolute guarantee" shape every other win-chance
    formula here already uses.

    When neither side has ever built a Battalion (still the common case
    before this branch gets used at all), the modifier is exactly 0 --
    ordinary population-share odds, completely unchanged from before this
    step existed."""
    tribe_might = compute_might(tribe)
    rival_might = compute_might(rival)
    if tribe_might == 0 and rival_might == 0:
        modifier = 0.0
    else:
        might_ratio = tribe_might / max(1, rival_might)
        modifier = max(
            config.MIGHT_MODIFIER_MIN, min(config.MIGHT_MODIFIER_MAX, (might_ratio - 1) * config.MIGHT_MODIFIER_SCALE)
        )
    return max(
        config.MIGHT_ADJUSTED_WIN_CHANCE_FLOOR,
        min(config.MIGHT_ADJUSTED_WIN_CHANCE_CEILING, base_win_chance + modifier),
    )


def _build_object_creator(sim, tribe, biome, target):
    """Object Creator era's signature building -- the factory that lets a
    tribe start inventing genuinely new things. One-time, permanent, same
    BUILD_FORGE-shaped gate (wood/stone cost + a free footprint slot), just a
    late-game one behind the era's own steep resource threshold (see
    eras.py's object_creator_era)."""
    if tribe.object_creator_built:
        return None
    if tribe.wood < config.OBJECT_CREATOR_WOOD_COST or tribe.stone < config.OBJECT_CREATOR_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "object_creator")
    if slot is None:
        return None
    tribe.wood -= config.OBJECT_CREATOR_WOOD_COST
    tribe.stone -= config.OBJECT_CREATOR_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["object_creator"]
    architect.record_building(tribe, "object_creator", slot[0], slot[1], w, h, sim.cycle)
    tribe.object_creator_built = True
    sim._award_trophy(tribe, "Visionary")
    return "the Object Creator hums to life -- the tribe can now design and build genuinely new things"


def _new_created_object(tribe) -> tuple[str, str]:
    """Picks a name and effect category for a freshly created item/structure
    -- shared by CREATE_ITEM/CREATE_USEFUL_STRUCTURE. The name is genuinely
    random/flavorful (same creative-but-bounded balance ITEM_NAMES_BY_TYPE
    already strikes for Forge items); the category is picked round-robin off
    this tribe's own creation count so far, not a hidden roll, cycling
    through all six of CREATED_OBJECT_CATEGORIES rather than gambling on the
    same one repeatedly. Explicit request: "let's limit the risk at this
    time knowing we will come back to it" -- full open-ended LLM-driven stat
    generation (parsing the model's own description into a novel mechanical
    effect) is a deliberate future follow-up, not built here."""
    name = random.choice(config.CREATED_OBJECT_NAMES)
    category = config.CREATED_OBJECT_CATEGORIES[len(tribe.created_objects) % len(config.CREATED_OBJECT_CATEGORIES)]
    return name, category


def _create_item(sim, tribe, biome, target):
    """The Object Creator's first real output: a genuinely new, permanent
    item with one bounded effect. population_boost is the one immediate,
    one-shot effect (a flat population grant, see config.
    CREATED_OBJECT_POPULATION_BONUS); every other category is read passively
    at its own real hook point (_created_object_bonus)."""
    if not tribe.object_creator_built:
        return None
    if tribe.wood < config.CREATE_ITEM_WOOD_COST or tribe.stone < config.CREATE_ITEM_STONE_COST:
        return None
    tribe.wood -= config.CREATE_ITEM_WOOD_COST
    tribe.stone -= config.CREATE_ITEM_STONE_COST
    name, category = _new_created_object(tribe)
    tribe.created_objects.append({"name": name, "category": category, "kind": "item"})
    if len(tribe.created_objects) == 1:
        sim._award_trophy(tribe, "Inventor")
    if category == "population_boost":
        tribe.population += config.CREATED_OBJECT_POPULATION_BONUS
        tribe.max_population = max(tribe.max_population, tribe.population)
        return (
            f"the {name} is unveiled -- {config.CREATED_OBJECT_POPULATION_BONUS} new people "
            "join the tribe, inspired by the invention"
        )
    return f"the {name} is unveiled -- a genuinely new {category.replace('_', ' ')} for the tribe"


def _create_useful_structure(sim, tribe, biome, target):
    """Same idea as CREATE_ITEM, but a real building footprint instead of a
    portable item -- "anything" the Object Creator can make spans both, per
    direct confirmation."""
    if not tribe.object_creator_built:
        return None
    if tribe.wood < config.CREATE_USEFUL_STRUCTURE_WOOD_COST or tribe.stone < config.CREATE_USEFUL_STRUCTURE_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "created_structure")
    if slot is None:
        return None
    tribe.wood -= config.CREATE_USEFUL_STRUCTURE_WOOD_COST
    tribe.stone -= config.CREATE_USEFUL_STRUCTURE_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["created_structure"]
    architect.record_building(tribe, "created_structure", slot[0], slot[1], w, h, sim.cycle)
    name, category = _new_created_object(tribe)
    tribe.created_objects.append({"name": name, "category": category, "kind": "structure"})
    if len(tribe.created_objects) == 1:
        sim._award_trophy(tribe, "Inventor")
    if category == "population_boost":
        tribe.population += config.CREATED_OBJECT_POPULATION_BONUS
        tribe.max_population = max(tribe.max_population, tribe.population)
        return (
            f"the {name} is built -- {config.CREATED_OBJECT_POPULATION_BONUS} new people "
            "join the tribe, drawn by the new structure"
        )
    return f"the {name} is built -- a genuinely new {category.replace('_', ' ')} structure for the tribe"


def _declare_conquest(sim, tribe, biome, target):
    """War and World Domination era's decisive war action -- unlike ordinary
    RAID's gradual population-siphon (several successful raids to fully
    absorb a rival), a tribe that's reached this era can commit everything
    to one all-in campaign against a real, known rival.

    Originally a single instant dice roll (win outright and absorb the
    rival via Simulation._merge_tribes, or fail at a small fixed population
    cost) -- confirmed via real run data it never fired even once across
    every recorded run before the Military branch existed, and working
    theory (2026-09-07) was that it read as an isolated, all-or-nothing
    gamble with nothing feeding into it.

    Explicit request, 2026-09-09, for a live "play by play blows, meters
    falling" popup once a fresh run reaches this same climactic action
    again: "Yes, real loss... we can 'win' and absorb the last 10% or so of
    the remaining pop." Redesigned into a bounded war of attrition
    (config.DECLARE_CONQUEST_MAX_ROUNDS rounds) -- each round rolls the
    same population-share/Might-adjusted chance the old single roll always
    used, but now the loser of THAT round takes a real, permanent
    population hit (config.DECLARE_CONQUEST_ROUND_LOSS_FRACTION_LOSER) and
    the round's winner takes a smaller one too
    (..._WINNER) -- "meters falling" plural, not just one side's. Ends the
    moment either side is fought down to
    config.DECLARE_CONQUEST_DEFEAT_THRESHOLD_FRACTION of its OWN starting
    population, and that side's survivors and stockpiles are fully absorbed
    into the other via _merge_tribes -- symmetric by design: the tribe that
    started this campaign can lose everything too, a real reason to build
    up Might with training first, not just a nice-to-have. A fight that
    never breaks either side within the round cap ends in a costly
    stalemate -- no merge, both sides keep whatever they have left.

    Returns the round-by-round record in the "battle" key of the
    recent_encounters entry this appends (see Simulation.recent_encounters'
    own "kind": "tribe_conquest_battle" -- the frontend replays this as an
    animated popup), not just a one-line result string."""
    tx, ty = target
    defender = None
    for other in sim.tribes.values():
        if other.id == tribe.id or other.extinct:
            continue
        if (other.x - tx) ** 2 + (other.y - ty) ** 2 <= config.RAID_PROXIMITY_RADIUS ** 2:
            defender = other
            break
    if defender is None:
        return "found no rival civilization there to conquer"
    if tribe.wood < config.DECLARE_CONQUEST_WOOD_COST or tribe.stone < config.DECLARE_CONQUEST_STONE_COST:
        return None
    tribe.wood -= config.DECLARE_CONQUEST_WOOD_COST
    tribe.stone -= config.DECLARE_CONQUEST_STONE_COST

    attacker_name, defender_name = tribe.name, defender.name
    attacker_start_population = tribe.population
    defender_start_population = defender.population
    attacker_might = compute_might(tribe)
    defender_might = compute_might(defender)
    attacker_armed = _armed_count(tribe)
    defender_armed = _armed_count(defender)
    rounds = []
    outcome = "stalemate"

    for round_number in range(1, config.DECLARE_CONQUEST_MAX_ROUNDS + 1):
        effective_attacker_population = tribe.population * (1 + _created_object_bonus(tribe, "combat_boost"))
        round_win_chance = effective_attacker_population / max(1, effective_attacker_population + defender.population)
        round_win_chance = _might_adjusted_win_chance(tribe, defender, round_win_chance)
        attacker_won_round = random.random() < round_win_chance
        winner, loser = (tribe, defender) if attacker_won_round else (defender, tribe)
        sim._lose_population(loser, round(loser.population * config.DECLARE_CONQUEST_ROUND_LOSS_FRACTION_LOSER), cause="conquest_round_loss")
        sim._lose_population(winner, round(winner.population * config.DECLARE_CONQUEST_ROUND_LOSS_FRACTION_WINNER), cause="conquest_round_loss")
        rounds.append({
            "round": round_number, "attacker_won": attacker_won_round,
            "attacker_population": tribe.population, "defender_population": defender.population,
        })

        if defender.extinct or defender.population <= defender_start_population * config.DECLARE_CONQUEST_DEFEAT_THRESHOLD_FRACTION:
            outcome = "attacker_wins"
            break
        if tribe.extinct or tribe.population <= attacker_start_population * config.DECLARE_CONQUEST_DEFEAT_THRESHOLD_FRACTION:
            outcome = "defender_wins"
            break

    resolved_by_surrender = False
    stalemate_note = None
    if outcome == "stalemate":
        # SURRENDER: DECLARE_CONQUEST_DEFEAT_THRESHOLD_FRACTION never fired within
        # the round cap (see config.py's note on why that check is nearly
        # unreachable by design of the round-loss fractions) -- rather than let
        # an unresolved war just reopen next cycle forever, whichever side ends
        # these MAX_ROUNDS decisively weaker concedes instead of fighting on.
        # A genuinely close fight (both sides within the ratio of each other)
        # still ends a true stalemate, no merge. Explicit follow-up request:
        # "surrender isn't allowed unless you lose 2 times already" -- the
        # first lopsided loss against a given rival is just noted, not
        # resolved; only the DECLARE_CONQUEST_SURRENDER_AFTER_LOSSES-th one
        # actually ends the war.
        weaker, stronger = (tribe, defender) if tribe.population <= defender.population else (defender, tribe)
        if weaker.population <= stronger.population * config.DECLARE_CONQUEST_SURRENDER_POPULATION_RATIO:
            losses = weaker.conquest_stalemate_losses.get(stronger.id, 0) + 1
            weaker.conquest_stalemate_losses[stronger.id] = losses
            if losses >= config.DECLARE_CONQUEST_SURRENDER_AFTER_LOSSES:
                outcome = "defender_wins" if weaker is tribe else "attacker_wins"
                resolved_by_surrender = True
            else:
                stalemate_note = (
                    f"{weaker.name}'s campaign against {stronger.name} ends in stalemate, battered and "
                    "clearly outmatched -- one more defeat like this and surrender will be inevitable"
                )

    if resolved_by_surrender:
        battle_outcome_key = "attacker_surrenders" if outcome == "defender_wins" else "defender_surrenders"
    else:
        battle_outcome_key = outcome

    battle_record = {
        "attacker_name": attacker_name, "defender_name": defender_name,
        "attacker_start_population": attacker_start_population, "defender_start_population": defender_start_population,
        "attacker_might": attacker_might, "defender_might": defender_might,
        "attacker_armed": attacker_armed, "defender_armed": defender_armed,
        "rounds": rounds, "outcome": battle_outcome_key,
    }

    if outcome == "attacker_wins":
        _record_combat(tribe, "Conquest", "won")
        _record_combat(defender, "Conquest Defense", "lost")
        sim.trauma.radiate_event_wave(defender.x, defender.y, config.RAID_TRAUMA_MAGNITUDE * 2, config.RAID_TRAUMA_RADIUS)
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_PRIDE_MAGNITUDE * 2, config.RAID_PRIDE_RADIUS)
        new_name = sim._merge_tribes(tribe, defender) if not defender.extinct else tribe.name
        sim.recent_encounters.append({
            "x": defender.x, "y": defender.y, "kind": "tribe_conquest_battle",
            "label": f"{attacker_name} conquers {defender_name}", "outcome": "won", "battle": battle_record,
        })
        if not resolved_by_surrender:
            return f"after {len(rounds)} rounds of real fighting, {attacker_name} breaks {defender_name} and wins outright -- {attacker_name} becomes {new_name}!"
        return f"after {config.DECLARE_CONQUEST_MAX_ROUNDS} brutal rounds neither side broke outright, but {defender_name} is left too battered to continue and surrenders -- {attacker_name} becomes {new_name}!"
    if outcome == "defender_wins":
        _record_combat(tribe, "Conquest", "lost")
        _record_combat(defender, "Conquest Defense", "won")
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_TRAUMA_MAGNITUDE * 2, config.RAID_TRAUMA_RADIUS)
        sim.trauma.radiate_event_wave(defender.x, defender.y, config.RAID_PRIDE_MAGNITUDE * 2, config.RAID_PRIDE_RADIUS)
        new_name = sim._merge_tribes(defender, tribe) if not tribe.extinct else defender.name
        sim.recent_encounters.append({
            "x": defender.x, "y": defender.y, "kind": "tribe_conquest_battle",
            "label": f"{defender_name} repels {attacker_name}'s all-in campaign and breaks them", "outcome": "lost", "battle": battle_record,
        })
        if not resolved_by_surrender:
            return f"after {len(rounds)} rounds of real fighting, {defender_name} breaks the campaign and absorbs {attacker_name} instead -- {defender_name} becomes {new_name}!"
        return f"after {config.DECLARE_CONQUEST_MAX_ROUNDS} brutal rounds neither side broke outright, but {attacker_name}'s campaign is left too battered to continue and surrenders -- {defender_name} becomes {new_name}!"
    _record_combat(tribe, "Conquest", "lost")
    _record_combat(defender, "Conquest Defense", "won")
    sim.recent_encounters.append({
        "x": defender.x, "y": defender.y, "kind": "tribe_conquest_battle",
        "label": f"{attacker_name} and {defender_name} fight to a standstill", "outcome": "stalemate", "battle": battle_record,
    })
    if stalemate_note:
        tribe.history.append(stalemate_note)
        defender.history.append(stalemate_note)
        return f"after {config.DECLARE_CONQUEST_MAX_ROUNDS} brutal rounds neither side breaks -- {stalemate_note}"
    return f"after {config.DECLARE_CONQUEST_MAX_ROUNDS} brutal rounds neither side breaks -- {attacker_name}'s campaign against {defender_name} ends in a costly stalemate"


def _plant_crop(sim, tribe, biome, target):
    """Only reachable at all once Simulation._prepare_turn's settled-near-water gate
    (Simulation._is_settled_near_water) allows it -- plains alone doesn't mean a tribe
    resettled somewhere with real water access, per the original design spec for
    farming. Growth itself is a passive per-cycle tick (Simulation._advance_farming),
    not something this action does directly -- planting just adds one more plot to
    tend."""
    if tribe.farm_plots >= config.MAX_FARM_PLOTS or tribe.wood < config.PLANT_CROP_WOOD_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "farm_plot")
    if slot is None:
        return None
    tribe.wood -= config.PLANT_CROP_WOOD_COST
    w, h = config.BUILDING_FOOTPRINTS["farm_plot"]
    architect.record_building(tribe, "farm_plot", slot[0], slot[1], w, h, sim.cycle)
    tribe.farm_plots += 1
    return f"a new plot is fenced and planted, scarecrow set to keep it clear -- {tribe.farm_plots} now growing"


def _gather_eggs(sim, tribe, biome, target):
    """Wild fowl near a real water source -- gated the same as PLANT_CROP (Simulation.
    _is_settled_near_water).

    Before a Coop exists, a find doesn't hatch here: this only sets
    tribe.pending_hatch; Simulation.step() resolves it with a real, non-scripted LLM
    call (backend/genetics.py's hatch()) the same cycle, the same pattern BREED already
    uses for pending_birth. Once the flock has at least two members, the two most
    recently hatched are what get crossed -- mirrors _eligible_breeding_pair preferring
    a fresh milestone over the whole population. This is the founding path (how a
    flock starts existing at all) and stays exactly as it always has.

    Once a Coop exists (see actions.py._build_coop), there's a real home for a caught
    fowl already -- a find now deposits into tribe.eggs (config.
    GATHER_EGGS_STOCKPILE_AMOUNT) instead of hatching directly. Simulation.
    _advance_flock is what actually incubates that stockpile into new flock from then
    on, once a Hatchery exists too. Explicit follow-up, 2026-09-11: "eggs gathered
    are put into the Hatchery... fowl caught are put into the Coop.\""""
    if tribe.coop_built:
        if random.random() >= config.GATHER_EGGS_SUCCESS_CHANCE:
            return "no eggs found this time"
        tribe.eggs += config.GATHER_EGGS_STOCKPILE_AMOUNT
        tribe.eggs_ever_gathered = True
        return f"an egg is found and brought back to the coop -- {tribe.eggs} now stored for the hatchery"
    if tribe.pending_hatch is not None:
        return "an egg is already being tended -- one thing at a time"
    if random.random() >= config.GATHER_EGGS_SUCCESS_CHANCE:
        return "no eggs found this time"
    parents = tribe.flock_lineage[-2:] if len(tribe.flock_lineage) >= 2 else None
    tribe.pending_hatch = {"parents": parents}
    tribe.eggs_ever_gathered = True  # see actions.py._build_hatchery's own prerequisite
    return "an egg is found and set aside to hatch"


def _catch_fish(sim, tribe, biome, target):
    """Only reachable once Simulation._prepare_turn's settled gate allows it, same as
    PLANT_CROP/GATHER_EGGS. "Learning to fish" isn't a separate knowledge system --
    the first successful catch just flips tribe.fishing_learned, which is all
    Simulation._advance_fish_supply checks to start a passive daily food supply from
    then on, the same "action unlocks a passive system" shape crops and water already
    use. Every catch (including the first) still pays out its own food too."""
    if random.random() >= config.CATCH_FISH_SUCCESS_CHANCE:
        return "no fish caught this time"
    caught = random.randint(config.FISHING_CATCH_FOOD_MIN, config.FISHING_CATCH_FOOD_MAX)
    # Dock's own bonus (config.DOCK_FISH_CATCH_BONUS_FRACTION) applies to the
    # passive daily supply now (see Simulation._advance_fish_supply), not here --
    # BUILD_DOCK requires fishing_learned already, and CATCH_FISH retires from
    # available_actions the instant fishing_learned is set, so a dock could
    # never actually exist while this manual catch was still reachable.
    # Code-quality pass: every other manual gather action (GATHER_WOOD/STONE/
    # WATER/FOOD, HUNT_DEER) already routes its gain through the storage cap --
    # this one was the odd one out, adding straight to tribe.food with no check.
    # sim._capped_add (not the message-returning _add_capped above) since the
    # "first catch" story below needs the real landed amount regardless of
    # whether it's the special first-catch message or the routine one.
    caught = sim._capped_add(tribe, "food", round(caught * _food_multiplier(tribe)))
    if not tribe.fishing_learned:
        tribe.fishing_learned = True
        sim._award_trophy(tribe, "Angler")
        if tribe.last_celebration_cycle != sim.cycle:
            sim._celebrate_fishing_learned(tribe)
        return f"the first catch! {caught} food landed, and fishing is now second nature to the tribe"
    return f"{caught} food caught fishing"


# A small, cheap flavor name for whoever is leading an expedition -- not a second LLM
# agent (that would double Ollama calls per tribe per cycle for a party that doesn't
# make its own strategic decisions anyway; the tribe already decided to send them).
# Just enough identity that "the exploration team" reads as a group of people with a
# lead, not an anonymous abstraction, matching how the tribe's own chief already has a
# name. Deterministic per (tribe, cycle) so re-reading the same expedition's state
# doesn't change who's leading it.
_SCOUT_NAME_SYLLABLES = (
    "Ka", "Ren", "Tor", "Vel", "Sha", "Nim", "Bri", "Kol", "Tal", "Ora", "Fen", "Mir",
)


def _generate_scout(tribe, cycle: int, base_days: int = None) -> dict:
    """A name plus a determination trait (0.0-1.0) that shifts this scout's own
    personal give-up point by up to EXPEDITION_DETERMINATION_DAY_VARIANCE days either
    side of the default -- some parties push a little harder, some turn back a little
    sooner, rather than every expedition behaving identically. `base_days` lets a
    different expedition kind (e.g. HUNTING_PARTY) use its own default patience instead
    of SCOUT's."""
    if base_days is None:
        base_days = config.EXPEDITION_MAX_DAYS
    seed = hash((tribe.id, cycle)) & 0xFFFFFFFF
    rng = random.Random(seed)
    name = "".join(rng.sample(_SCOUT_NAME_SYLLABLES, 2))
    determination = rng.random()
    day_bonus = round((determination - 0.5) * 2 * config.EXPEDITION_DETERMINATION_DAY_VARIANCE)
    return {
        "name": name,
        "determination": determination,
        "max_days": base_days + day_bonus,
    }


# Explicit request: "Raiders incoming need a Label, like 'Terrible Knoxit RAIDS!!!'"
# -- a live report that every raider approach/sighting read as the same bare
# "Raiders" text everywhere on the map, indistinguishable from every other one
# a tribe has ever seen. Same deterministic seeded-combination shape
# _generate_scout uses above (no LLM call, no state to track beyond the seed
# inputs), just its own title/syllable pool so a raid reads as a distinct,
# memorable threat rather than another named individual.
_RAIDER_NAME_TITLES = ("Terrible", "Savage", "Dread", "Merciless", "Bloodfang", "Ruthless", "Wicked", "Feral")
_RAIDER_NAME_SYLLABLES = ("Grak", "Thok", "Mor", "Zul", "Krag", "Skarn", "Vor", "Drex", "Rok", "Gnash", "Kro", "Fell")


def _generate_raider_name(tribe_id: str, cycle: int) -> str:
    """Deterministic per (tribe, cycle) the same way _generate_scout's own name
    is -- re-reading the same approach's state doesn't change who's riding in."""
    seed = hash((tribe_id, cycle, "raiders")) & 0xFFFFFFFF
    rng = random.Random(seed)
    title = rng.choice(_RAIDER_NAME_TITLES)
    name = "".join(rng.sample(_RAIDER_NAME_SYLLABLES, 2))
    return f"{title} {name}"


# Moved to physics.reflect_into_grid so RELOCATE's raw model target (simulation.py)
# and _hunting_party's own target (below) can reuse it too -- kept as a thin alias
# here since this is where it originated and existing callers/tests import it from
# this module.
_reflect_into_grid = physics.reflect_into_grid


def _expedition_launch_point(tribe, angle_radians: float, grid_size: int) -> tuple[int, int]:
    """Explicit request: "All Scouting, Hunting, Exploration, etc. should use
    starting points off the edge of the Territory boundary, not the center."
    Once territory exists, a party sets out already at the wall's own edge,
    along the same heading it's actually traveling, instead of fanning out from
    the exact town-hall tile every single dispatch. Falls back to the tribe's
    own current position before territory exists (a nomadic band with no walls
    yet has no "edge" to start from)."""
    if tribe.territory_center is None:
        return tribe.x, tribe.y
    cx, cy = tribe.territory_center
    lx = _reflect_into_grid(round(cx + math.cos(angle_radians) * tribe.territory_radius), grid_size)
    ly = _reflect_into_grid(round(cy + math.sin(angle_radians) * tribe.territory_radius), grid_size)
    return lx, ly


def _push_past_visited_ground(
    tribe, ox: int, oy: int, angle_radians: float, base_distance: float, grid_size: int,
) -> tuple[int, int]:
    """Explicit request: prevent "incessant 'survey's an area' nonsense... they
    found it, it's good for XYZ, move on, no need to explore it again." Once a
    heading's own target would land in a Tribe Map sector already marked
    visited, pushes farther out along that exact same heading instead of
    settling for already-covered ground -- the same "keep walking this
    direction" idea physics.extend_ray_to_grid_edge already uses once an
    ordinary search reaches its target early with days left. Bounded (6 tries)
    and fails open: if every attempt is still visited (a small or heavily
    covered map), returns the farthest one tried rather than refusing to
    launch."""
    tx = ty = None
    for step in range(6):
        distance = base_distance * (1 + step * 0.5)
        tx = _reflect_into_grid(ox + round(math.cos(angle_radians) * distance), grid_size)
        ty = _reflect_into_grid(oy + round(math.sin(angle_radians) * distance), grid_size)
        if sector_of(tx, ty) not in tribe.visited_sectors:
            break
    return tx, ty


def _scout(sim, tribe, biome, target):
    """Dispatches an expedition -- your most capable people, out searching, not an
    instant look. They travel and camp under their own supply (no drain on the
    tribe's stockpile), for up to config.EXPEDITION_MAX_DAYS before turning back
    empty-handed if they've found nothing. If they reach real fresh water or their
    intended destination first, they turn back immediately to report it -- but the
    finding only becomes real, actionable knowledge once they've walked all the way
    home (Simulation._advance_expeditions runs the day-by-day travel; this handler
    only launches or no-ops one). Only after that can the tribe's own reasoning
    choose to RELOCATE the whole camp there. This replaced an instant per-turn
    terrain check and, before that, handing a newly-elected chief water's exact
    coordinates outright (see leadership.py) -- water and distant terrain should be
    things a tribe discovers by actually sending people to go look, not facts the
    simulation gifts for free.

    Explicit request: "they can't reason about closeness to the discover, they
    have to get to a pre-assigned location and explore along the way... scout
    directions rotate on a 20 degree angle starting with the South East."
    target_vector is deliberately NOT read here anymore -- live runs showed two
    scouts launched back to back heading the exact same direction, since small
    models repeatedly failed to turn compass-direction facts (or even their own
    prior choices) into coordinates that actually covered new ground. Each real
    dispatch advances tribe.scout_rotation_index by one step
    (config.SCOUT_ROTATION_STEP_DEGREES), so coverage spreads out over time
    regardless of what the model reasons about geometry -- projected out to the
    grid edge along that heading (physics.extend_ray_to_grid_edge), the same
    "keep walking this direction" logic an ordinary search already pushes onward
    with once it reaches its own original target.

    A tribe can have up to expedition_capacity(tribe) parties out at once (scaling with
    population past config.MAX_CONCURRENT_EXPEDITIONS' floor -- see that function), any
    mix of scouting and hunting -- capped rather than unlimited since nothing currently
    deducts population to launch one."""
    blocked = _expedition_dispatch_blocked(tribe, "scout")
    if blocked is not None:
        return blocked

    angle_degrees = (
        config.SCOUT_ROTATION_START_ANGLE_DEGREES
        + config.SCOUT_ROTATION_STEP_DEGREES * tribe.scout_rotation_index
    ) % 360
    tribe.scout_rotation_index += 1
    angle_radians = math.radians(angle_degrees)
    # Bug report: "they go big long lines like they are flying, possibly too far."
    # This used to project all the way to the grid's true edge (physics.
    # extend_ray_to_grid_edge, up to ~99 tiles distant) and, if a party reached it
    # early with days left, push even further -- a ruler-straight, cross-map dash
    # every single dispatch. SCOUT_PATROL_DISTANCE bounds a single dispatch to a
    # local patrol instead; the rotating heading (scout_rotation_index) still sweeps
    # a new direction each real dispatch, so coverage keeps spreading over many
    # shorter trips rather than one long one. EXPEDITION_SPEED is untouched -- this
    # is deliberately about how far a trip is aimed, not how fast it's walked.
    # Explicit request: "I want to prevent this incessant 'survey's an area'
    # nonsense" -- pushes past the Tribe Map's already-visited ground along this
    # same heading instead of landing on a sector already confirmed.
    #
    # Live correction: "the Scout take off point is right, but should be at a
    # 90 degree angle away from the boundary, not parallel to it." Computing
    # the target from tribe.x/y (its own live position, which drifts up to
    # territory_radius away from territory_center after settling) while the
    # launch point is placed from territory_center meant the two weren't
    # necessarily on the same ray -- the walk from launch point to target could
    # angle off to one side instead of continuing straight outward. Launching
    # from the same point the party actually starts at keeps that walk exactly
    # radial (perpendicular to the boundary), matching the launch point's own
    # placement.
    lx, ly = _expedition_launch_point(tribe, angle_radians, sim.world.grid_size)
    tx, ty = _push_past_visited_ground(tribe, lx, ly, angle_radians, config.SCOUT_PATROL_DISTANCE, sim.world.grid_size)
    scout = _generate_scout(tribe, sim.cycle)
    tribe.expeditions.append({
        "kind": "scout",
        "pos": [lx, ly],
        "origin": [lx, ly],
        "target": [tx, ty],
        "day": 0,
        "phase": "outbound",
        "found": None,
        "terrain_report": None,
        "food_gathered": 0,
        "water_gathered": 0,
        "lead_scout": scout["name"],
        "determination": scout["determination"],
        "max_days": scout["max_days"],
        # Everywhere this expedition has actually walked this trip -- the persistent
        # world-trail mechanic (Landscape.trails) only lights up once a route gets
        # reused, so a single fresh journey barely shows anything even while it's
        # actively happening. This is just this one party's breadcrumb line, cleared
        # when they get home, not a permanent feature of the map.
        "path": [[lx, ly]],
    })
    tribe.expeditions_launched += 1
    return f"scouts led by {scout['name']} depart camp to explore toward ({tx},{ty})"


def _exploration_party(sim, tribe, biome, target):
    """A deeper, more deliberate expedition than SCOUT -- explicit request:
    "a smart Chief will send one Scout and one Exploration Party." Where SCOUT
    is a fast, discovery-only dash, an Exploration Party travels longer,
    gathers real wood/stone along the way (Simulation._advance_exploration_
    party_outbound) up to a real carrying-capacity limit, and can stumble on
    a rival settlement or a Landmark -- on top of everything SCOUT's own
    return already discovers (water, resource sites, raider camps), shared
    via Simulation._advance_one_expedition's common fallthrough. Own rotating
    heading (tribe.explore_rotation_index, offset from SCOUT's own sweep) so
    the two parties don't retrace each other's ground."""
    blocked = _expedition_dispatch_blocked(tribe, "explore")
    if blocked is not None:
        return blocked

    angle_degrees = (
        config.SCOUT_ROTATION_START_ANGLE_DEGREES
        + config.SCOUT_ROTATION_STEP_DEGREES * tribe.explore_rotation_index
        + 180  # offset from SCOUT's own sweep so the two don't retrace each other
    ) % 360
    tribe.explore_rotation_index += 1
    angle_radians = math.radians(angle_degrees)
    # Live-run finding: 829 EXPLORATION_PARTY dispatches averaged 1.02 days
    # before turning back, against a 6-day budget -- reusing SCOUT_PATROL_
    # DISTANCE meant it never actually went any farther than a plain SCOUT.
    # EXPLORATION_PARTY_PATROL_DISTANCE gives it a real, longer reach of its
    # own, and the same Tribe Map push-past used for SCOUT.
    #
    # Same launch-point-as-origin correction as _scout's own comment -- keeps
    # the walk from launch point to target strictly radial (perpendicular to
    # the territory boundary), not skewed off to one side by tribe.x/y's own
    # drift away from territory_center.
    lx, ly = _expedition_launch_point(tribe, angle_radians, sim.world.grid_size)
    tx, ty = _push_past_visited_ground(
        tribe, lx, ly, angle_radians, config.EXPLORATION_PARTY_PATROL_DISTANCE, sim.world.grid_size
    )
    scout = _generate_scout(tribe, sim.cycle, base_days=config.EXPLORATION_PARTY_MAX_DAYS)
    tribe.expeditions.append({
        "kind": "explore",
        "pos": [lx, ly],
        "origin": [lx, ly],
        "target": [tx, ty],
        "day": 0,
        "phase": "outbound",
        "found": None,
        "terrain_report": None,
        "food_gathered": 0,
        "water_gathered": 0,
        "wood_gathered": 0,
        "stone_gathered": 0,
        "lead_scout": scout["name"],
        "determination": scout["determination"],
        "max_days": scout["max_days"],
        "path": [[lx, ly]],
    })
    tribe.expeditions_launched += 1
    return f"an exploration party led by {scout['name']} departs camp to chart new ground toward ({tx},{ty})"


def _hunting_party(sim, tribe, biome, target):
    """A multi-day alternative to instant HUNT_DEER, sharing the exact same expedition
    list and day-by-day travel machinery as SCOUT (up to config.MAX_CONCURRENT_
    EXPEDITIONS parties out at once, any mix of hunting and scouting). Persists day
    over day -- moving toward target_vector, camping under its own supply -- rolling a
    fresh catch chance each day (scaled by wherever they currently stand's own game
    yield) until something is caught or config.HUNTING_PARTY_MAX_DAYS runs out, and
    carries the same wolf-pack hazard risk as an instant hunt on every single day out,
    not just once.

    The catch only becomes real food the moment the party walks back into camp -- same
    "findings aren't real until you're home" rule as SCOUT. That's the deliberate,
    testable tension: a tribe that's starving *right now* gets no relief from a hunt
    that's still out in the field, no matter how promising, and every extra day spent
    searching is another chance at a hazard, not a free wait."""
    blocked = _expedition_dispatch_blocked(tribe, "hunt")
    if blocked is not None:
        return blocked

    tx, ty = target
    # Explicit request: "the bounds-safe function is too loose at the edges of our
    # board." A plain clamp here collapsed any model-chosen overshoot onto the exact
    # boundary tile -- the same class of bug _reflect_into_grid was already built to
    # fix for SCOUT/EXPLORATION_PARTY's own targets.
    tx = _reflect_into_grid(tx, sim.world.grid_size)
    ty = _reflect_into_grid(ty, sim.world.grid_size)
    # Explicit request: "All Scouting, Hunting, Exploration, etc. should use
    # starting points off the edge of the Territory boundary, not the center."
    # HUNTING_PARTY trusts the model's own target_vector rather than a computed
    # compass heading (see this function's own docstring), so the heading used
    # for the launch point is derived from tribe -> target instead.
    angle_radians = math.atan2(ty - tribe.y, tx - tribe.x)
    lx, ly = _expedition_launch_point(tribe, angle_radians, sim.world.grid_size)
    scout = _generate_scout(tribe, sim.cycle, base_days=config.HUNTING_PARTY_MAX_DAYS)
    tribe.expeditions.append({
        "kind": "hunt",
        "pos": [lx, ly],
        "origin": [lx, ly],
        "target": [tx, ty],
        "day": 0,
        "phase": "outbound",
        "food_caught": 0,
        "food_gathered": 0,
        "water_gathered": 0,
        "lead_scout": scout["name"],
        "determination": scout["determination"],
        "max_days": scout["max_days"],
        "path": [[lx, ly]],
    })
    tribe.expeditions_launched += 1
    return f"a hunting party led by {scout['name']} departs camp toward ({tx},{ty})"


def _relocate(sim, tribe, biome, target):
    """The only action that actually moves the tribe -- a deliberate decision to
    relocate the whole camp, not an automatic side effect of doing something else.
    Costs stamina (food/water, on top of ordinary upkeep) -- marching is tiring, and
    without a cost here relocating would be strictly free compared to every gathering
    action, which all cost time and risk. Moving through a well-worn trail is faster
    than breaking new ground, and relocating wears that trail a little more."""
    tribe.food = max(0, tribe.food - config.RELOCATE_FOOD_COST)
    tribe.water = max(0, tribe.water - config.RELOCATE_WATER_COST)
    tx, ty = target
    bonus = sim.world.trail_speed_bonus(tribe.x, tribe.y, config.MAX_TRAIL_BONUS_SPEED)
    # Explicit request ("everyone moving on the board moves at the pace of 1
    # sky tick"): a settled tribe relocating uses the slower, uniform pace;
    # pre-settlement keeps the faster MOVEMENT_SPEED since that march (toward
    # newly-confirmed water) is life-or-death and needs to stay fast.
    speed_base = config.MOVEMENT_SPEED if not tribe.has_ever_settled else config.SETTLED_MOVEMENT_SPEED
    base_speed = speed_base + bonus
    # Explicit request: "travel speed is 5x on toll roads."
    if sim.world.is_toll_road(tribe.x, tribe.y):
        base_speed *= config.TOLL_ROAD_SPEED_MULTIPLIER
    nx, ny = physics.terrain_aware_step(tribe.x, tribe.y, tx, ty, base_speed=base_speed, has_boat=tribe.boat_built)
    nx, ny = sim._resolve_toll(tribe, tribe.x, tribe.y, nx, ny)
    # Live bug: territory/wall_rings/buildings are placed once, permanently, at
    # tribe.territory_center the instant a tribe first settles (Simulation.
    # _found_territory) -- "territory is not founded until settled," so this only
    # ever applies afterward, never during ordinary pre-settlement wandering. A
    # single RELOCATE step can cover 15-20+ tiles with trail/toll-road speed
    # bonuses stacked, which a live run showed carrying an already-settled tribe
    # clean outside its own walls in one turn -- cycles_since_relocate didn't even
    # reset (both ends of the jump independently qualified as "settled enough"
    # ground), so the tribe was left permanently detached from its own city:
    # every future build kept landing back at the abandoned territory_center,
    # nowhere near where the tribe actually stood. Clamped to the tribe's own
    # territory_radius instead of left unbounded -- a settled tribe can still
    # move freely anywhere within its own city, just can't step outside it.
    if tribe.territory_center is not None:
        tcx, tcy = tribe.territory_center
        dist = ((nx - tcx) ** 2 + (ny - tcy) ** 2) ** 0.5
        if dist > tribe.territory_radius:
            scale = tribe.territory_radius / dist
            nx = round(tcx + (nx - tcx) * scale)
            ny = round(tcy + (ny - tcy) * scale)
    sim.world.wear_trail(nx, ny, config.TRAIL_WEAR_PER_PASS, tribe.color, tribe.id)
    mark_visited_sector(tribe, nx, ny)
    tribe.x, tribe.y = nx, ny
    # Explicit correction: "the volcano is a Hazard they will die if they go
    # there." Unlike the river's drowning hazard (Simulation._expedition_river_
    # hazard, expedition movement only), RELOCATE moving the whole camp onto/
    # through the volcano needs the same real consequence -- this is exactly the
    # kind of "they went there" the hazard is meant to catch.
    sim._volcano_hazard(tribe, nx, ny)
    # Same coverage, for cliffs/ocean/shoals -- see Simulation._cliffs_hazard/
    # _ocean_hazard/_shoals_hazard's own comments.
    sim._cliffs_hazard(tribe, nx, ny)
    sim._ocean_hazard(tribe, nx, ny)
    sim._shoals_hazard(tribe, nx, ny)
    return None


def _eligible_breeding_pair(tribe) -> tuple[str, str] | None:
    """The named-individual pool this tribe can draw parents from: whoever is
    currently chief, plus whoever currently holds a trophy (see Simulation.
    _award_trophy's `individual` param) -- not the whole population, which stays an
    anonymous count. Most-recently-earned trophy holders are preferred as the second
    parent, so a fresh milestone is what actually gets acted on. Returns None if fewer
    than two distinct named individuals exist yet."""
    candidates = []
    if tribe.chief_name:
        candidates.append(tribe.chief_name)
    for trophy in reversed(tribe.trophies):
        name = trophy["chief"]
        if name not in candidates:
            candidates.append(name)
    if len(candidates) < 2:
        return None
    return candidates[0], candidates[1]


def _eligible_new_battalion_leader(tribe) -> str | None:
    """Military branch, step 1, redesigned 2026-09-10: "We already name
    Warriors when they get Trophies. If they have one they can let it lead a
    battalion." No more chief-chosen NAME_WARRIOR action -- whoever has the
    most trophies personally credited to them (config.
    BATTALION_LEADER_TROPHY_THRESHOLD or more, now just 1) is automatically
    eligible to lead a Battalion, excluding the Chief (the Chief leads the
    tribe, doesn't personally lead troops) and excluding anyone already
    leading one of tribe.battalions (each named leader gets one Battalion,
    not several). Ties keep whichever name reached that count first
    (tribe.trophies is already in the order each was actually earned, and
    dict iteration order matches insertion order). Returns None if nobody
    new qualifies yet -- called from _allocate_battalion_strength whenever
    growth needs a home, not from any chief-facing action."""
    already_leading = {b["leader"] for b in tribe.battalions}
    counts: dict[str, int] = {}
    for trophy in tribe.trophies:
        name = trophy["chief"]
        if name == tribe.chief_name or name in already_leading:
            continue
        counts[name] = counts.get(name, 0) + 1
    best_name, best_count = None, 0
    for name, count in counts.items():
        if count > best_count:
            best_name, best_count = name, count
    return best_name if best_count >= config.BATTALION_LEADER_TROPHY_THRESHOLD else None


def _allocate_battalion_strength(tribe, added: int) -> None:
    """Military branch, steps 1-2, redesigned 2026-09-10: "if they have more
    than 1, they can have many Battallions. If they have a lot, we need some
    restrictions." Called by both _build_barracks and _train_battalion
    whenever tribe.battalion_size actually grows -- decides whose command
    that new strength joins. A fresh, not-yet-leading trophy-holder (see
    _eligible_new_battalion_leader) gets a brand new Battalion of their own,
    up to config.MAX_CONCURRENT_BATTALIONS concurrent leaders (SECOND-
    OPINION(Sonnet 5, 2026-09-10): a real leadership hierarchy above this
    flat cap -- e.g. a General over several Battalion leaders -- was also on
    the table; deferred as the smaller v1 per Scott's own "possibly" hedge).
    Once the cap is reached, or nobody's proven themselves yet, the growth
    reinforces whichever existing Battalion is currently smallest instead.
    If neither a new leader nor an existing Battalion is available, the
    strength simply sits unled for now (a real militia headcount,
    tribe.battalion_size, that's outpaced who's actually stepped up to lead
    it) -- compute_might already treats unled headcount as contributing no
    trophy bonus, the same as an unnamed Warrior always did.

    Explicit follow-up, 2026-09-10: "It's probably smart to 'promote' the
    Warrior leaders. The Tribe should know and maybe be happy, maybe even
    celebrate" -- then corrected the same day: "I think we are doing
    celebrations in a very complex way... if there are multiple reasons to
    celebrate in a day, they are packaged to celebrate all in one big
    festive party." A brand new leader no longer fires their own pride
    wave/Fame bump immediately (that used to bypass the shared celebration
    cooldown every other celebration in this project respects, and would
    have meant a separate party for each of up to MAX_CONCURRENT_BATTALIONS
    leaders even if they were all named the same day) -- their name just
    joins tribe.pending_battalion_celebrations; Simulation.
    _advance_battalion_celebrations is what actually throws the party, once
    per cooldown window, naming everyone still waiting at once."""
    if added <= 0:
        return
    candidate = _eligible_new_battalion_leader(tribe)
    if candidate is not None and len(tribe.battalions) < config.MAX_CONCURRENT_BATTALIONS:
        tribe.battalions.append({"leader": candidate, "size": added})
        tribe.history.append(f"{candidate} steps up to lead a new Battalion of {added} for {tribe.name}, proven by their own deeds")
        tribe.pending_battalion_celebrations.append(candidate)
        return
    if tribe.battalions:
        weakest = min(tribe.battalions, key=lambda b: b["size"])
        weakest["size"] += added


def _battalion_leaders_text(tribe) -> str:
    """Flavor text for _train_battalion's own result string -- "under
    {warrior_name}" doesn't make sense once a tribe can have several named
    leaders at once (or none yet, if nobody's earned a trophy)."""
    if not tribe.battalions:
        return "its own officers"
    return ", ".join(b["leader"] for b in tribe.battalions)


def _breed(sim, tribe, biome, target):
    """Two named individuals from the tribe -- its chief and whoever holds a trophy,
    see _eligible_breeding_pair -- start a family. A solo cost paid by this one tribe
    (see config.BREED_FOOD_COST/WATER_COST), distinct from the shared/split cost a
    future tribe-to-tribe merge would use. The actual outcome (the child's name, a
    flavor note) isn't decided here -- this only sets tribe.pending_birth; Simulation.
    step() resolves it with a real, non-scripted LLM call (backend/breeding.py) the
    same cycle, the same pattern _install_chief already uses for pending_chief_context."""
    if not _has_room_to_grow(tribe):
        return "no room to raise a family right now -- the tribe has outgrown what it can currently sustain"
    pair = _eligible_breeding_pair(tribe)
    if pair is None:
        return "no one with enough standing in the tribe yet to start a family"
    if tribe.food < config.BREED_FOOD_COST or tribe.water < config.BREED_WATER_COST:
        return "too little food and water spared to support a new family right now"

    tribe.food -= config.BREED_FOOD_COST
    tribe.water -= config.BREED_WATER_COST
    parent_a, parent_b = pair
    tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
    return f"{parent_a} and {parent_b} decide to start a family together"


def _find_minor_settlement(sim, x, y):
    """Shared by RAID and TRADE -- a settlement mid-respawn (raids_remaining <= 0)
    isn't a valid target for either until it's back. Reuses RAID_PROXIMITY_RADIUS
    as the search distance, same as a rival tribe -- not a separate, wider net."""
    for ms in sim.minor_settlements:
        if ms["raids_remaining"] <= 0:
            continue
        if (ms["x"] - x) ** 2 + (ms["y"] - y) ** 2 <= config.RAID_PROXIMITY_RADIUS ** 2:
            return ms
    return None


def _raid_minor_settlement(sim, tribe, settlement):
    """No people, no chief, no LLM on the other side -- explicit request: 'no
    advanced logic like battle... stealing only.' A raid here always succeeds, at
    no population risk, unlike raiding a real rival tribe. Only 3 uses
    (config.MINOR_SETTLEMENT_MAX_RAIDS) before it's exhausted and needs to
    respawn (Simulation._advance_minor_settlements).

    Live-bug-adjacent finding, not yet confirmed to have fired in a real run
    but a real structural risk: a settlement respawns holding an exact copy of
    whichever tribe currently has the highest population (Simulation.
    _biggest_tribe_snapshot). If the biggest tribe is the one doing the
    raiding, it's raiding a mirror of its own stockpile -- and this addition
    used to setattr the stolen amount directly, bypassing the storage cap, the
    same gap Simulation._resolve_raider_attack's own fix just closed. Capped
    here the same way: the settlement still loses the full stolen amount (a
    real, permanent loss toward its own depletion), but only what actually
    fits under the tribe's own storage cap lands in its stockpile.

    Code-quality pass: routed through sim._capped_add instead of inlining the
    same cap arithmetic that helper already implements."""
    looted = {}
    for resource in ("wood", "stone", "food", "water"):
        stolen = round(settlement[resource] * config.MINOR_SETTLEMENT_RAID_STEAL_FRACTION)
        settlement[resource] -= stolen
        looted[resource] = sim._capped_add(tribe, resource, stolen)
    settlement["raids_remaining"] -= 1
    if settlement["raids_remaining"] <= 0:
        settlement["depleted_at_cycle"] = sim.cycle
    sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
    sim.recent_encounters.append({
        "x": settlement["x"], "y": settlement["y"], "kind": "minor_settlement_raid",
        "label": "Settlement raided", "outcome": "won",
    })
    return (
        f"raided an outlying settlement -- {looted['wood']} wood, {looted['stone']} stone, "
        f"{looted['food']} food, and {looted['water']} water taken"
    )


def _raid(sim, tribe, biome, target):
    """Attempt to raid a rival tribe found at target_vector -- the mechanical outlet
    for an aggressive/warlord chief philosophy (leadership.py can already generate one)
    that otherwise has nothing to act on. Real risk on both sides: win chance starts as
    just the attacker's share of the two tribes' combined population, then gets a
    bounded Might-based adjustment on top (_might_adjusted_win_chance -- Military
    branch, plan file valiant-forging-falcon.md), so a smaller raiding party can
    still lose to a larger, better-armed defender, and even a winning raid costs the
    attacker people -- violence isn't a free lever here. Also checks for an
    unaffiliated minor settlement first (Simulation._spawn_minor_settlements) -- a
    much safer, weaker target than a real rival, guaranteed to succeed, unaffected by
    Might (RAID against a minor settlement never touches this path)."""
    tx, ty = target
    settlement = _find_minor_settlement(sim, tx, ty)
    if settlement is not None:
        return _raid_minor_settlement(sim, tribe, settlement)

    defender = None
    for other in sim.tribes.values():
        if other.id == tribe.id or other.extinct:
            continue
        if (other.x - tx) ** 2 + (other.y - ty) ** 2 <= config.RAID_PROXIMITY_RADIUS ** 2:
            defender = other
            break

    if defender is None:
        return "found no rival encampment there to raid"

    # Object Creator era's combat_boost effect applies here too, not just
    # DECLARE_CONQUEST -- see _created_object_bonus.
    effective_population = tribe.population * (1 + _created_object_bonus(tribe, "combat_boost"))
    attacker_win_chance = effective_population / max(1, effective_population + defender.population)
    attacker_win_chance = _might_adjusted_win_chance(tribe, defender, attacker_win_chance)
    if random.random() < attacker_win_chance:
        # Code-quality pass: the attacker's own gain here used to setattr the
        # stolen amount directly -- the same uncapped-mutation bug already fixed
        # on the raider-camp-strike/minor-settlement-raid paths (see
        # _raid_minor_settlement's own comment), just missed on the one path
        # where a tribe raids a real rival tribe instead of a camp/settlement.
        for resource in ("wood", "stone", "food", "water"):
            stolen = round(getattr(defender, resource) * config.RAID_STEAL_FRACTION)
            setattr(defender, resource, getattr(defender, resource) - stolen)
            sim._capped_add(tribe, resource, stolen)
        tribe.raids_won += 1
        _record_combat(tribe, "Raiding", "won")
        _record_combat(defender, "Raid Defense", "lost")
        if tribe.raids_won == 1:
            sim._award_trophy(tribe, "First Conquest")
        sim._check_custom_awards(tribe, "raiding")

        # Population moves rather than just vanishing -- captured or defecting
        # survivors, not pointless casualties. Enough raids like this eventually
        # absorb the defender entirely (see Simulation._merge_tribes) instead of a
        # flat, repeatable loss with no benefit to the winner beyond stolen goods.
        absorbed = min(defender.population, max(1, round(defender.population * config.RAID_POPULATION_ABSORB_FRACTION)))
        defender.population -= absorbed
        tribe.population += absorbed
        tribe.max_population = max(tribe.max_population, tribe.population)

        sim._lose_population(tribe, config.RAID_ATTACKER_POPULATION_LOSS_ON_WIN, cause="raid_losses")
        sim.trauma.radiate_event_wave(defender.x, defender.y, config.RAID_TRAUMA_MAGNITUDE, config.RAID_TRAUMA_RADIUS)
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
        sim.recent_encounters.append({
            "x": defender.x, "y": defender.y, "kind": "tribe_raid",
            "label": f"{tribe.name} raids {defender.name}", "outcome": "won",
        })

        if defender.population <= 0:
            old_name = tribe.name
            defender_name = defender.name
            new_name = sim._merge_tribes(tribe, defender)
            return f"raided {defender_name}, fully absorbing its survivors -- {old_name} becomes {new_name}!"

        defender.history.append(f"{tribe.name}'s raid carried off {absorbed} of {defender.name}'s people")
        return f"raided {defender.name}, seized supplies, and absorbed {absorbed} survivors"
    else:
        # Explicit request: "Raids that fail give the winning Tribe people and
        # inventory" -- the win branch above already lets the attacker loot and
        # absorb population on success; a repelled defense used to only ever avoid
        # loss, never actually gain anything beyond a counter. Mirrors the win
        # branch, roles reversed: the defender is the one who just won.
        #
        # Ordering matters here: the pre-existing small attrition cost
        # (_lose_population, below) runs FIRST and can itself mark the attacker
        # extinct through its own normal channel (cause="failed_raid" -- they
        # really did just die from the failed attempt). Only if they survive THAT
        # does the new absorption apply on top, which can separately finish them
        # off through _merge_tribes. Never both in the same pass -- an
        # already-extinct tribe has nothing left to absorb.
        tribe.raids_lost += 1
        defender.raids_defended += 1
        _record_combat(tribe, "Raiding", "lost")
        _record_combat(defender, "Raid Defense", "won")
        if defender.raids_defended == 1:
            sim._award_trophy(defender, "Raid Breaker")
        # Code-quality pass: same uncapped-mutation fix as the win branch above,
        # mirrored -- the defender's gain here was equally uncapped.
        for resource in ("wood", "stone", "food", "water"):
            stolen = round(getattr(tribe, resource) * config.RAID_STEAL_FRACTION)
            setattr(tribe, resource, getattr(tribe, resource) - stolen)
            sim._capped_add(defender, resource, stolen)

        sim._lose_population(tribe, config.RAID_ATTACKER_POPULATION_LOSS_ON_LOSS, cause="failed_raid")
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_TRAUMA_MAGNITUDE, config.RAID_TRAUMA_RADIUS)
        sim.trauma.radiate_event_wave(defender.x, defender.y, config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
        sim.recent_encounters.append({
            "x": defender.x, "y": defender.y, "kind": "tribe_raid",
            "label": f"{tribe.name} repelled by {defender.name}", "outcome": "lost",
        })

        if tribe.extinct:
            return f"attempted to raid {defender.name} and was wiped out in the failed attempt"

        absorbed = min(tribe.population, max(1, round(tribe.population * config.RAID_POPULATION_ABSORB_FRACTION)))
        tribe.population -= absorbed
        defender.population += absorbed
        defender.max_population = max(defender.max_population, defender.population)

        if tribe.population <= 0:
            old_name = defender.name
            attacker_name = tribe.name
            new_name = sim._merge_tribes(defender, tribe)
            return f"attempted to raid {attacker_name}, but was fully repelled and absorbed -- {old_name} becomes {new_name}!"

        defender.history.append(f"{defender.name} repelled {tribe.name}'s raid and carried off {absorbed} of its people")
        return f"attempted to raid {defender.name} and was repelled, losing supplies and people in the process"


def _strike_raider_camp(sim, tribe, biome, target):
    """A tribe that has scouted a raider camp (Simulation._advance_one_expedition's
    raider-sighting roll, tribe.raider_sightings) can strike it directly once
    organized enough (Bronze Age) -- turning a known threat into an actionable target
    instead of only ever defending against it. Instant, like RAID, not a multi-day
    expedition. Win chance is population-scaled since the camp itself has no
    simulated population to compare against, unlike RAID's ratio-based chance."""
    camp = tuple(target)
    if camp not in tribe.raider_sightings:
        return "no known raider camp at that location"

    win_chance = min(
        config.STRIKE_RAIDER_CAMP_MAX_WIN_CHANCE,
        config.STRIKE_RAIDER_CAMP_BASE_WIN_CHANCE
        + (tribe.population // 10) * config.STRIKE_RAIDER_CAMP_POPULATION_BONUS_PER_10,
    )
    if random.random() < win_chance:
        tribe.raider_sightings.remove(camp)
        # Live bug, same shape as Simulation._resolve_raider_attack's own fix:
        # this used to setattr a fraction of the tribe's OWN current food
        # directly, bypassing the storage cap -- not loot recovered from the
        # camp, just an uncapped compound multiplier on the tribe's own
        # stockpile every time this is won. sim._capped_add returns the amount
        # actually added, since this action's own return line needs the real
        # number, not just a message.
        looted = sim._capped_add(tribe, "food", round(tribe.food * config.STRIKE_RAIDER_CAMP_LOOT_FRACTION))
        sim.trauma.radiate_event_wave(camp[0], camp[1], config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
        sim.recent_encounters.append({
            "x": camp[0], "y": camp[1], "kind": "raider_camp_strike",
            "label": "Raider camp destroyed", "outcome": "won",
        })
        _record_combat(tribe, "Raider Camp Strike", "won")
        return f"the raider camp at {camp} is destroyed -- {looted} food recovered"

    sim._lose_population(tribe, config.STRIKE_RAIDER_CAMP_POPULATION_LOSS_ON_FAILURE, cause="failed_raider_strike")
    sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_TRAUMA_MAGNITUDE, config.RAID_TRAUMA_RADIUS)
    sim.recent_encounters.append({
        "x": camp[0], "y": camp[1], "kind": "raider_camp_strike",
        "label": "Strike failed", "outcome": "lost",
    })
    _record_combat(tribe, "Raider Camp Strike", "lost")
    return f"the strike on the raider camp at {camp} failed -- they escaped into the wilds"


def _expel_raiders_from_territory(sim, tribe, biome, target):
    """A proactive, whole-tribe alternative to just waiting for
    Simulation._resolve_raider_attack's passive defense once raiders are
    already inbound (tribe.raiders_approaching) -- "a full population frenzy
    repelling all Raiders from the Territory Boundary." Win chance is
    population-scaled like STRIKE_RAIDER_CAMP's own (an approaching raider
    party has no simulated population of its own to compare against, same
    reasoning that function's comment already gives), just with a higher
    base/ceiling since committing the whole population is a stronger showing
    than an ordinary strike party.

    Explicit follow-up: "if they lose, they lose but redouble their efforts in
    the same turn... if they have to try again, populations are lost and the
    gains reduce." A failed wave doesn't end the action outright -- anger
    fuels an immediate retry, up to config.EXPEL_RAIDERS_MAX_WAVES total in
    this one call, each wave still a real cost (population lost, a
    RAID_STEAL_FRACTION-sized cut of resources, same shape RAID's own failure
    branch uses) so retrying is never free, and the eventual reward shrinks
    each wave it took to actually win. Bounded rather than unbounded so an
    overwhelming force can't loop forever in one action -- if every wave here
    fails (or the tribe goes extinct partway through), tribe.raiders_approaching
    is left exactly as it was, falling back to the existing passive countdown/
    defense rather than inventing a second failure outcome for the same
    threat.

    A win doesn't just clear the threat -- the raiders are "cast elsewhere on
    the map," reusing the exact relocate-after-ambush mechanic an expedition's
    own raider encounter already has (Simulation._relocate_raider_sighting_
    after_ambush), so they become a real, later-strikeable STRIKE_RAIDER_CAMP
    target instead of vanishing without a trace."""
    approach = tribe.raiders_approaching
    if approach is None:
        return "no raiders are currently approaching the territory to expel"

    ax, ay = approach["x"], approach["y"]
    reward_multiplier = 1.0
    for wave in range(1, config.EXPEL_RAIDERS_MAX_WAVES + 1):
        win_chance = min(
            config.EXPEL_RAIDERS_MAX_WIN_CHANCE,
            config.EXPEL_RAIDERS_BASE_WIN_CHANCE
            + (tribe.population // 10) * config.EXPEL_RAIDERS_WIN_CHANCE_POPULATION_BONUS_PER_10,
        )
        if random.random() < win_chance:
            tribe.raiders_approaching = None
            # Code-quality pass: uncapped tribe.food += -- same bug class as the
            # rest of this file. A large, established tribe's population-scaled
            # loot here can be substantial, easily enough to overrun the cap in
            # one call.
            gained_food = sim._capped_add(
                tribe, "food",
                max(1, round(tribe.population * config.EXPEL_RAIDERS_LOOT_PER_POPULATION * reward_multiplier)),
            )
            gained_population = max(1, round(tribe.population * config.EXPEL_RAIDERS_POPULATION_GAIN_FRACTION * reward_multiplier))
            tribe.population += gained_population
            tribe.max_population = max(tribe.max_population, tribe.population)
            sim._relocate_raider_sighting_after_ambush(tribe, ax, ay)
            sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
            sim.recent_encounters.append({
                "x": ax, "y": ay, "kind": "raider_attack",
                "label": "Raiders expelled", "outcome": "repelled",
            })
            _record_combat(tribe, "Expel Raiders", "won")
            wave_note = "" if wave == 1 else f" after {wave} furious waves of resistance"
            return (
                f"the tribe rises as one and drives the raiders from the territory boundary{wave_note} -- "
                f"{gained_food} food seized and {gained_population} routed raiders join the tribe"
            )

        sim._lose_population(tribe, config.EXPEL_RAIDERS_POPULATION_LOSS_PER_FAILED_WAVE, cause="expel_raiders_failed")
        _record_combat(tribe, "Expel Raiders", "lost")
        if tribe.extinct:
            return "the frenzied defense collapses entirely -- nothing left to expel with"
        for resource in ("wood", "stone", "food"):
            stolen = round(getattr(tribe, resource) * config.RAID_STEAL_FRACTION)
            setattr(tribe, resource, getattr(tribe, resource) - stolen)
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_TRAUMA_MAGNITUDE, config.RAID_TRAUMA_RADIUS)
        reward_multiplier = max(
            config.EXPEL_RAIDERS_MIN_REWARD_MULTIPLIER, reward_multiplier - config.EXPEL_RAIDERS_REWARD_REDUCTION_PER_WAVE
        )

    return (
        f"wave after wave, the frenzy can't break them after {config.EXPEL_RAIDERS_MAX_WAVES} attempts -- "
        "battered, the tribe falls back to defend the camp instead"
    )


def _territory_threat_radius(tribe) -> float:
    """See config.TERRITORY_CLEARING_RADIUS_MARGIN's own comment -- CLEAR_TERRITORY
    (and the real-construction menu-lock in Simulation._prepare_turn) both reach
    slightly past the territory boundary itself, not just strictly inside it."""
    return tribe.territory_radius + config.TERRITORY_CLEARING_RADIUS_MARGIN


def _territory_has_nearby_threats(tribe) -> bool:
    """Shared by AFFORDABILITY_CHECKS["CLEAR_TERRITORY"] and _prepare_turn's
    real-construction menu-lock -- one real number, not two independently
    drifting distance checks for the same question."""
    if tribe.territory_center is None:
        return False
    cx, cy = tribe.territory_center
    radius = _territory_threat_radius(tribe)
    return any(math.hypot(x - cx, y - cy) <= radius for x, y in tribe.raider_sightings)


def _clear_territory(sim, tribe, biome, target):
    """Explicit request, 2026-09-09: "Territory boundaries must be cleared of
    threats before they can really start building anything really. This is
    not a passive action. The Chief must clear the area." Reverses an
    earlier same-session version (Simulation._advance_territory_clearing,
    a fully autonomous per-cycle sweep with no chief action at all) -- the
    Chief now has to actually choose this, the same "not passive" shape
    _expel_raiders_from_territory above already established for whole-
    boundary defense, rather than the tribe defending itself for free every
    cycle.

    Sweeps every raider camp within _territory_threat_radius, not just
    strictly inside the boundary -- explicit follow-up: "make the Clearing
    radius a little larger than the Boundary area so they clear any Raider
    just on the line or outside it." Reuses ACTION_REGISTRY["STRIKE_RAIDER_
    CAMP"] directly per camp, the same established reuse pattern
    Simulation._advance_battalion_patrol's own docstring explains (real
    win-chance/loot logic, not duplicated here) -- bypassing
    STRIKE_RAIDER_CAMP's own era gate is fine since this is an internal
    call, not exposing that action itself; this action has its own,
    earlier era unlock (see eras.py, primitive_dawn) precisely so it's
    available before real construction ever is.

    Iterates a snapshot, not the live list -- resolving one call can mutate
    tribe.raider_sightings (a win removes the camp; see
    _strike_raider_camp), and a still-recovering tribe hitting a wave of
    camps more than once in the same sweep isn't the intent, just clearing
    whatever's near the boundary right now."""
    if tribe.territory_center is None:
        return "there is no territory yet to clear"
    cx, cy = tribe.territory_center
    radius = _territory_threat_radius(tribe)
    targets = [camp for camp in tribe.raider_sightings if math.hypot(camp[0] - cx, camp[1] - cy) <= radius]
    if not targets:
        return "no raiders are camped near the territory boundary to clear"

    cleared = 0
    for camp in targets:
        if camp not in tribe.raider_sightings:
            continue  # already resolved earlier in this same sweep
        result = ACTION_REGISTRY["STRIKE_RAIDER_CAMP"](sim, tribe, biome, camp)
        if result and "destroyed" in result:
            cleared += 1
    if cleared == 0:
        return "the tribe pushed toward the raiders camped near the boundary but couldn't clear any of them this time"
    return f"the tribe clears {cleared} raider camp{'s' if cleared != 1 else ''} from around the territory boundary"


def _record_trade(tribe, resource: str, given: int = 0, received: int = 0) -> None:
    """Shared by every trade path (_execute_trade, _trade_with_minor_settlement)
    -- explicit request: sidebar boxes for "an elastic and running total of
    things traded away/received." A keyed dict per resource rather than a
    fixed field per resource, so a newly-tradeable resource (Fur, a Mine's own
    named ore, a crafted item's type) doesn't need a hardcoded slot. Zero
    amounts are skipped entirely rather than padding the dict with a resource
    that never actually moved this trade."""
    if given:
        tribe.trade_given[resource] = tribe.trade_given.get(resource, 0) + given
    if received:
        tribe.trade_received[resource] = tribe.trade_received.get(resource, 0) + received


def _record_combat(tribe, kind: str, outcome: str) -> None:
    """Shared by every real win/lose combat outcome (RAID, raid defense, home
    NPC raid defense, STRIKE_RAIDER_CAMP, expedition raider ambush) --
    explicit request: a sidebar box for "each type of combat with W/L
    totals." `outcome` is "won" or "lost"; elastic like _record_trade above --
    a combat kind this tribe has never faced simply doesn't appear until it
    does."""
    record = tribe.combat_record.setdefault(kind, {"won": 0, "lost": 0})
    record[outcome] += 1


def _execute_trade(sim, tribe, partner) -> str:
    """The actual exchange, shared by instant TRADE and SEND_TRADE_EMISSARY once
    either has found a real partner -- both sides give up the same fraction of what
    they're currently holding and receive the same fraction back, unconditional
    once initiated (like RAID, this doesn't ask the other side's permission)."""
    # Code-quality pass: each side's received gift used to be folded straight into
    # the same setattr as its own given gift, uncapped -- same bug class as the
    # rest of this file/simulation.py. Give (a plain subtraction, no cap concern)
    # and receive (routed through sim._capped_add) are now separate steps; the
    # trade ledger below records what actually landed, not the nominal gift.
    for resource in ("wood", "stone", "food", "water"):
        tribe_amount = getattr(tribe, resource)
        partner_amount = getattr(partner, resource)
        tribe_gift = round(tribe_amount * config.TRADE_GIFT_FRACTION)
        partner_gift = round(partner_amount * config.TRADE_GIFT_FRACTION)
        setattr(tribe, resource, tribe_amount - tribe_gift)
        setattr(partner, resource, partner_amount - partner_gift)
        tribe_received = sim._capped_add(tribe, resource, partner_gift)
        partner_received = sim._capped_add(partner, resource, tribe_gift)
        _record_trade(tribe, resource, given=tribe_gift, received=tribe_received)
        _record_trade(partner, resource, given=partner_gift, received=partner_received)

    # Explicit request: "maybe some hunters want a Tannery and they can trade
    # furs too." A Mine/Tannery's named resource (Fur, Orosite Ore, ...) used
    # to have nowhere to go -- trade only ever swapped the same four generic
    # resources, the exact gap the original "Mine & unique resource" design
    # note called out. Same fractional-gift shape as the loop above, over
    # whichever named resources either side actually holds.
    #
    # Code-quality pass: same give/receive split and cap fix as the generic
    # resource loop above -- this one was equally uncapped.
    for resource in set(tribe.unique_resources) | set(partner.unique_resources):
        tribe_amount = tribe.unique_resources.get(resource, 0)
        partner_amount = partner.unique_resources.get(resource, 0)
        tribe_gift = round(tribe_amount * config.TRADE_GIFT_FRACTION)
        partner_gift = round(partner_amount * config.TRADE_GIFT_FRACTION)
        tribe.unique_resources[resource] = tribe_amount - tribe_gift
        partner.unique_resources[resource] = partner_amount - partner_gift
        tribe_received = sim._capped_unique_add(tribe, resource, partner_gift)
        partner_received = sim._capped_unique_add(partner, resource, tribe_gift)
        _record_trade(tribe, resource, given=tribe_gift, received=tribe_received)
        _record_trade(partner, resource, given=partner_gift, received=partner_received)

    # A forged item is a discrete, indivisible thing -- can't hand over a "fraction"
    # of one the way the fractional gifts above work, so each side that actually has
    # any items gives up its oldest one. Snapshot both gifts before appending either,
    # so a tribe that had zero items doesn't immediately hand back the very item it
    # was just given.
    tribe_item_gift = tribe.items.pop(0) if tribe.items else None
    partner_item_gift = partner.items.pop(0) if partner.items else None
    if tribe_item_gift is not None:
        partner.items.append(tribe_item_gift)
        _record_trade(tribe, tribe_item_gift["type"], given=1)
        _record_trade(partner, tribe_item_gift["type"], received=1)
    if partner_item_gift is not None:
        tribe.items.append(partner_item_gift)
        _record_trade(partner, partner_item_gift["type"], given=1)
        _record_trade(tribe, partner_item_gift["type"], received=1)

    tribe.trades_completed += 1
    partner.trades_completed += 1
    if tribe.trades_completed == 1:
        sim._award_trophy(tribe, "First Contact")
    if partner.trades_completed == 1:
        sim._award_trophy(partner, "First Contact")
    sim._check_custom_awards(tribe, "trading")
    sim._check_custom_awards(partner, "trading")
    sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.TRADE_PRIDE_MAGNITUDE, config.TRADE_PRIDE_RADIUS)
    sim.trauma.radiate_event_wave(partner.x, partner.y, config.TRADE_PRIDE_MAGNITUDE, config.TRADE_PRIDE_RADIUS)
    return f"opened trade with {partner.name} -- goods exchanged both ways"


def _find_trade_partner(sim, tribe, x, y):
    for other in sim.tribes.values():
        if other.id == tribe.id or other.extinct:
            continue
        if (other.x - x) ** 2 + (other.y - y) ** 2 <= config.TRADE_PROXIMITY_RADIUS ** 2:
            return other
    return None


def _trade_with_minor_settlement(sim, tribe, settlement):
    """The peaceful, repeatable alternative to raiding the same target -- explicit
    idea: 'we can reuse some code and make it raid or trade independently on
    occurrence.' Smaller and safer than a raid (MINOR_SETTLEMENT_TRADE_FRACTION <<
    MINOR_SETTLEMENT_RAID_STEAL_FRACTION) and doesn't touch raids_remaining -- there's
    no one on the other side to actually negotiate with or give anything back, so
    this is a one-way, guaranteed-safe take, not a real two-way exchange.

    Same storage-cap fix as _raid_minor_settlement's own -- a settlement is
    seeded from whichever tribe is currently biggest (Simulation.
    _biggest_tribe_snapshot), and since this never depletes the way raiding
    does, it's a repeatable enough channel that the receiving side's own
    storage cap must actually hold.

    Code-quality pass: routed through sim._capped_add instead of inlining the
    same cap arithmetic that helper already implements."""
    gained = {}
    for resource in ("wood", "stone", "food", "water"):
        taken = round(settlement[resource] * config.MINOR_SETTLEMENT_TRADE_FRACTION)
        settlement[resource] -= taken
        gained[resource] = sim._capped_add(tribe, resource, taken)
        _record_trade(tribe, resource, received=gained[resource])  # one-way -- nothing given up
    tribe.trades_completed += 1
    if tribe.trades_completed == 1:
        sim._award_trophy(tribe, "First Contact")
    sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.TRADE_PRIDE_MAGNITUDE, config.TRADE_PRIDE_RADIUS)
    return (
        f"traded peacefully with an outlying settlement -- {gained['wood']} wood, {gained['stone']} stone, "
        f"{gained['food']} food, and {gained['water']} water received"
    )


def _trade(sim, tribe, biome, target):
    """Attempt to open trade with a rival tribe found at target_vector -- the peaceful
    counterpart to RAID, and the mechanical outlet for a cooperative/community-minded
    chief philosophy that otherwise has nothing to act on. Instant: only works if a
    rival already happens to be within TRADE_PROXIMITY_RADIUS of target_vector right
    now -- SEND_TRADE_EMISSARY is the longer-reach alternative, for a rival already
    discovered (tribe.discovered_rivals) but too far for this tight a radius. Also
    checks for an unaffiliated minor settlement first, the same way RAID does -- a
    safer, smaller, one-sided exchange rather than a real trade."""
    tx, ty = target
    settlement = _find_minor_settlement(sim, tx, ty)
    if settlement is not None:
        return _trade_with_minor_settlement(sim, tribe, settlement)

    partner = _find_trade_partner(sim, tribe, tx, ty)
    if partner is None:
        return "found no rival encampment there to trade with"
    return _execute_trade(sim, tribe, partner)


def _nearest_rival(sim, tribe, x, y):
    """Explicit correction: "they can't make an ALLIANCE if they have not made
    contact with another Tribe or Settlement." Used to have no radius cutoff at
    all -- a geopolitical stance toward a rival nobody had ever actually gotten
    close to. Originally enforced with a live config.DIPLOMACY_CONTACT_RADIUS
    distance check between the two tribes' own positions -- but once both are
    settled (home position fixed) and spawned far apart by design, that live
    check could never pass again after founding, even after real contact had
    already happened. Gated on tribe.discovered_rivals instead (see
    Simulation._note_rival_discovery): once a rival's camp has genuinely been
    found -- by home proximity or by a scout physically closing the distance --
    "we've made contact" is a lasting fact, not a snapshot that a fixed home
    position could stop satisfying the moment it's checked again. target_vector
    still only disambiguates *which* discovered rival is meant when more than
    one exists."""
    best, best_dist = None, None
    for other in sim.tribes.values():
        if other.id == tribe.id or other.extinct or other.id not in tribe.discovered_rivals:
            continue
        dist = (other.x - x) ** 2 + (other.y - y) ** 2
        if best is None or dist < best_dist:
            best, best_dist = other, dist
    return best


def _declare_alliance(sim, tribe, biome, target):
    """Explicit follow-up from the Agentic Evolution spec reconciliation (Age 4's
    Declare_Geopolitical_Posture): a persistent, per-rival relationship a tribe can
    actually declare, unlike instant RAID/TRADE which resolve once and leave no
    lasting record of how two tribes feel about each other. Symmetric -- a real
    declaration both sides now live under, not a private opinion only one side
    holds, since only one side ever gets to "choose" this in a given cycle. Also
    doubles as suing for peace out of a declared war (the same action either way,
    simpler than a separate CEASEFIRE verb for what's mechanically the same state
    change). State-only: this doesn't itself change RAID/TRADE odds, just gives the
    tribe (and its rival) a real, persistent fact to reason from.

    Explicit request, 2026-09-09: "let's make sure they can't take any waring
    or alliance type actions until they build a Barracks." Real guard here,
    not just the menu-filtering AFFORDABILITY_CHECKS entry -- same
    belt-and-suspenders shape _build_kitchen's own cooking_learned/
    long_houses_built check already uses. No soft-lock risk for suing for
    peace: a tribe can only ever have reached WAR in the first place via
    this same barracks_built gate."""
    if tribe.barracks_built <= 0:
        return "a barracks must be built before any formal stance toward a rival tribe is worth declaring"
    tx, ty = target
    rival = _nearest_rival(sim, tribe, tx, ty)
    if rival is None:
        return "no rival tribe has been encountered nearby yet to declare a stance toward"
    was_war = tribe.stance_toward.get(rival.id) == "WAR"
    already_allied = tribe.stance_toward.get(rival.id) == "ALLIED"
    tribe.stance_toward[rival.id] = "ALLIED"
    rival.stance_toward[tribe.id] = "ALLIED"
    sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.NEGOTIATE_PRIDE_MAGNITUDE, config.NEGOTIATE_PRIDE_RADIUS)
    sim.trauma.radiate_event_wave(rival.x, rival.y, config.NEGOTIATE_PRIDE_MAGNITUDE, config.NEGOTIATE_PRIDE_RADIUS)
    if not already_allied:
        # See Simulation._resolve_cultural_crossover -- only on genuinely becoming
        # allies, not every redundant re-declaration while already allied.
        tribe.pending_cultural_crossover = rival.id
    if was_war:
        return f"{tribe.name} sues for peace with {rival.name} -- the war ends, both now allied"
    return f"{tribe.name} declares an alliance with {rival.name}"


def _mutual_ally_at_top_era(sim, tribe):
    """The rival this tribe could build a Joint Castle with -- both at the
    one era DECLARE_CONQUEST/BUILD_JOINT_CASTLE exist in, and genuinely,
    mutually allied (not just this tribe's own one-sided declaration).
    Returns None otherwise. See Simulation._has_active_alliance_at_top_era
    for the mirrored check step() uses to hold off era_ceiling while this is
    still in progress."""
    from .eras import next_era
    if next_era(tribe.era) is not None:
        return None
    for other in sim.tribes.values():
        if other.id == tribe.id or other.extinct or next_era(other.era) is not None:
            continue
        if tribe.stance_toward.get(other.id) == "ALLIED" and other.stance_toward.get(tribe.id) == "ALLIED":
            return other
    return None


def _build_joint_castle(sim, tribe, biome, target):
    """Explicit request, 2026-09-10: "If they form an alliance, should we
    revise the menu to allow full builds all the way until both reach
    Castle-state... I love building the Castle together. 1 big piece in the
    middle of the Tribes." A genuinely shared structure, tracked on
    sim.joint_castle (not either Tribe -- it belongs to neither one alone),
    positioned at the midpoint between both territories the moment
    construction actually starts. Staged like CONSTRUCT_WALL/TRAIN_
    BATTALION -- built up over several calls from EITHER chief, not a
    one-shot flip, and each call's contribution scales with that tribe's own
    population the same way actions._labor_multiplier already does
    everywhere else staged progress exists.

    A deliberately different path to "Castle-state" than the ordinary
    Fortress+long-house ladder (_build_castle) -- this is the two tribes'
    shared monument to their alliance, not a bigger version of either one's
    own defensive works, so it doesn't inherit that ladder's prerequisites.
    Completion sets castle_built on BOTH tribes (see Simulation.step's own
    golden_age check, which confirms completion against sim.joint_castle's
    own progress specifically, not just "both tribes have castle_built" --
    either could in principle also reach that flag via the ordinary solo
    ladder, unrelated to any alliance)."""
    rival = _mutual_ally_at_top_era(sim, tribe)
    if rival is None:
        return None
    if tribe.castle_built and rival.castle_built:
        return None
    joint = sim.joint_castle
    if joint is None or set(joint["tribe_ids"]) != {tribe.id, rival.id}:
        if tribe.territory_center is None or rival.territory_center is None:
            return None
        mx = (tribe.territory_center[0] + rival.territory_center[0]) // 2
        my = (tribe.territory_center[1] + rival.territory_center[1]) // 2
        joint = {"tribe_ids": [tribe.id, rival.id], "x": mx, "y": my, "wood": 0, "stone": 0}
        sim.joint_castle = joint

    contribution = round(config.JOINT_CASTLE_CONTRIBUTION_PER_ACTION_BASE * _labor_multiplier(tribe.population))
    added_wood = max(0, min(tribe.wood, config.JOINT_CASTLE_WOOD_COST - joint["wood"], contribution))
    added_stone = max(0, min(tribe.stone, config.JOINT_CASTLE_STONE_COST - joint["stone"], contribution))
    if added_wood <= 0 and added_stone <= 0:
        return None
    tribe.wood -= added_wood
    tribe.stone -= added_stone
    joint["wood"] += added_wood
    joint["stone"] += added_stone

    if joint["wood"] >= config.JOINT_CASTLE_WOOD_COST and joint["stone"] >= config.JOINT_CASTLE_STONE_COST:
        tribe.castle_built = True
        rival.castle_built = True
        sim._award_trophy(tribe, "Architects of Peace")
        sim._award_trophy(rival, "Architects of Peace")
        sim.trauma.radiate_event_wave(joint["x"], joint["y"], config.ERA_ADVANCE_PRIDE_MAGNITUDE, config.ERA_ADVANCE_PRIDE_RADIUS)
        message = (
            f"the Joint Castle at ({joint['x']},{joint['y']}) is complete -- {tribe.name} and {rival.name} "
            "raise it together, a lasting monument to their alliance"
        )
        tribe.history.append(message)
        rival.history.append(message)
        return message
    return (
        f"{tribe.name} contributes to the Joint Castle at ({joint['x']},{joint['y']}) with {rival.name} -- "
        f"{joint['wood']}/{config.JOINT_CASTLE_WOOD_COST} wood, {joint['stone']}/{config.JOINT_CASTLE_STONE_COST} stone"
    )


def _declare_war(sim, tribe, biome, target):
    """The hostile counterpart to _declare_alliance -- same symmetric, state-only
    shape, including the same explicit barracks_built guard (see _declare_
    alliance's own docstring)."""
    if tribe.barracks_built <= 0:
        return "a barracks must be built before any formal stance toward a rival tribe is worth declaring"
    tx, ty = target
    rival = _nearest_rival(sim, tribe, tx, ty)
    if rival is None:
        return "no rival tribe has been encountered nearby yet to declare a stance toward"
    if tribe.stance_toward.get(rival.id) == "WAR":
        return f"{tribe.name} is already at war with {rival.name}"
    tribe.stance_toward[rival.id] = "WAR"
    rival.stance_toward[tribe.id] = "WAR"
    sim.trauma.radiate_event_wave(rival.x, rival.y, config.RAID_TRAUMA_MAGNITUDE, config.RAID_TRAUMA_RADIUS)
    return f"{tribe.name} declares war on {rival.name}"


def _send_trade_emissary(sim, tribe, biome, target):
    """Explicit correction, after a live 986-cycle run completed zero trades:
    "Looks like Trade needs the Alliance treatment... I meant for the Trade
    treatment to be like the 'ALLIANCE/DECLARE_WAR' not Scout." This used to
    dispatch a real multi-day expedition toward the model's own guessed
    target_vector, checking every day of travel whether a rival happened to be
    within instant TRADE's own tight proximity of wherever the emissary
    currently stood -- with no way to aim that guess at where a rival actually
    is, a live run showed it almost always wandering nowhere close and
    reporting "found no one." _declare_alliance/_declare_war solved the exact
    same problem for geopolitical stance by requiring real, already-established
    contact (_nearest_rival, tribe.discovered_rivals) instead of a guess --
    this now does the same: instant, contact-gated, no travel to simulate. This
    remains the deliberate, longer-reach option TRADE isn't (discovery can
    happen from as far as RIVAL_PRECISE_AWARENESS_RADIUS, wider than instant
    TRADE's own tight TRADE_PROXIMITY_RADIUS), just resolved immediately rather
    than over several days once contact already exists.

    Explicit request: "it's unwise to Trade before we have a full Wall" --
    unlike instant TRADE (a chance encounter, not a deliberate choice to
    expose the tribe), reaching out to a known rival is a real, deliberate
    decision a tribe shouldn't make before it can defend what it has at
    home.

    Explicit follow-up: "they can also seek the Raidable Settlements too...
    Trade is good to offer" -- checks for an unaffiliated minor settlement
    first, the same way instant TRADE/RAID already do (_find_minor_settlement,
    a fixed map feature found by target_vector, not a contact requirement),
    before falling back to the contact-gated rival check above."""
    if not tribe.wall_rings or not city_layout.ring_fully_built(tribe.wall_rings[0]):
        return "the first wall ring must be finished before it's wise to go looking for strangers to trade with"
    tx, ty = target
    settlement = _find_minor_settlement(sim, tx, ty)
    if settlement is not None:
        return _trade_with_minor_settlement(sim, tribe, settlement)
    rival = _nearest_rival(sim, tribe, tx, ty)
    if rival is None:
        return "no rival tribe has been encountered nearby yet to send an emissary to"
    return _execute_trade(sim, tribe, rival)


ACTION_REGISTRY = {
    "GATHER_WOOD": _gather_wood,
    "GATHER_STONE": _gather_stone,
    "GATHER_WATER": _gather_water,
    "GATHER_FOOD": _forage,
    "HUNT_DEER": _hunt_deer,
    "BUILD_FIRE": _build_fire,
    "COOK_FOOD": _cook_food,
    "CONSTRUCT_WALL": _construct_wall,
    "BUILD_LONG_HOUSE": _build_long_house,
    "UPGRADE_LONG_HOUSE": _upgrade_long_house,
    "BUILD_CASTLE": _build_castle,
    "BUILD_ROAD": _build_road,
    "BUILD_DOCK": _build_dock,
    "BUILD_FISHERY": _build_fishery,
    "BUILD_SAWMILL": _build_sawmill,
    "BUILD_QUARRY": _build_quarry,
    "BUILD_MINE": _build_mine,
    "GATHER_ORE": _gather_ore,
    "BUILD_TANNERY": _build_tannery,
    "BUILD_DEER_PEN": _build_deer_pen,
    "BUILD_HATCHERY": _build_hatchery,
    "BUILD_COOP": _build_coop,
    "BUILD_BATH_HOUSE": _build_bath_house,
    "BUILD_LIBRARY": _build_library,
    "RESEARCH": _research,
    "BUILD_WELL": _build_well,
    "BUILD_WAREHOUSE": _build_warehouse,
    "UPGRADE_WAREHOUSE": _upgrade_warehouse,
    "BUILD_BARRACKS": _build_barracks,
    "UPGRADE_BARRACKS": _upgrade_barracks,
    "TRAIN_BATTALION": _train_battalion,
    "BUILD_FORGE": _build_forge,
    "FORGE_ITEM": _forge_item,
    "USE_ITEM": _use_item,
    "BUILD_OBJECT_CREATOR": _build_object_creator,
    "CREATE_ITEM": _create_item,
    "CREATE_USEFUL_STRUCTURE": _create_useful_structure,
    "DECLARE_CONQUEST": _declare_conquest,
    "BUILD_JOINT_CASTLE": _build_joint_castle,
    "BUILD_KITCHEN": _build_kitchen,
    "BUILD_MOAT": _build_moat,
    "BUILD_KEEP": _build_keep,
    "BUILD_FORTRESS": _build_fortress,
    "PLANT_CROP": _plant_crop,
    "GATHER_EGGS": _gather_eggs,
    "CATCH_FISH": _catch_fish,
    "SCOUT": _scout,
    "EXPLORATION_PARTY": _exploration_party,
    "HUNTING_PARTY": _hunting_party,
    "RELOCATE": _relocate,
    "BREED": _breed,
    "RAID": _raid,
    "STRIKE_RAIDER_CAMP": _strike_raider_camp,
    "EXPEL_RAIDERS_FROM_TERRITORY": _expel_raiders_from_territory,
    "CLEAR_TERRITORY": _clear_territory,
    "TRADE": _trade,
    "DECLARE_ALLIANCE": _declare_alliance,
    "DECLARE_WAR": _declare_war,
    "SEND_TRADE_EMISSARY": _send_trade_emissary,
}

# Plain mechanical facts about what each verb does, handed to the model in the prompt
# (see prompts.py) so it can reason about tradeoffs instead of guessing from a bare
# action name -- live testing showed tribes repeatedly deciding they "must relocate to
# find water" while starving, apparently never realizing GATHER_WATER already works
# wherever they stand (just at a lower yield than a river tile gets). This is the same
# category as the nearest_water fact already given to a founding chief: information the
# simulation legitimately has, not an instruction about what to pick.
ACTION_DESCRIPTIONS = {
    "GATHER_WOOD": "Harvest wood at your current tile -- forest yields the most, plains and river tiles some, mountains and ocean almost none. Yield also drops the more this exact spot has been harvested recently.",
    "GATHER_STONE": "Harvest stone at your current tile -- mountains yield the most by far, every other biome almost none. Yield also drops the more this exact spot has been harvested recently.",
    "GATHER_WATER": "Harvest water at your current tile -- works in any biome, though a river tile yields more than elsewhere. Small drowning risk if you're on a river.",
    "GATHER_FOOD": "Forage for berries, fruit, and wild plants at your current tile -- plains yields the most, forest some, mountains and ocean almost none. No hazard, unlike hunting, but a lower yield ceiling. Yield also drops the more this exact spot has been foraged recently.",
    "HUNT_DEER": "Attempt to harvest food at your current tile -- forest has the most game, plains and river tiles some, mountains and ocean almost none. Small risk of losing a hunter to a wolf pack, most likely in forest.",
    "BUILD_FIRE": "Build a fire at your current tile using stored wood. Does nothing if one is already built here.",
    "COOK_FOOD": "Learn to cook -- only possible once you've successfully hunted or foraged, and successfully built a fire, at some point. A one-time skill, usable anywhere from then on: every future forage, hunt, or catch brings home three times as much food, and every future celebration feast costs less.",
    "CONSTRUCT_WALL": "Work on your wall using stored wood and stone -- a real defensive structure built up over several turns, not finished in one. Automatically does whatever the wall needs next: unlocks a new section if none is currently open, continues an unlocked section's progress (more per turn with more people to put to the work), reinforces a completed section with another tier, or -- once a whole ring is fully built and reinforced -- opens a brand new ring further out. A more complete wall meaningfully improves your odds of defending against a raider attack. Repeatable; does nothing further once maxed out.",
    "BUILD_LONG_HOUSE": "Build a long house at your current tile using stored wood and stone -- real, lasting shelter for the tribe, one house at a time. Repeatable as population grows, up to 5; UPGRADE_LONG_HOUSE takes over from there.",
    "UPGRADE_LONG_HOUSE": "Expand the long houses already standing to support more households -- only worth considering once 5 long houses already stand. No new structure, no placement needed. Repeatable, but each upgrade costs more than the last.",
    "BUILD_CASTLE": "Build a castle at your current tile using stored wood and stone -- only possible once a fortress stands and enough long houses have been built. A one-time, permanent structure that adds real defense on top of whatever your wall already provides.",
    "BUILD_ROAD": "Build a road at your current tile using stored wood and stone. A one-time, permanent improvement: every future scouting party, hunting party, or exploration party you send out travels faster from then on.",
    "BUILD_DOCK": "Build a dock at your current tile using stored wood -- only possible once the tribe has settled here and has already learned to fish (a real successful catch). A one-time, permanent structure: every future fish caught here pays out more from then on.",
    "BUILD_FISHERY": "Build a fishery using stored wood and stone -- only possible once a dock already stands. A one-time, permanent structure: the settlement's passive daily fish supply flows in even more steadily from then on.",
    "BUILD_SAWMILL": "Build a sawmill using stored wood and stone -- only possible once wood has actually been gathered here at least once. A one-time, permanent structure at your settlement: every future load of gathered wood is worth six times as much from then on.",
    "BUILD_QUARRY": "Build a quarry using stored wood and stone -- only possible once stone has actually been gathered here at least once. A one-time, permanent structure at your settlement: every future load of harvested stone is worth three times as much from then on.",
    "BUILD_MINE": "Excavate a mine at a vein your scouts have already found, using stored wood and stone -- only possible once a quarry stands and at least one vein is known. A one-time, permanent structure, but its unique resource has to actually be fetched (GATHER_ORE) before it starts flowing in steadily.",
    "GATHER_ORE": "Fetch the Mine's unique resource -- only possible once a mine has been excavated. The first successful fetch also starts a small, permanent daily supply from then on, the same way fishing works once learned.",
    "BUILD_TANNERY": "Build a tannery using stored wood and stone -- only possible once a hunt has actually succeeded. A one-time, permanent structure at your settlement: Fur flows in steadily from then on, and every successful hunt yields extra meat from then on.",
    "BUILD_DEER_PEN": "Build a deer pen using stored wood and stone -- only possible once a tannery already stands and several hunts have actually succeeded. A one-time, permanent structure: a small captive herd starts immediately, breeds on its own if fed, and feeds the tannery extra Fur every cycle on top of what it already produces.",
    "BUILD_HATCHERY": "Build a hatchery using stored wood and stone -- only possible once a wild egg has actually been found and hatched. A one-time, permanent structure at your settlement: the flock grows on its own much more reliably from then on.",
    "BUILD_COOP": "Build a coop using stored wood and stone -- only possible once the flock has at least one member. A one-time, permanent structure: paired with a Hatchery, gathered and laid eggs are actually incubated into new flock automatically from then on, instead of the flock only growing by chance.",
    "BUILD_BATH_HOUSE": "Build a bath house using stored wood and stone -- no prerequisite beyond being settled. A one-time, permanent structure at your settlement: the tribe's daily food and water consumption drops from then on.",
    "BUILD_LIBRARY": "Build a library using stored wood and stone -- only possible once at least one long house stands. A one-time, permanent structure: unlocks RESEARCH, a real way to reach the next era sooner.",
    "RESEARCH": "Study the tribe's own remembered history at the library, using a little stored wood -- only possible once a library stands. Distills what's been lived through into a permanent Library entry, and permanently shortens the path to the next era a little further. Repeatable.",
    "BUILD_WELL": "Build a well using stored wood and stone -- no prerequisite beyond being settled. A one-time, permanent structure at your settlement: the tribe's daily passive water supply flows in faster from then on.",
    "BUILD_WAREHOUSE": "Build a warehouse using stored wood and stone. Raises how much of every resource can be stored at once -- gathering more than storage allows is wasted. Repeatable up to 5 warehouses; UPGRADE_WAREHOUSE takes over from there.",
    "UPGRADE_WAREHOUSE": "Reinforce the warehouses already standing to raise storage capacity further -- only worth considering once 5 warehouses already stand. No new structure, no placement needed. Repeatable, but each upgrade costs more than the last.",
    "BUILD_BARRACKS": "Build a barracks using stored wood and stone -- only possible once a Keep stands. Repeatable up to 5; each one raises how large a Battalion can ever be trained. Real housing for a standing military, the first building of the Military branch.",
    "UPGRADE_BARRACKS": "Reinforce the barracks already standing to raise Battalion capacity further -- only worth considering once 5 barracks already stand. No new structure, no placement needed. Repeatable, but each upgrade costs more than the last.",
    "TRAIN_BATTALION": "Train soldiers for your Battalion, led by your Warrior -- only possible once a Warrior is named and a Barracks stands. Costs food, not wood/stone. Built up over several turns like a wall section, not finished in one -- more people trains faster. Repeatable up to your Barracks' own capacity.",
    "BUILD_FORGE": "Build a forge using stored wood and stone -- only possible once a mine stands and at least one unit of its ore is already in stock. A one-time, permanent structure: from then on, ore can be worked into real tools, weapons, and inventions.",
    "FORGE_ITEM": "Work stored ore and wood into a real item at your forge -- a tool, a weapon, or a small invention, picked at random. No durability to track: each item just carries a flat value, usable later or given away in a trade.",
    "USE_ITEM": "Redeem your oldest crafted item for its stored value, converted into wood and stone. Does nothing if you have no items.",
    "BUILD_OBJECT_CREATOR": "Build the Object Creator using stored wood and stone -- a one-time, permanent factory that lets the tribe start inventing genuinely new items and structures from then on.",
    "CREATE_ITEM": "Design and craft a genuinely new item at the Object Creator -- a real, permanent effect (a bonus to gathering, combat, defense, celebrations, exploration speed, or an immediate population grant), picked for you. Only possible once the Object Creator stands.",
    "CREATE_USEFUL_STRUCTURE": "Design and build a genuinely new structure at the Object Creator -- same real, permanent effects as CREATE_ITEM, but a building instead of a portable item. Only possible once the Object Creator stands.",
    "DECLARE_CONQUEST": "An all-in campaign to fully and immediately conquer a rival tribe near target_vector, in one decisive stroke rather than several raids. A win absorbs them completely; a loss costs far more than an ordinary failed raid. Does nothing if no rival is there.",
    "BUILD_JOINT_CASTLE": "Contribute wood and stone toward a Joint Castle raised together with a genuinely, mutually allied rival tribe -- a shared monument to the alliance, built up over several turns from either side. Completing it marks both tribes as having reached Castle-state. Only possible once truly allied, not just once one side has declared it.",
    "BUILD_KITCHEN": "Build a kitchen using stored wood and stone -- only possible once cooking is known and a long house stands. A one-time, permanent structure: stacks with cooking for nine times as much food from every future forage, hunt, or catch, instead of only three.",
    "BUILD_MOAT": "Dig a moat using stored wood and stone -- only possible once the wall has been reinforced with a second layer. A one-time, permanent structure, cheaper than another wall layer: a further defense bonus.",
    "BUILD_KEEP": "Build a keep using stored wood and stone -- only possible once enough long houses stand. A one-time, permanent structure: a further defense bonus for the settlement.",
    "BUILD_FORTRESS": "Build a fortress using stored wood and stone -- only possible once a keep stands and enough long houses have been built. A one-time, permanent structure: a further defense bonus for the settlement.",
    "PLANT_CROP": "Plant a farm plot at your current tile, fenced and set with a scarecrow using stored wood -- only possible once the tribe has settled here. A planted plot grows on its own over the following cycles and yields food automatically once mature; no further action needed to harvest it. Up to a few plots can be tended at once.",
    "GATHER_EGGS": "Search for wild fowl nests near your current tile -- only possible once the tribe has settled here. A found egg is set aside and hatches on its own, growing the tribe's flock by one.",
    "CATCH_FISH": "Attempt to harvest food by fishing at your current tile -- only possible once the tribe has settled here. Pays out food immediately on a catch, and the very first successful catch also starts a small, permanent daily food supply from then on -- fishing, once learned, is never unlearned.",
    "SCOUT": "Dispatch an expedition to explore -- the direction is chosen automatically to spread coverage out over time, not from target_vector. They travel and camp on their own supply, searching up to a few days before turning back if they find nothing. What they find only becomes known once they've walked all the way home. Your tribe can have several parties out at once (scouting or hunting, any mix -- more as your population grows) -- choosing SCOUT again sends another one if there's room, or just reports on whoever's already out once you're at capacity.",
    "EXPLORATION_PARTY": "Dispatch a deeper, longer-ranging expedition than SCOUT -- direction chosen automatically, its own sweep separate from SCOUT's. Gathers real wood and stone along the way on top of the food and water any expedition forages, until they're carrying as much as they can manage, then heads home. Can discover anything SCOUT can (water, resource sites, raider camps) plus rival settlements and Landmarks -- rare points of interest that yield a real, unique treasure the moment they're found. Shares the same expedition capacity as SCOUT/HUNTING_PARTY.",
    "HUNTING_PARTY": "Send a hunting party toward target_vector -- shares the same expedition capacity as SCOUT (several parties, scouting or hunting in any mix, can be out at once -- more as your population grows). They travel and hunt on their own supply for up to several days, facing the same wolf-pack risk as an instant hunt on every day out, until they catch something or give up. Any food caught only becomes real, usable food once they've walked all the way home -- a hunt still in the field does nothing for hunger right now, no matter how promising.",
    "RELOCATE": "Move your whole tribe several tiles toward target_vector this cycle, possibly over several cycles for a far destination. Produces no resources while traveling and costs extra food and water for the effort.",
    "BREED": "Your chief and whoever currently holds a trophy start a family together, costing food and water and growing your population by one child if it succeeds. Does nothing if fewer than two named individuals (a chief plus at least one trophy-holder) exist yet, or if food/water can't cover the cost.",
    "RAID": "Attempt to raid a rival tribe if one is near target_vector. A win steals some of their stockpile but still costs you people; a loss costs you more. An unaffiliated minor settlement near target_vector is a much safer alternative -- no people of its own, so a raid there always succeeds with no risk, though it can only be raided a few times before it's exhausted and needs time to recover. Does nothing if neither is there.",
    "STRIKE_RAIDER_CAMP": "Attack a raider camp your scouts have already found (see your raider sighting reports) -- only possible once you know where one is. Success destroys it and recovers some food; failure costs a life and leaves the camp standing.",
    "EXPEL_RAIDERS_FROM_TERRITORY": "Turn the whole population out to drive off raiders currently approaching (only possible while raiders are actually inbound). A win seizes real plunder and wins over stragglers, scaled by your own population -- and the raiders are cast off elsewhere, not gone for good. A loss costs people and supplies, but doesn't end the fight: anger fuels an immediate second and third wave in the same breath, each cheaper in reward and costlier in lives than the last.",
    "CLEAR_TERRITORY": "Sweep every raider camp near the territory boundary (a little past the boundary line itself, not just strictly inside it) -- only possible while one is actually camped there. Real construction (walls and every building but a basic fire) is blocked until this is done, so a fresh settlement isn't left building next to a standing threat.",
    "TRADE": "Attempt to open trade with a rival tribe if one is near target_vector. Both sides give up a small fraction of everything they hold and receive the same fraction back -- a mutual exchange, no risk of loss. An unaffiliated minor settlement near target_vector can also be traded with -- smaller and one-sided (nothing is given up), but it never depletes the way raiding one does. Does nothing if neither is there.",
    "DECLARE_ALLIANCE": "Declare a lasting alliance with whichever rival tribe is nearest target_vector -- a real, persistent stance both tribes will remember, not a one-time exchange. Also ends a war you'd previously declared with that same rival. Only possible once a Barracks stands. Does nothing if no rival tribe exists.",
    "DECLARE_WAR": "Declare a lasting state of war with whichever rival tribe is nearest target_vector -- a real, persistent stance both tribes will remember. Does not attack them directly (see RAID for that); this only sets how the two tribes now stand. Only possible once a Barracks stands. Does nothing if no rival tribe exists, or if already at war with them.",
    "SEND_TRADE_EMISSARY": "Reach out to open trade with a rival tribe already in contact (a much longer reach than TRADE's tight radius, but requires the rival to have actually been encountered before -- same requirement as ALLIANCE/DECLARE_WAR). An unaffiliated minor settlement near target_vector can also be traded with, the same as TRADE -- safer than RAIDing it, and it never depletes the way raiding does. Instant: goods exchange immediately if either is found.",
}
