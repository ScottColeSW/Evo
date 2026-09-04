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
    return multiplier


def _labor_multiplier(population: int) -> float:
    """More hands means more gathered per action -- upkeep (Simulation._apply_upkeep)
    already scales with population, but yield never did, so a bigger tribe was strictly
    worse off per-capita: identical output from one GATHER_WOOD regardless of whether 3
    or 30 people stood behind it, against a food/water cost that only ever grew.
    POPULATION_YIELD_BASELINE matches Tribe.__init__'s own starting population, so this
    never scales a tribe at or below starting size down -- only ever rewards growth
    past it."""
    return max(1.0, population / config.POPULATION_YIELD_BASELINE)


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
    out at once. config.MAX_CONCURRENT_EXPEDITIONS is only ever the floor now, not a
    hard ceiling -- a tribe of 8 (starting population) still gets exactly that many, but
    a larger tribe can spare more search bandwidth, the same way its upkeep cost already
    scales with population (see Simulation._apply_upkeep)."""
    return max(config.MAX_CONCURRENT_EXPEDITIONS, tribe.population // config.EXPEDITION_SLOT_POPULATION_DIVISOR)


def _storage_cap(tribe) -> int:
    """A generous ceiling, not a tight one -- STORAGE_CAP_BASE alone already clears
    every era's own resource requirement, so this is never the reason a tribe can't
    advance. It only ever catches genuinely excessive hoarding (explicit request,
    after a live run showed a tribe pile wood up to 200+ while permanently starved
    on stone). Repeatable Warehouses raise it further -- 'expansion of the tribe
    will allow that to scale storage with building needs.'"""
    return config.STORAGE_CAP_BASE + tribe.warehouses_built * config.WAREHOUSE_STORAGE_BONUS_PER_BUILDING


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
    if _already_built(sim, tribe, "fire") or tribe.wood < 10:
        return None
    tribe.wood -= 10
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
    reason about which section that is."""
    target_section = city_layout.next_wall_work_section(tribe)
    if target_section is None:
        return "no wall section is currently unlocked and incomplete -- EXPAND_TERRITORY unlocks the next one"
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
        # User's own refinement: "allow 1 Long House per section, then once it is
        # 100% Wall Ring 1 Layer, they can build the rest on demand" -- a banked
        # credit so housing doesn't stall out during a long wall push.
        tribe.wall_lock_long_house_credits += 1
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


def _build_long_house(sim, tribe, biome, target):
    """Explicit request, gated on the wall already being complete first -- defense
    before shelter. Explicit correction: "most structures they only need 1 of.
    but house builds are dependant on population needs" -- repeatable, not a
    one-time flag, gated each time on real population need
    (config.HOUSING_POPULATION_PER_LONG_HOUSE) so a tribe can't spam housing it
    doesn't need. tribe.long_houses_built is also the real proxy the Keep/
    Fortress/Castle tier reads for how established this settlement has become.

    User's own refinement on the wall-commitment lock (see Simulation._prepare_
    turn): locking out housing for an entire ring's construction would stall it
    too long, so a banked wall_lock_long_house_credits (one per section
    completed, see _construct_wall) lets exactly one Long House through early,
    before ring 0 is actually finished."""
    ring0_done = bool(tribe.wall_rings) and city_layout.ring_fully_built(tribe.wall_rings[0])
    if not ring0_done and tribe.wall_lock_long_house_credits <= 0:
        return "the first wall ring must be finished before a long house is worth building here (or bank a credit by completing another wall section)"
    houses_needed = max(1, -(-tribe.population // config.HOUSING_POPULATION_PER_LONG_HOUSE))
    if tribe.long_houses_built >= houses_needed:
        return None
    if tribe.wood < config.LONG_HOUSE_WOOD_COST or tribe.stone < config.LONG_HOUSE_STONE_COST:
        return None
    slot = architect.find_free_slot(sim.world, tribe, "long_house")
    if slot is None:
        return None
    tribe.wood -= config.LONG_HOUSE_WOOD_COST
    tribe.stone -= config.LONG_HOUSE_STONE_COST
    w, h = config.BUILDING_FOOTPRINTS["long_house"]
    architect.record_building(tribe, "long_house", slot[0], slot[1], w, h, sim.cycle)
    tribe.long_houses_built += 1
    if not ring0_done:
        tribe.wall_lock_long_house_credits -= 1
    if tribe.long_houses_built == 1:
        sim._award_trophy(tribe, "Master Builder")
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        return "a long house rises -- the tribe has real, lasting shelter for the first time"
    return f"another long house rises -- {tribe.long_houses_built} now stand"


def _build_keep(sim, tribe, biome, target):
    """Explicit request: "they can have 10 houses before they build a Keep."
    First tier of the defensive ladder after Long House -- a real additional
    defense bonus stacked on top of the wall's own (Simulation.
    _resolve_raider_attack)."""
    if tribe.keep_built:
        return None
    if tribe.long_houses_built < config.KEEP_LONG_HOUSES_REQUIRED:
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
    if tribe.long_houses_built < config.FORTRESS_LONG_HOUSES_REQUIRED:
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
    if tribe.long_houses_built < config.CASTLE_LONG_HOUSES_REQUIRED:
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


def _expand_territory(sim, tribe, biome, target):
    """2026-09-02 redesign: unlocks exactly one new wall section per call, in
    fixed compass order -- "expansion must be done for each wall section," no
    exception for ring 0. Once every section in the outermost ring is both
    unlocked and fully reinforced, the next call opens a whole new ring
    further out instead (backend/city_layout.build_ring) -- no limit on ring
    count beyond land availability.

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
        tribe.wall_rings.append(city_layout.build_ring(sim.world, tribe, len(tribe.wall_rings)))
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
    real spent investment rather than something freely re-gathered."""
    return config.ITEM_STORAGE_CAP_BASE + tribe.warehouses_built * config.ITEM_STORAGE_CAP_PER_WAREHOUSE


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
    stone_gain = round(item["value"] * config.USE_ITEM_STONE_SHARE)
    wood_gain = item["value"] - stone_gain
    tribe.wood += wood_gain
    tribe.stone += stone_gain
    return f"the {item['name']} is put to use -- {wood_gain} wood and {stone_gain} stone recovered from its worth"


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
    return f"a new plot is planted -- {tribe.farm_plots} now growing"


def _gather_eggs(sim, tribe, biome, target):
    """Wild fowl near a real water source -- gated the same as PLANT_CROP (Simulation.
    _is_settled_near_water). A find doesn't hatch here: this only sets
    tribe.pending_hatch; Simulation.step() resolves it with a real, non-scripted LLM
    call (backend/genetics.py's hatch()) the same cycle, the same pattern BREED already
    uses for pending_birth. Once the flock has at least two members, the two most
    recently hatched are what get crossed -- mirrors _eligible_breeding_pair preferring
    a fresh milestone over the whole population."""
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
    caught = round(caught * _food_multiplier(tribe))
    tribe.food += caught
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
    if len(tribe.expeditions) >= expedition_capacity(tribe):
        fields = ", ".join(
            f"{e['lead_scout']} (day {e['day']}/{e['max_days']}, {e['phase']})" for e in tribe.expeditions
        )
        return f"no one left to send -- every party is already out: {fields}"

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
    tx, ty = _push_past_visited_ground(
        tribe, tribe.x, tribe.y, angle_radians, config.SCOUT_PATROL_DISTANCE, sim.world.grid_size
    )
    lx, ly = _expedition_launch_point(tribe, angle_radians, sim.world.grid_size)
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
    if len(tribe.expeditions) >= expedition_capacity(tribe):
        fields = ", ".join(
            f"{e['lead_scout']} (day {e['day']}/{e['max_days']}, {e['phase']})" for e in tribe.expeditions
        )
        return f"no one left to send -- every party is already out: {fields}"

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
    tx, ty = _push_past_visited_ground(
        tribe, tribe.x, tribe.y, angle_radians, config.EXPLORATION_PARTY_PATROL_DISTANCE, sim.world.grid_size
    )
    lx, ly = _expedition_launch_point(tribe, angle_radians, sim.world.grid_size)
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
    if len(tribe.expeditions) >= expedition_capacity(tribe):
        fields = ", ".join(
            f"{e['lead_scout']} (day {e['day']}/{e['max_days']}, {e['phase']})" for e in tribe.expeditions
        )
        return f"no one left to send -- every party is already out: {fields}"

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
    base_speed = config.MOVEMENT_SPEED + bonus
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


def _breed(sim, tribe, biome, target):
    """Two named individuals from the tribe -- its chief and whoever holds a trophy,
    see _eligible_breeding_pair -- start a family. A solo cost paid by this one tribe
    (see config.BREED_FOOD_COST/WATER_COST), distinct from the shared/split cost a
    future tribe-to-tribe merge would use. The actual outcome (the child's name, a
    flavor note) isn't decided here -- this only sets tribe.pending_birth; Simulation.
    step() resolves it with a real, non-scripted LLM call (backend/breeding.py) the
    same cycle, the same pattern _install_chief already uses for pending_chief_context."""
    if tribe.population >= config.POPULATION_GROWTH_CAP:
        return "no room to raise a family right now -- the tribe is already at capacity"
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
    respawn (Simulation._advance_minor_settlements)."""
    looted = {}
    for resource in ("wood", "stone", "food", "water"):
        stolen = round(settlement[resource] * config.MINOR_SETTLEMENT_RAID_STEAL_FRACTION)
        settlement[resource] -= stolen
        setattr(tribe, resource, getattr(tribe, resource) + stolen)
        looted[resource] = stolen
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
    that otherwise has nothing to act on. Real risk on both sides: win chance is just
    the attacker's share of the two tribes' combined population, so a smaller raiding
    party can still lose to a larger defender, and even a winning raid costs the
    attacker people -- violence isn't a free lever here. Also checks for an
    unaffiliated minor settlement first (Simulation._spawn_minor_settlements) -- a
    much safer, weaker target than a real rival, guaranteed to succeed."""
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

    attacker_win_chance = tribe.population / max(1, tribe.population + defender.population)
    if random.random() < attacker_win_chance:
        for resource in ("wood", "stone", "food", "water"):
            stolen = round(getattr(defender, resource) * config.RAID_STEAL_FRACTION)
            setattr(defender, resource, getattr(defender, resource) - stolen)
            setattr(tribe, resource, getattr(tribe, resource) + stolen)
        tribe.raids_won += 1
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
        if defender.raids_defended == 1:
            sim._award_trophy(defender, "Raid Breaker")
        for resource in ("wood", "stone", "food", "water"):
            stolen = round(getattr(tribe, resource) * config.RAID_STEAL_FRACTION)
            setattr(tribe, resource, getattr(tribe, resource) - stolen)
            setattr(defender, resource, getattr(defender, resource) + stolen)

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
        looted = round(tribe.food * config.STRIKE_RAIDER_CAMP_LOOT_FRACTION)
        tribe.food += looted
        sim.trauma.radiate_event_wave(camp[0], camp[1], config.RAID_PRIDE_MAGNITUDE, config.RAID_PRIDE_RADIUS)
        sim.recent_encounters.append({
            "x": camp[0], "y": camp[1], "kind": "raider_camp_strike",
            "label": "Raider camp destroyed", "outcome": "won",
        })
        return f"the raider camp at {camp} is destroyed -- {looted} food recovered"

    sim._lose_population(tribe, config.STRIKE_RAIDER_CAMP_POPULATION_LOSS_ON_FAILURE, cause="failed_raider_strike")
    sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_TRAUMA_MAGNITUDE, config.RAID_TRAUMA_RADIUS)
    sim.recent_encounters.append({
        "x": camp[0], "y": camp[1], "kind": "raider_camp_strike",
        "label": "Strike failed", "outcome": "lost",
    })
    return f"the strike on the raider camp at {camp} failed -- they escaped into the wilds"


def _execute_trade(sim, tribe, partner) -> str:
    """The actual exchange, shared by instant TRADE and SEND_TRADE_EMISSARY once
    either has found a real partner -- both sides give up the same fraction of what
    they're currently holding and receive the same fraction back, unconditional
    once initiated (like RAID, this doesn't ask the other side's permission)."""
    for resource in ("wood", "stone", "food", "water"):
        tribe_amount = getattr(tribe, resource)
        partner_amount = getattr(partner, resource)
        tribe_gift = round(tribe_amount * config.TRADE_GIFT_FRACTION)
        partner_gift = round(partner_amount * config.TRADE_GIFT_FRACTION)
        setattr(tribe, resource, tribe_amount - tribe_gift + partner_gift)
        setattr(partner, resource, partner_amount - partner_gift + tribe_gift)

    # Explicit request: "maybe some hunters want a Tannery and they can trade
    # furs too." A Mine/Tannery's named resource (Fur, Orosite Ore, ...) used
    # to have nowhere to go -- trade only ever swapped the same four generic
    # resources, the exact gap the original "Mine & unique resource" design
    # note called out. Same fractional-gift shape as the loop above, over
    # whichever named resources either side actually holds.
    for resource in set(tribe.unique_resources) | set(partner.unique_resources):
        tribe_amount = tribe.unique_resources.get(resource, 0)
        partner_amount = partner.unique_resources.get(resource, 0)
        tribe_gift = round(tribe_amount * config.TRADE_GIFT_FRACTION)
        partner_gift = round(partner_amount * config.TRADE_GIFT_FRACTION)
        tribe.unique_resources[resource] = tribe_amount - tribe_gift + partner_gift
        partner.unique_resources[resource] = partner_amount - partner_gift + tribe_gift

    # A forged item is a discrete, indivisible thing -- can't hand over a "fraction"
    # of one the way the fractional gifts above work, so each side that actually has
    # any items gives up its oldest one. Snapshot both gifts before appending either,
    # so a tribe that had zero items doesn't immediately hand back the very item it
    # was just given.
    tribe_item_gift = tribe.items.pop(0) if tribe.items else None
    partner_item_gift = partner.items.pop(0) if partner.items else None
    if tribe_item_gift is not None:
        partner.items.append(tribe_item_gift)
    if partner_item_gift is not None:
        tribe.items.append(partner_item_gift)

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
    this is a one-way, guaranteed-safe take, not a real two-way exchange."""
    gained = {}
    for resource in ("wood", "stone", "food", "water"):
        taken = round(settlement[resource] * config.MINOR_SETTLEMENT_TRADE_FRACTION)
        settlement[resource] -= taken
        setattr(tribe, resource, getattr(tribe, resource) + taken)
        gained[resource] = taken
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
    now -- SEND_TRADE_EMISSARY is the deliberate, patient alternative that actually
    goes looking. Also checks for an unaffiliated minor settlement first, the same
    way RAID does -- a safer, smaller, one-sided exchange rather than a real trade."""
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
    close to. config.DIPLOMACY_CONTACT_RADIUS reuses the same "close enough to
    exchange real information" distance BROADCAST_HEARING_RADIUS already
    represents for linguistic convergence -- looser than RAID/TRADE's tight
    physical-encounter radius (an envoy covering that ground is plausible;
    outright combat/trade goods changing hands isn't at the same range), but
    still a real contact requirement, not target_vector alone."""
    best, best_dist = None, None
    for other in sim.tribes.values():
        if other.id == tribe.id or other.extinct:
            continue
        dist = (other.x - x) ** 2 + (other.y - y) ** 2
        if best is None or dist < best_dist:
            best, best_dist = other, dist
    if best is None:
        return None
    # The contact check is real distance between the two tribes themselves, not
    # between target_vector and the candidate -- target_vector only disambiguates
    # *which* rival is meant when more than one exists, same as before.
    contact_dist = (best.x - tribe.x) ** 2 + (best.y - tribe.y) ** 2
    if contact_dist > config.DIPLOMACY_CONTACT_RADIUS ** 2:
        return None
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
    tribe (and its rival) a real, persistent fact to reason from."""
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


def _declare_war(sim, tribe, biome, target):
    """The hostile counterpart to _declare_alliance -- same symmetric, state-only
    shape."""
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
    """A deliberate, patient search for a rival tribe to trade with -- unlike TRADE
    (instant, only works if a rival already happens to be within
    TRADE_PROXIMITY_RADIUS right now), this dispatches a real, multi-day expedition
    that actively looks, sharing the exact day-by-day travel/give-up machinery
    HUNTING_PARTY already uses (see Simulation._advance_trade_emissary_outbound) --
    nearly the same mechanic, per explicit confirmation. Finding a rival executes
    the exchange immediately, at the point of contact -- the emissary still has to
    walk home to report it, the same "not real until you're home" rule as a hunting
    party's catch, but only for this tribe's own knowledge of what happened; the
    exchange itself already moved both sides' goods the moment contact was made.

    Explicit request: "it's unwise to Trade before we have a full Wall" --
    unlike instant TRADE (a chance encounter, not a deliberate choice to expose
    the tribe), sending an emissary out looking for strangers is a real,
    deliberate decision a tribe shouldn't make before it can defend what it
    has at home."""
    if not tribe.wall_rings or not city_layout.ring_fully_built(tribe.wall_rings[0]):
        return "the first wall ring must be finished before it's wise to go looking for strangers to trade with"
    if len(tribe.expeditions) >= expedition_capacity(tribe):
        fields = ", ".join(
            f"{e['lead_scout']} (day {e['day']}/{e['max_days']}, {e['phase']})" for e in tribe.expeditions
        )
        return f"no one left to send -- every party is already out: {fields}"

    tx, ty = target
    tx = max(0, min(sim.world.grid_size - 1, tx))
    ty = max(0, min(sim.world.grid_size - 1, ty))
    scout = _generate_scout(tribe, sim.cycle, base_days=config.TRADE_EMISSARY_MAX_DAYS)
    tribe.expeditions.append({
        "kind": "trade",
        "pos": [tribe.x, tribe.y],
        "origin": [tribe.x, tribe.y],
        "target": [tx, ty],
        "day": 0,
        "phase": "outbound",
        "food_gathered": 0,
        "water_gathered": 0,
        "lead_scout": scout["name"],
        "determination": scout["determination"],
        "max_days": scout["max_days"],
        "path": [[tribe.x, tribe.y]],
    })
    tribe.expeditions_launched += 1
    return f"an emissary led by {scout['name']} departs camp toward ({tx},{ty}), seeking a tribe to trade with"


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
    "BUILD_CASTLE": _build_castle,
    "BUILD_ROAD": _build_road,
    "EXPAND_TERRITORY": _expand_territory,
    "BUILD_DOCK": _build_dock,
    "BUILD_FISHERY": _build_fishery,
    "BUILD_SAWMILL": _build_sawmill,
    "BUILD_QUARRY": _build_quarry,
    "BUILD_MINE": _build_mine,
    "GATHER_ORE": _gather_ore,
    "BUILD_TANNERY": _build_tannery,
    "BUILD_HATCHERY": _build_hatchery,
    "BUILD_BATH_HOUSE": _build_bath_house,
    "BUILD_LIBRARY": _build_library,
    "RESEARCH": _research,
    "BUILD_WELL": _build_well,
    "BUILD_WAREHOUSE": _build_warehouse,
    "BUILD_FORGE": _build_forge,
    "FORGE_ITEM": _forge_item,
    "USE_ITEM": _use_item,
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
    "COOK_FOOD": "Learn to cook -- only possible once you've successfully hunted and successfully built a fire at some point. A one-time skill, usable anywhere from then on: once learned, stored food goes much further and every future celebration feast costs less.",
    "CONSTRUCT_WALL": "Work on a wall at your current tile using stored wood and stone -- a real defensive structure built up over several turns, not finished in one. Each turn spent on it adds real progress (more so with more people to put to the work), and a more complete wall meaningfully improves your odds of defending against a raider attack. Does nothing further once complete.",
    "BUILD_LONG_HOUSE": "Build a long house at your current tile using stored wood and stone -- only possible once your wall is fully complete. Repeatable as population grows: real, lasting shelter for the tribe, one house at a time.",
    "BUILD_CASTLE": "Build a castle at your current tile using stored wood and stone -- only possible once a fortress stands and enough long houses have been built. A one-time, permanent structure that adds real defense on top of whatever your wall already provides.",
    "BUILD_ROAD": "Build a road at your current tile using stored wood and stone. A one-time, permanent improvement: every future scouting party, hunting party, or trade emissary you send out travels faster from then on.",
    "EXPAND_TERRITORY": "Grow your settlement's real owned territory using stored wood and stone, unlocking the next wall section to build. Repeatable -- once a whole wall ring is fully unlocked and reinforced, this opens a brand new ring further out instead.",
    "BUILD_DOCK": "Build a dock at your current tile using stored wood -- only possible once the tribe has settled here and has already learned to fish (a real successful catch). A one-time, permanent structure: every future fish caught here pays out more from then on.",
    "BUILD_FISHERY": "Build a fishery using stored wood and stone -- only possible once a dock already stands. A one-time, permanent structure: the settlement's passive daily fish supply flows in even more steadily from then on.",
    "BUILD_SAWMILL": "Build a sawmill using stored wood and stone -- only possible once wood has actually been gathered here at least once. A one-time, permanent structure at your settlement: every future load of gathered wood is worth three times as much from then on.",
    "BUILD_QUARRY": "Build a quarry using stored wood and stone -- only possible once stone has actually been gathered here at least once. A one-time, permanent structure at your settlement: every future load of harvested stone is worth three times as much from then on.",
    "BUILD_MINE": "Excavate a mine at a vein your scouts have already found, using stored wood and stone -- only possible once a quarry stands and at least one vein is known. A one-time, permanent structure, but its unique resource has to actually be fetched (GATHER_ORE) before it starts flowing in steadily.",
    "GATHER_ORE": "Fetch the Mine's unique resource -- only possible once a mine has been excavated. The first successful fetch also starts a small, permanent daily supply from then on, the same way fishing works once learned.",
    "BUILD_TANNERY": "Build a tannery using stored wood and stone -- only possible once a hunt has actually succeeded. A one-time, permanent structure at your settlement: Fur flows in steadily from then on, and every successful hunt yields extra meat from then on.",
    "BUILD_HATCHERY": "Build a hatchery using stored wood and stone -- only possible once a wild egg has actually been found and hatched. A one-time, permanent structure at your settlement: the flock grows on its own much more reliably from then on.",
    "BUILD_BATH_HOUSE": "Build a bath house using stored wood and stone -- no prerequisite beyond being settled. A one-time, permanent structure at your settlement: the tribe's daily food and water consumption drops from then on.",
    "BUILD_LIBRARY": "Build a library using stored wood and stone -- only possible once at least one long house stands. A one-time, permanent structure: unlocks RESEARCH, a real way to reach the next era sooner.",
    "RESEARCH": "Study the tribe's own remembered history at the library, using a little stored wood -- only possible once a library stands. Distills what's been lived through into a permanent Library entry, and permanently shortens the path to the next era a little further. Repeatable.",
    "BUILD_WELL": "Build a well using stored wood and stone -- no prerequisite beyond being settled. A one-time, permanent structure at your settlement: the tribe's daily passive water supply flows in faster from then on.",
    "BUILD_WAREHOUSE": "Build a warehouse using stored wood and stone. Raises how much of every resource can be stored at once -- gathering more than storage allows is wasted. Repeatable: each one raises the limit further.",
    "BUILD_FORGE": "Build a forge using stored wood and stone -- only possible once a mine stands and at least one unit of its ore is already in stock. A one-time, permanent structure: from then on, ore can be worked into real tools, weapons, and inventions.",
    "FORGE_ITEM": "Work stored ore and wood into a real item at your forge -- a tool, a weapon, or a small invention, picked at random. No durability to track: each item just carries a flat value, usable later or given away in a trade.",
    "USE_ITEM": "Redeem your oldest crafted item for its stored value, converted into wood and stone. Does nothing if you have no items.",
    "BUILD_KITCHEN": "Build a kitchen using stored wood and stone -- only possible once cooking is known and a long house stands. A one-time, permanent structure: cooked meals become excellent food, stretching stores even further from then on.",
    "BUILD_MOAT": "Dig a moat using stored wood and stone -- only possible once the wall has been reinforced with a second layer. A one-time, permanent structure, cheaper than another wall layer: a further defense bonus.",
    "BUILD_KEEP": "Build a keep using stored wood and stone -- only possible once enough long houses stand. A one-time, permanent structure: a further defense bonus for the settlement.",
    "BUILD_FORTRESS": "Build a fortress using stored wood and stone -- only possible once a keep stands and enough long houses have been built. A one-time, permanent structure: a further defense bonus for the settlement.",
    "PLANT_CROP": "Plant a farm plot at your current tile using stored wood -- only possible once the tribe has settled here. A planted plot grows on its own over the following cycles and yields food automatically once mature; no further action needed to harvest it. Up to a few plots can be tended at once.",
    "GATHER_EGGS": "Search for wild fowl nests near your current tile -- only possible once the tribe has settled here. A found egg is set aside and hatches on its own, growing the tribe's flock by one.",
    "CATCH_FISH": "Attempt to harvest food by fishing at your current tile -- only possible once the tribe has settled here. Pays out food immediately on a catch, and the very first successful catch also starts a small, permanent daily food supply from then on -- fishing, once learned, is never unlearned.",
    "SCOUT": "Dispatch an expedition to explore -- the direction is chosen automatically to spread coverage out over time, not from target_vector. They travel and camp on their own supply, searching up to a few days before turning back if they find nothing. What they find only becomes known once they've walked all the way home. Your tribe can have a couple of parties out at once (scouting or hunting, any mix) -- choosing SCOUT again sends another one if there's room, or just reports on whoever's already out once you're at capacity.",
    "EXPLORATION_PARTY": "Dispatch a deeper, longer-ranging expedition than SCOUT -- direction chosen automatically, its own sweep separate from SCOUT's. Gathers real wood and stone along the way on top of the food and water any expedition forages, until they're carrying as much as they can manage, then heads home. Can discover anything SCOUT can (water, resource sites, raider camps) plus rival settlements and Landmarks -- rare points of interest that yield a real, unique treasure the moment they're found. Shares the same expedition capacity as SCOUT/HUNTING_PARTY/SEND_TRADE_EMISSARY.",
    "HUNTING_PARTY": "Send a hunting party toward target_vector -- shares the same expedition capacity as SCOUT (a couple of parties, scouting or hunting in any mix, can be out at once). They travel and hunt on their own supply for up to several days, facing the same wolf-pack risk as an instant hunt on every day out, until they catch something or give up. Any food caught only becomes real, usable food once they've walked all the way home -- a hunt still in the field does nothing for hunger right now, no matter how promising.",
    "RELOCATE": "Move your whole tribe several tiles toward target_vector this cycle, possibly over several cycles for a far destination. Produces no resources while traveling and costs extra food and water for the effort.",
    "BREED": "Your chief and whoever currently holds a trophy start a family together, costing food and water and growing your population by one child if it succeeds. Does nothing if fewer than two named individuals (a chief plus at least one trophy-holder) exist yet, or if food/water can't cover the cost.",
    "RAID": "Attempt to raid a rival tribe if one is near target_vector. A win steals some of their stockpile but still costs you people; a loss costs you more. An unaffiliated minor settlement near target_vector is a much safer alternative -- no people of its own, so a raid there always succeeds with no risk, though it can only be raided a few times before it's exhausted and needs time to recover. Does nothing if neither is there.",
    "STRIKE_RAIDER_CAMP": "Attack a raider camp your scouts have already found (see your raider sighting reports) -- only possible once you know where one is. Success destroys it and recovers some food; failure costs a life and leaves the camp standing.",
    "TRADE": "Attempt to open trade with a rival tribe if one is near target_vector. Both sides give up a small fraction of everything they hold and receive the same fraction back -- a mutual exchange, no risk of loss. An unaffiliated minor settlement near target_vector can also be traded with -- smaller and one-sided (nothing is given up), but it never depletes the way raiding one does. Does nothing if neither is there.",
    "DECLARE_ALLIANCE": "Declare a lasting alliance with whichever rival tribe is nearest target_vector -- a real, persistent stance both tribes will remember, not a one-time exchange. Also ends a war you'd previously declared with that same rival. Does nothing if no rival tribe exists.",
    "DECLARE_WAR": "Declare a lasting state of war with whichever rival tribe is nearest target_vector -- a real, persistent stance both tribes will remember. Does not attack them directly (see RAID for that); this only sets how the two tribes now stand. Does nothing if no rival tribe exists, or if already at war with them.",
    "SEND_TRADE_EMISSARY": "Dispatch an emissary toward target_vector to actively search for a rival tribe to trade with -- unlike TRADE, this doesn't need one nearby right now, only somewhere along the way over the next few days. Shares the same expedition capacity as SCOUT and HUNTING_PARTY. Finding a partner exchanges goods immediately; the emissary still has to walk home to report what happened.",
}
