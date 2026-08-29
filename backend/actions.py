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

import random

from . import config, physics
from .world import BIOME_LABELS, biome_at


# Real biomes don't hand out every resource equally -- a mountain has essentially no
# game to hunt, a forest has no stone to quarry. Previously GATHER_WOOD/GATHER_STONE/
# HUNT_DEER paid the same flat yield in every biome (only local depletion scaled them),
# so a tribe standing on a bare mountain peak could "hunt deer" as effectively as one
# deep in a forest -- resources were an abstract number with no connection to what was
# actually around them. GATHER_WATER was already biome-aware (river vs. elsewhere);
# this brings the other three in line with it.
BIOME_YIELD_MULTIPLIER = {
    "wood": {"forest": 1.0, "plains": 0.4, "river": 0.3, "mountains": 0.15, "ocean": 0.0},
    "stone": {"mountains": 1.0, "forest": 0.1, "plains": 0.1, "river": 0.1, "ocean": 0.0},
    "game": {"forest": 1.0, "plains": 0.6, "river": 0.3, "mountains": 0.15, "ocean": 0.0},
}


def _harvest(sim, tribe, resource_key, base_yield, biome):
    """Shared depletion logic: yield at this tile shrinks the more it's been harvested
    recently, and harvesting here raises that further. Capped below total depletion
    (config.MAX_SCARCITY) so staying put is costly, not a guaranteed dead end. Also
    scales by how much this biome actually supports the resource in the first place --
    see BIOME_YIELD_MULTIPLIER."""
    biome_factor = BIOME_YIELD_MULTIPLIER.get(resource_key, {}).get(biome, 1.0)
    scarcity = sim.world.scarcity(resource_key, tribe.x, tribe.y)
    yield_amount = round(base_yield * biome_factor * (1 - scarcity))
    sim.world.deplete(resource_key, tribe.x, tribe.y, config.DEPLETION_PER_HARVEST, config.MAX_SCARCITY)
    return yield_amount


def _gather_wood(sim, tribe, biome, target):
    tribe.wood += _harvest(sim, tribe, "wood", 10, biome)
    return None


def _gather_stone(sim, tribe, biome, target):
    tribe.stone += _harvest(sim, tribe, "stone", 10, biome)
    return None


def _gather_water(sim, tribe, biome, target):
    if biome == "river" and random.random() < config.DROWNING_HAZARD_CHANCE:
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.DROWNING_HAZARD_POPULATION_LOSS)
        return "the river's current pulled someone under"
    base = config.WATER_YIELD_RIVER if biome == "river" else config.WATER_YIELD_OFF_RIVER
    tribe.water += _harvest(sim, tribe, "water", base, biome)
    return None


def _hunt_deer(sim, tribe, biome, target):
    if biome == "forest" and random.random() < config.HUNT_HAZARD_CHANCE:
        tribe.food = max(0, tribe.food - config.HUNT_HAZARD_FOOD_LOSS)
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.HUNT_HAZARD_TRAUMA_MAGNITUDE, config.HUNT_HAZARD_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.HUNT_HAZARD_POPULATION_LOSS)
        return "a wolf pack struck the hunting party"
    tribe.food += _harvest(sim, tribe, "game", 15, biome)
    return None


def _already_built(sim, tribe, kind):
    existing = sim.world.constructions.get((tribe.x, tribe.y))
    return existing is not None and existing["type"] == kind


def _build_fire(sim, tribe, biome, target):
    # Without this, repeatedly choosing BUILD_FIRE at an already-built tile radiated
    # more ancestral pride every time at zero additional benefit -- a self-reinforcing
    # loop that made staying in one spot forever look increasingly attractive. A second
    # fire where one already burns accomplishes nothing.
    if _already_built(sim, tribe, "fire") or tribe.wood < 10:
        return None
    tribe.wood -= 10
    sim.world.add_construction(tribe.x, tribe.y, "fire", sim.cycle)
    sim.trauma.radiate_event_wave(
        tribe.x, tribe.y, config.BUILD_FIRE_PRIDE_MAGNITUDE, config.BUILD_FIRE_PRIDE_RADIUS
    )
    return None


def _construct_wall(sim, tribe, biome, target):
    if _already_built(sim, tribe, "wall") or tribe.wood < 15 or tribe.stone < 15:
        return None
    tribe.wood -= 15
    tribe.stone -= 15
    sim.world.add_construction(tribe.x, tribe.y, "wall", sim.cycle)
    return None


def _scout(sim, tribe, biome, target):
    """Dispatches an expedition toward target_vector -- your most capable people, out
    searching, not an instant look. They travel and camp under their own supply (no
    drain on the tribe's stockpile), for up to config.EXPEDITION_MAX_DAYS before turning
    back empty-handed if they've found nothing. If they reach real fresh water or their
    intended destination first, they turn back immediately to report it -- but the
    finding only becomes real, actionable knowledge once they've walked all the way home
    (Simulation._advance_expeditions runs the day-by-day travel; this handler only
    launches or no-ops one). Only after that can the tribe's own reasoning choose to
    RELOCATE the whole camp there. This replaced an instant per-turn terrain check and,
    before that, handing a newly-elected chief water's exact coordinates outright (see
    leadership.py) -- water and distant terrain should be things a tribe discovers by
    actually sending people to go look, not facts the simulation gifts for free."""
    if tribe.expedition is not None:
        exp = tribe.expedition
        return f"scouts remain in the field (day {exp['day']}/{config.EXPEDITION_MAX_DAYS}, {exp['phase']})"

    tx, ty = target
    tx = max(0, min(sim.world.grid_size - 1, tx))
    ty = max(0, min(sim.world.grid_size - 1, ty))
    tribe.expedition = {
        "pos": [tribe.x, tribe.y],
        "origin": [tribe.x, tribe.y],
        "target": [tx, ty],
        "day": 0,
        "phase": "outbound",
        "found": None,
        "terrain_report": None,
        "food_gathered": 0,
        "water_gathered": 0,
    }
    return f"scouts depart camp to explore toward ({tx},{ty})"


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
    speed = round(config.MOVEMENT_SPEED + bonus)
    nx, ny = physics.calculate_next_step(tribe.x, tribe.y, tx, ty, speed=speed)
    sim.world.wear_trail(nx, ny, config.TRAIL_WEAR_PER_PASS)
    tribe.x, tribe.y = nx, ny
    return None


def _idle(sim, tribe, biome, target):
    return None


ACTION_REGISTRY = {
    "GATHER_WOOD": _gather_wood,
    "GATHER_STONE": _gather_stone,
    "GATHER_WATER": _gather_water,
    "HUNT_DEER": _hunt_deer,
    "BUILD_FIRE": _build_fire,
    "CONSTRUCT_WALL": _construct_wall,
    "SCOUT": _scout,
    "RELOCATE": _relocate,
    "IDLE": _idle,
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
    "HUNT_DEER": "Attempt to harvest food at your current tile -- forest has the most game, plains and river tiles some, mountains and ocean almost none. Small risk of losing a hunter to a wolf pack, most likely in forest.",
    "BUILD_FIRE": "Build a fire at your current tile using stored wood. Does nothing if one is already built here.",
    "CONSTRUCT_WALL": "Build a wall at your current tile using stored wood and stone. Does nothing if one is already built here.",
    "SCOUT": "Dispatch an expedition toward target_vector. They travel and camp on their own supply, searching up to a few days before turning back if they find nothing. What they find only becomes known once they've walked all the way home -- choosing SCOUT again while one is already out just checks on it, it doesn't send a second one.",
    "RELOCATE": "Move your whole tribe several tiles toward target_vector this cycle, possibly over several cycles for a far destination. Produces no resources while traveling and costs extra food and water for the effort.",
    "IDLE": "Do nothing this cycle.",
}
