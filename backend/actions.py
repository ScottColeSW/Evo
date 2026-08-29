"""The action registry: each action name maps to a handler function that mutates a
tribe's state and returns an optional hazard note for the chronicle. Adding a new
action means registering a handler here, not extending an if/elif chain in
Simulation -- this is the Registry Factory referenced in the README/design notes.

Handler signature: (sim: Simulation, tribe: Tribe, biome: str) -> str | None
"""

import random

from . import config


def _harvest(sim, tribe, resource_key, base_yield):
    """Shared depletion logic: yield at this tile shrinks the more it's been harvested
    recently, and harvesting here raises that further. Capped below total depletion
    (config.MAX_SCARCITY) so staying put is costly, not a guaranteed dead end."""
    scarcity = sim.world.scarcity(resource_key, tribe.x, tribe.y)
    yield_amount = round(base_yield * (1 - scarcity))
    sim.world.deplete(resource_key, tribe.x, tribe.y, config.DEPLETION_PER_HARVEST, config.MAX_SCARCITY)
    return yield_amount


def _gather_wood(sim, tribe, biome):
    tribe.wood += _harvest(sim, tribe, "wood", 10)
    return None


def _gather_stone(sim, tribe, biome):
    tribe.stone += _harvest(sim, tribe, "stone", 10)
    return None


def _gather_water(sim, tribe, biome):
    if biome == "river" and random.random() < config.DROWNING_HAZARD_CHANCE:
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.DROWNING_HAZARD_POPULATION_LOSS)
        return "the river's current pulled someone under"
    base = config.WATER_YIELD_RIVER if biome == "river" else config.WATER_YIELD_OFF_RIVER
    tribe.water += _harvest(sim, tribe, "water", base)
    return None


def _hunt_deer(sim, tribe, biome):
    if biome == "forest" and random.random() < config.HUNT_HAZARD_CHANCE:
        tribe.food = max(0, tribe.food - config.HUNT_HAZARD_FOOD_LOSS)
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.HUNT_HAZARD_TRAUMA_MAGNITUDE, config.HUNT_HAZARD_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.HUNT_HAZARD_POPULATION_LOSS)
        return "a wolf pack struck the hunting party"
    tribe.food += _harvest(sim, tribe, "game", 15)
    return None


def _already_built(sim, tribe, kind):
    existing = sim.world.constructions.get((tribe.x, tribe.y))
    return existing is not None and existing["type"] == kind


def _build_fire(sim, tribe, biome):
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


def _construct_wall(sim, tribe, biome):
    if _already_built(sim, tribe, "wall") or tribe.wood < 15 or tribe.stone < 15:
        return None
    tribe.wood -= 15
    tribe.stone -= 15
    sim.world.add_construction(tribe.x, tribe.y, "wall", sim.cycle)
    return None


def _idle(sim, tribe, biome):
    return None


ACTION_REGISTRY = {
    "GATHER_WOOD": _gather_wood,
    "GATHER_STONE": _gather_stone,
    "GATHER_WATER": _gather_water,
    "HUNT_DEER": _hunt_deer,
    "BUILD_FIRE": _build_fire,
    "CONSTRUCT_WALL": _construct_wall,
    "IDLE": _idle,
}
