"""The action registry: each action name maps to a handler function that mutates a
tribe's state and returns an optional hazard note for the chronicle. Adding a new
action means registering a handler here, not extending an if/elif chain in
Simulation -- this is the Registry Factory referenced in the README/design notes.

Handler signature: (sim: Simulation, tribe: Tribe, biome: str) -> str | None
"""

import random

from . import config


def _gather_wood(sim, tribe, biome):
    tribe.wood += 10
    return None


def _gather_stone(sim, tribe, biome):
    tribe.stone += 10
    return None


def _gather_water(sim, tribe, biome):
    if biome == "river" and random.random() < config.DROWNING_HAZARD_CHANCE:
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.DROWNING_HAZARD_POPULATION_LOSS)
        return "the river's current pulled someone under"
    tribe.water += config.WATER_YIELD_RIVER if biome == "river" else config.WATER_YIELD_OFF_RIVER
    return None


def _hunt_deer(sim, tribe, biome):
    if biome == "forest" and random.random() < config.HUNT_HAZARD_CHANCE:
        tribe.food = max(0, tribe.food - config.HUNT_HAZARD_FOOD_LOSS)
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.HUNT_HAZARD_TRAUMA_MAGNITUDE, config.HUNT_HAZARD_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.HUNT_HAZARD_POPULATION_LOSS)
        return "a wolf pack struck the hunting party"
    tribe.food += 15
    return None


def _build_fire(sim, tribe, biome):
    if tribe.wood < 10:
        return None
    tribe.wood -= 10
    sim.world.add_construction(tribe.x, tribe.y, "fire", sim.cycle)
    sim.trauma.radiate_event_wave(
        tribe.x, tribe.y, config.BUILD_FIRE_PRIDE_MAGNITUDE, config.BUILD_FIRE_PRIDE_RADIUS
    )
    return None


def _construct_wall(sim, tribe, biome):
    if tribe.wood < 15 or tribe.stone < 15:
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
