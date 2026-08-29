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


def _harvest(sim, tribe, resource_key, base_yield):
    """Shared depletion logic: yield at this tile shrinks the more it's been harvested
    recently, and harvesting here raises that further. Capped below total depletion
    (config.MAX_SCARCITY) so staying put is costly, not a guaranteed dead end."""
    scarcity = sim.world.scarcity(resource_key, tribe.x, tribe.y)
    yield_amount = round(base_yield * (1 - scarcity))
    sim.world.deplete(resource_key, tribe.x, tribe.y, config.DEPLETION_PER_HARVEST, config.MAX_SCARCITY)
    return yield_amount


def _gather_wood(sim, tribe, biome, target):
    tribe.wood += _harvest(sim, tribe, "wood", 10)
    return None


def _gather_stone(sim, tribe, biome, target):
    tribe.stone += _harvest(sim, tribe, "stone", 10)
    return None


def _gather_water(sim, tribe, biome, target):
    if biome == "river" and random.random() < config.DROWNING_HAZARD_CHANCE:
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.DROWNING_HAZARD_POPULATION_LOSS)
        return "the river's current pulled someone under"
    base = config.WATER_YIELD_RIVER if biome == "river" else config.WATER_YIELD_OFF_RIVER
    tribe.water += _harvest(sim, tribe, "water", base)
    return None


def _hunt_deer(sim, tribe, biome, target):
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
    """Looks at a distant tile without moving the tribe -- what a scouting party would
    report back, not a relocation. Written into memory so it can surface in a later
    turn's recall, the same channel real experience already uses. Reports any nearby
    structures too -- a rival's fires or walls across a river aren't automatically
    visible (nearby_structures is only checked around the tribe's own position), so
    noticing a neighbor's growth has to be earned by actually scouting toward them."""
    tx, ty = target
    tx = max(0, min(sim.world.grid_size - 1, tx))
    ty = max(0, min(sim.world.grid_size - 1, ty))
    scouted_biome = biome_at(tx, ty)
    label = BIOME_LABELS.get(scouted_biome, scouted_biome)
    report = f"Scouts report {label} terrain at ({tx},{ty})."
    found = sim.world.nearby_structures(tx, ty, radius=6)
    if found:
        structures = ", ".join(f"{s['type']}@({s['x']},{s['y']})" for s in found)
        report += f" Structures observed: {structures}."
    tribe.memory.remember(report, sim.cycle, weight=0.5)
    return f"scouts venture toward ({tx},{ty}) and report {label}" + (" with signs of habitation" if found else "")


def _relocate(sim, tribe, biome, target):
    """The only action that actually moves the tribe -- a deliberate decision to
    relocate the whole camp, not an automatic side effect of doing something else.
    Costs stamina (food/water, on top of ordinary upkeep) -- marching is tiring, and
    without a cost here relocating would be strictly free compared to every gathering
    action, which all cost time and risk."""
    tribe.food = max(0, tribe.food - config.RELOCATE_FOOD_COST)
    tribe.water = max(0, tribe.water - config.RELOCATE_WATER_COST)
    tx, ty = target
    nx, ny = physics.calculate_next_step(tribe.x, tribe.y, tx, ty, speed=config.MOVEMENT_SPEED)
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
