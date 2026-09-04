"""Wall-ring geometry and defense math. Pure structural computation -- unlike
backend/architect.py's placement decision, this never needs to become "smart," it's
just polygon math, kept in its own module for that reason.

A tribe's wall is a set of concentric rings around its territory_center, each made of
config.WALL_RING_SECTION_COUNT positioned 1x5 sections (a compass octagon). A section
whose footprint crosses a natural hazard biome (config.UNBUILDABLE_BIOMES -- ocean,
river, lake, cliffs, shoals) substitutes as a free "natural barrier," never needing
construction or reinforcement, at a weaker defense value than a real built section.
"""

import math

from . import config

_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _is_natural_barrier(world, x: int, y: int, w: int, h: int) -> bool:
    """Any tile under the section's footprint sitting on a hazard biome makes the
    whole section a free natural barrier -- one river/ocean/cliff tile under a 5x1
    strip is enough of a hazard to substitute for construction there."""
    return any(
        world.biome(tx, ty) in config.UNBUILDABLE_BIOMES
        for tx in range(x, x + w) for ty in range(y, y + h)
    )


def build_ring(world, tribe, ring_index: int) -> dict:
    """Computes one ring's full geometry, called once, lazily, the moment it's first
    needed (ring 0 at founding; ring i>0 the first time EXPAND_TERRITORY opens it)."""
    radius = config.WALL_RING_RADIUS_STEP * (ring_index + 1)
    assert radius >= config.WALL_MIN_RING_RADIUS  # tripwire, not expected to ever fire
    cx, cy = tribe.territory_center
    sections = []
    for i, direction in enumerate(_DIRECTIONS):
        # -pi/2 offset so index 0 ("N") points toward decreasing y (this codebase's
        # world has y increasing southward, matching _compass_direction elsewhere),
        # then increasing angle sweeps clockwise through NE, E, SE, ... matching
        # _DIRECTIONS' own order.
        angle = -math.pi / 2 + 2 * math.pi * i / config.WALL_RING_SECTION_COUNT
        sx = cx + round(radius * math.cos(angle))
        sy = cy + round(radius * math.sin(angle))
        # An edge whose outward normal points more horizontally than vertically runs
        # a vertical section (spans north-south); a more-vertical normal runs a
        # horizontal section (spans east-west) -- the section lies *along* the ring,
        # perpendicular to the direction pointing away from town center.
        horizontal = abs(math.sin(angle)) >= abs(math.cos(angle))
        w, h = (config.WALL_SECTION_LENGTH, config.WALL_SECTION_WIDTH) if horizontal else (config.WALL_SECTION_WIDTH, config.WALL_SECTION_LENGTH)
        x, y = sx - w // 2, sy - h // 2
        natural = _is_natural_barrier(world, x, y, w, h)
        sections.append({
            "index": i, "direction": direction, "x": x, "y": y, "w": w, "h": h,
            "orientation": "horizontal" if horizontal else "vertical",
            "natural_barrier": natural, "unlocked": natural, "progress": 0, "tier": 0,
        })
    return {"radius": radius, "sections": sections}


def next_wall_work_section(tribe) -> tuple[int, int] | None:
    """(ring_index, section_index) for CONSTRUCT_WALL to act on next: unfinished
    construction first, across every ring/section in fixed order, then
    reinforcement. None once every unlocked section is fully built and maxed out."""
    for ring_i, ring in enumerate(tribe.wall_rings):
        for sec in ring["sections"]:
            if sec["unlocked"] and not sec["natural_barrier"] and sec["progress"] < 100:
                return ring_i, sec["index"]
    for ring_i, ring in enumerate(tribe.wall_rings):
        for sec in ring["sections"]:
            if sec["unlocked"] and not sec["natural_barrier"] and sec["progress"] >= 100 and sec["tier"] < config.WALL_MAX_LAYERS:
                return ring_i, sec["index"]
    return None


def ring_fully_built(ring: dict) -> bool:
    """Every section either a natural barrier or fully raised (progress >= 100) --
    doesn't require reinforcement, just the base wall standing all the way around."""
    return all(s["natural_barrier"] or s["progress"] >= 100 for s in ring["sections"])


def ring_fully_reinforced(ring: dict) -> bool:
    """Every section either a natural barrier or reinforced to the tier cap."""
    return all(s["natural_barrier"] or s["tier"] >= config.WALL_MAX_LAYERS for s in ring["sections"])


def next_unlockable_section(tribe) -> tuple[int, int] | None:
    """(ring_index, section_index) of the next section EXPAND_TERRITORY should
    unlock: the first not-yet-unlocked, non-natural section in fixed order across
    existing rings. None if every existing ring is fully unlocked (a new ring must
    be built via build_ring before there's anything left to unlock)."""
    for ring_i, ring in enumerate(tribe.wall_rings):
        for sec in ring["sections"]:
            if not sec["unlocked"] and not sec["natural_barrier"]:
                return ring_i, sec["index"]
    return None


def _section_fraction(sec: dict) -> float:
    if sec["natural_barrier"]:
        return config.NATURAL_BARRIER_DEFENSE_FRACTION
    if not sec["unlocked"]:
        return 0.0
    return min(1.0, (sec["progress"] / 100 + sec["tier"]) / (1 + config.WALL_MAX_LAYERS))


def wall_defense_fraction(tribe) -> float:
    """Outer-ring average -- the perimeter a raider actually has to cross first.
    0.0 for a tribe with no wall rings at all yet."""
    if not tribe.wall_rings:
        return 0.0
    outer = tribe.wall_rings[-1]["sections"]
    return sum(_section_fraction(s) for s in outer) / len(outer)


def inner_ring_defense_bonus(tribe) -> float:
    """Defense-in-depth: a small extra bonus per ring behind the outermost one,
    once that inner ring is fully built and reinforced."""
    if len(tribe.wall_rings) <= 1:
        return 0.0
    maxed_inner = sum(1 for ring in tribe.wall_rings[:-1] if ring_fully_reinforced(ring))
    return maxed_inner * config.RAIDER_DEFENSE_PER_INNER_RING_BONUS


def breach_outer_ring(tribe) -> None:
    """A failed defense damages the newest (outermost) perimeter, but not the whole
    thing. Explicit feedback (live playtest): wiping every real section back to 0
    read as "the whole wall falls" -- overly harsh, especially for a tribe still on
    its first ring, where the outer ring *is* the whole wall. Now finds the single
    weakest real (non-natural) section that's actually taken any damage-worthy
    progress (skips untouched sections -- nothing to breach there) and knocks it
    down exactly one layer: a reinforced section loses one tier, a merely-complete
    (tier 0) section loses half its construction progress. Older, inner rings and
    every other section of the outer ring are untouched. Natural-barrier sections
    are terrain, never touched."""
    if not tribe.wall_rings:
        return
    candidates = [
        sec for sec in tribe.wall_rings[-1]["sections"]
        if not sec["natural_barrier"] and (sec["progress"] > 0 or sec["tier"] > 0)
    ]
    if not candidates:
        return
    weakest = min(candidates, key=lambda sec: sec["progress"] / 100 + sec["tier"])
    if weakest["tier"] > 0:
        weakest["tier"] -= 1
    else:
        weakest["progress"] = max(0, weakest["progress"] - 50)
