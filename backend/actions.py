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
    "wood": {"forest": 1.0, "plains": 0.4, "river": 0.3, "lake": 0.3, "mountains": 0.15,
             "cliffs": 0.0, "shoals": 0.05, "ocean": 0.0},
    "stone": {"mountains": 1.0, "forest": 0.1, "plains": 0.1, "river": 0.1, "lake": 0.1,
              "cliffs": 0.5, "shoals": 0.05, "ocean": 0.0},
    "game": {"forest": 1.0, "plains": 0.6, "river": 0.3, "lake": 0.3, "mountains": 0.15,
             "cliffs": 0.05, "shoals": 0.1, "ocean": 0.0},
    # Foraging (berries, fruit, wild plants) used to not exist at all -- food only ever
    # came from HUNT_DEER/HUNTING_PARTY, both carrying the same wolf-pack risk, so there
    # was no low-risk food option the way GATHER_WATER is a low-risk (if lower-yield)
    # alternative to a river tile. Deliberately profiled opposite to "game": plains is
    # the best foraging ground (open land, berries, roots), forest is only secondary --
    # real tension between forest's higher-risk/higher-yield hunting and plains' safe,
    # steady foraging, rather than one biome just being strictly best at everything.
    "forage": {"plains": 1.0, "forest": 0.6, "river": 0.4, "lake": 0.4, "mountains": 0.1,
               "cliffs": 0.0, "shoals": 0.1, "ocean": 0.0},
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
}


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


def _gather_wood(sim, tribe, biome, target):
    tribe.wood += _harvest(sim, tribe, "wood", 10, biome)
    return None


def _gather_stone(sim, tribe, biome, target):
    tribe.stone += _harvest(sim, tribe, "stone", 10, biome)
    return None


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
    tribe.water += _harvest(sim, tribe, "water", base, biome)
    return None


def _hunt_deer(sim, tribe, biome, target):
    if biome == "forest" and random.random() < config.HUNT_HAZARD_CHANCE:
        tribe.food = max(0, tribe.food - config.HUNT_HAZARD_FOOD_LOSS)
        sim.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.HUNT_HAZARD_TRAUMA_MAGNITUDE, config.HUNT_HAZARD_TRAUMA_RADIUS
        )
        sim._lose_population(tribe, config.HUNT_HAZARD_POPULATION_LOSS, cause="wolf_attack")
        return "a wolf pack struck the hunting party"
    tribe.food += _harvest(sim, tribe, "game", 15, biome)
    return None


def _forage(sim, tribe, biome, target):
    """Berries, fruit, and wild plants -- a real low-risk food option, unlike
    HUNT_DEER/HUNTING_PARTY which both carry wolf-pack risk. Lower base yield than
    hunting (10 vs. 15) since safety is the whole point: foraging trades hunting's
    higher ceiling for a guaranteed, no-hazard return."""
    tribe.food += _harvest(sim, tribe, "forage", 10, biome)
    return None


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
    sim.trauma.radiate_event_wave(
        tribe.x, tribe.y, config.BUILD_FIRE_PRIDE_MAGNITUDE, config.BUILD_FIRE_PRIDE_RADIUS
    )
    return None


def _construct_wall(sim, tribe, biome, target):
    """Explicit request: a wall is built in stages, like a crop, not finished in one
    action -- "30% of a wall can be built through a day with a team of 3." Reuses
    _labor_multiplier (the same "more hands get more done" concept _harvest already
    uses) instead of a separate team-size notion: at Tribe.__init__'s starting
    population (labor multiplier 1.0), one action adds ~30% progress, reaching
    completion in ~4 actions; a larger tribe builds faster. Total cost is unchanged
    from the old instant version (15 wood, 15 stone), just paid proportionally to
    the progress each action actually adds, so a tribe can start without the full
    amount banked yet."""
    if _already_built(sim, tribe, "wall"):
        return None
    existing = sim.world.constructions.get((tribe.x, tribe.y))
    current_progress = existing["progress"] if existing and existing["type"] == "wall" else 0

    added = min(100 - current_progress, round(config.WALL_PROGRESS_PER_ACTION_BASE * _labor_multiplier(tribe.population)))
    wood_cost = round(config.WALL_WOOD_COST_TOTAL * added / 100)
    stone_cost = round(config.WALL_STONE_COST_TOTAL * added / 100)
    if tribe.wood < wood_cost or tribe.stone < stone_cost:
        return None

    tribe.wood -= wood_cost
    tribe.stone -= stone_cost
    new_progress = current_progress + added
    sim.world.add_construction(tribe.x, tribe.y, "wall", sim.cycle, progress=new_progress)
    if new_progress >= 100:
        return "the wall is complete -- the camp is properly defended now"
    return f"wall construction continues -- {new_progress}% complete"


def _plant_crop(sim, tribe, biome, target):
    """Only reachable at all once Simulation._prepare_turn's settled-near-water gate
    (Simulation._is_settled_near_water) allows it -- plains alone doesn't mean a tribe
    resettled somewhere with real water access, per the original design spec for
    farming. Growth itself is a passive per-cycle tick (Simulation._advance_farming),
    not something this action does directly -- planting just adds one more plot to
    tend."""
    if tribe.farm_plots >= config.MAX_FARM_PLOTS or tribe.wood < config.PLANT_CROP_WOOD_COST:
        return None
    tribe.wood -= config.PLANT_CROP_WOOD_COST
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
    return "an egg is found and set aside to hatch"


def _gather_fish(sim, tribe, biome, target):
    """Only reachable once Simulation._prepare_turn's settled-near-water gate allows
    it, same as PLANT_CROP/GATHER_EGGS. "Learning to fish" isn't a separate knowledge
    system -- the first successful catch just flips tribe.fishing_learned, which is
    all Simulation._advance_fish_supply checks to start a passive daily food supply
    from then on, the same "action unlocks a passive system" shape crops and water
    already use. Every catch (including the first) still pays out its own food too."""
    if random.random() >= config.GATHER_FISH_SUCCESS_CHANCE:
        return "no fish caught this time"
    caught = random.randint(config.FISHING_CATCH_FOOD_MIN, config.FISHING_CATCH_FOOD_MAX)
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
    actually sending people to go look, not facts the simulation gifts for free.

    A tribe can have up to expedition_capacity(tribe) parties out at once (scaling with
    population past config.MAX_CONCURRENT_EXPEDITIONS' floor -- see that function), any
    mix of scouting and hunting -- capped rather than unlimited since nothing currently
    deducts population to launch one."""
    if len(tribe.expeditions) >= expedition_capacity(tribe):
        fields = ", ".join(
            f"{e['lead_scout']} (day {e['day']}/{e['max_days']}, {e['phase']})" for e in tribe.expeditions
        )
        return f"no one left to send -- every party is already out: {fields}"

    tx, ty = target
    tx = max(0, min(sim.world.grid_size - 1, tx))
    ty = max(0, min(sim.world.grid_size - 1, ty))
    scout = _generate_scout(tribe, sim.cycle)
    tribe.expeditions.append({
        "kind": "scout",
        "pos": [tribe.x, tribe.y],
        "origin": [tribe.x, tribe.y],
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
        "path": [[tribe.x, tribe.y]],
    })
    tribe.expeditions_launched += 1
    return f"scouts led by {scout['name']} depart camp to explore toward ({tx},{ty})"


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
    tx = max(0, min(sim.world.grid_size - 1, tx))
    ty = max(0, min(sim.world.grid_size - 1, ty))
    scout = _generate_scout(tribe, sim.cycle, base_days=config.HUNTING_PARTY_MAX_DAYS)
    tribe.expeditions.append({
        "kind": "hunt",
        "pos": [tribe.x, tribe.y],
        "origin": [tribe.x, tribe.y],
        "target": [tx, ty],
        "day": 0,
        "phase": "outbound",
        "food_caught": 0,
        "food_gathered": 0,
        "water_gathered": 0,
        "lead_scout": scout["name"],
        "determination": scout["determination"],
        "max_days": scout["max_days"],
        "path": [[tribe.x, tribe.y]],
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
    nx, ny = physics.terrain_aware_step(tribe.x, tribe.y, tx, ty, base_speed=base_speed)
    sim.world.wear_trail(nx, ny, config.TRAIL_WEAR_PER_PASS, tribe.color)
    tribe.x, tribe.y = nx, ny
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


# Explicit request: IDLE isn't offered as a real choice anymore (see eras.py's
# stone_age.unlocks_actions and config.PRE_SETTLEMENT_ACTIONS) -- a tribe always has
# something worth doing. This handler stays registered purely as the internal safe
# no-op Simulation._resolve_action dispatches to when a decision genuinely can't be
# understood at all, or when a syntactically real action name just isn't unlocked
# right now (wrong era, not settled, etc.) -- never something a model is told it can
# pick.
def _idle(sim, tribe, biome, target):
    return None


def _raid(sim, tribe, biome, target):
    """Attempt to raid a rival tribe found at target_vector -- the mechanical outlet
    for an aggressive/warlord chief philosophy (leadership.py can already generate one)
    that otherwise has nothing to act on. Real risk on both sides: win chance is just
    the attacker's share of the two tribes' combined population, so a smaller raiding
    party can still lose to a larger defender, and even a winning raid costs the
    attacker people -- violence isn't a free lever here."""
    tx, ty = target
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
        tribe.raids_lost += 1
        defender.raids_defended += 1
        sim._lose_population(tribe, config.RAID_ATTACKER_POPULATION_LOSS_ON_LOSS, cause="failed_raid")
        sim.trauma.radiate_event_wave(tribe.x, tribe.y, config.RAID_TRAUMA_MAGNITUDE, config.RAID_TRAUMA_RADIUS)
        sim.recent_encounters.append({
            "x": defender.x, "y": defender.y, "kind": "tribe_raid",
            "label": f"{tribe.name} repelled by {defender.name}", "outcome": "lost",
        })
        return f"attempted to raid {defender.name} and was repelled"


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


def _trade(sim, tribe, biome, target):
    """Attempt to open trade with a rival tribe found at target_vector -- the peaceful
    counterpart to RAID, and the mechanical outlet for a cooperative/community-minded
    chief philosophy that otherwise has nothing to act on. Both sides give up the same
    fraction of what they're currently holding and receive the same fraction back --
    a real, mutual exchange, unconditional once initiated (like RAID, this doesn't ask
    the other side's permission; it's the tribe's own choice to reach out peacefully,
    not something requiring the other model to also choose TRADE the same cycle,
    which would make this nearly impossible to ever actually trigger)."""
    tx, ty = target
    partner = None
    for other in sim.tribes.values():
        if other.id == tribe.id or other.extinct:
            continue
        if (other.x - tx) ** 2 + (other.y - ty) ** 2 <= config.TRADE_PROXIMITY_RADIUS ** 2:
            partner = other
            break

    if partner is None:
        return "found no rival encampment there to trade with"

    for resource in ("wood", "stone", "food", "water"):
        tribe_amount = getattr(tribe, resource)
        partner_amount = getattr(partner, resource)
        tribe_gift = round(tribe_amount * config.TRADE_GIFT_FRACTION)
        partner_gift = round(partner_amount * config.TRADE_GIFT_FRACTION)
        setattr(tribe, resource, tribe_amount - tribe_gift + partner_gift)
        setattr(partner, resource, partner_amount - partner_gift + tribe_gift)

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


ACTION_REGISTRY = {
    "GATHER_WOOD": _gather_wood,
    "GATHER_STONE": _gather_stone,
    "GATHER_WATER": _gather_water,
    "GATHER_FOOD": _forage,
    "HUNT_DEER": _hunt_deer,
    "BUILD_FIRE": _build_fire,
    "CONSTRUCT_WALL": _construct_wall,
    "PLANT_CROP": _plant_crop,
    "GATHER_EGGS": _gather_eggs,
    "GATHER_FISH": _gather_fish,
    "SCOUT": _scout,
    "HUNTING_PARTY": _hunting_party,
    "RELOCATE": _relocate,
    "BREED": _breed,
    "RAID": _raid,
    "STRIKE_RAIDER_CAMP": _strike_raider_camp,
    "TRADE": _trade,
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
    "GATHER_FOOD": "Forage for berries, fruit, and wild plants at your current tile -- plains yields the most, forest some, mountains and ocean almost none. No hazard, unlike hunting, but a lower yield ceiling. Yield also drops the more this exact spot has been foraged recently.",
    "HUNT_DEER": "Attempt to harvest food at your current tile -- forest has the most game, plains and river tiles some, mountains and ocean almost none. Small risk of losing a hunter to a wolf pack, most likely in forest.",
    "BUILD_FIRE": "Build a fire at your current tile using stored wood. Does nothing if one is already built here.",
    "CONSTRUCT_WALL": "Work on a wall at your current tile using stored wood and stone -- a real defensive structure built up over several turns, not finished in one. Each turn spent on it adds real progress (more so with more people to put to the work), and a more complete wall meaningfully improves your odds of defending against a raider attack. Does nothing further once complete.",
    "PLANT_CROP": "Plant a farm plot at your current tile using stored wood -- only possible once the tribe has settled somewhere with real water access. A planted plot grows on its own over the following cycles and yields food automatically once mature; no further action needed to harvest it. Up to a few plots can be tended at once.",
    "GATHER_EGGS": "Search for wild fowl nests near your current tile -- only possible on the same settled ground with reliable water access that farming needs (fowl nest near water, not in it). A found egg is set aside and hatches on its own, growing the tribe's flock by one.",
    "GATHER_FISH": "Fish the water at your current tile -- only possible on the same settled ground with reliable water access that farming needs. Pays out food immediately on a catch, and the very first successful catch also starts a small, permanent daily food supply from then on -- fishing, once learned, is never unlearned.",
    "SCOUT": "Dispatch an expedition toward target_vector. They travel and camp on their own supply, searching up to a few days before turning back if they find nothing. What they find only becomes known once they've walked all the way home. Your tribe can have a couple of parties out at once (scouting or hunting, any mix) -- choosing SCOUT again sends another one if there's room, or just reports on whoever's already out once you're at capacity.",
    "HUNTING_PARTY": "Send a hunting party toward target_vector -- shares the same expedition capacity as SCOUT (a couple of parties, scouting or hunting in any mix, can be out at once). They travel and hunt on their own supply for up to several days, facing the same wolf-pack risk as an instant hunt on every day out, until they catch something or give up. Any food caught only becomes real, usable food once they've walked all the way home -- a hunt still in the field does nothing for hunger right now, no matter how promising.",
    "RELOCATE": "Move your whole tribe several tiles toward target_vector this cycle, possibly over several cycles for a far destination. Produces no resources while traveling and costs extra food and water for the effort.",
    "BREED": "Your chief and whoever currently holds a trophy start a family together, costing food and water and growing your population by one child if it succeeds. Does nothing if fewer than two named individuals (a chief plus at least one trophy-holder) exist yet, or if food/water can't cover the cost.",
    "RAID": "Attempt to raid a rival tribe if one is near target_vector. A win steals some of their stockpile but still costs you people; a loss costs you more. Does nothing if no rival is there.",
    "STRIKE_RAIDER_CAMP": "Attack a raider camp your scouts have already found (see your raider sighting reports) -- only possible once you know where one is. Success destroys it and recovers some food; failure costs a life and leaves the camp standing.",
    "TRADE": "Attempt to open trade with a rival tribe if one is near target_vector. Both sides give up a small fraction of everything they hold and receive the same fraction back -- a mutual exchange, no risk of loss. Does nothing if no rival is there.",
    # IDLE deliberately has no entry here -- it's never offered as a real choice (see
    # backend/actions.py._idle), so it has no description to show the model.
}
