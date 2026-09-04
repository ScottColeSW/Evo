OLLAMA_URL = "http://localhost:11434"

TICK_SECONDS = 0.5

GRID_SIZE = 100
MAX_TRIBES = 4

# GATHER_WOOD/GATHER_STONE used to be available from the moment a tribe existed, before
# it had even decided where to actually live -- a nomadic band stockpiling timber and
# quarried stone before choosing a home. Simulation._is_camped gates both behind
# actually staying put somewhere farmable first: SETTLEMENT_STABILITY_CYCLES consecutive
# cycles without choosing RELOCATE, standing on one of FARMABLE_BIOMES (open land with
# real water access). A tribe's starting stockpile (Tribe.__init__) still covers early
# BUILD_FIRE/CONSTRUCT_WALL needs before that -- this gates *replenishing* the economy,
# not survival itself (GATHER_WATER/GATHER_FOOD/HUNT_DEER are never touched).
SETTLEMENT_STABILITY_CYCLES = 10
FARMABLE_BIOMES = ("plains", "river", "lake")

# Explicit request: "the proposed settlement sites, water found, are making it
# hard to Settle. I think we can make this an initial territory with a bounding
# area around it that is larger than the Discovery." A single confirmed water
# tile was too fragile a RELOCATE target -- landing one tile off onto
# non-qualifying ground (a riverbank's cliff edge, a mountain slope) meant
# never actually settling despite being right next to real water. Any tile
# within this Chebyshev radius of a confirmed water site now counts as good
# enough ground too (see Simulation._near_confirmed_water), on top of the
# existing exact-biome-match check, not instead of it.
# Tightened 6 -> 4 (2026-09-02): a settlement exactly at the old radius's edge
# read as visually disconnected from the water it was supposedly settled near.
SETTLEMENT_WATER_TERRITORY_RADIUS = 4

# Toll roads (backend/world.py.wear_trail/is_toll_road/road_owner, Simulation.
# _resolve_toll): explicit request -- "trails that have been traversed more
# than 5 times by anyone will automatically evolve into visible and owned
# roads that others may travel for a fee... The first trailblazer gets the
# ownership and tolls (automatically collected when used or crossed). can't
# pay, can't cross." A real, cumulative, never-decaying crossing count is
# separate from the existing decaying `wear` value (cosmetic/speed-bonus
# only) -- 5 crossings is the literal number named in the request.
ROAD_EVOLVE_CROSSINGS = 5
TOLL_FEE_WOOD = 5
# Explicit request: "travel speed is 5x on toll roads." A real reason to want
# one running toward you, not just a toll to dread -- far stronger than an
# ordinary trail's own wear-scaled MAX_TRAIL_BONUS_SPEED.
TOLL_ROAD_SPEED_MULTIPLIER = 5

# Explicit request: a weak model faced with the full ~13-action Stone Age list from
# cycle one has no structural push toward the single most important early decision --
# settling somewhere real. Before a tribe has EVER settled next to real water (see
# Simulation.Tribe.has_ever_settled/Simulation._is_settled_near_water), its choices are
# narrowed to just enough to survive and actually go find a home; everything requiring
# an actual camp (building, hunting parties, trading) unlocks permanently the first
# time it genuinely settles, not re-locked if it later relocates again. Still the
# model's own choice among what's offered, same gating principle as GATHER_WOOD/STONE
# being locked pre-settlement -- just applied to the whole early action set at once.
#
# Explicit correction: BREED and RAID are never locked behind settling -- neither
# needs a fixed camp to happen (a nomadic band can still fight or start a family),
# they're just naturally rare this early since eligibility (a trophy holder for
# BREED, a rival tribe nearby for RAID) is harder to come by before settling down.
PRE_SETTLEMENT_ACTIONS = ("GATHER_WATER", "GATHER_FOOD", "SCOUT", "RELOCATE", "BREED", "RAID")

# Tiles moved per axis per cycle toward target_vector. At 1 (the original value), crossing
# the 100-tile grid takes 100+ cycles minimum -- at roughly 2 seconds of real inference
# time per cycle, a "good" decision to travel far was still visually imperceptible for
# minutes. This doesn't change AI decision quality, just how fast a given decision reads
# on screen.
MOVEMENT_SPEED = 4

# Marching costs stamina -- without this, RELOCATE was a free action while every gathering
# action costs time/risk, which would make endless relocation strictly better than settling
# anywhere. Paid on top of ordinary upkeep, same resources, not a separate stat.
#
# At 2 each, a real multi-cycle journey (the kind SCOUT/RELOCATE was split out to
# encourage) stacked with ordinary upkeep to drain 3 food + 3 water per cycle with zero
# income -- an 8-cycle, 30-tile crossing cost 24 of a 40/30 starting stockpile, making
# the escape route out of a depleted home tile nearly as lethal as staying. Verified via
# direct computation before lowering: at 1 each, the same journey costs 16, leaving real
# margin while relocating is still twice as costly per cycle as standing still.
RELOCATE_FOOD_COST = 1
RELOCATE_WATER_COST = 1

# The self-modification engine lets a model rewrite backend/physics.py on disk and
# hot-reloads it when turns get slow. It's validated with an AST parse + cooldown
# lockout, but it is still an LLM writing code to your machine — off by default.
ENABLE_SELF_MODIFICATION = False
SELF_MOD_LATENCY_THRESHOLD_MS = 4000
SELF_MOD_COOLDOWN_CYCLES = 20

MEMORY_CONSOLIDATE_EVERY_N_CYCLES = 40

# The "night cycle" (backend/reflection.py): periodically, a larger model reviews a
# tribe's own recent history and decides for itself whether its guiding philosophy
# should change -- distinct from the fast small model handling every live turn. The
# reviewer model is deliberately not configurable per-tribe -- it's meant to be a
# consistently larger, slower, less-often-run reviewer regardless of which small model
# a tribe actually plays with. mistral:7b is a real step up from the 2-3B models
# tribes actually play with live, without the 26B extremes of the original gemma4:26b
# default -- gemma4:26b was removed from this machine as too large to actually use.
NIGHT_CYCLE_EVERY_N_CYCLES = 30
NIGHT_CYCLE_REVIEWER_MODEL = "mistral:7b"
NIGHT_CYCLE_HISTORY_WINDOW = 20

# Explicit request: "can we have some random breeding in the over-night cycle?" Every
# existing breeding side-effect (Simulation._celebrate_*) fires off a specific
# milestone -- a tribe that never crosses one of those particular triggers had no path
# to a family besides the model explicitly choosing BREED itself, which this session's
# live data shows these models almost never reach for on their own. Night is the one
# recurring beat every settled tribe passes regardless of what else happened that day,
# so it's a natural home for an occasional chance encounter -- same eligibility rule
# and $0 cost as every other breeding path (see BREED_FOOD_COST below), just
# probabilistic instead of tied to a specific milestone.
NIGHT_CYCLE_RANDOM_BREED_CHANCE = 0.25

# An innate tradition, not a chief's choice (see Simulation._hold_tribal_gathering):
# every tribe gathers once per in-game day to take stock, whatever its philosophy or
# model. Mirrors frontend/index.html's own DAY_LENGTH_CYCLES for the sun/moon arc --
# kept as two separately-defined constants (JS can't import this module) but they must
# stay equal, or the gathering stops landing at the in-game dawn it's meant to mark.
DAY_LENGTH_CYCLES = 20

# Rough pre-flight sanity check against a model's on-disk size (backend/vram_guard.py).
# Not a live enforcement layer -- see that module's docstring for why.
VRAM_LIMIT_GB = 14.0

# Inference temperature: bumped when a tribe stands on ancestrally traumatic ground,
# so panic/urgency actually reads as less predictable model output, not just flavor text.
DEFAULT_TEMPERATURE = 0.55
ANCESTRAL_DREAD_TEMPERATURE = 1.15

# Ancestral trauma matrix event weights (see backend/ancestral_matrix.py). Pride/dread
# bias text only appears once a tile's score crosses +-0.35 (ancestral_matrix.py), so
# a single-event magnitude below that never actually surfaces -- keep these above it.
BUILD_FIRE_PRIDE_MAGNITUDE = 0.4
BUILD_FIRE_PRIDE_RADIUS = 5
CITY_FOUNDED_PRIDE_MAGNITUDE = 0.5
CITY_FOUNDED_PRIDE_RADIUS = 8
HUNT_HAZARD_TRAUMA_MAGNITUDE = -0.4
HUNT_HAZARD_TRAUMA_RADIUS = 6

# Hunting in the forest carries a real risk of a wolf encounter — this is what actually
# generates negative ancestral events; without it the trauma matrix only ever sees pride.
HUNT_HAZARD_CHANCE = 0.12
HUNT_HAZARD_FOOD_LOSS = 10
HUNT_HAZARD_POPULATION_LOSS = 1

# HUNTING_PARTY: a multi-day alternative to instant HUNT_DEER, reusing the SCOUT
# expedition state machine (see actions.py._hunting_party, simulation.py's
# _advance_expedition) -- persist day over day until something is caught, rather than
# one guaranteed-yield roll on the spot. The catch itself only becomes real food once
# the party walks home (same "findings aren't real until you're home" rule as SCOUT),
# so a tribe that's starving *right now* gets no relief from a hunt still in the field
# -- and every extra day out is another HUNT_HAZARD_CHANCE roll, not just one.
HUNTING_PARTY_MAX_DAYS = 4
HUNTING_PARTY_CATCH_CHANCE_BASE = 0.35
HUNTING_PARTY_CATCH_FOOD_MIN = 20
HUNTING_PARTY_CATCH_FOOD_MAX = 35

# A tribe previously had no way to be told wildlife was actually present -- the only
# fact touching game was local resource scarcity (see below), which only ever reports
# past depletion, never announces a live sighting before any hunting has happened
# there. This is a random per-cycle chance (scaled by the richest nearby tile's own
# game yield multiplier, so a mountain or ocean tile essentially never triggers one) of
# a real, named encounter fact appearing in the tribe's visible entities. The radius is
# small and deliberately tighter than BROADCAST_HEARING_RADIUS/nearby_structures --
# this is "close enough to hear or spot," not "somewhere on the regional map."
GAME_SIGHTING_CHANCE_BASE = 0.3
GAME_SIGHTING_RADIUS = 2

# Local resource depletion (backend/world.py, backend/actions.py). Each harvest of a
# given resource at a given tile raises that tile's scarcity; yield there is scaled down
# by (1 - scarcity). Capped below 1.0 so a tribe that never moves still gets a trickle,
# rather than a hard lock -- the point is pressure to relocate, not a guaranteed death
# sentence for staying put. Regeneration is global and constant, independent of whether
# anyone is currently there.
DEPLETION_PER_HARVEST = 0.15
DEPLETION_REGEN_PER_CYCLE = 0.02
MAX_SCARCITY = 0.8

# Water (backend/actions.py, backend/eras.py). River tiles yield far more than
# scrounging elsewhere, which is what actually gives river tiles strategic pull.
STARTING_WATER = 30
WATER_YIELD_RIVER = 15
WATER_YIELD_OFF_RIVER = 3

# Explicit request ("like relocate, gather water becomes irrelevant once they have
# settled"): a tribe genuinely settled next to real water (Simulation.
# _is_settled_near_water) shouldn't need to keep manually choosing GATHER_WATER every
# cycle just to stand still -- the same "passive consequence, not a discrete action"
# category as crop growth (Simulation._advance_farming). GATHER_WATER still works and
# still adds more on top; this just means the tap never really runs dry once settled.
#
# Bug report: "we have hit a food and water scaling problem... I'm not sure why
# water is still a problem when they are settled." Confirmed: this used to be a
# flat SETTLED_WATER_SUPPLY_PER_CYCLE = 10, while _apply_upkeep's real drain
# (population // UPKEEP_POPULATION_DIVISOR) grows with the tribe -- past
# population ~100 the flat income could no longer keep up, a structural deficit
# that only got worse the bigger (more "successful") a tribe got. Now scales
# with the same per-capita upkeep base instead of a fixed number, so a settled
# tribe's water income keeps pace at any population -- this multiplier is the
# "easy factor variable" to turn up if tribes are still running dry.
SETTLED_WATER_SUPPLY_MULTIPLIER = 1.5

# Explicit request: "if they are lucky enough to have a resource... in the
# territory they settle in, it's a daily allotted freebie they never have to
# gather from." Same per-capita-upkeep scaling as SETTLED_WATER_SUPPLY_MULTIPLIER/
# FISHING_SUPPLY_MULTIPLIER above/below, but deliberately smaller -- a lucky
# in-territory site is a bonus on top of real gathering/infrastructure, not a
# replacement for it.
IN_TERRITORY_SITE_YIELD_MULTIPLIER = 0.5

# Reaching a new era (backend/eras.py) radiates a pride event at the tribe's location,
# same mechanism as BUILD_FIRE -- advancement is a genuine "the ground remembers this"
# moment, not just a silent counter change.
ERA_ADVANCE_PRIDE_MAGNITUDE = 0.5
ERA_ADVANCE_PRIDE_RADIUS = 8

# Population upkeep (backend/simulation.py._apply_upkeep). Larger tribes cost more to
# sustain each tick -- this is what makes growth an ongoing pressure instead of a
# one-time threshold crossed once and then irrelevant.
UPKEEP_POPULATION_DIVISOR = 10  # cost per tick = max(1, population // this)

# Population growth (backend/simulation.py._grow_population). Originally food > 80
# costing 30 -- verified live that a real 79-cycle run under realistic mixed play
# (gathering, hunting, scouting, not a maximally-optimized single-resource grind)
# never got food above ~38 for either tribe, starting from 40. Reaching Bronze Age
# needs population 20, which at the old cost was 12 growth events needing ~360+
# cumulative food surplus above ordinary living costs -- not a difficulty tuning
# choice, just unreachable under any plausible play. Lowered once already to 50/15;
# this session's headless A/B testing (multiple real 100-cycle runs, see
# logs/experiments.jsonl) showed most tribes still never sustaining food above the
# high 30s/low 40s even once wildlife sighting and hunting-party fixes were added --
# 50 was still frequently out of reach. Loosened further, still same shape (a real
# food cost per growth tick, not free), just inside the range tribes actually reach.
POPULATION_GROWTH_FOOD_THRESHOLD = 25
POPULATION_GROWTH_FOOD_COST = 8
# Explicit request: "we should not put a cap on population" -- live runs tonight
# showed multiple tribes actually reaching the old cap (80) and sitting there,
# which was the whole point of removing it rather than just raising the number.
# Infinity rather than deleting every `population < POPULATION_GROWTH_CAP` check
# across simulation.py/actions.py -- same real per-growth-tick food cost gates
# growth either way, there's just no longer a ceiling on top of that.
POPULATION_GROWTH_CAP = float("inf")

# Farming (backend/actions.py PLANT_CROP, Simulation._advance_farming): gated on
# genuinely settled ground with real water access, not just any farmable biome -- see
# Simulation._is_settled_near_water. "Plains" alone doesn't mean a tribe resettled
# somewhere with easy water, per the original design spec for this feature.
FARMING_REQUIRES_ADJACENT_WATER = ("river", "lake")
PLANT_CROP_WOOD_COST = 10
MAX_FARM_PLOTS = 4
CROP_GROWTH_PER_CYCLE = 10  # a plot matures in ~10 cycles once planted
CROP_HARVEST_YIELD = 15  # food per plot, per harvest
CROP_WATER_PER_PLOT_PER_CYCLE = 2  # a plot that goes unwatered withers outright

UNBUILDABLE_BIOMES = ("ocean", "river", "lake", "cliffs", "shoals")

# Redesigned 2026-09-02 ("these shouldn't be disconnected... look at it as a whole"):
# retires the old abstract city_buildings counter (population-driven, unrelated to any
# real named building) in favor of real placed building footprints (backend/
# architect.py) inside a real owned territory (backend/city_layout.py). Granted the
# instant tribe.has_ever_settled becomes True -- the earlier "Settled" milestone, not
# the later, stricter founded_city (Monolithic Era + a real Long House).
#
# Live-run correction (2026-09-02, same day): the original TERRITORY_FOUNDING_REGION=10
# (radius 40 on a GRID_SIZE=100 map) was way too big in practice -- a single tribe's
# starting territory spanned 80% of the map's width, guaranteeing overlap with every
# other tribe's before either side ever took an EXPAND_TERRITORY action. Dropped to the
# smallest value that still respects WALL_MIN_RING_RADIUS below (the real geometric
# floor for 8 wall sections to stay properly spaced) rather than picking an arbitrary
# smaller number.
TERRITORY_FOUNDING_REGION = 3  # base radius = SETTLEMENT_WATER_TERRITORY_RADIUS * this = 12 tiles
WALL_RING_RADIUS_STEP = SETTLEMENT_WATER_TERRITORY_RADIUS * TERRITORY_FOUNDING_REGION  # 12; ring i sits at 12*(i+1)

# EXPAND_TERRITORY unlocks exactly one new wall section per call, in fixed compass
# order -- "expansion must be done for each wall section," no exception for ring 0.
# tribe.territory_radius (see actions._expand_territory) is always derived as
# WALL_RING_RADIUS_STEP * (ring count) -- explicit correction after live data showed
# it drifting far past the wall's own real geometry when it used to grow by its own
# separately-scaled increment every call instead.

# The wall is a real polygon of positioned sections around the territory, not one
# progress-bar tile. WALL_RING_SECTION_COUNT=8 (a compass octagon) needs
# 2*radius*sin(pi/8) >= WALL_MIN_SECTION_SPACING (3 long-house-widths, 3*3=9) --
# WALL_MIN_RING_RADIUS=12 is that real floor, solved from the same formula, and
# WALL_RING_RADIUS_STEP is now set exactly at it (chord spacing ~9.2) rather than
# comfortably past it -- the tightest radius that's still provably correct, not a
# runtime path expected to ever violate the tripwire below.
WALL_RING_SECTION_COUNT = 8
WALL_SECTION_LENGTH = 5
WALL_SECTION_WIDTH = 1
WALL_MIN_SECTION_SPACING = 9
WALL_MIN_RING_RADIUS = 12

# A water/cliff/etc-bordered section substitutes as a free "natural barrier" needing
# no construction/reinforcement -- weaker than a real built section, never fully
# impassable (matches how the tribe's own people can still reach real water; only the
# defense math is affected here).
NATURAL_BARRIER_DEFENSE_FRACTION = 0.5
# Defense-in-depth: each ring behind the outermost one, once fully built+reinforced,
# adds a small extra bonus on top of the outer ring's own defense_fraction -- a raider
# that breaches the outer wall still has to get through however many maxed rings
# stand behind it.
RAIDER_DEFENSE_PER_INNER_RING_BONUS = 0.05

# backend/architect.py's placement algorithm inflates every occupied rect (buildings
# and wall sections alike) by this many tiles on each side before checking overlap,
# so placed structures never sit edge-to-edge touching.
BUILDING_PLACEMENT_PADDING = 1

# Real footprints (tiles, w x h) for every placeable structure. Moat and Road are
# deliberately excluded -- the moat is a property of the wall ring, not a placeable
# rect, and the road already follows worn trail tiles (world.trails); both keep their
# existing boolean-flag mechanics untouched.
BUILDING_FOOTPRINTS = {
    "town_hall": (5, 5), "long_house": (3, 2), "wall_section": (5, 1),
    "keep": (4, 4), "fortress": (6, 6), "castle": (8, 8),
    "sawmill": (3, 3), "quarry": (3, 3), "mine": (3, 3), "forge": (2, 2),
    "warehouse": (3, 3),
    "kitchen": (2, 2), "tannery": (2, 2), "dock": (2, 2), "fishery": (2, 4),
    "farm_plot": (3, 3), "flock_pen": (2, 2), "fire": (1, 1), "hatchery": (2, 2),
    "boat": (2, 3), "bath_house": (2, 2), "library": (3, 3), "well": (2, 2),
}

# BUILD_FISHERY (backend/actions.py): a new building, unlocked once a Dock already
# stands. Stacks a further multiplier onto the existing passive daily fish supply
# (FISHING_SUPPLY_MULTIPLIER) rather than replacing it -- a real reason to build both.
FISHERY_WOOD_COST = 25
FISHERY_STONE_COST = 15
FISHERY_SUPPLY_BONUS_MULTIPLIER = 1.5

# wellbeing.compute_wellbeing's self-actualization tier: how many real placed
# buildings (backend/architect.py) count as "fully built out" for a 1.0 score,
# replacing the old fixed MAX_CITY_BUILDINGS=6 ceiling now that building count has
# no hard cap.
SELF_ACTUALIZATION_BUILDING_REFERENCE = 15

# Egg-gathering/flock genetics (backend/actions.py GATHER_EGGS, Simulation._resolve_hatch,
# backend/genetics.py hatch()): gated the same as farming (settled + real water access) --
# wild fowl near a confirmed water source, not a separate condition.
GATHER_EGGS_SUCCESS_CHANCE = 0.4
# A flock isn't just a one-way counter -- it eats, and an established flock can also
# breed on its own (Simulation._advance_flock), the same "passive consequence, not a
# discrete action" category as crop growth. Real stakes both ways: undersized on feed
# and it shrinks, big enough and it can grow without another GATHER_EGGS at all.
FLOCK_UPKEEP_FOOD_PER_MEMBER = 1
FLOCK_MIN_SIZE_TO_BREED = 2
FLOCK_NATURAL_HATCH_CHANCE = 0.15

# Explicit request: "let them feast and use Eggs and Chickens/Flock for food
# after the stock grows... let them use everything more than a dozen each."
# tribe.eggs is a real, separate stockpile from tribe.flock (a living flock
# lays eggs passively, distinct from GATHER_EGGS finding a wild nest to hatch
# into a new flock member) -- see Simulation._advance_flock_eggs. Once either
# stockpile grows past LIVESTOCK_SURPLUS_THRESHOLD ("a dozen"), the surplus is
# automatically eaten as food each cycle (Simulation._advance_livestock_feast)
# instead of piling up forever with no payoff -- the same "don't let it just
# sit there" shape the storage cap already applies to bulk resources, except
# here the overflow becomes real food instead of being capped away.
# BUILD_BATH_HOUSE (backend/actions.py, Simulation._apply_upkeep): explicit
# request -- "bath house bolsters Well-Being upkeep once built." No special
# prerequisite beyond being settled and affordable (same as Warehouse/Road) --
# hygiene isn't gated behind a proven success the way hunting/fishing/mining
# are. Reduces the tribe's real per-cycle food/water consumption, which
# directly raises wellbeing.py's physiological tier score too (it's computed
# straight from the same upkeep-buffer formula this multiplies).
BATH_HOUSE_WOOD_COST = 20
BATH_HOUSE_STONE_COST = 15
BATH_HOUSE_UPKEEP_MULTIPLIER = 0.85

# BUILD_LIBRARY/RESEARCH (backend/actions.py, Simulation._advance_era_if_ready):
# explicit request -- a Library condenses the tribe's own TribeMemory (backend/
# memory.py) into permanent, readable entries (own frontend tab, not just an
# establishment line), and unlocks RESEARCH: a real, repeatable "growth and
# innovation" payoff, not a flat stat nudge. Gated on long_houses_built > 0 (real
# housing already established), same "building homes" signal Kitchen/Sawmill/
# Quarry already use -- a Library only makes sense once people actually live here.
LIBRARY_WOOD_COST = 30
LIBRARY_STONE_COST = 25
RESEARCH_WOOD_COST = 10
# Each completed RESEARCH permanently discounts the *next* era's population/
# resource thresholds and its advancement cost by this fraction, capped so
# advancement can never become free -- a tribe that invests in research
# genuinely reaches the next era sooner, the concrete "boosts growth" this was
# built for. Applied fresh against next_era() each check (Simulation.
# _advance_era_if_ready), not baked into eras.py's own numbers.
INNOVATION_ERA_DISCOUNT_PER_RESEARCH = 0.04
INNOVATION_ERA_DISCOUNT_CAP = 0.5
# How many of the tribe's own highest-weight memories get folded into one Library
# entry -- a real distillation (see TribeMemory.consolidate's own top-3 taboo
# ranking, which this deliberately mirrors), not the full raw log dumped in.
LIBRARY_ENTRY_MEMORY_COUNT = 3

# BUILD_WELL (backend/actions.py, Simulation._advance_water_supply): explicit
# request -- water's only passive-income lever was a single flat formula tied to
# population, with no equivalent of Fishery/Dock's stacking bonus for food. Same
# "infrastructure from the moment it's unlocked" shape Bath House/Warehouse
# already use (no proven-success gate), stacking onto SETTLED_WATER_SUPPLY_
# MULTIPLIER exactly the way FISHERY_SUPPLY_BONUS_MULTIPLIER stacks onto
# FISHING_SUPPLY_MULTIPLIER for food.
WELL_WOOD_COST = 20
WELL_STONE_COST = 20
WELL_SUPPLY_BONUS_MULTIPLIER = 1.5

EGGS_LAID_PER_FLOCK_PER_CYCLE_DIVISOR = 5  # 1 egg per 5 flock members per cycle
LIVESTOCK_SURPLUS_THRESHOLD = 12
EGG_FEAST_FOOD_VALUE = 2  # food per surplus egg eaten
FLOCK_FEAST_FOOD_VALUE = 8  # food per surplus flock member eaten

# BUILD_HATCHERY (backend/actions.py): explicit follow-up -- "the Flock and the
# Eggs self generate. So, maybe after they GATHER_EGGS in the wild, they can
# have a Hatchery." Same "a real proven success gates the building, not a
# scouted site or another building" pattern as Sawmill/Quarry/Tannery -- gated
# on tribe.eggs_ever_gathered (a real wild GATHER_EGGS find, see actions.py.
# _gather_eggs), not flock size alone. A hatchery is where eggs get incubated
# into new flock faster, so it boosts the natural-hatch chance (Simulation.
# _advance_flock) rather than the passive laying rate (_advance_flock_eggs).
HATCHERY_WOOD_COST = 15
HATCHERY_STONE_COST = 10
HATCHERY_HATCH_CHANCE_MULTIPLIER = 2.0

# Fishing (backend/actions.py CATCH_FISH, Simulation._advance_fish_supply): gated the
# same as farming/eggs -- once settled, no separate real-water check (explicit
# correction: that extra distinction was "bogus," just a settled gate like everything
# else here). "Learning to fish" isn't a separate knowledge/skill system -- it's the
# same "an action unlocks a passive system" shape crops and water already use. The
# first successful catch flips Tribe.fishing_learned, which is all _advance_fish_
# supply checks; every CATCH_FISH after that (including the first) still pays out
# its own immediate catch too.
#
# Explicit request (2026-08-31): a settled tribe fishes right at its own tile -- no
# expedition, no travel time, unlike HUNTING_PARTY's multi-day trip -- so fishing
# should read as strictly the best food return once available: higher success odds
# and a higher catch than HUNT_DEER's base_yield=15 (which also risks a wolf-pack
# hazard) or GATHER_FOOD's base_yield=10 (both in actions.py), and it already carries
# no resource cost the way PLANT_CROP spends wood. Expected value per attempt is now
# 0.8 * 19 = 15.2, above both of those bases, with zero hazard risk.
CATCH_FISH_SUCCESS_CHANCE = 0.8
FISHING_CATCH_FOOD_MIN = 14
FISHING_CATCH_FOOD_MAX = 24
# Bug report: "we have hit a food and water scaling problem... we should have an
# easy factor vaiable we can we turn up for food." Same flat-vs-scaling flaw
# SETTLED_WATER_SUPPLY_MULTIPLIER was fixed for -- a flat 8 food/cycle couldn't
# keep pace with population-scaled upkeep past a certain tribe size. Now scales
# with the same per-capita upkeep base; turn this multiplier up if tribes are
# still going hungry once large.
FISHING_SUPPLY_MULTIPLIER = 1.5
# Explicit request: fish fertilizer -- once fishing is learned, a farm plot's growth
# rate roughly doubles (halving the season), reusing tribe.fishing_learned rather than
# a separate fertilizer resource/action. See Simulation._advance_farming.
FISH_FERTILIZER_GROWTH_MULTIPLIER = 2.0

# Chief trophies (backend/simulation.py._check_chief_trophies): a lightweight legacy
# system credited to whichever chief is in power the moment each is first earned, once
# per tribe's lifetime. "Water Bringer" is deliberately the standout -- reliable water
# access is the single hardest survival problem this simulation poses (see the whole
# expedition/nearest_water design), so it's the trophy that actually means something.
FOOD_TROPHY_THRESHOLD = 60

# Celebrations (backend/simulation.py._check_for_celebration): automatic and threshold-
# based, same pattern as era advancement/trophies/population growth -- NOT a discrete
# action the model has to remember to pick. This session's own data (BREED sat free and
# genuinely eligible for 20+ live cycles without ever being chosen; HUNTING_PARTY is
# barely picked either) suggests these models rarely reach for a new discrete choice at
# all, so a reward gated behind choosing one more action would likely suffer the same
# fate. Fires on a real resource surplus (the same threshold FOOD_TROPHY_THRESHOLD
# already uses for "Well Fed") OR a genuine new discovery (any memory entry just
# recorded this cycle at or above CELEBRATION_DISCOVERY_WEIGHT -- the same weight that
# already promotes a memory into a permanent taboo/lesson, i.e. this tribe's own
# definition of "something big enough to remember forever"). Spends a real fraction of
# the surplus (the "mass gathering effort"), radiates real pride through the area, and
# -- if two distinct named individuals are already eligible -- is what naturally brings
# them together, without needing the model to separately choose BREED.
CELEBRATION_DISCOVERY_WEIGHT = 0.75
CELEBRATION_RESOURCE_COST_FRACTION = 0.3
# Explicit finding: at 30% of *current* food with no ceiling, a thriving tribe's
# celebrations get more expensive in absolute terms the wealthier it gets --
# "we spend a lot of time on Parties." Capped so a rich tribe doesn't bleed
# proportionally more just for being rich.
CELEBRATION_MAX_COST = 60
CELEBRATION_PRIDE_MAGNITUDE = 0.5
CELEBRATION_PRIDE_RADIUS = 6

# Fame: a new well-being measurement (wellbeing.py's 6th, non-Maslow tier).
# Explicit request: "Finding and marking Landmarks increases Fame... We can tie
# this to big events too like Roads and Walls, etc." Every real celebration
# (every _celebrate_* hook plus the generic _check_for_celebration) awards a
# flat amount; a Landmark discovery awards its own, larger amount when it falls
# inside the tribe's own territory.
FAME_PER_CELEBRATION = 2
FAME_PER_LANDMARK = 3
FAME_PER_LANDMARK_IN_TERRITORY = 6
# Normalizes tribe.fame into the same 0..1 scale every other wellbeing tier
# uses -- reaching "satisfied" (wellbeing.TIER_SATISFIED_THRESHOLD) takes
# roughly 6 celebrations' worth, deliberately slower than esteem's own trophy
# count since fame is meant to accumulate over a whole run, not an early spike.
FAME_SCORE_REFERENCE = 12
CELEBRATION_COOLDOWN_CYCLES = 20
# _check_for_celebration's surplus-only branch (no real discovery, just "food is
# comfortably above FOOD_TROPHY_THRESHOLD") can otherwise fire every single cooldown
# window forever -- real, but not a fresh reason to spend food indefinitely once a
# tribe has proven it can reliably sustain a surplus. Retires after this many, the
# same "generalist narrows to specialist" shape GATHER_FOOD's own retirement uses --
# the discovery branch (always a genuinely new, distinct thing) never retires.
CELEBRATION_SURPLUS_RETIREMENT_COUNT = 3

# Explicit request: "Celebrations can even be cheaper if they learn how to cook
# food... a pot luck event where they all go out and hunt and gather for a feast."
# COOK_FOOD (backend/actions.py) is gated on real prerequisites (a successful hunt
# and a successfully-built fire, ever -- see tribe.hunt_ever_succeeded/
# fire_ever_built) rather than needing a fire currently standing at this exact
# tile -- explicit correction: cooking is a skill learned once, not something tied
# to a specific structure. Once learned (tribe.cooking_learned, the same "learn
# once, keep forever" shape fishing_learned uses), every future celebration costs
# less: real food is being contributed and prepared efficiently, not just handed
# over from the stockpile.
CELEBRATION_COOKING_COST_MULTIPLIER = 0.5

# Redesigned 2026-09-02 ("that's a mess, let's break it down and build it back up
# properly"): cooking used to divide food *consumption* (Simulation._apply_upkeep)
# by 3 instead of multiplying food *production*, the odd one out against
# SAWMILL_WOOD_MULTIPLIER/QUARRY_STONE_MULTIPLIER/DOCK_FISH_CATCH_BONUS_FRACTION,
# which all apply their bonus at the harvest point instead. COOKING_FOOD_MULTIPLIER
# now matches that shape exactly: applied in actions._food_multiplier at every real
# food-production point (GATHER_FOOD, HUNT_DEER/HUNTING_PARTY, CATCH_FISH, passive
# fish supply, crop harvest) -- not to loot/pillage transfers, which move existing
# stockpiled food rather than producing new food.
COOKING_FOOD_MULTIPLIER = 3

# Milestone trophies (backend/simulation.py._award_trophy's `individual` param): unlike
# the chief-credited trophies above, these are earned by a specific named scout or
# hunter and credit them by name, not the chief. Also the pool of "named individuals"
# the breeding design draws from alongside the chief -- see BREED_FOOD_COST below.
MILESTONE_SCOUT_SUCCESSES = 5
MILESTONE_HUNT_SUCCESSES = 5

# BREED (backend/actions.py._breed, backend/breeding.py). Was free (0/0) -- the two
# real eligible windows watched in an early session both landed inside a full
# starvation death spiral (0 food/water), so a positive cost would have blocked BREED
# during the exact moments eligibility was most likely to appear. Live-run correction
# (2026-09-02): "BREEDing is free after all. it should cost" -- a weaker model
# (llama3.2:1b) latched onto BREED as its reflexive default with nothing weighing
# against it (63.8% of turns in one run), the same lexical-fixation pattern already
# seen with GATHER_FOOD, just on a different verb. Costs roughly one gathering
# action's worth of each resource now -- real friction against reflexive spam, but
# still affordable outside an actual crisis, unlike a cost scaled to feel
# "significant" against a healthy stockpile.
BREED_FOOD_COST = 8
BREED_WATER_COST = 5

# Survival instinct thresholds (backend/instincts.py), expressed as cycles of upkeep
# remaining rather than a flat stockpile number -- a flat "food <= 20" meant wildly
# different things depending on population (20 cycles of buffer for a population-8
# tribe paying 1/cycle, only 4 cycles for a population-50 tribe paying 5/cycle), so the
# same number wasn't actually a consistent signal across tribe sizes. "Critical" also
# raises inference temperature, same as ancestral dread -- real panic, not just
# different wording.
HUNGER_WARNING_CYCLES_LEFT = 4
HUNGER_CRITICAL_CYCLES_LEFT = 1
THIRST_WARNING_CYCLES_LEFT = 4
THIRST_CRITICAL_CYCLES_LEFT = 1

# Real runs showed tribes starving/dehydrating while sitting on 100+ wood or stone --
# gathering more of a resource that was never the bottleneck, apparently without
# realizing the stockpile was already well past any near-term building need. Set above
# every currently-real wood/stone cost (BUILD_FIRE=10, CONSTRUCT_WALL=15/15, the
# priciest era-advancement cost so far is 40) so crossing it genuinely means "more than
# any real use," not an arbitrary number. Only surfaced alongside an actual food/water
# warning (see Simulation._prepare_turn) -- a fact about the mismatch, not a standing
# nudge to stop gathering.
MATERIAL_SURPLUS_THRESHOLD = 50

# What happens when upkeep can't be paid -- someone dies, and the ground remembers it.
# Magnitude matches the other hazard deaths (wolf attack, drowning) so any death reliably
# clears the -0.35 dread threshold in ancestral_matrix.py, not just repeated ones.
STARVATION_POPULATION_LOSS = 1
STARVATION_TRAUMA_MAGNITUDE = -0.4
STARVATION_TRAUMA_RADIUS = 5
DEHYDRATION_POPULATION_LOSS = 1
DEHYDRATION_TRAUMA_MAGNITUDE = -0.4
DEHYDRATION_TRAUMA_RADIUS = 5

# Drowning: gathering water from a river carries real risk, mirroring the forest
# hunting hazard. Lower than the wolf-attack chance since it's a deliberate act, not
# an ambush, but water gathering elsewhere is still slower -- a genuine risk/reward.
DROWNING_HAZARD_CHANCE = 0.08
DROWNING_HAZARD_POPULATION_LOSS = 1
DROWNING_TRAUMA_MAGNITUDE = -0.4
DROWNING_TRAUMA_RADIUS = 6

# A scout doesn't need to physically step into a river or lake to know it's there --
# running water carries, and a lake is visible well before its shore. Without this, a
# scout could pass within a tile or two of a lake and report nothing, which read as a
# missing "proximity" sense rather than a close call. Only the exact-tile case still
# carries DROWNING_HAZARD_CHANCE -- hearing water from a safe distance carries no risk.
WATER_SENSING_RADIUS = 6

# SCOUT's actual heading (backend/actions.py._scout): explicit request --
# "they can't reason about closeness to the discover, they have to get to a
# pre-assigned location and explore along the way. let's say for now, scout
# directions rotate on a 20 degree angle starting with the South East." Small
# models repeatedly failed to translate compass-direction facts (or even their
# own prior targets) into coordinates that actually covered new ground -- live
# runs showed two scouts launched back to back heading the exact same
# direction. target_vector is no longer read for SCOUT specifically (still
# used for RELOCATE/HUNTING_PARTY/RAID/TRADE/etc, all of which have a real
# reason to point somewhere the model actually chose): each dispatch advances
# to the next angle in a fixed rotation instead, guaranteeing coverage spreads
# out over time regardless of what the model reasons about geometry.
# Changed from 45 (southeast) to 135 (southwest) -- live bug report: Forest
# Tribe's very first scout (every tribe's rotation starts at index 0, so this
# is the one heading every tribe's opening scout shares) walked southeast
# straight into the cramped river/cliffs strip at (83,58) that later caused
# the city-founding/land-cap bugs fixed the same session. Southwest was
# picked as a quick, reversible nudge away from that specific corner of the
# map, not a claim that it's provably better in general -- 135 degrees
# matches _compass_direction's own convention for southwest.
SCOUT_ROTATION_START_ANGLE_DEGREES = 135
# Explicit request: "after that is used, change the wall ghost line and
# orthogonal angle by 12 degrees, for the next, and rotate it for the next."
SCOUT_ROTATION_STEP_DEGREES = 12

# Live report (2026-09-03): "the scouts went exact the same way" -- confirmed
# against a fresh run's own snapshots (Forest Tribe's opening SCOUT and Mountain
# Tribe's opening SCOUT computed the identical (dx,dy) offset from their own
# position, both at scout_rotation_index=0). The comment above already flagged
# this exact gap ("every tribe's rotation starts at index 0, so this is the one
# heading every tribe's opening scout shares") but only nudged the shared angle
# away from one bad map corner, never gave tribes distinct rotations. Tribe.
# __init__ now seeds scout_rotation_index/explore_rotation_index at
# tribe_index * this stagger instead of always 0 -- 7 is coprime with the
# 18-step full rotation (360/20), so no two tribes' sequences realign for any
# tribe count this project supports (config.MAX_TRIBES).
SCOUT_ROTATION_TRIBE_STAGGER_STEPS = 7

# Live bug report (2026-09-02): "they go big long lines like they are flying,
# possibly too far." A fresh dispatch used to target the grid's true edge (up to ~99
# tiles away) and get pushed even further if it arrived early with days left --
# confirmed live, a scout covered 26 tiles in just 2 days on a dead-straight heading.
# Bounds a single SCOUT dispatch to a local patrol instead; EXPEDITION_SPEED (10/day,
# an earlier deliberate choice) is untouched -- this only shortens how far a trip is
# aimed, not how fast it walks. At the low end of determination variance (max_days=2,
# see EXPEDITION_DETERMINATION_DAY_VARIANCE), 25 tiles takes a full day's speed to
# spare either way, so determination still meaningfully affects whether a party
# finishes its patrol before giving up.
SCOUT_PATROL_DISTANCE = 25

# Live-run finding: "Long Explorations have not manifested" -- confirmed against
# a real run's own logs, 829 EXPLORATION_PARTY dispatches averaged 1.02 days
# before turning back, against a 6-day budget (EXPLORATION_PARTY_MAX_DAYS).
# Root cause: _exploration_party reused this same SCOUT_PATROL_DISTANCE, so it
# never actually went any farther than a plain SCOUT and always "arrived"
# almost immediately. Sized so a full outbound leg at EXPEDITION_SPEED (10/day)
# takes ~4.5 days -- genuinely uses most of the day budget without regularly
# blowing past it before determination variance/trail bonuses are even
# considered.
EXPLORATION_PARTY_PATROL_DISTANCE = 45

# A wandering storm cloud (Simulation._advance_weather) -- weather that exists whether
# or not any tribe is watching, not triggered by or aimed at anyone. Rare to spawn
# (checked once per cycle only while no storm is active), rare to strike once present,
# gone after STORM_LIFESPAN_CYCLES either way. A tribe standing exactly where it
# strikes takes a real, small hazard (same _lose_population channel as every other
# hazard, so immortality still protects it); a tribe merely nearby just gets a fact
# about it (see Simulation._build_visible_entities) -- the same "real event, no
# scripted reaction" pattern as a wildlife sighting.
STORM_SPAWN_CHANCE = 0.02
STORM_LIFESPAN_CYCLES = 20
STORM_SPEED = 3
STORM_HEADING_JITTER = 0.4  # radians/cycle -- wanders, doesn't fly a dead-straight line
LIGHTNING_STRIKE_CHANCE = 0.15
LIGHTNING_STRIKE_RADIUS = 3  # a tribe within this many tiles notices the strike as a fact
LIGHTNING_HAZARD_POPULATION_LOSS = 1
LIGHTNING_TRAUMA_MAGNITUDE = -0.5
LIGHTNING_TRAUMA_RADIUS = 6

# Chief mortality (backend/simulation.py._lose_population). Previously a chief, once
# elected, was permanent flavor text -- it never mattered who was actually still alive.
# Now any population loss (starvation, thirst, a hazard, a lost raid) carries a real
# chance the chief is among the casualties, clearing the tribe's leadership and forcing
# a fresh, model-generated succession contest (see _install_chief) rather than leaving
# the tribe leaderless forever. This is a real consequence of resource mismanagement,
# not a scripted nudge -- what the tribe does about a leadership vacuum is still its
# own turn-by-turn call.
CHIEF_DEATH_CHANCE_ON_LOSS = 0.2

# A tribe can now actually go extinct (population 0) instead of being propped up at a
# permanent population-1 floor. Extinction is a far larger trauma event than an ordinary
# death -- it should be visible on the map for a long time afterward.
EXTINCTION_TRAUMA_MAGNITUDE = -0.6
EXTINCTION_TRAUMA_RADIUS = 10

# Raiding (backend/actions.py._raid): the mechanical outlet for an aggressive/warlord
# chief philosophy (backend/leadership.py can already generate one) that otherwise has
# nothing to actually act on. Real risk on both sides -- a smaller raiding party can
# still lose to a larger defender, and even a winning raid costs the attacker
# something. Available from the Stone Age (see eras.py): inter-tribal conflict is at
# least as old as inter-tribal cooperation, not a later "advanced" capability.
RAID_PROXIMITY_RADIUS = 3
RAID_STEAL_FRACTION = 0.3
RAID_ATTACKER_POPULATION_LOSS_ON_WIN = 1
RAID_ATTACKER_POPULATION_LOSS_ON_LOSS = 2
RAID_TRAUMA_MAGNITUDE = -0.5
RAID_TRAUMA_RADIUS = 6
RAID_PRIDE_MAGNITUDE = 0.4
RAID_PRIDE_RADIUS = 5

# Every raid win transfers a slice of the defender's current population to the
# attacker (replacing the old flat, one-sided population loss) -- captured or
# defecting survivors, not just casualties. Once enough raids have driven a defender's
# population to zero this way, Simulation._merge_tribes turns the winner into a new,
# more advanced entity instead of just leaving a hole where the loser was.
RAID_POPULATION_ABSORB_FRACTION = 0.2

# Trade (backend/actions.py._trade): the peaceful counterpart to RAID, and the
# mechanical outlet for a cooperative/community-minded chief philosophy
# (leadership.py can already generate one, e.g. "prioritizes cooperation... believing
# in sharing resources") that otherwise has nothing to act on. Both sides give up the
# same fraction of what they're currently holding and receive the same fraction back
# -- a real, mutual exchange, not a one-sided gift or a raid without the violence.
TRADE_PROXIMITY_RADIUS = 3
TRADE_GIFT_FRACTION = 0.15
TRADE_PRIDE_MAGNITUDE = 0.3
TRADE_PRIDE_RADIUS = 4

# Minor settlements (backend/simulation.py._spawn_minor_settlements, backend/
# actions.py._raid/_trade): explicit request -- neutral, non-AI raid/trade targets
# scattered on the map, distinct from tribe-vs-tribe RAID/TRADE. "Quick and dirty":
# no population, no chief, no LLM call, no battle roll -- RAID against one always
# succeeds (there's no defense to lose to), at no population risk either way.
# Snapshot-based, not simulated: each one's stockpile mirrors whichever real
# tribe currently has the highest population at the moment it spawns/respawns, so
# its loot scales with how developed the world actually is instead of a flat
# invented number.
MINOR_SETTLEMENT_COUNT = 3
MINOR_SETTLEMENT_MAX_RAIDS = 3
MINOR_SETTLEMENT_RESPAWN_CYCLES = 7
# Raiding empties it out fast (only 3 uses before it's exhausted); trading is the
# smaller, repeatable, risk-free alternative to raiding the same target -- reuses
# RAID/TRADE's own existing proximity radii, not a separate search distance.
MINOR_SETTLEMENT_RAID_STEAL_FRACTION = 0.4
MINOR_SETTLEMENT_TRADE_FRACTION = 0.1
# Explicit request: "when they start to build a Wall we need to force existing
# Raider sites out of the Territory and for some distance away from the
# Territory boundary." Extra margin beyond a tribe's own territory_radius kept
# clear when placing/relocating a minor settlement -- not just literally inside
# the wall, but a real buffer past it too.
MINOR_SETTLEMENT_TERRITORY_BUFFER = 8

# Tribe Map: a coarse "ground we've actually walked" record, distinct from the
# positive-find lists (lumber_sites etc., which only record a discovery, not
# mere passage). Bucketed rather than per-tile so it stays small over a long
# run and reads as "this general area," not a literal breadcrumb trail
# (Landscape.trails already covers that at the per-tile level). Feeds the
# survey-spam fix (see Simulation._advance_exploration_party_outbound) so a
# tribe stops re-confirming ground it's already covered.
TRIBE_MAP_SECTOR_SIZE = 10

# DECLARE_ALLIANCE/DECLARE_WAR (backend/actions.py): a persistent, per-rival
# geopolitical stance (Age 4's Declare_Geopolitical_Posture from the Agentic
# Evolution spec reconciliation) -- unlike instant RAID/TRADE, this leaves a real,
# lasting record of how two tribes stand, surfaced back as a fact each tribe can
# reason from. No proximity gate (see actions.py._nearest_rival) -- a policy
# declaration isn't a physical encounter the way RAID/TRADE are.
NEGOTIATE_PRIDE_MAGNITUDE = 0.3
NEGOTIATE_PRIDE_RADIUS = 4

# backend/threat.py -- the reconciled, non-overriding version of the Agentic
# Evolution spec's Module A (calculate_threat_proximity). Same distance-weighted
# exponential-decay shape the spec itself proposed (w_r * exp(-alpha * dist)),
# scoped to declared-WAR rivals specifically -- raider proximity already has its
# own honest fact (see Simulation._advance_raider_approach).
THREAT_DECAY_RATE = 0.05
THREAT_ASSESSMENT_MIN_LEVEL = 0.05

# SEND_TRADE_EMISSARY (backend/actions.py): TRADE itself is instant and only works if
# a rival already happens to be within TRADE_PROXIMITY_RADIUS right now -- a tribe can
# never deliberately reach out to a rival it merely knows the rough direction of. This
# is a real, multi-day expedition instead, sharing the exact day-by-day travel/give-up
# machinery HUNTING_PARTY already uses (nearly the same mechanic, per explicit
# confirmation) -- same max-days pacing as a hunting party, same proximity check as
# instant TRADE once it's actually looking.
TRADE_EMISSARY_MAX_DAYS = 4

# Raider hazard (backend/simulation.py._check_raider_attack): a real, population-
# scaled mechanic, not a scripted "your people are not safe" directive (a hardcoded
# HUNT_DEER nudge was already reverted once on this exact principle). Gated behind
# tribe.has_ever_settled -- a nomadic band with nothing built and nothing stockpiled
# has nothing worth raiding yet. Chance scales with population up to a cap rather than
# applying full force the moment a tribe settles; a cooldown (mirrors
# CELEBRATION_COOLDOWN_CYCLES) keeps this reading as discrete events, not noise.
# Live feedback: "we have introduced about 75% too many Raiders on the map, they
# are interfering." Most of that was the sighting-marker list never getting
# pruned (see RAIDER_SIGHTING_MAX_REMEMBERED) -- but actual attack frequency gets
# a modest tune-down here too, not a redesign.
RAIDER_HAZARD_MAX_CHANCE = 0.09
RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE = 60
RAIDER_HAZARD_COOLDOWN_CYCLES = 35

# Defense resolution once an attack triggers. Population alone gives some defensive
# chance (more hands to fight back -- same shape as RAID's own population-ratio win
# chance); a wall at the tribe's own tile adds more, scaled continuously by its own
# construction progress (see WALL_PROGRESS_* below) -- a half-built wall gives roughly
# half this bonus, not zero and not full. Capped below 1.0: even a maximally-defended
# tribe isn't literally immune.
RAIDER_DEFENSE_BASE_CHANCE = 0.25
RAIDER_DEFENSE_POPULATION_BONUS_PER_10 = 0.05
RAIDER_DEFENSE_WALL_BONUS_AT_FULL_PROGRESS = 0.35
RAIDER_DEFENSE_MAX_CHANCE = 0.85
# Explicit request: a river/lake tile is a natural partial barrier -- a settled-near-
# water tribe (Simulation._is_settled_near_water) needs less constructed wall to reach
# the same real protection, not a separate wall requirement.
RAIDER_DEFENSE_WATER_BONUS = 0.15
# Explicit finding: raiders were being repelled too consistently -- the raiding force
# itself never scaled with what it was actually attacking, so any moderately-sized
# tribe's population/wall bonuses alone could reliably clear RAIDER_DEFENSE_MAX_CHANCE.
# Scales with the same population signal that already drives whether an attack happens
# at all (RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE) -- a bigger, wealthier tribe draws a
# genuinely stronger raiding force, not the same fixed threat every time. This is what
# makes a wall (and water) actually matter, not just population.
RAIDER_STRENGTH_DEFENSE_PENALTY_AT_MAX = 0.35

# A failed defense costs population and stockpile -- mitigated continuously by wall
# progress, never fully negated. Reuses RAID_TRAUMA_MAGNITUDE/RADIUS and
# RAID_PRIDE_MAGNITUDE/RADIUS (above) rather than new ones: mechanically the same kind
# of violence event as a tribe-vs-tribe raid, not a new trauma category.
RAIDER_ATTACK_POPULATION_LOSS_UNDEFENDED = 2
RAIDER_ATTACK_POPULATION_LOSS_AT_FULL_WALL = 1
RAIDER_STEAL_FRACTION = 0.25

# Explicit request: "the repelling Tribe better get some good rewards from that.
# it's huge for them!" -- a successful defense used to yield only pride and a
# counter, nothing tangible. Scales with raider_strength (the same signal driving
# how dangerous the attack was) so a genuinely tough repelled raid leaves real,
# substantial spoils behind, not a token amount.
RAIDER_DEFEAT_LOOT_FRACTION = 0.2

# Scout early-warning (Simulation._advance_one_expedition's arrival-home branch): an
# independent roll, rolled separately from every resource-site chance above, so a
# party can plausibly spot both a resource site and raider sign on the same trip.
# Radiates a small dread event AT THE SIGHTING COORDINATE, not the tribe's camp -- a
# genuinely new pattern: this is about a place now known to be dangerous, not
# something that happened at home.
# Live-run feedback (2026-09-02): "tone down a little the Raider sites, the Raids
# are fine" -- this is only the map-marker/rumor rate, not RAIDER_APPROACH_CYCLES or
# any actual attack odds, both left untouched. Halved from 0.1.
RAIDER_SIGHTING_CHANCE = 0.05
RAIDER_SIGHTING_TRAUMA_MAGNITUDE = -0.4
RAIDER_SIGHTING_TRAUMA_RADIUS = 4
# Bug report: "we have a lot of Raider camps right on top of a resource." Both the
# raider sighting and a resource-site discovery (terrain_report) get recorded at the
# exact same exp["target"] coordinate, so whenever both independent rolls succeed on
# the same trip they land on the literal same tile by construction -- "on the same
# trip" doesn't have to mean "at the identical spot." Nudges the raider sighting by
# up to this many tiles off the target instead, still nearby, no longer stacked.
RAIDER_SIGHTING_OFFSET = 5
# Bug report: "we introduced about 75% too many Raiders on the map, they are
# interfering." Unlike the other LANDMARK_TYPES lists (a lumber/water/quarry site
# stays a real, permanently-true fact), a raider sighting is transient danger --
# tribe.raider_sightings was append-only with nothing ever trimming it, so a long
# run (day 86, this report) accumulated dozens of permanent (frontend/index.html
# LANDMARK_TYPES) swords markers per tribe that were never actually cleaned up.
# Kept small and most-recent-first: old sighting are stale intel, not history.
RAIDER_SIGHTING_MAX_REMEMBERED = 6

# Explicit request: "I do want to see RAIDERs ride in over time" -- an attack used to
# resolve entirely in one invisible instant (roll, resolve, done). Now a triggered
# attack (Simulation._check_raider_attack) starts a real, visible, multi-cycle
# approach (Simulation._advance_raider_approach) before it actually resolves -- real
# advance warning the tribe can act on (finish a wall) before the attack lands, not
# just a surprise. Explicit follow-up request: "let's start raiders 10 cycles away" --
# raised from 3 to give a tribe genuinely enough real time to react (finish a wall
# in progress, etc.) rather than a token few-cycle heads-up.
RAIDER_APPROACH_CYCLES = 10
RAIDER_APPROACH_START_DISTANCE = 8

# Explicit request: "It would be interesting to see a Scout encounter a RAIDER
# group" -- a real, in-the-field ambush during expedition travel, distinct from the
# settlement-level attack above and distinct from the report-based sighting roll
# (RAIDER_SIGHTING_CHANCE) -- this is a party physically running into raiders, not a
# rumor or a distant attack on the camp. Ends the trip immediately, the same way the
# wolf-pack hazard ends a hunt outright (Simulation._advance_hunting_party_outbound).
# Gated behind has_ever_settled the same as the settlement attack -- raiders being
# active against a tribe at all is itself tied to that tribe having something worth
# raiding.
EXPEDITION_RAIDER_AMBUSH_CHANCE = 0.04
EXPEDITION_RAIDER_AMBUSH_POPULATION_LOSS = 1

# Staged wall construction (backend/actions.py._construct_wall): reuses
# _labor_multiplier(population) -- the same "more hands get more done per action"
# concept _harvest already uses -- rather than inventing a separate team-size notion.
# At Tribe.__init__'s starting population (POPULATION_YIELD_BASELINE=8, multiplier
# 1.0), one action adds ~30% progress ("a team of 3... 30% of a wall... through a
# day" from design conversation), reaching completion in ~4 actions; a larger tribe
# builds faster. Total cost (15 wood, 15 stone -- unchanged from the old instant
# version) is paid proportionally to the progress each action actually adds, not up
# front, so a tribe can start a wall without having the full amount banked yet.
WALL_PROGRESS_PER_ACTION_BASE = 30
WALL_WOOD_COST_TOTAL = 15
WALL_STONE_COST_TOTAL = 15

# A second wall layer, reinforcing an already-complete section (backend/city_layout.py/
# actions.py._construct_wall): explicit request -- "Torches can be a freebie for
# building walls 2 levels" and "a Moat should be available after 2 layers of walls
# have been built." A flat cost, not another multi-action progress bar the way the
# very first pass on a section was -- reinforcing a standing section is simpler than
# raising one from nothing. Capped at WALL_MAX_LAYERS per section: 2 layers/tiers is
# the whole point named in both requests, not an arbitrary stopping point. Once every
# section in a ring is at this tier, growth continues via a whole new ring further
# out (backend/city_layout.py.build_ring) -- no limit on ring count beyond land.
WALL_MAX_LAYERS = 2
WALL_LAYER_WOOD_COST = 20
WALL_LAYER_STONE_COST = 20

# Torches (backend/simulation.py._resolve_raider_attack): explicit request --
# free once a tribe has fire and a second wall layer, no action or cost of its
# own, just a real defense bonus applied directly in the raid formula.
TORCHES_DEFENSE_BONUS = 0.05

# BUILD_MOAT (backend/actions.py._build_moat): explicit request, "a Moat should
# be available after 2 layers of walls have been built." A cheaper alternative
# investment once wall layers are maxed out, not a replacement for the wall
# already standing -- smaller cost, smaller bonus than a wall layer.
MOAT_WOOD_COST = 15
MOAT_STONE_COST = 10
MOAT_DEFENSE_BONUS = 0.08

# BUILD_LONG_HOUSE (backend/actions.py._build_long_house): explicit correction --
# "most structures they only need 1 of. but house builds are dependant on
# population needs." Repeatable now, gated on real population need
# (HOUSING_POPULATION_PER_LONG_HOUSE) rather than a single one-time flag -- a
# growing tribe keeps needing more shelter, the same way farm plots keep
# growing rather than capping at one. Still gated on the wall already being
# complete first -- defense before shelter.
LONG_HOUSE_WOOD_COST = 25
LONG_HOUSE_STONE_COST = 20
# Raised 8 -> 30 (explicit request: "how many people to a Long House? 30 maybe at
# most") -- a real, human-scale capacity per building rather than a small number
# that mostly just controlled how fast Long Houses accumulated.
HOUSING_POPULATION_PER_LONG_HOUSE = 30

# The defensive tier ladder after Long House (backend/actions.py._build_keep/
# _build_fortress/_build_castle): explicit request -- "they can have 10 houses
# before they build a Keep, 40 until they reach a Fortress, 70 until they can
# build castles." Gated on tribe.long_houses_built (a real proxy for how
# established the settlement has become) rather than era or population alone,
# each stage requiring the previous one already standing. Each is a real,
# additional defense bonus stacked on top of the wall's own (Simulation.
# _resolve_raider_attack's RAIDER_DEFENSE_WALL_BONUS_AT_FULL_PROGRESS).
KEEP_LONG_HOUSES_REQUIRED = 10
KEEP_WOOD_COST = 30
KEEP_STONE_COST = 35
KEEP_DEFENSE_BONUS = 0.10

FORTRESS_LONG_HOUSES_REQUIRED = 40
FORTRESS_WOOD_COST = 50
FORTRESS_STONE_COST = 60
FORTRESS_DEFENSE_BONUS = 0.20

CASTLE_LONG_HOUSES_REQUIRED = 70
CASTLE_WOOD_COST = 40
CASTLE_STONE_COST = 50
CASTLE_DEFENSE_BONUS = 0.15

# BUILD_ROAD (backend/actions.py._build_road): a permanent, tribe-built version of
# the same trail_speed_bonus a well-worn path already grants expeditions (World.
# trail_speed_bonus) -- flat, not distance-decayed like a trail, since a road exists
# deliberately rather than wearing in from repeated travel.
ROAD_WOOD_COST = 30
ROAD_STONE_COST = 15
ROAD_SPEED_BONUS = 2

# EXPAND_TERRITORY (backend/actions.py._expand_territory): grows the tribe's real
# territory_radius (see TERRITORY_FOUNDING_REGION above) and unlocks the next wall
# section in fixed compass order -- one call per section, no exception for ring 0.
TERRITORY_EXPANSION_WOOD_COST = 60
TERRITORY_EXPANSION_STONE_COST = 60

# BUILD_DOCK (backend/actions.py._build_dock): explicit request, "once they have
# Settled in hopes they will figure out fishing" -- a real fishing yield bonus once
# built, not just flavor, to actually reward betting on fishing early.
DOCK_WOOD_COST = 20
DOCK_FISH_CATCH_BONUS_FRACTION = 0.5

# BUILD_SAWMILL/BUILD_QUARRY (backend/actions.py): explicit request, "I think they
# should build a saw mill and a quarry after they have farming and fishing down and
# are building homes. saw mill turns 1 wood into 3 wood. quarried stone is also
# worth 3 times as much as a harvested stone." Same "3x via a multiplier applied
# once at the point of harvest" shape cooked food uses (config.
# COOKING_FOOD_MULTIPLIER) -- not a conversion action spending wood to make more
# wood, a permanent multiplier on every future GATHER_WOOD/GATHER_STONE. Gated on
# tribe.long_house_built ("building homes") and tribe.fishing_learned ("fishing
# down"), the two real facts named in the request, not era alone.
# Live data (day 86 run): a tribe with a Sawmill built still sat at wood: 6 vs.
# stone: 1618 after 1700+ cycles -- wood is a cost on far more building types than
# stone is (nearly every building spends some wood; far fewer spend stone), so the
# same multiplier as Quarry's wasn't enough to keep up. Doubled as a first pass --
# revisit with a real building-cost audit only if a future run still shows the
# same imbalance.
SAWMILL_WOOD_COST = 30
SAWMILL_STONE_COST = 15
SAWMILL_WOOD_MULTIPLIER = 6
QUARRY_WOOD_COST = 15
QUARRY_STONE_COST = 30
QUARRY_STONE_MULTIPLIER = 3

# BUILD_WAREHOUSE + storage caps (backend/actions.py): explicit request, after a
# live run showed Forest Tribe pile wood up to 200+ while permanently starved on
# stone -- unlimited storage meant nothing ever pushed a tribe to reconsider what
# it was gathering. STORAGE_CAP_BASE alone (150) already clears every era's own
# resource requirement (the highest, Cosmic Post-Human, needs at most 120), so a
# tribe is never blocked by this before ever building a Warehouse -- it only ever
# catches genuinely excessive hoarding. Repeatable, like Long House
# (HOUSING_POPULATION_PER_LONG_HOUSE) -- "expansion of the tribe will allow that to
# scale storage with building needs" -- each Warehouse adds a further flat bonus,
# same shape, same fixed footprint every time (see config.BUILDING_FOOTPRINTS --
# explicit request: "a building that never changes its footprint regardless of how
# much it is holding").
STORAGE_CAP_BASE = 150

# Explicit request: "these guys need punishment for choosing the wrong thing.
# like for waste when they overfill the storage." Gathering into an
# already-full (or nearly full) store used to just narrate the waste with no
# real consequence -- now radiates a real negative trauma wave (actions.py.
# _add_capped), the same "the ancestors remember what happened here" idiom
# already used for drowning/wolf attacks/starvation. Matches those real
# hazards' own magnitude (-0.4) rather than a diluted version -- AncestralTrauma
# Matrix.bias_string only surfaces DREAD past -0.35, and a real, immediately
# felt consequence on the very first waste (not just after several repeats at
# the same tile) is the point.
WASTE_TRAUMA_MAGNITUDE = -0.4
WASTE_TRAUMA_RADIUS = 5
WAREHOUSE_STORAGE_BONUS_PER_BUILDING = 100
WAREHOUSE_WOOD_COST = 25
WAREHOUSE_STONE_COST = 20

# Resource-site discovery (lumber/wildlife/quarry/mine): superseded 2026-09-02 --
# see world.py's SITE_SEED_GRID_CELL_SIZE/SITE_SEED_FILL_PROBABILITY/
# SITE_DISCOVERY_RADIUS. Sites used to be decided fresh via an independent chance
# roll on whatever exact tile a scout's report landed on (a brief intermediate
# fix for the *fairness* problem: lumber/wildlife were guaranteed on forest and
# quarry on mountains, structurally locking out a tribe that never scouted the
# right biome). Explicit follow-up request -- "a twisted sparse matrix assignment
# based on the existing map" -- replaced that with real, pre-seeded site locations
# a scout discovers by landing nearby, which fixes the earlier fix's own remaining
# gap for free (two site types can no longer stack on the same tile, since each
# has its own independent seed set).

# BUILD_MINE + the per-biome unique resource (backend/actions.py): explicit
# request -- "Mines can [also] contain the Unique Resource of the Biome (these
# locations are scattered about the map)." A mine site is discovered the same
# pre-seeded way lumber/quarry sites are (see world.py/Simulation.
# _advance_one_expedition's terrain report); its resource name is read off
# whatever real biome the pre-seeded point itself sits on, across any biome, not
# just mountains. Gated on tribe.quarry_built: excavating a named seam is a
# deeper extension of already knowing how to quarry, not a parallel unrelated
# skill.
MINE_WOOD_COST = 20
MINE_STONE_COST = 30
MINE_YIELD_PER_CYCLE = 5

# GATHER_ORE (backend/actions.py): explicit correction -- "GATHER_ORE only
# comes in if they Discover a Mine. They do not harvest on a Discovery, so
# they have to fetch it once." Unlike Sawmill/Quarry (multipliers on an
# existing manual action), a Mine produces a brand new named resource with no
# manual counterpart at all -- it used to start flowing the instant mine_built
# was set, with no real fetch ever required. Now mirrors fishing_learned's own
# shape exactly: Simulation._advance_mine_yield's passive daily flow doesn't
# start until tribe.ore_ever_gathered is set by a real GATHER_ORE success.
GATHER_ORE_BASE_YIELD = 8

# BUILD_TANNERY (backend/actions.py): explicit request -- "maybe some hunters
# want a Tannery and they can trade furs too." Mirrors _build_mine exactly:
# gated on a real discovered site (a Rabbit Warren, from tribe.wildlife_sites),
# locks in the exact site used, and pays out its named resource ("Fur") into
# the same tribe.unique_resources dict mines already use -- one shared pool,
# not a second parallel resource system.
TANNERY_WOOD_COST = 15
TANNERY_STONE_COST = 15
TANNERY_YIELD_PER_CYCLE = 4
# Explicit request: "it also gives the meat to the kitchen (2 meat per catch)
# which cooks it (multiplier)" -- a flat bonus added to every successful hunt's
# food yield once the Tannery is built (see actions.py._hunt_deer and
# Simulation._report_hunting_party_home), on the theory that a real tannery
# means less of the catch goes to waste.
TANNERY_MEAT_BONUS_PER_HUNT = 2

# BUILD_KITCHEN (backend/actions.py): explicit follow-up -- "we might have to let
# them build a kitchen which improves cooked food to excellent food yielding 3
# per cooked item." Stacks on top of COOKING_FOOD_MULTIPLIER (see actions.
# _food_multiplier) rather than replacing it -- excellent food is 3x as good
# as cooked food, not just 3x raw. Gated on cooking_learned + long_house_built.
KITCHEN_WOOD_COST = 20
KITCHEN_STONE_COST = 10
KITCHEN_FOOD_MULTIPLIER = 3

# BUILD_FORGE/FORGE_ITEM/USE_ITEM (backend/actions.py): explicit request -- a Mine's
# named ore had nowhere real to go once excavated ("we skipped a beat" between mining
# and doing anything with it). Gated on tribe.mine_built plus at least one unit of
# that mine's own resource already in stock ("built after they get 1 Ore"), not a
# separate discovery mechanic like Mine/Tannery's own site-scouting gate. Deliberately
# simple per explicit request: "We do not need to track durability, but they can
# provide value" -- each crafted item just carries a flat, type-based value, no wear.
FORGE_WOOD_COST = 25
FORGE_STONE_COST = 20
FORGE_ITEM_ORE_COST = 1
FORGE_ITEM_WOOD_COST = 10
ITEM_TYPES = ("tool", "weapon", "innovation")
ITEM_NAMES_BY_TYPE = {
    "tool": ("Iron Plow", "Whetstone", "Forged Hoe", "Tempered Chisel"),
    "weapon": ("Iron Spearhead", "Bronze Axe", "Reinforced Bow", "War Hammer"),
    # Explicit steer: flavorful, not sci-fi -- no Joby eVTOLs or spaceships, just
    # small mechanical curiosities a forge could plausibly produce.
    "innovation": ("Geared Wheel", "Pressure Valve", "Tempered Spring", "Balanced Hinge"),
}
ITEM_VALUE_BY_TYPE = {"tool": 8, "weapon": 12, "innovation": 15}

# FORGE_ITEM's own storage cap (backend/actions.py._item_storage_cap): explicit
# follow-up to the passive-income storage-cap fix -- that one closed the gap for
# wood/stone/food/water/unique_resources, but tribe.items (crafted tools/weapons/
# innovations) was left as a plain list with no ceiling at all, the same
# unbounded-hoarding shape STORAGE_CAP_BASE was built to close. A much smaller
# scale than STORAGE_CAP_BASE (150) is deliberate -- each item already represents
# a real investment (FORGE_ITEM_ORE_COST + FORGE_ITEM_WOOD_COST spent per craft),
# unlike a bulk resource; 5 uncashed items sitting around is already a lot.
ITEM_STORAGE_CAP_BASE = 5
ITEM_STORAGE_CAP_PER_WAREHOUSE = 1
# USE_ITEM redeems a crafted item for its value, split across wood/stone -- the
# straightforward cash-out for a value that otherwise just sits on the tribe.
USE_ITEM_STONE_SHARE = 0.5

# Bronze Age counter-offensive (backend/actions.py._strike_raider_camp): a tribe that
# has scouted a raider camp (raider_sightings) can strike it directly once organized
# enough -- turning a warning into an actionable target instead of only ever
# defending. Instant, like RAID, not a multi-day expedition. Win chance is
# population-scaled since the camp itself has no simulated population to compare
# against, unlike tribe-vs-tribe RAID's ratio-based chance.
STRIKE_RAIDER_CAMP_BASE_WIN_CHANCE = 0.5
STRIKE_RAIDER_CAMP_POPULATION_BONUS_PER_10 = 0.03
STRIKE_RAIDER_CAMP_MAX_WIN_CHANCE = 0.85
STRIKE_RAIDER_CAMP_POPULATION_LOSS_ON_FAILURE = 1
STRIKE_RAIDER_CAMP_LOOT_FRACTION = 0.15  # of the tribe's own food, representing recovered supplies

# A tribe can only overhear another tribe's broadcast (and therefore only converge on
# shared vocabulary with them) within this Euclidean distance -- previously broadcasts
# were audible map-wide regardless of distance, which gave away free information and
# removed any incentive to actually travel toward another tribe.
BROADCAST_HEARING_RADIUS = 15

# DECLARE_ALLIANCE/DECLARE_WAR (backend/actions.py._nearest_rival): explicit
# correction -- "they can't make an ALLIANCE if they have not made contact with
# another Tribe or Settlement." Reuses BROADCAST_HEARING_RADIUS's own "close
# enough to exchange real information" distance rather than inventing a second
# number for the same underlying idea.
DIPLOMACY_CONTACT_RADIUS = BROADCAST_HEARING_RADIUS

# Cross-tribe proximity awareness, independent of whether the other tribe has ever
# broadcast anything -- real data this session showed every single run (25/25 tribe-
# reports) ending with zero trades and zero raids. The default two-tribe spawn distance
# is ~62 tiles, well beyond BROADCAST_HEARING_RADIUS, so tribes essentially never had
# any way to become aware of each other's existence at all, let alone converge on the
# same ground. Two tiers, mirroring how you'd actually notice a distant camp: exact
# coordinates only once close (RIVAL_PRECISE_AWARENESS_RADIUS), just a rough compass
# direction -- no coordinates, since you can't see exact GPS from that far -- out to
# RIVAL_DISTANT_SIGHTING_RADIUS, set above the real default spawn distance so a fresh
# two-tribe game is aware of the other from cycle one instead of remaining permanently
# blind to a rival that's simply never going to wander within 15 tiles by chance.
RIVAL_PRECISE_AWARENESS_RADIUS = 20
RIVAL_DISTANT_SIGHTING_RADIUS = 70

# Worn trails (backend/world.py, backend/actions.py, backend/simulation.py's
# _advance_expedition): the inverse of resource depletion. Repeatedly relocating or
# scouting through the same tile wears a trail there, which speeds up both RELOCATE and
# expedition travel through it later -- rewarding a tribe for reusing a route it (or
# another tribe) already traveled, rather than every journey being an equally slow trek
# through untouched ground. Low wear-per-pass and slow decay are deliberate: a single
# trip barely matters (0.03 wear -> ~0.09 speed at MAX_TRAIL_BONUS_SPEED=3), but a route
# used repeatedly compounds into a real shortcut, and durably so -- a destination just
# out of one expedition's EXPEDITION_MAX_DAYS reach can become reachable a few attempts
# later along the same path, without any distance rule being overridden.
# At the original 0.03/0.002 pair, a single unreused pass fully decayed in
# 0.03/0.002 = 15 cycles -- maybe 30-60 seconds of real time, so a viewer could watch a
# trail form and vanish again before ever really registering it existed. Decay is now
# zero: a trail, once worn, is permanent infrastructure -- it only ever gets more worn
# (and faster) with reuse, never fades on its own.
TRAIL_WEAR_PER_PASS = 0.03
TRAIL_DECAY_PER_CYCLE = 0.0
MAX_TRAIL_BONUS_SPEED = 3  # added to MOVEMENT_SPEED/EXPEDITION_SPEED at full wear

# Scouting expeditions (backend/actions.py._scout, backend/simulation.py._advance_expeditions).
# A small, self-sufficient party travels out looking for water or distant terrain, turning
# back the moment they succeed or after EXPEDITION_MAX_DAYS with nothing -- either way, the
# finding (if any) isn't real, actionable knowledge for the tribe until they've walked all
# the way home. Faster than a full RELOCATE since it's a handful of unburdened people, not
# the whole camp and its belongings.
# Movement speed scales by whatever terrain is currently being crossed -- previously
# every RELOCATE/expedition step was a pure straight-line vector toward the target with
# zero awareness of what lay in between, so a mountain range or a river crossing cost
# exactly the same as open plains. Ocean gets a multiplier of 0.0, which
# physics.terrain_aware_step treats as genuinely impassable (no boats exist yet in this
# Stone Age simulation) and deflects around along a single axis, rather than just being
# slow -- the one real "obstacle" in the world right now.
TERRAIN_MOVEMENT_MULTIPLIER = {
    "plains": 1.0,
    "forest": 0.8,
    "mountains": 0.4,
    "river": 0.3,
    "ocean": 0.0,
}

# Boat (Simulation._advance_automatic_boat, backend/physics.py.terrain_aware_step):
# explicit request, automatic like fire once a Dock stands and fishing is
# mastered -- "give the boat mobility in the clean water, not the sea." River is
# normally the slowest passable terrain (0.3x above); a boat turns it into a real
# advantage over dry land instead of an obstacle. Ocean stays exactly as
# impassable as ever -- this is deliberately NOT an ocean-crossing mechanic.
BOAT_WATER_BIOMES = {"river", "lake"}
BOAT_WATER_MOVEMENT_MULTIPLIER = 1.2

EXPEDITION_SPEED = 10
EXPEDITION_MAX_DAYS = 3

# A tribe could previously only ever have one party (scouting or hunting) in the field
# at a time -- a chief with real people to spare had no way to send out more than a
# single expedition regardless of population. Capped rather than unlimited: nothing
# currently deducts population to launch a party, so an uncapped tribe could spam
# expeditions for free.
MAX_CONCURRENT_EXPEDITIONS = 2

# actions.expedition_capacity() lets a larger tribe spare more search parties at once --
# MAX_CONCURRENT_EXPEDITIONS above is only ever the floor now, not a hard ceiling. A
# tribe of 8 (the starting population) still gets exactly 2; a tribe of 20+ can spare
# more. Real per-capita capacity, the same way upkeep already scales with population
# (see Simulation._apply_upkeep) -- population growth used to buy a tribe nothing on
# the scouting/hunting side no matter how large it got.
EXPEDITION_SLOT_POPULATION_DIVISOR = 5

# actions._labor_multiplier() lets a larger tribe gather/hunt/forage more per action --
# upkeep (_apply_upkeep) already scales with population, but yield from GATHER_WOOD/
# STONE/WATER/FOOD and HUNT_DEER never did, so a bigger tribe was strictly worse off
# per-capita: identical output, more mouths to feed. POPULATION_YIELD_BASELINE matches
# Tribe.__init__'s own starting population, so a tribe at or below starting size sees no
# change at all -- this only ever rewards growth past it, never penalizes a small tribe.
POPULATION_YIELD_BASELINE = 8

# Every expedition's lead scout gets a procedurally-generated determination trait (see
# actions.py._generate_scout) that shifts their own personal give-up point by up to
# this many days either side of EXPEDITION_MAX_DAYS -- a stubborn scout searches a
# little longer before turning back, a cautious one a little less. Not a second LLM
# agent making its own choices, just per-expedition character instead of every party
# behaving identically.
EXPEDITION_DETERMINATION_DAY_VARIANCE = 1

# A traveling party forages and hunts along the way rather than being a pure resource
# black hole -- more on the outbound leg (fresh, unpicked ground, no urgency yet), less
# on the way back (already-passed terrain, hurrying home with news). Delivered to the
# tribe's stockpile only on arrival home, same as the water/terrain finding itself --
# still self-sufficient enough not to starve in the field, but not free income either.
EXPEDITION_OUTBOUND_DAILY_FOOD = 3
EXPEDITION_OUTBOUND_DAILY_WATER = 2
EXPEDITION_RETURN_DAILY_FOOD = 1
EXPEDITION_RETURN_DAILY_WATER = 1

# EXPLORATION_PARTY (backend/actions.py._exploration_party, Simulation.
# _advance_exploration_party_outbound): explicit request -- "a smart Chief
# will send one Scout and one Exploration Party... anything out there can be
# discovered including settlements, raider camps, ocean, whatever they
# find... leave Landmarks (with a reason to go there)... some limits on how
# long they can stay out and how much they can carry." Where SCOUT is a fast,
# discovery-only dash, an Exploration Party is a deeper, deliberate trip:
# real wood/stone gathered along the way (on top of the food/water every
# expedition already forages), a real carrying-capacity limit (not just a day
# count), and a chance at spotting a rival settlement or a Landmark. It shares
# everything SCOUT's own return already discovers (water, resource sites,
# raider camps) via Simulation._advance_one_expedition's shared fallthrough --
# this only adds what SCOUT doesn't.
EXPLORATION_PARTY_MAX_DAYS = 6
EXPLORATION_PARTY_DAILY_WOOD = 3
EXPLORATION_PARTY_DAILY_STONE = 3
EXPLORATION_PARTY_CARRY_CAPACITY = 60  # combined wood+stone+food+water before forced return
SETTLEMENT_SIGHTING_RADIUS = 12

# A Landmark is a real, persistent point of interest (drawn on the map like
# any other site) with its own one-time reward -- a fun, flavorful unique
# resource, deliberately not ore (Mine already owns that niche). The place
# name and the thing found there are separate on purpose: you don't carry
# home "12 units of a sacred spring," you carry home a relic found near one.
LANDMARK_DISCOVERY_CHANCE = 0.08  # per outbound day
LANDMARK_REWARD_MIN = 10
LANDMARK_REWARD_MAX = 25
LANDMARK_NAMES = (
    "Ancient Grove", "Sacred Spring", "Sunken Idol", "Whispering Stones",
    "Old Watchtower", "Hidden Falls", "Standing Stones", "Forgotten Shrine",
)
LANDMARK_RESOURCE_NAMES = (
    "Amber Charm", "Carved Totem", "Silver Trinket", "Bone Flute",
    "Painted Shell", "Gilded Feather", "Polished Stone", "Woven Talisman",
)

# Action-repetition throttle (Simulation._apply_turn/_prepare_turn): explicit
# request, after a live run showed one tribe choose GATHER_STONE on 49% of all
# 728 turns (and a different run's tribe choose BREED on 63.8%) while other real
# needs went untouched -- the same "models fixate on one verb regardless of
# payoff" pattern this project already documented for GATHER_FOOD/GATHER_WOOD.
# Once an action has been chosen this many cycles in a row, it's pulled from
# available_actions for a cooldown, forcing a genuinely different choice.
# RELOCATE is exempt (Simulation._apply_turn) -- a real, sustained multi-cycle
# journey is documented, desired behavior (see README), not fixation.
# Threshold=4 is the minimum that still lets CONSTRUCT_WALL finish building one
# full section (WALL_PROGRESS_PER_ACTION_BASE=30 -- 30/60/90/100, exactly 4
# actions at baseline population) before ever throttling -- reinforcing a
# section to WALL_MAX_LAYERS afterward can still eat one cooldown, a smaller,
# accepted cost since that only gates Moat/Torches, not core progression.
ACTION_REPETITION_THROTTLE_THRESHOLD = 4
ACTION_REPETITION_THROTTLE_COOLDOWN = 7
