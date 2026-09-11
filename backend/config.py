OLLAMA_URL = "http://localhost:11434"

TICK_SECONDS = 0.5

GRID_SIZE = 100
MAX_TRIBES = 4

# Simulation.debug_snapshot / Tribe.debug_transcript: explicit request,
# 2026-09-09 -- "a separate page that shows me, in a 4 column format, live,
# what we tell the llm, how it responses... I want to see the raw info we
# sent and how they reply." How many recent turns of raw prompt/response
# each tribe keeps for that live debug view -- a capped deque, not the full
# run's history, so this can't grow unbounded over a long run. Called out
# explicitly as a number worth raising later if 20 turns of scrollback isn't
# enough.
DEBUG_TRANSCRIPT_HISTORY_LIMIT = 20

# Map dream, phase 2: "another attempt at increasing the Ocean/unplayable
# area" -- a real island, ocean wrapping the north/south/west edges too, not
# just the existing east coast (world._coast_boundary_x). Same "fixed inset +
# two sine waves" shape as every other wavy boundary in world.py.
#
# First attempt at this used base=18 with the same +-7 wave amplitude every
# other boundary in this file uses -- explicit correction: "the Ocean was only
# supposed to be a 'frame'." That depth ate into the mountains, forced the
# volcano and every spawn point to relocate, and produced narrow sandbar-like
# spits of land near the corners where two deep insets overlapped. This base
# (6) is paired with a much smaller wave amplitude in world.py's own
# _west/_north/_south_coast_boundary (+-2, not +-7) so the real reach of the
# frame stays within the requested 4-8 tiles everywhere, not just at its
# calmest point.
WEST_COAST_INSET_BASE = 6
NORTH_COAST_INSET_BASE = 6
SOUTH_COAST_INSET_BASE = 6

# Natural river/lake rework, phase 1: "I'd love the river and lake to look
# better and more natural." backend/world_hydrology_data.py is a ONE-TIME baked
# tile set (see scripts/generate_hydrology.py), not something world.py computes
# per call -- this seed is what that offline generator used, kept here so it's
# reproducible and so a future map-variety phase has an obvious place to thread
# a per-run seed through instead of this fixed constant (not implemented yet).
HYDROLOGY_SEED = 20260906

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

# Tiles moved per axis per cycle toward target_vector, before a tribe has ever
# settled. At 1 (the original value), crossing the 100-tile grid takes 100+
# cycles minimum -- at roughly 2 seconds of real inference time per cycle, a
# "good" decision to travel far was still visually imperceptible for minutes.
# This doesn't change AI decision quality, just how fast a given decision reads
# on screen. Kept fast specifically for the pre-settlement march to newly-
# confirmed water -- see SETTLED_MOVEMENT_SPEED below for the slower, uniform
# pace that applies once a tribe has actually settled.
#
# Explicit request ("improve the Tribe movement speed to the Scout speed
# pre-settlement"), after has_ever_settled was tightened to require real
# confirmed water: once water IS found, the whole tribe's own march there
# shouldn't be slower than the scout that found it. Matches EXPEDITION_SPEED
# (10, below) directly -- kept as a literal since EXPEDITION_SPEED is defined
# later in this file; keep the two in sync if either is retuned.
MOVEMENT_SPEED = 10

# Explicit request ("everyone that is moving on the board moves at the pace of
# 1 sky tick"): once a tribe has settled, RELOCATE (and, via
# SETTLED_EXPEDITION_SPEED below, a hunting or exploration party) advances at
# this one uniform per-cycle pace instead of MOVEMENT_SPEED/EXPEDITION_SPEED's
# larger, once-a-day-feeling jumps -- movement should read as continuous and
# consistent, not different speeds for different standing orders. Deliberately
# NOT applied before a tribe has ever settled: the confirmed-water march that
# ends a pre-founding tribe's search is life-or-death (see the founding-death
# regression MOVEMENT_SPEED/EXPEDITION_SPEED's own pre-settlement speed
# already exists to prevent), so that march keeps the faster pace. A plain
# SCOUT is also exempt regardless of settlement -- see EXPEDITION_SPEED's own
# comment below.
SETTLED_MOVEMENT_SPEED = 1

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
NIGHT_CYCLE_REVIEWER_MODEL = "phi4-mini:latest"
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

# Explicit request, 2026-09-08: "Fire should require > 10 wood so it's a touch
# harder but still necessary." Was a bare hardcoded 10 in actions._build_fire
# (and mirrored in simulation.AFFORDABILITY_CHECKS) -- a real named constant now,
# same as every other action's own cost.
BUILD_FIRE_WOOD_COST = 15

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

# Live bug report: "never landed a clean hunt? that's very intolerant." A real
# 346-cycle run showed every single HUNTING_PARTY dispatch come home empty --
# BIOME_YIELD_MULTIPLIER["game"] drops as low as 0.05 in cliffs/desert (0.35 *
# 0.05 = a 1.75% catch chance per day roll, only HUNTING_PARTY_MAX_DAYS rolls
# before giving up), so a party unlucky enough to keep landing in poor game
# terrain could plausibly never catch anything across many dispatches. Floors
# any already-nonzero game multiplier up to this value before the catch roll
# -- true zero-game biomes (ocean, volcano) stay genuinely un-huntable, this
# only rescues the biomes where hunting is nominally possible but was rolling
# at next to nothing.
HUNTING_PARTY_MIN_GAME_MULTIPLIER = 0.15
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

# Explicit request: "every water source a Tribe finds, adds to the passive
# Water income, so they should just get an Infinity sign for water once they
# find 2 or 3." Checked against SETTLED_WATER_SUPPLY_MULTIPLIER's own formula
# and confirmed_water_sites was never actually part of it -- passive income
# only ever scaled with upkeep/farm_draw/well_built, so a second or third
# confirmed source genuinely did nothing extra, and a real live run (day 12,
# population climbing into the thousands) showed water declining for 100+
# straight cycles despite multiple confirmed sources on record. Real state
# now: a tribe that's confirmed this many distinct sources gets water
# entirely off the management board for good -- same "permanent one-way
# mastery" shape fishing_learned/cooking_learned already give their own
# resource, and naturally one-way already since confirmed_water_sites only
# ever grows (Tribe.confirmed_water_sites.append, never removed), so no
# separate flag is needed to make this stick.
WATER_SECURITY_SITE_THRESHOLD = 3

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
# Explicit request: "much bigger populations going to war" -- growth used to be a
# flat +1/cycle regardless of tribe size (confirmed against real run data: every
# model tested grew at ~1 population/cycle, no exceptions), which made a
# late-era population requirement a pure real-time tax with nothing to actually
# play around -- reaching population 1500 at that flat rate would mean ~1500
# additional cycles, several hours of continued Ollama inference for no new
# decisions. Growth now scales with the tribe's own current size (same shape
# actions.expedition_capacity already uses for a different stat), so a small
# tribe grows at essentially today's pace (population 8 // 20 still floors to
# the same +1) while a large one compounds -- population 1500 is reached in
# roughly 111 cycles instead of ~1500, real growth still gated behind the same
# food surplus, just no longer flat once a civilization is actually large.
POPULATION_GROWTH_SCALE_DIVISOR = 20
# Explicit follow-up: "can we use an actual population growth model based on
# Well-Being?" -- the scaled base above only ever reflected raw tribe size, with
# nothing to say about whether that population is actually thriving. wellbeing.
# compute_wellbeing's physiological tier (food/water buffer -- already computed
# every turn, already reaches the tribe's own prompt as a fact) is a real,
# already-tracked signal for exactly that.
#
# Live bug, confirmed against a real run: this used to average all five Maslow
# tiers with a floor (so growth could slow but never truly stop) -- esteem/
# self_actualization (trophies, era progress) have nothing to do with feeding
# more mouths, and kept the average propped up even while physiological sat at
# 0.0, so a tribe in total, sustained famine still grew at ~90% of full speed.
# Confirmed live: population 129 -> 10,835 in ~110 cycles while food never
# recovered -- a floored multiplier on a population-proportional base can only
# ever slow down, never reach zero, so it's unbounded by construction no
# matter how low the floor is. Keyed on physiological alone now, no floor: a
# real, sustained famine brings growth to an honest zero, same as every other
# food-gated system here already can. See Simulation._grow_population's own
# docstring for the full trace. Not a replacement for the population-scaled
# base above (self_actualization used to be part of why a pure wellbeing-only
# model risked a struggling tribe never reaching a late population target at
# all -- moot now that it's physiological-only, but the scaled base is still
# what lets a THRIVING tribe's growth compound with its own size) -- a
# multiplier on it instead. A perfectly-fed tribe (physiological == 1.0) grows
# at base_growth * this multiplier -- kept above 1x so a genuinely thriving
# tribe still grows faster than the old flat +1/cycle ever did, not just
# avoids the famine case above.
POPULATION_GROWTH_WELLBEING_MAX_MULTIPLIER = 2.0
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
# Explicit request (2026-09-05): "make it cost 25 wood for a simple fence and
# scarecrow, it's common sense" -- a real plot needs more than seed money to
# keep wildlife off it while it grows.
PLANT_CROP_WOOD_COST = 25
MAX_FARM_PLOTS = 4
CROP_GROWTH_PER_CYCLE = 10  # a plot matures in ~10 cycles once planted
CROP_HARVEST_YIELD = 15  # food per plot, per harvest
CROP_WATER_PER_PLOT_PER_CYCLE = 2  # a plot that goes unwatered withers outright

# Explicit correction, after a live 986-cycle run: "we have scaled the Farm a
# bit hard. We can back it down and let them react." A harvest applies
# actions._labor_multiplier (population / POPULATION_YIELD_BASELINE=8)
# uncapped, same as every gather action -- but a gather action is throttled by
# the model actually choosing to do it repeatedly, while a harvest fires
# automatically the moment crop_growth hits 100, no choice involved. At the
# run's real peak populations (530 and 887), that multiplier reached 66x-111x,
# so CROP_HARVEST_YIELD(15) * up to MAX_FARM_PLOTS(4) plots produced single
# harvests of several thousand food against a few-hundred-food storage cap --
# almost all of it wasted every single time, confirmed live (267 "stores
# nearly full" harvest-waste events in that one run). Capped well below
# gather actions' own effectively-unbounded scaling so growth still helps
# (up to a real 5x boost) without guaranteeing overflow -- a tribe now has to
# actually build Warehouses/manage storage to capture a big harvest, not
# just always lose most of it.
FARM_LABOR_MULTIPLIER_CAP = 5.0

UNBUILDABLE_BIOMES = ("ocean", "river", "lake", "cliffs", "shoals", "volcano")

# Map dream, phase 1 (user's own sketch): a new Desert biome in the south of the
# map, harsh but real -- same "buildable but not farmable, slow and low-yield"
# treatment mountains already gets (see TERRAIN_MOVEMENT_MULTIPLIER/
# BIOME_YIELD_MULTIPLIER below), not just a recolor. world._desert_north_boundary
# uses this as its base, the same "fixed constant + two sine waves" shape every
# other wavy zone boundary in world.py already follows.
DESERT_NORTH_BOUNDARY_BASE = 78

# The volcano is a real hazard, not decoration -- explicit correction: "the
# volcano is a Hazard they will die if they go there." "Inactive" describes its
# look (dormant, not erupting on screen), not its danger -- a real inactive
# volcano still kills via toxic gas/heat/unstable ground. Chance set far above
# DROWNING_HAZARD_CHANCE (0.08) -- this needs to read as a serious, well-known
# danger, not a mild river crossing. See Simulation._volcano_hazard.
#
# Explicit design correction: "the way we handle hazard is a scenario. Any
# party goes out, they discover a hazard, 1 person in the party dies, they
# run home to report the death and the hazard area." POPULATION_LOSS used to
# be 5 here -- the one environmental hazard out of line with every sibling
# (drowning/cliffs/ocean/wolf-pack/raider-ambush all cost exactly 1). Traced
# live: a young, pre-settlement tribe (population 13, mid-relocation, 8/10
# cycles into settling) took three volcano hits in three consecutive cycles
# from a single unlucky scout and was wiped out outright -- a flat -5 against
# a tribe that size isn't "one person," it's near-instant extinction. Matches
# every other hazard now.
VOLCANO_HAZARD_CHANCE = 0.75
VOLCANO_HAZARD_POPULATION_LOSS = 1
VOLCANO_TRAUMA_MAGNITUDE = -0.6  # more severe dread than DROWNING_TRAUMA_MAGNITUDE (-0.4)
VOLCANO_TRAUMA_RADIUS = 8  # wider than DROWNING_TRAUMA_RADIUS (6) -- a bigger, more memorable disaster

# Explicit request (2026-09-06): "we need to add back the Cliffs hazard or
# those Scouts stay there." Cliffs form the whole coastal ring around the
# island (unlike the volcano's small, localized danger zone) and were fully
# passable with zero consequence -- a scout patrolling that direction simply
# parked there forever with nothing to gain and nothing to fear. Moderate,
# not volcano-severe: this is routine coastal terrain a scout crosses often,
# not a rare landmark to avoid outright -- real risk for lingering, not a
# near-certain death sentence for going anywhere near the coastline.
CLIFFS_HAZARD_CHANCE = 0.2
CLIFFS_HAZARD_POPULATION_LOSS = 1
CLIFFS_TRAUMA_MAGNITUDE = -0.4
CLIFFS_TRAUMA_RADIUS = 6

# Explicit spec (2026-09-06): "Ocean is instant kill 1, report, gravemarker."
# physics.terrain_aware_step already deflects ordinary movement around ocean
# (it's the one biome real navigation treats as impassable) -- this is a
# safety net for the known edge case where a reflected/overshot target can
# still land exactly on one (see physics.reflect_into_grid's own docstring),
# not a routine occurrence. Certain rather than a rolled chance, matching
# "instant" -- see Simulation._ocean_hazard.
OCEAN_HAZARD_POPULATION_LOSS = 1

# Explicit design correction: "volcano, cliff, beach, ocean should all have
# the same treatment." "Beach" is this map's shoals biome (world.py's
# BIOME_LABELS -- "The Glass Shallows"), the flat, sandy counterpart to
# cliffs along the same coastline (world.biome_at picks one or the other per
# coastal tile). Had no hazard at all before this -- a scout could sit on it
# indefinitely with nothing to fear, the same gap cliffs itself used to have.
# Chance matches CLIFFS_HAZARD_CHANCE for now (same "routine coastal terrain,
# real risk for lingering, not a near-certain death sentence" reasoning,
# rather than volcano-severe or ocean-certain) -- a first pass, not tuned
# against live data yet. See Simulation._shoals_hazard.
SHOALS_HAZARD_CHANCE = 0.2
SHOALS_HAZARD_POPULATION_LOSS = 1
SHOALS_TRAUMA_MAGNITUDE = -0.4
SHOALS_TRAUMA_RADIUS = 6

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

# Explicit request (2026-09-06): "we need to eliminate a 3 ring and begin to
# reinforce them with Concrete... 2 rings is enough. they will have to build
# outside the walls once they hit that point." Wall expansion (CONSTRUCT_WALL's
# expansion fallback since the 2026-09-08 EXPAND_TERRITORY merge) used to open a
# whole new ring, unconditionally, every time the outermost was fully
# reinforced -- no ceiling beyond raw land availability, confirmed live (a
# real run built well past 2). Once a tribe already has this many rings, all
# fully reinforced, CONSTRUCT_WALL retires from the choice set for good, the
# same one-way "generalist narrows once its job is done" shape BUILD_FIRE/
# COOK_FOOD already use -- a real ceiling instead of an unbounded ratchet.
MAX_WALL_RINGS = 2

# Lowered back to 1 (explicit correction, 2026-09-06): "with 2 built, they
# feel too safe even in an open field." Briefly raised to 3 the night before
# after a settling tile with 3 natural barriers looked good on inspection --
# but watching a full run through to its era ceiling showed 2+ free natural
# wall sections reads as too much passive safety, not "good, defensible
# ground." Back to the original bar: only a single freebie natural-barrier
# section is tolerated before _choose_territory_center goes looking for
# somewhere less water-dominated.
TERRITORY_MAX_ACCEPTABLE_NATURAL_BARRIERS = 1

# CONSTRUCT_WALL's expansion fallback unlocks exactly one new wall section per
# call, in fixed compass order -- "expansion must be done for each wall section,"
# no exception for ring 0.
# tribe.territory_radius (see actions._expand_wall_territory) is always derived as
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
    "object_creator": (3, 3), "created_structure": (2, 2),
    "barracks": (3, 3),
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
# Explicit request: "Esteem is scaled wrong" -- trophies alone (min(1.0, count/5))
# didn't distinguish a tribe that just won a minor distinction from one that
# raised a real monument. A weighted point total instead, one line per real
# achievement -- a trophy is common and worth little on its own; a completed
# Keep or Castle is a substantial, rare undertaking and worth much more.
# ESTEEM_POINTS_PER_SPACE_STATION is a placeholder for a real building this
# project doesn't have yet (a future rockets/interplanetary-exploration era,
# per an explicit "harder to get to rockets or a new map, not impossible, but
# a real climb" design goal) -- the number is banked now, at the user's own
# specified value, so nothing needs renumbering once that content exists, but
# nothing currently sets tribe.space_stations_built or reads this constant.
ESTEEM_POINTS_PER_TROPHY = 1
ESTEEM_POINTS_PER_TOLL_ROAD = 5
ESTEEM_POINTS_PER_KEEP = 10
ESTEEM_POINTS_PER_CASTLE = 20
ESTEEM_POINTS_PER_SPACE_STATION = 50  # not wired to anything yet -- see comment above
# Matches FAME_SCORE_REFERENCE's own value/reasoning below: high enough that
# esteem keeps climbing across a whole run instead of saturating on the first
# few trophies, low enough that a tribe that actually raises real monuments
# (not just trophies) can still reach it before the run ends. Will likely need
# retuning once Space Station (or any other high-value future structure)
# actually exists and real runs show where tribes land.
ESTEEM_SCORE_REFERENCE = 100

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

# BUILD_WELL (backend/actions.py, Simulation._advance_water_supply/
# _is_water_secure): explicit request -- water's only passive-income lever was
# a single flat formula tied to population, with no equivalent of Fishery/
# Dock's stacking bonus for food. Same "infrastructure from the moment it's
# unlocked" shape Bath House/Warehouse already use (no proven-success gate).
#
# Explicit correction, 2026-09-09: "Tribe 2 built a Well which should have
# gotten them to the infinity Water." A Well used to only stack a
# WELL_SUPPLY_BONUS_MULTIPLIER (retired) onto the passive formula, never
# granting full water security the way Kitchen+a proven source does for food
# -- now well_built is its own sufficient path to _is_water_secure, same as
# every other resource's building-based mastery route.
WELL_WOOD_COST = 20
WELL_STONE_COST = 20

EGGS_LAID_PER_FLOCK_PER_CYCLE_DIVISOR = 5  # 1 egg per 5 flock members per cycle
# Live report: "crazy villagers" eating the whole flock/every egg the moment
# either crossed a dozen -- this was a flat cap regardless of tribe size, tuned
# back when population sat around 20-50 (a dozen was a real fraction of that).
# It never got revisited once population scaling changed elsewhere (expedition
# capacity, upkeep, growth all already scale with population -- this was the
# one that didn't), so a tribe of any size was still permanently capped at the
# same dozen, converting all real herd growth into an immediate meal instead of
# a bigger, more valuable flock. LIVESTOCK_SURPLUS_THRESHOLD is now the floor
# (still exactly "a dozen" for a small tribe), not the ceiling -- see
# Simulation._livestock_surplus_threshold, same max(floor, population-scaled)
# shape actions.expedition_capacity already uses for a different stat.
LIVESTOCK_SURPLUS_THRESHOLD = 12
# Live report ("eggs and fowl aren't contributing like they should"): confirmed
# via a live run (run_20260908_112524) -- both tribes' eggs sat exactly at their
# own threshold (164/164 and 190/190), trimmed back to it every time they poked
# over, and flock itself (21-22) was nowhere near its own threshold and never
# could be. //10 made the threshold outrun what the mechanic can actually
# produce: flock only grows +1 per successful natural hatch (FLOCK_NATURAL_
# HATCH_CHANCE, 15%/cycle) and actively shrinks under food pressure, so it
# can't keep pace with a threshold scaling linearly with population -- at
# population 5000 that's a threshold of 500 against a flock that realistically
# tops out in the 20s-40s, making FLOCK_FEAST_FOOD_VALUE's payoff (4x eggs')
# functionally dead past a few hundred population, most of the game. //100 is
# a 10x gentler curve -- population 5000 now yields a threshold of 50, within
# reach of a real flock/eggs stockpile, while a large tribe still gets a real
# ceiling above the flat floor, the original point of this scaling.
LIVESTOCK_SURPLUS_POPULATION_DIVISOR = 100
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
# uses. Explicit correction (2026-09-06): "Fame is scaled too high" -- at the
# original 12, "satisfied" only took 6 celebrations' worth, which a live
# 756-cycle (38-day) run blew past almost immediately (real celebrations fire
# far more often than the original estimate assumed); tribe.fame ended that
# run at 104-233, meaning the score had been pinned at 1.0 for nearly the
# entire run instead of "accumulating over a whole run" as originally
# intended. Raised to roughly match a real full run's own endgame fame, so
# the score keeps climbing across the run instead of saturating in the first
# day or two.
FAME_SCORE_REFERENCE = 100
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

# Military branch, step 1 (plan file valiant-forging-falcon.md) -- redesigned
# 2026-09-10. Explicit correction: "Name_Warrior is wrong. We already name
# Warriors when they get Trophies. If they have one they can let it lead a
# battalion. if they have more than 1, they can have many Battallions." The
# original design required a separate, chief-chosen NAME_WARRIOR action and a
# steep 3-trophy threshold -- confirmed via two independent live runs that
# this whole chain (NAME_WARRIOR/BUILD_BARRACKS/TRAIN_BATTALION) had never
# fired together, and a later run showed an eligible individual sitting
# unappointed for 38+ days simply because nothing ever forced the chief to
# spend a turn on it. Naming is now automatic and implicit the moment a
# personally-credited trophy exists (actions._eligible_new_battalion_leader)
# -- no action, no chief decision, no NAME_WARRIOR. Threshold dropped to 1:
# _award_trophy still only pays out each named trophy type once per tribe's
# entire lifetime, so this remains a real, earned distinction, just no longer
# an artificially rare one gating the whole branch behind it.
BATTALION_LEADER_TROPHY_THRESHOLD = 1
# "if they have a lot, we need some restrictions" -- a tribe can field this
# many concurrently-led Battalions at once (actions._allocate_battalion_
# strength). A flat, invented first-pass default (SECOND-OPINION(Sonnet 5,
# 2026-09-10): a real leadership hierarchy -- e.g. a General over several
# Battalion leaders once a tribe has "a lot" of eligible individuals -- was
# also on the table and explicitly left for a later pass per Scott's own
# "possibly leadership tiering/structure" hedge; this flat cap is the
# smaller, easily-retuned v1), not tuned against live data yet -- revisit if
# a run shows tribes routinely maxing this out with many more eligible
# individuals going unled.
MAX_CONCURRENT_BATTALIONS = 3

# Military branch, step 2: BUILD_BARRACKS. Repeatable, like BUILD_WAREHOUSE --
# each one raises how large a Battalion TRAIN_BATTALION can ever train
# (BATTALION_CAPACITY_PER_BARRACKS per Barracks), the same "repeatable
# building raises a real cap" shape Warehouse already uses for storage,
# rather than a population-fraction formula -- army size stays visibly tied
# to something the player actually built. Gated on tribe.keep_built (not just
# affordability) -- continues the existing Wall -> Long House -> Keep ->
# Fortress/Castle defensive ladder rather than sitting unconnected to it.
# Costs invented defaults, priced near Keep's own (30/35) since Barracks is
# the next real investment past it, not tuned against live data yet --
# revisit if a real run shows tribes never actually building this, the same
# way Sawmill/Quarry/Mine costs were once flagged as unvalidated guesses.
BARRACKS_WOOD_COST = 25
BARRACKS_STONE_COST = 25
BATTALION_CAPACITY_PER_BARRACKS = 20

# Explicit request, 2026-09-10: "Looks like we have to scale Barracks like we
# have done with Warehouse, among others." Same real problem, same fix shape:
# a flat-cost, uncapped repeatable BUILD_* let a tribe build 15-57 of these in
# a single run (confirmed live, the same run that surfaced the Warrior
# redesign above). Real BUILD_BARRACKS ceiling; UPGRADE_BARRACKS (backend/
# actions.py._upgrade_barracks) takes over past this point, with an
# escalating cost (BARRACKS_UPGRADE_COST_GROWTH per tier, tribe.
# barracks_upgrades) so it doesn't become the same infinite-spam problem
# under a new name -- mirrors WAREHOUSE_MAX_COUNT/UPGRADE_WAREHOUSE
# (this file, further down) exactly.
BARRACKS_MAX_COUNT = 5
BARRACKS_UPGRADE_WOOD_COST_BASE = 35
BARRACKS_UPGRADE_STONE_COST_BASE = 35
BARRACKS_UPGRADE_COST_GROWTH = 0.5

# Military branch, step 3: TRAIN_BATTALION. Staged, like CONSTRUCT_WALL --
# "built up over several turns, more with more people" -- rather than a
# one-shot flip; a standing force materializing in one action reads as too
# cheap. Reuses actions._labor_multiplier the same way CONSTRUCT_WALL's own
# progress-per-action does, so a larger tribe trains faster. At the starting
# population (labor multiplier 1.0), one action trains ~5 soldiers -- a full
# single-Barracks capacity (20) in ~4 actions, matching how many actions
# CONSTRUCT_WALL's own base value takes to finish one section. Costs real
# food per soldier trained (feeding real people, the same "a real resource
# cost, not just wood/stone" shape BREED already uses), not wood/stone --
# Barracks itself already paid the building cost.
BATTALION_TRAINING_PER_ACTION_BASE = 5
BATTALION_TRAINING_FOOD_COST_PER_SOLDIER = 2

# Explicit confirmation, 2026-09-10, resolving the "how independent should
# multiple Battalions be" fork from the same-day Warrior redesign: "my idea
# for simplification presently, they can have a Tribe readiness pool and 1
# patrol at a time... just make the variables so we can increase those
# patrols, if we need." Named here so it's a real, deliberate number instead
# of an implicit fact of tribe.battalion_patrol being a single dict rather
# than a list -- raising this later would mean converting battalion_patrol
# into a list of that many concurrent patrol dicts (a real structural
# change, not just bumping this constant), so it stays at 1 until that's
# actually built.
MAX_CONCURRENT_PATROLS = 1

# Military branch, step 4: the autonomous patrol -- explicit request, "I do
# want to add it as a visual player on the board not just a passthru." A
# real, moving entity (Simulation._advance_battalion_patrol, tribe.
# battalion_patrol), not a hidden formula tweak -- kept entirely separate
# from tribe.expeditions (a Battalion isn't chief-dispatched and must never
# compete with SCOUT/HUNTING_PARTY for expedition_capacity's own limited
# slots). Explicit follow-up: "they can patrol for a set number of cycles,
# then go back to training. Cooldown for 3 whole days" -- a real cycle, not a
# standing-forever presence: BATTALION_PATROL_DURATION_DAYS out, then
# BATTALION_PATROL_COOLDOWN_DAYS resting/training (tribe.
# battalion_cooldown_until_cycle) before the next patrol can start.
# "cooldown on patrol, training has its own controls" -- this cooldown only
# ever gates a new patrol from starting; TRAIN_BATTALION's own affordability
# gate (BATTALION_TRAINING_FOOD_COST_PER_SOLDIER etc. above) is completely
# separate and untouched by this.
BATTALION_PATROL_DURATION_DAYS = 1
BATTALION_PATROL_COOLDOWN_DAYS = 3
# Moves every cycle while patrolling, not gated to once-per-day the way a
# settled expedition's own movement is -- a 1-day patrol duration would only
# ever take a single step under that gating, nowhere near enough to reach a
# real target. Same baseline as EXPEDITION_SPEED.
BATTALION_PATROL_SPEED = 10

# Military branch, step 5: Might (plan file valiant-forging-falcon.md,
# compute_might). Explicit request, 2026-09-08: "Might now should include a
# Training factor (not overpowered, more like bolster and upkeep)." tribe.
# battalion_readiness (0.0-1.0) is a real, maintained meter, not a one-time
# flip -- it has to be earned AND kept up, the same "sustained investment,
# not fire-and-forget" shape BATTALION_PATROL's own cooldown already gives
# the patrol side of this branch. Bolster: actions._train_battalion nudges
# it up a little every time it's called -- recruiting new soldiers while
# still under capacity, or (new) a cheaper maintenance drill once the
# Battalion is already at full headcount, so there's always something worth
# doing to keep readiness from draining even after "done" training. Upkeep:
# Simulation._advance_battalion_readiness_upkeep drains it a small fixed
# amount every cycle regardless -- a standing force goes stale without
# continued drilling, it isn't just recruited once and forgotten.
# Explicit request, 2026-09-10: "they need some bounds around training Might.
# Say 3 rounds gives them 100% Might (we have build in degradation)." Was
# 0.15 (~7 calls to reach full readiness) -- raised so 3 real TRAIN_BATTALION
# calls reliably clear 1.0 (0.34 * 3 = 1.02, capped). Applies identically to
# both of TRAIN_BATTALION's branches (recruiting under capacity, or the
# post-capacity maintenance drill) -- one constant, one bound, for either
# phase. The "build in degradation" Scott confirmed is already real:
# BATTALION_READINESS_DECAY_PER_CYCLE just below, applied every cycle
# regardless (Simulation._advance_battalion_readiness_upkeep) -- unchanged.
BATTALION_READINESS_BOLSTER_PER_ACTION = 0.34
BATTALION_READINESS_DECAY_PER_CYCLE = 0.005
# Deliberately cheaper than BATTALION_TRAINING_FOOD_COST_PER_SOLDIER times a
# real batch of soldiers -- nobody's being newly fed here, just drilled.
BATTALION_READINESS_UPKEEP_FOOD_COST = 8

# compute_might(tribe) itself: each term below is an independent, named
# multiplier stacked on tribe.battalion_size (the base), the same "stack
# multiple named multipliers" shape _food_multiplier already uses for
# kitchen/cooking/created-object bonuses. All four originally-planned inputs
# (equipment, defensive tier, Warrior trophies, Well-Being) plus training/
# readiness now. Every value below is an invented first-pass default, not
# tuned against live data yet -- revisit once a real run shows Might numbers
# in practice, the same way every other unvalidated constant in this project
# has been flagged.
MIGHT_WEAPON_BONUS = 0.5  # a fully-armed Battalion (1 Forge weapon per soldier): +50%
MIGHT_TIER_BONUS_PER_TIER = 0.15  # Keep/Fortress/Castle: +15%/+30%/+45%
MIGHT_TROPHY_BONUS_PER_TROPHY = 0.1  # each trophy personally credited to the Warrior: +10%
# Explicit request: "Well-Being might be included some also" -- deliberately
# the smallest-magnitude term (+-15% across the full 0.0-1.0 range), since
# Well-Being already drives population growth elsewhere and shouldn't double
# up as a dominant lever here too.
MIGHT_WELLBEING_WEIGHT = 0.3
# Explicit request: "not overpowered, more like bolster and upkeep" --
# capped well under weapon's own swing, comparable to the tier ladder's.
MIGHT_TRAINING_BONUS = 0.25

# Military branch, step 6: RAID(rival)/DECLARE_CONQUEST integration
# (actions._might_adjusted_win_chance). Might is layered on TOP of the
# existing population-share win chance, never a replacement for it -- same
# "real ceiling, never an absolute guarantee either direction" shape every
# other win-chance formula in this project already uses. When neither side
# has ever built a Battalion (still the common case before this branch gets
# used at all), the modifier is exactly 0 -- ordinary RAID/DECLARE_CONQUEST
# behavior for every tribe that hasn't touched Military stays completely
# unchanged.
MIGHT_MODIFIER_MIN = -0.2
MIGHT_MODIFIER_MAX = 0.2
MIGHT_MODIFIER_SCALE = 0.15
# The combined (population + Might) win chance still can't be a sure thing
# either way -- same reasoning STRIKE_RAIDER_CAMP_MAX_WIN_CHANCE/
# EXPEL_RAIDERS_MAX_WIN_CHANCE already codify for their own fights.
MIGHT_ADJUSTED_WIN_CHANCE_FLOOR = 0.05
MIGHT_ADJUSTED_WIN_CHANCE_CEILING = 0.95

# Military branch, step 7: the DECLARE_CONQUEST eligibility nudge
# (Simulation._prepare_turn) -- "the Chief has to actually be able to reach
# DECLARE_CONQUEST... an eligibility nudge once it's genuinely a good bet."
# Only fires once this tribe's own Might is at least 30% ahead of a known,
# nearby rival's -- a real, named advantage, not just any nonzero edge.
DECLARE_CONQUEST_NUDGE_MIGHT_RATIO = 1.3

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

# Explicit request ("suspend crisis for 10 cycles beyond Territory lock"): the
# march to reach confirmed water is itself expensive (RELOCATE's own food/
# water cost, several cycles running at MOVEMENT_SPEED), so a tribe often
# arrives and founds its city already right at the survival-crisis threshold
# -- immediately cutting its menu down to SURVIVAL_CRISIS_ACTIONS the instant
# it's finally able to build/farm/settle in would be a harsh, undeserved
# start. See Tribe.settled_at_cycle and _prepare_turn's crisis-check block.
SETTLEMENT_CRISIS_GRACE_CYCLES = 10

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
#
# Live bug, confirmed against a real run: this was the one population-linked
# constant in the whole project that never got the population-scaling
# treatment everything else did (upkeep, _labor_multiplier, expedition
# capacity, livestock surplus threshold all scale with tribe size). A tribe of
# 3232, in near-total sustained famine (physiological 0.01-0.03) the entire
# back half of a real run, kept growing 3-7/cycle from windfall-spike growth
# (see LABOR_MULTIPLIER_CAP's own comment) while every "starvation claimed
# lives" event cost it exactly -1 regardless -- a flat death toll that could
# never meaningfully offset a population-proportional growth spike once a
# tribe is this large. Simulation._scaled_population_loss applies the same
# max(floor, population // divisor) shape _livestock_surplus_threshold already
# uses -- still exactly 1 for a small tribe, a real deterrent for a large one.
POPULATION_LOSS_DIVISOR = 50
STARVATION_TRAUMA_MAGNITUDE = -0.4
STARVATION_TRAUMA_RADIUS = 5

# Simulation._advance_population_pressure: explicit request, 2026-09-09,
# after real data showed the reactive starvation/thirst drip alone wasn't
# reliably enough of a check on population -- "cull an additional 11% less
# than the supportable population... naturally, meaning as needed... checked
# per-cycle." Distinct from starvation/thirst (reactive, fires only once
# food/water actually run out): this is proactive, keyed on population
# alone against _sustainable_population, and catches the real case that
# growth's own gradual taper can't -- a winning RAID/DECLARE_CONQUEST
# absorbing a rival's population in one shot, pushing a tribe over its own
# sustainable line instantly rather than gradually.
POPULATION_CARRYING_CAPACITY_TARGET_FRACTION = 0.89
# Only this fraction of the excess (population above the target line) is
# culled per cycle -- "naturally, as needed," not a one-time snap cull. A
# large overshoot drains down over several cycles instead of vanishing in
# one tick.
POPULATION_PRESSURE_CULL_FRACTION = 0.05
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
#
# Reverted (explicit correction, 2026-09-06): "I wanted only wandering Raids on
# the board to take the 75% ... If the Settlement gets raided, we already have
# that down cold." The "kill 1, lose 75%" spec was meant for an expedition
# ambushed while out in the field (see config.EXPEDITION_RAIDER_AMBUSH_LOOT_
# FRACTION), not this settlement-defense mechanic -- restored to its original
# values, wall progress still meaningfully reduces the cost of a failed
# defense here.
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
# Explicit follow-up, after a test flake exposed the real gap: Simulation.
# _relocate_raider_sighting_after_ambush used to pick independent x/y offsets
# in [-RAIDER_SIGHTING_OFFSET, +RAIDER_SIGHTING_OFFSET], which could land back
# on the exact tile just cleared (both offsets rolling 0) -- "cast elsewhere
# on the map" should never mean "didn't actually move." A minimum radius (not
# zero) on the angle+distance version of that same nudge guarantees real
# displacement every time, not just most of the time.
RAIDER_SIGHTING_MIN_OFFSET = 2
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

# Explicit spec (2026-09-06): "I wanted only wandering Raids on the board to
# take the 75% of their collected holdings and the 1 life, if they lose. so
# they might loose 4 meat, 3 wood, 2 stone... Given that's a 75% cut of what
# they hold." Population loss above already matched (1); this is the new
# piece -- a lost ambush now also cuts whatever the party is actually
# carrying (exp["food_gathered"]/"water_gathered"/"wood_gathered"/
# "stone_gathered"), not just the tribe's population, since a wandering
# party's real "holdings" are what it's carrying in the field, not the whole
# settlement's stockpile.
EXPEDITION_RAIDER_AMBUSH_LOOT_FRACTION = 0.75

# Explicit finding, after a live 986-cycle run: "how many times did they
# defend and get loot? in an ambush scenario" -- turned out to be zero, a
# structural guarantee, not just bad luck: an on-foot party ambushed in the
# field had no defend/win branch at all (only an automatic win if already on
# a boat over water). Real chance added, in the spirit of _resolve_raider_
# attack's own population-scaled defense, but deliberately lower/capped --
# a traveling party carries no wall, keep, moat, or torches with it, just its
# own numbers.
EXPEDITION_AMBUSH_DEFENSE_BASE_CHANCE = 0.2
EXPEDITION_AMBUSH_DEFENSE_POPULATION_BONUS_PER_10 = 0.03
EXPEDITION_AMBUSH_DEFENSE_MAX_CHANCE = 0.5

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
# growing rather than capping at one.
#
# Explicit correction, 2026-09-09: "I'm very tempted to remove the Wall
# restriction on it" -- dropped; Long House no longer requires the wall at
# all. See _build_long_house's own docstring for the real-run finding this
# was grounded in.
LONG_HOUSE_WOOD_COST = 25
LONG_HOUSE_STONE_COST = 20
# Raised 8 -> 30 (explicit request: "how many people to a Long House? 30 maybe at
# most") -- a real, human-scale capacity per building rather than a small number
# that mostly just controlled how fast Long Houses accumulated.
HOUSING_POPULATION_PER_LONG_HOUSE = 30

# Explicit request, 2026-09-09: "modify long houses to scale like warehouse."
# Same shape as WAREHOUSE_MAX_COUNT/UPGRADE_WAREHOUSE (real data showed a
# warehouse count with no ceiling spiraling to 47 in one run) -- real
# BUILD_LONG_HOUSE builds now cap here; UPGRADE_LONG_HOUSE
# (backend/actions.py._upgrade_long_house) takes over from there, with an
# escalating cost so it can't become the same infinite-spam problem under a
# new name. Necessary given FORTRESS_LONG_HOUSES_REQUIRED/CASTLE_LONG_
# HOUSES_REQUIRED below (40/70) -- no tribe should need 40-70 literal
# buildings placed on the map to reach those tiers.
LONG_HOUSE_MAX_COUNT = 5
LONG_HOUSE_UPGRADE_WOOD_COST_BASE = 30
LONG_HOUSE_UPGRADE_STONE_COST_BASE = 25
LONG_HOUSE_UPGRADE_COST_GROWTH = 0.35

# Explicit request: "'furs' can make the Long Houses more comfortable and
# easier to build" -- Tannery Fur (TANNERY_YIELD_PER_CYCLE, tribe.
# unique_resources["Fur"]) already existed but nothing ever spent it. Each
# Fur used knocks a fixed amount off both costs -- a real speed-up, not a
# free ride: explicit correction, "reduce the cost, don't fully substitute
# it," floors the discount well above zero. (25, 20) -> as low as (16, 11) at
# the 3-Fur cap (both floors coincide at exactly 3 furs), a 36-45% reduction,
# and every Fur actually used is deducted from the pile (see
# actions._long_house_fur_discount) -- "be sure they are consumed as used."
FUR_LONG_HOUSE_WOOD_DISCOUNT = 3
FUR_LONG_HOUSE_STONE_DISCOUNT = 3
FUR_LONG_HOUSE_MIN_WOOD_COST = 16
FUR_LONG_HOUSE_MIN_STONE_COST = 11

# The defensive tier ladder after Long House (backend/actions.py._build_keep/
# _build_fortress/_build_castle): explicit request (original) -- "they can
# have 10 houses before they build a Keep, 40 until they reach a Fortress, 70
# until they can build castles." Gated on tribe.long_houses_built PLUS
# tribe.long_house_upgrades now (a real proxy for how established the
# settlement has become) rather than era or population alone, each stage
# requiring the previous one already standing. Each is a real, additional
# defense bonus stacked on top of the wall's own (Simulation.
# _resolve_raider_attack's RAIDER_DEFENSE_WALL_BONUS_AT_FULL_PROGRESS).
#
# Explicit correction, 2026-09-09: "lower long houses restriction to 3" --
# real data showed the original 10 was already a steep bar even before
# LONG_HOUSE_MAX_COUNT (5) existed; Fortress/Castle's own much higher
# thresholds now lean on UPGRADE_LONG_HOUSE to stay reachable at all.
KEEP_LONG_HOUSES_REQUIRED = 3
KEEP_WOOD_COST = 30
KEEP_STONE_COST = 35
KEEP_DEFENSE_BONUS = 0.10

# Explicit correction, 2026-09-09: "I think the costs on Fortress and Castle
# are wrong in terms of very restricted... we need to allow it with some
# effort not a nearly impossible goal." The original 40/70 were tuned
# against the old unbounded (population-scaled) Long House count -- against
# the new LONG_HOUSE_MAX_COUNT=5 cap, that would have meant 35/65
# UPGRADE_LONG_HOUSE calls on an escalating cost curve, effectively
# unreachable. Rescaled proportionally to the new model instead: Fortress
# needs 3 upgrades past the real cap (8 total), Castle needs 7 (12 total)
# -- a real, escalating-but-affordable step up from Keep's 3, not a
# separate order of magnitude.
FORTRESS_LONG_HOUSES_REQUIRED = 8
FORTRESS_WOOD_COST = 50
FORTRESS_STONE_COST = 60
FORTRESS_DEFENSE_BONUS = 0.20

CASTLE_LONG_HOUSES_REQUIRED = 12
CASTLE_WOOD_COST = 40
CASTLE_STONE_COST = 50
CASTLE_DEFENSE_BONUS = 0.15

# Explicit request, 2026-09-10: "If they form an alliance, should we revise
# the menu to allow full builds all the way until both reach Castle-state...
# I love building the Castle together. 1 big piece in the middle of the
# Tribes." A genuinely shared structure (Simulation.joint_castle,
# actions._build_joint_castle) -- a separate path to "Castle-state" from the
# ordinary Fortress+long-house ladder above, open only to two top-era tribes
# that are mutually allied. Costed higher than a solo Castle (roughly 2x)
# since it's the two tribes' combined effort, not a bigger footprint for one.
JOINT_CASTLE_WOOD_COST = 80
JOINT_CASTLE_STONE_COST = 100
# Same "bigger tribe contributes more per call" shape BATTALION_TRAINING_
# PER_ACTION_BASE/CONSTRUCT_WALL's own progress-per-action already use
# (actions._labor_multiplier) -- staged across several calls from either
# chief, not a one-shot flip.
JOINT_CASTLE_CONTRIBUTION_PER_ACTION_BASE = 10

# BUILD_ROAD (backend/actions.py._build_road): a permanent, tribe-built version of
# the same trail_speed_bonus a well-worn path already grants expeditions (World.
# trail_speed_bonus) -- flat, not distance-decayed like a trail, since a road exists
# deliberately rather than wearing in from repeated travel.
ROAD_WOOD_COST = 30
ROAD_STONE_COST = 15
ROAD_SPEED_BONUS = 2

# CONSTRUCT_WALL's expansion fallback (backend/actions.py._expand_wall_territory):
# grows the tribe's real territory_radius (see TERRITORY_FOUNDING_REGION above) and
# unlocks the next wall section in fixed compass order -- one call per section, no
# exception for ring 0.
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

# Explicit request: "if they Build a Sawmill... and have discovered and are
# using a Timber Grove to get wood, they can have the treatment" -- the same
# permanent-mastery idea as water/food's own security fixes, softened by
# explicit design choice (see Simulation._advance_wood_supply's own
# docstring): unlike food/water, wood has no automatic per-cycle drain to
# guard against, so a real passive income (same shape MINE_YIELD_PER_CYCLE/
# TANNERY_YIELD_PER_CYCLE already use for their own resource) instead of
# "always topped to the cap" -- large enough to feel like real security
# without making every future building free outright. Runs every cycle, the
# same cadence fish/mine/tannery's own passive income already use.
WOOD_SECURITY_DAILY_INCOME = 10
QUARRY_WOOD_COST = 15
QUARRY_STONE_COST = 30
QUARRY_STONE_MULTIPLIER = 3

# Stone's own version of WOOD_SECURITY_DAILY_INCOME above -- same amount, but
# a higher bar by explicit request: a Quarry alone isn't enough for real stone
# mastery, a Mine is required too. See Simulation._is_stone_secure.
STONE_SECURITY_DAILY_INCOME = 10

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
# Raised 150 -> 300 (explicit request, 2026-09-06: "let's give the Warehouse
# more basic capacity. They are wasting a lot of cycles on building them.")
# Confirmed live: a tribe built 12 Warehouses over a 756-cycle run (1350
# total capacity) while wood/food/stone were still in the low thousands --
# each individual Warehouse's bonus was too small relative to a mature
# economy's real accumulation rate, so the model kept spending real turns on
# more of them instead of anything else.
STORAGE_CAP_BASE = 300

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
# Raised 100 -> 400 (same request as STORAGE_CAP_BASE above): each Warehouse
# now covers 4x as much capacity for the same cost, so a mature economy needs
# far fewer of them to keep pace -- fewer turns spent on repeat construction,
# not a change to what a single Warehouse costs.
WAREHOUSE_STORAGE_BONUS_PER_BUILDING = 400
WAREHOUSE_WOOD_COST = 25
WAREHOUSE_STONE_COST = 20
# Explicit request, 2026-09-08: "we have to limit build_warehouse when they
# don't have a need" -- Simulation._warehouse_needed's own "a resource is
# already this close to the cap" half, on top of the population-vs-cap check
# it already used for the very first warehouse.
WAREHOUSE_NEED_NEAR_CAP_FRACTION = 0.85

# Explicit request, 2026-09-09: real data showed a run where one tribe built 47
# warehouses (no count cap, trivial flat cost, 700->far beyond in ~50 cycles).
# "They shouldn't build more than 5 I think. The rest of the capacity comes
# from the build or improvement calls being made now for a new build." Real
# BUILD_WAREHOUSE ceiling; UPGRADE_WAREHOUSE (backend/actions.py.
# _upgrade_warehouse) takes over past this point, with an escalating cost
# (WAREHOUSE_UPGRADE_COST_GROWTH per tier) so it doesn't just become the same
# infinite-spam problem under a new name.
WAREHOUSE_MAX_COUNT = 5
WAREHOUSE_UPGRADE_WOOD_COST_BASE = 40
WAREHOUSE_UPGRADE_STONE_COST_BASE = 35
WAREHOUSE_UPGRADE_COST_GROWTH = 0.5

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

# Object Creator era (replaces the old empty mechanization_era/silicon_era
# reserved slots -- see eras.py): "they can create anything they want and we
# have to somehow support it." BUILD_OBJECT_CREATOR is the one-time factory,
# same BUILD_FORGE-shaped gate (wood/stone cost + a free footprint slot).
# CREATE_ITEM/CREATE_USEFUL_STRUCTURE are deliberately risk-bounded: the name
# is genuinely random/flavorful (same "no invisible dice on what matters,
# some dice on what it's called" precedent ITEM_NAMES_BY_TYPE already sets),
# but the mechanical effect is drawn from a small fixed category menu at a
# single capped magnitude -- explicit request: "let's limit the risk at this
# time knowing we will come back to it" (full open-ended LLM-driven stat
# generation is a deliberate future follow-up, not built now).
OBJECT_CREATOR_WOOD_COST = 80
OBJECT_CREATOR_STONE_COST = 80
CREATE_ITEM_WOOD_COST = 20
CREATE_ITEM_STONE_COST = 20
CREATE_USEFUL_STRUCTURE_WOOD_COST = 40
CREATE_USEFUL_STRUCTURE_STONE_COST = 40
CREATED_OBJECT_NAMES = (
    "Auto-Loom", "Sky Anchor", "Glass Compass", "Wind Ledger", "Storm Kiln",
    "Echo Frame", "Sun Lattice", "Tide Engine", "Quiet Forge", "Signal Cairn",
)
# Bounded effect menu -- one category per creation, picked round-robin off
# tribe.created_objects's own length (deterministic, not a hidden roll) so
# a tribe that keeps creating things cycles through every category rather
# than gambling on the same one repeatedly. Six categories spanning six
# different axes of tribe life (gathering, war, defense, culture, exploration,
# growth), not just one narrow effect repeated -- explicit invitation: "if you
# want to add different ways the new things can alter the Tribes, I'm ok with
# creativity." See actions.py._created_object_bonus for where each is read.
CREATED_OBJECT_CATEGORIES = (
    "gather_boost", "combat_boost", "defense_boost",
    "celebration_discount", "expedition_boost", "population_boost",
)
# A flat, modest +20% per creation, same order of magnitude as SAWMILL_WOOD_
# MULTIPLIER/DOCK_FISH_CATCH_BONUS_FRACTION -- not KITCHEN_FOOD_MULTIPLIER's
# 3x stacking, deliberately smaller given these come with no real prerequisite
# beyond the Object Creator itself. Stacks additively across multiple created
# objects in the same category, same shape as the raider-defense bonus stack.
# Used by gather_boost/combat_boost/defense_boost/celebration_discount --
# expedition_boost/population_boost use their own differently-scaled
# constants below since they're flat tile/population amounts, not percentages.
CREATED_OBJECT_MAGNITUDE = 0.2
# expedition_boost: a flat extra tiles/cycle for every expedition and RELOCATE
# this tribe sends out from then on (see physics.terrain_aware_step's
# base_speed callers) -- measured in tiles, not a percentage, so it gets its
# own constant rather than reusing CREATED_OBJECT_MAGNITUDE.
CREATED_OBJECT_EXPEDITION_SPEED_BONUS = 2
# population_boost: a one-time flat population grant the moment it's created
# (not an ongoing multiplier) -- the simplest, safest of the six effects by
# construction, since there's nothing left to keep re-checking afterward.
CREATED_OBJECT_POPULATION_BONUS = 3

# War and World Domination era (replaces the old empty cosmic_post_human
# slot): the ladder's real final era. DECLARE_CONQUEST is a deliberate,
# all-in escalation of the ordinary RAID a tribe has had since tribal_synapse
# -- win, and the rival is fully and immediately absorbed (Simulation.
# _merge_tribes) in one campaign rather than several successful raids
# gradually grinding them down; lose, and the cost is much steeper than an
# ordinary repelled raid, since this is a full campaign, not a hit-and-run.
DECLARE_CONQUEST_WOOD_COST = 100
DECLARE_CONQUEST_STONE_COST = 100

# Explicit request, 2026-09-09: "I do want a play by play blows, meters
# falling, informational popup for only the one and only WAR... Yes, real
# loss." Redesigned from a single instant dice roll into a bounded war of
# attrition -- each round rolls the same population-share/Might-adjusted
# chance DECLARE_CONQUEST always used, but now the loser of THAT round
# takes a real, permanent population hit (and the round's winner takes a
# smaller one too -- "meters falling" plural, not just the loser's) instead
# of the whole campaign resolving on one roll. Ends the moment either side
# is fought down to DECLARE_CONQUEST_DEFEAT_THRESHOLD_FRACTION of its OWN
# starting population -- "we can 'win' and absorb the last 10% or so of the
# remaining pop" -- and _merge_tribes absorbs the loser's survivors and
# stockpiles into the winner exactly like it already does for an ordinary
# RAID reducing someone to zero. Symmetric by design: the side that
# initiated the campaign can lose it all too, not just pay a bounded
# penalty -- a real reason "to build up their Might with Training" first,
# not just a nice-to-have. A fight that never breaks either side within
# DECLARE_CONQUEST_MAX_ROUNDS ends in a costly stalemate -- no merge,
# both sides keep whatever they have left.
DECLARE_CONQUEST_MAX_ROUNDS = 6
DECLARE_CONQUEST_DEFEAT_THRESHOLD_FRACTION = 0.10
DECLARE_CONQUEST_ROUND_LOSS_FRACTION_LOSER = 0.30
DECLARE_CONQUEST_ROUND_LOSS_FRACTION_WINNER = 0.08

# Grounded 2026-09-11 against run_20260911_065718 (758 cycles): with the
# constants above, even a side that loses every single one of the 6 rounds
# only falls to 0.7 ** 6 ~= 0.1176 of its starting population -- ABOVE
# DECLARE_CONQUEST_DEFEAT_THRESHOLD_FRACTION (0.10), so the in-round defeat
# check above can never actually fire. Confirmed live: 18/18 real
# DECLARE_CONQUEST attempts in that run ended "costly stalemate", none
# ever decisive, while population repeatedly crashed to single/low double
# digits and rebuilt between wars -- an endless grind, never a resolution.
# User's diagnosis watching it happen: "we need to add a surrender...
# I don't think anyone will ever win." Rather than re-tuning the round
# math itself (loss fractions above look deliberately chosen), a side that
# ends the war meaningfully weaker than the other concedes instead of the
# battle silently reopening next cycle -- see _declare_conquest's
# post-round-cap surrender check.
DECLARE_CONQUEST_SURRENDER_POPULATION_RATIO = 0.5

# Explicit follow-up, 2026-09-11: "surrender isn't allowed unless you lose 2
# times already" -- don't let the very first lopsided stalemate against a
# given rival end the war outright. Tribe.conquest_stalemate_losses tracks,
# per rival id, how many times this tribe has already been the weaker side
# in a surrender-eligible stalemate; only once that count reaches this many
# does the NEXT such stalemate actually resolve as a surrender. Below the
# threshold it's still a true stalemate (no merge), just with a history note
# that they're clearly outmatched, so the next loss like this ends it.
DECLARE_CONQUEST_SURRENDER_AFTER_LOSSES = 2

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

# Explicit request: "a full population frenzy repelling all Raiders from the
# Territory Boundary... bigger population scales up the loot and pop. gained" --
# a proactive alternative to just waiting for _resolve_raider_attack's passive
# defense once tribe.raiders_approaching is set. Win-chance shape mirrors
# STRIKE_RAIDER_CAMP's own population-scaled formula (a higher base/ceiling here,
# since "the whole population" committing is a stronger showing than an ordinary
# strike party), not RAID's ratio-based one -- an approaching raider party has no
# simulated population of its own to compare against, same reasoning
# STRIKE_RAIDER_CAMP's own comment already gives.
EXPEL_RAIDERS_BASE_WIN_CHANCE = 0.6
EXPEL_RAIDERS_WIN_CHANCE_POPULATION_BONUS_PER_10 = 0.03
EXPEL_RAIDERS_MAX_WIN_CHANCE = 0.9
# On a win: new resources and people, not just recovered losses -- the raiders'
# own plunder and stragglers, scaled by the tribe's own size the same way
# RAID's absorbed-population and _resolve_raider_attack's raider_strength both
# already scale with population.
EXPEL_RAIDERS_LOOT_PER_POPULATION = 0.6
EXPEL_RAIDERS_POPULATION_GAIN_FRACTION = 0.03
# Explicit follow-up ("if they lose, they lose but redouble their efforts in
# the same turn... if they have to try again, populations are lost and the
# gains reduce"): a failed wave doesn't end the action -- anger fuels an
# immediate retry, up to this many total waves in one action call, each still
# a real cost (population lost, a RAID_STEAL_FRACTION-sized cut of resources)
# so retrying is never free, and the eventual reward shrinks per wave it took.
# Bounded rather than unbounded so a truly overwhelming raiding force (or a
# tribe unlucky enough to keep failing) can't loop forever in one action --
# the passive approach/defense (_advance_raider_approach) is still there as a
# fallback if every wave here fails.
EXPEL_RAIDERS_MAX_WAVES = 3
EXPEL_RAIDERS_POPULATION_LOSS_PER_FAILED_WAVE = 2
EXPEL_RAIDERS_REWARD_REDUCTION_PER_WAVE = 0.35
EXPEL_RAIDERS_MIN_REWARD_MULTIPLIER = 0.2

# CLEAR_TERRITORY (backend/actions.py._clear_territory): explicit request,
# 2026-09-09: "Territory boundaries must be cleared of threats before they
# can really start building anything really. This is not a passive action.
# The Chief must clear the area... make the Clearing radius a little larger
# than the Boundary area so they clear any Raider just on the line or
# outside it." Extra tiles beyond tribe.territory_radius that both this
# action and the real-construction menu-lock (Simulation._prepare_turn) reach
# to find a raider camp.
TERRITORY_CLEARING_RADIUS_MARGIN = 5

# A tribe can only overhear another tribe's broadcast (and therefore only converge on
# shared vocabulary with them) within this Euclidean distance -- previously broadcasts
# were audible map-wide regardless of distance, which gave away free information and
# removed any incentive to actually travel toward another tribe.
BROADCAST_HEARING_RADIUS = 15

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
#
# RIVAL_PRECISE_AWARENESS_RADIUS is also the discovery range for
# Simulation._note_rival_discovery (tribe.discovered_rivals) -- explicit request,
# 2026-09-09: "i suggest they narrow the target as they get closer." Once a home
# camp or a live expedition gets this close to a rival, that rival counts as
# discovered for good, which is also what now gates DECLARE_ALLIANCE/DECLARE_WAR/
# SEND_TRADE_EMISSARY's "have we made contact" check (backend/actions.py.
# _nearest_rival) -- the old dedicated DIPLOMACY_CONTACT_RADIUS constant (a live
# home-to-home distance check) is retired: once both tribes are settled and spawn
# deliberately far apart, that live check could never pass again after founding,
# even after real contact had already happened once.
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
# Explicit request: "we need to ensure we are not overloading the browser, are
# being efficient with communications." exp["path"] is a pure visualization
# breadcrumb (backend game logic reads terrain_checkpoints/target, never this
# -- see Simulation._discover_sites_along_route) that grows by one point every
# single cycle an expedition is out, with no cap -- and a party can now
# legitimately stay out for hundreds of cycles (EXPEDITION_MAX_DAYS raised,
# and a search is no longer allowed to give up from day count alone). Every
# point ever walked was still being resent over the websocket in full, every
# tick, for that party's entire remaining lifetime, and stored the same way in
# every board_history.db snapshot. The frontend's own per-frame redraw cost
# was already fixed to be O(1) regardless of length (a cached, incrementally-
# extended Path2D) -- this caps the underlying data those points come from,
# so a months-long push doesn't also mean an ever-growing wire payload and
# database row. Simply stops recording new points past this length rather
# than a sliding window (dropping the oldest to make room): a plain list
# append staying under a fixed cap is exactly the shape the frontend's
# path.length-grew-by-one incremental cache already expects -- a FIFO window
# would make length stop changing once full, forcing that cache back to a
# full rebuild every single tick forever, the exact per-frame cost this was
# built to avoid in the first place. A visually static tail past ~150 tiles
# already walked is a reasonable tradeoff for a trip that long.
EXPEDITION_PATH_MAX_POINTS = 150
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
    "desert": 0.5,  # tough sand, harsher than plains/forest but not a climb
    # Deliberately non-zero -- unlike ocean's 0.0 (which makes physics.
    # terrain_aware_step treat it as impassable and deflect around it), the
    # volcano hazard (Simulation._volcano_hazard) needs a tribe to actually be
    # able to step onto the tile to be killed by it.
    "volcano": 0.4,
}

# Boat (Simulation._advance_automatic_boat, backend/physics.py.terrain_aware_step):
# explicit request, automatic like fire once a Dock stands and fishing is
# mastered -- "give the boat mobility in the clean water, not the sea." River is
# normally the slowest passable terrain (0.3x above); a boat turns it into a real
# advantage over dry land instead of an obstacle. Ocean stays exactly as
# impassable as ever -- this is deliberately NOT an ocean-crossing mechanic.
BOAT_WATER_BIOMES = {"river", "lake"}
BOAT_WATER_MOVEMENT_MULTIPLIER = 1.2

# Explicit request ("Scouts in particular should get their speed bonus back,
# stealthing past observations, and moving quick to report findings and
# foragings"): a plain SCOUT (Simulation._advance_one_expedition's
# is_scout check) always moves at this fast, once-a-day-batch pace, regardless
# of settlement status -- speed IS the scout's whole role, and a single fast
# jump through danger is inherently lower-exposure than a hunting/exploration
# party's slower, smoother SETTLED_EXPEDITION_SPEED pace. This is also the
# actual fix for a live tribe wipe: a settled scout moving the slow uniform
# pace lingered near the volcano for many consecutive cycles, each one
# independently rolling a hazard chance tuned for one roll per real day.
EXPEDITION_SPEED = 10
# See SETTLED_MOVEMENT_SPEED's own comment -- same uniform "1 sky tick" pace,
# applied to expedition movement once a tribe has settled. EXPEDITION_SPEED
# itself stays reserved for a not-yet-settled tribe's search, which still
# needs to move fast (see Simulation._advance_one_expedition's is_new_day
# split: movement now happens every cycle regardless of settlement, only the
# per-cycle distance and the once-a-day bookkeeping -- day count, "daily"
# food/water/wood/stone gains, hunting rolls -- differ by settlement status).
SETTLED_EXPEDITION_SPEED = 12
# Raised 3 -> 5 (explicit request, same conversation that diagnosed a tribe
# dying of thirst before ever founding): "give the Scout a 5 day start. They
# are not penalized if they come back late, the Tribe is, given someone
# dies." A scout that gives up early just means dispatching another one from
# scratch -- more patience per dispatch costs the scout nothing and gives a
# real search more room to actually find water before the tribe's own
# survival clock runs out.
EXPEDITION_MAX_DAYS = 10

# A tribe could previously only ever have one party (scouting or hunting) in the field
# at a time -- a chief with real people to spare had no way to send out more than a
# single expedition regardless of population. Capped rather than unlimited: nothing
# currently deducts population to launch a party, so an uncapped tribe could spam
# expeditions for free.
MAX_CONCURRENT_EXPEDITIONS = 3

# actions.expedition_capacity() lets a larger tribe spare more search parties at once --
# MAX_CONCURRENT_EXPEDITIONS above is only ever the floor now, not a hard ceiling. A
# tribe of 8 (the starting population) still gets exactly 2; a tribe of 20+ can spare
# more. Real per-capita capacity, the same way upkeep already scales with population
# (see Simulation._apply_upkeep) -- population growth used to buy a tribe nothing on
# the scouting/hunting side no matter how large it got.
EXPEDITION_SLOT_POPULATION_DIVISOR = 5

# Live bug report: "we need to limit the number of scouts and gatherers at a
# time... my system was throttled." A real run reached population 352, which
# uncapped would allow 70 simultaneous expeditions for one tribe alone -- the
# map visibly buried in scout icons alongside real, confirmed system load.
# This was a deliberate "no hard ceiling" design once (see EXPEDITION_SLOT_
# POPULATION_DIVISOR's own comment), but real scale exposed that as wrong -- a
# genuine ceiling now caps expedition_capacity()'s per-capita growth.
#
# Follow-up live report (run_20260908_082234, both tribes well past this
# ceiling in population): "the board is very crowded with outgoing and
# returning. There should only be 6 at a time total." 8 was already close
# enough to read as flooded once two large tribes both hit it at once --
# lowered to the number actually confirmed comfortable to watch.
MAX_CONCURRENT_EXPEDITIONS_CEILING = 6

# Explicit request: "I am concerned about excess chatter... a lot of players
# on the board." A real prompt reconstructed from a live run (population 329,
# 7 concurrent expeditions -- MAX_CONCURRENT_EXPEDITIONS_CEILING's own ceiling
# of 8 was already close) showed the "Still in the field" fact naming every
# single party in one run-on sentence, one clause each, with no limit -- and
# removing the per-kind dispatch cap (2026-09-07, a separate, correct fix for
# tribes getting artificially stuck at one party) makes reaching this ceiling
# more common, not less. Below this many parties, full per-party detail (who,
# what day, headed where) is cheap and still worth showing whole. At or above
# it, Simulation._prepare_turn switches to a kind+count summary instead --
# except right at the capacity ceiling itself, where knowing exactly who's
# about to come home is a real decision input (whether waiting one more cycle
# is worth it), so full detail always shows there regardless of this
# threshold.
FIELD_REPORT_DETAIL_THRESHOLD = 3

# actions._labor_multiplier() lets a larger tribe gather/hunt/forage more per action --
# upkeep (_apply_upkeep) already scales with population, but yield from GATHER_WOOD/
# STONE/WATER/FOOD and HUNT_DEER never did, so a bigger tribe was strictly worse off
# per-capita: identical output, more mouths to feed. POPULATION_YIELD_BASELINE matches
# Tribe.__init__'s own starting population, so a tribe at or below starting size sees no
# change at all -- this only ever rewards growth past it, never penalizes a small tribe.
POPULATION_YIELD_BASELINE = 8
# Live bug, confirmed against a real run: _labor_multiplier was left uncapped
# here even after FARM_LABOR_MULTIPLIER_CAP closed the identical problem for
# farm harvests -- a single HUNT_DEER at population 3232 (multiplier 404x)
# generated 1,854 raw food in one action, briefly spiking tribe.food (and
# therefore wellbeing's physiological tier) far above what upkeep would
# otherwise sustain. That transient spike was enough to reopen population
# growth (Simulation._grow_population) even during what should have been a
# real, sustained famine -- the tribe's own daily gathering summaries showed
# growth spikes of 50-167 population in the same run this was traced against.
#
# 2026-09-08: _labor_multiplier itself moved from a flat, unbounded ratio to
# sqrt(population / POPULATION_YIELD_BASELINE) -- see its own docstring for
# the full reasoning. A flat 5.0x cap on a linear ratio meant every tribe past
# population 40 got the exact same per-action yield as one of exactly 40 --
# confirmed against a live day-12 run sitting on 16 wood at population 4767,
# unable to ever afford CONSTRUCT_WALL's 60-wood threshold. The sqrt curve
# only reaches 5.0x at population 200 now and keeps climbing gently past it
# (~20x at 3232, ~45x at 16,574) instead of flatlining -- this cap is raised
# to match: a real backstop far out past anything the curve produces at any
# population actually reached in a real run, not the everyday ceiling a flat
# 5.0x had quietly become.
LABOR_MULTIPLIER_CAP = 100.0

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

# A hazard landmark is the warning-flavored sibling of the reward Landmark
# above -- explicit design spec: "if anyone discovers a hazard, even if no
# one dies, they landmark it... if its a landmark, it needs a 'dangerous'
# sounding name." Land hazards (volcano/cliffs/ocean/shoals) get one on
# first discovery, dedup'd against Tribe.hazard_landmarks the same way
# LANDMARK_NAMES dedups against tribe.landmarks -- "no party should ever
# linger" repeating the same sighting every day. Unconditional on the
# separate, chance-based death roll (see each hazard function's own
# HAZARD_CHANCE) -- merely knowing the ground is dangerous doesn't require
# losing someone first.
HAZARD_LANDMARK_NAMES = (
    "Widow's Reach", "Skull Hollow", "The Bleeding Ground", "Deadfall Ridge",
    "Cursed Hollow", "The Gnawed Bones", "Ashen Scar", "The Last Warning",
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
