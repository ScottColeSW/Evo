OLLAMA_URL = "http://localhost:11434"

TICK_SECONDS = 0.5

GRID_SIZE = 100
MAX_TRIBES = 4

# GATHER_WOOD/GATHER_STONE used to be available from the moment a tribe existed, before
# it had even decided where to actually live -- a nomadic band stockpiling timber and
# quarried stone before choosing a home. Simulation._is_settled gates both behind
# actually staying put somewhere farmable first: SETTLEMENT_STABILITY_CYCLES consecutive
# cycles without choosing RELOCATE, standing on one of FARMABLE_BIOMES (open land with
# real water access). A tribe's starting stockpile (Tribe.__init__) still covers early
# BUILD_FIRE/CONSTRUCT_WALL needs before that -- this gates *replenishing* the economy,
# not survival itself (GATHER_WATER/GATHER_FOOD/HUNT_DEER are never touched).
SETTLEMENT_STABILITY_CYCLES = 10
FARMABLE_BIOMES = ("plains", "river", "lake")

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
SETTLED_WATER_SUPPLY_PER_CYCLE = 10

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
POPULATION_GROWTH_CAP = 80

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

# City growth (backend/simulation.py._advance_city_growth): once a tribe founds a city
# (Era.founds_city), one more building appears every time population crosses another
# multiple of this step, up to MAX_CITY_BUILDINGS -- a small, legible stand-in for real
# city-layout simulation, not an attempt at one.
CITY_BUILDING_POPULATION_STEP = 5
MAX_CITY_BUILDINGS = 6

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
FISHING_SUPPLY_PER_CYCLE = 8
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

# Explicit request: "cooked food is worth 3 raw food... so only the task of
# cooking is needed to improve the meals and maslow stats." Applied at the single
# point food is actually consumed (Simulation._apply_upkeep) rather than at every
# scattered food-gain call site (GATHER_FOOD, HUNT_DEER, HUNTING_PARTY, CATCH_FISH,
# farm harvest) -- economically equivalent (the same stockpile now covers 3x the
# need) but one clean touch point instead of six. wellbeing.py's physiological tier
# and instincts.py's hunger thresholds both read the same effective food-upkeep so
# a tribe that's learned to cook doesn't get a false "starving" warning under the
# old, harsher rate.
COOKING_UPKEEP_DIVISOR = 3

# Milestone trophies (backend/simulation.py._award_trophy's `individual` param): unlike
# the chief-credited trophies above, these are earned by a specific named scout or
# hunter and credit them by name, not the chief. Also the pool of "named individuals"
# the breeding design draws from alongside the chief -- see BREED_FOOD_COST below.
MILESTONE_SCOUT_SUCCESSES = 5
MILESTONE_HUNT_SUCCESSES = 5

# BREED (backend/actions.py._breed, backend/breeding.py). Free -- the two real eligible
# windows watched live this session both landed inside a full starvation death spiral
# (0 food/water), meaning the tribe couldn't have afforded any positive cost even if it
# had chosen BREED. The eligibility gate (two distinct named individuals) is still the
# real constraint; this just stops affordability from being a second one stacked on top
# of it during the exact moments eligibility is most likely to appear.
BREED_FOOD_COST = 0
BREED_WATER_COST = 0

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
RAIDER_HAZARD_MAX_CHANCE = 0.12
RAIDER_HAZARD_POPULATION_FOR_MAX_CHANCE = 60
RAIDER_HAZARD_COOLDOWN_CYCLES = 25

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
# independent roll, NOT tied to the biome of whatever terrain_report was rolled --
# unlike lumber/wildlife/quarry sites, raiders aren't a biome feature, so a party can
# plausibly spot both a resource site and raider sign on the same trip. Radiates a
# small dread event AT THE SIGHTING COORDINATE, not the tribe's camp -- a genuinely
# new pattern: this is about a place now known to be dangerous, not something that
# happened at home.
RAIDER_SIGHTING_CHANCE = 0.1
RAIDER_SIGHTING_TRAUMA_MAGNITUDE = -0.4
RAIDER_SIGHTING_TRAUMA_RADIUS = 4

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

# BUILD_LONG_HOUSE (backend/actions.py._build_long_house): explicit request, gated on
# the wall already being complete first -- defense before shelter. A one-time flag on
# the tribe (tribe.long_house_built), not a second world.constructions entry at the
# same tile the wall already occupies -- that dict holds one record per tile, so a
# second type there would silently overwrite the wall's own progress (see
# Landscape.add_construction). Costs more than the wall itself: a real communal
# building, not another defensive structure.
LONG_HOUSE_WOOD_COST = 25
LONG_HOUSE_STONE_COST = 20

# BUILD_CASTLE (backend/actions.py._build_castle): the construction tier after
# BUILD_LONG_HOUSE, gated on the long house already standing -- a real, additional
# defense bonus stacked on top of the wall's own (Simulation._resolve_raider_attack's
# RAIDER_DEFENSE_WALL_BONUS_AT_FULL_PROGRESS), not just a bigger cosmetic building.
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

# EXPAND_TERRITORY (backend/actions.py._expand_territory): only meaningful once
# automatic city growth (Simulation._advance_city_growth) has already reached its
# normal ceiling -- a deliberate push past MAX_CITY_BUILDINGS, not redundant with
# growth that already happens on its own. Capped at double the normal max so the
# territory visual (frontend's drawTerritory, which scales off city_buildings)
# doesn't grow without bound.
TERRITORY_EXPANSION_WOOD_COST = 60
TERRITORY_EXPANSION_STONE_COST = 60
TERRITORY_EXPANSION_BUILDINGS_BONUS = 2

# BUILD_DOCK (backend/actions.py._build_dock): explicit request, "once they have
# Settled in hopes they will figure out fishing" -- a real fishing yield bonus once
# built, not just flavor, to actually reward betting on fishing early.
DOCK_WOOD_COST = 20
DOCK_FISH_CATCH_BONUS_FRACTION = 0.5

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
