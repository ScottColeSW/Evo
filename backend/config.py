OLLAMA_URL = "http://localhost:11434"

TICK_SECONDS = 0.5

GRID_SIZE = 100
MAX_TRIBES = 4

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

# Chief trophies (backend/simulation.py._check_chief_trophies): a lightweight legacy
# system credited to whichever chief is in power the moment each is first earned, once
# per tribe's lifetime. "Water Bringer" is deliberately the standout -- reliable water
# access is the single hardest survival problem this simulation poses (see the whole
# expedition/nearest_water design), so it's the trophy that actually means something.
FOOD_TROPHY_THRESHOLD = 60

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

# A tribe can only overhear another tribe's broadcast (and therefore only converge on
# shared vocabulary with them) within this Euclidean distance -- previously broadcasts
# were audible map-wide regardless of distance, which gave away free information and
# removed any incentive to actually travel toward another tribe.
BROADCAST_HEARING_RADIUS = 15

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
