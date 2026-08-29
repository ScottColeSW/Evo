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
RELOCATE_FOOD_COST = 2
RELOCATE_WATER_COST = 2

# The self-modification engine lets a model rewrite backend/physics.py on disk and
# hot-reloads it when turns get slow. It's validated with an AST parse + cooldown
# lockout, but it is still an LLM writing code to your machine — off by default.
ENABLE_SELF_MODIFICATION = False
SELF_MOD_LATENCY_THRESHOLD_MS = 4000
SELF_MOD_COOLDOWN_CYCLES = 20

MEMORY_CONSOLIDATE_EVERY_N_CYCLES = 40

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

# Survival instinct thresholds (backend/instincts.py). "Critical" also raises inference
# temperature, same as ancestral dread -- real panic, not just different wording.
HUNGER_WARNING_THRESHOLD = 20
HUNGER_CRITICAL_THRESHOLD = 5
THIRST_WARNING_THRESHOLD = 15
THIRST_CRITICAL_THRESHOLD = 5

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

# A tribe can now actually go extinct (population 0) instead of being propped up at a
# permanent population-1 floor. Extinction is a far larger trauma event than an ordinary
# death -- it should be visible on the map for a long time afterward.
EXTINCTION_TRAUMA_MAGNITUDE = -0.6
EXTINCTION_TRAUMA_RADIUS = 10

# A tribe can only overhear another tribe's broadcast (and therefore only converge on
# shared vocabulary with them) within this Euclidean distance -- previously broadcasts
# were audible map-wide regardless of distance, which gave away free information and
# removed any incentive to actually travel toward another tribe.
BROADCAST_HEARING_RADIUS = 15
