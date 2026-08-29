OLLAMA_URL = "http://localhost:11434"

TICK_SECONDS = 0.5

GRID_SIZE = 100
MAX_TRIBES = 4

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
