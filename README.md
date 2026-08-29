# Evolution2Civ (aka Project Chronos)

A local, spectator-mode sandbox where LLM-driven tribes (via [Ollama](https://ollama.com))
gather resources, invent their own language, and try to grow from a handful of survivors
into a founded settlement — entirely on their own. You pick which local model runs each
tribe from a picker screen, hit start, and watch.

This started as a rambling voice-to-text brainstorm about factory patterns, multi-agent
adversarial games, and evolutionary civilization sims, then grew into a pitch deck framing
it as an "algorithmic survival environment" for stress-testing local models. Neither the
original transcript nor the deck's code snippets are included verbatim — both were full of
UI chrome, citation ads, or code that didn't actually run (broken string ops, calls to
functions that were never defined, mismatched method signatures, an async loop calling a
synchronous HTTP client). This repo is a clean, working implementation of the design they
converged on.

## What's actually happening

- **`backend/world.py`** — a 100x100 tile landscape with Earth-like hydrology: mountains
  in the northwest, an ocean along the entire east edge (`OCEAN_X_START`), and a river that
  originates in the highlands and winds (a sine-meander, not a straight line) down through
  plains and forest to the coast, rather than being an arbitrary diagonal band unrelated to
  anything else on the map. Also tracks the structures tribes build on it.
- **`backend/ancestral_matrix.py`** — a separate emotional-scar layer over the same grid.
  Triumphant or traumatic events radiate outward with a quadratic falloff (clamped to
  [-1, 1]) and get quietly folded into future prompts near that coordinate as an
  unexplained ancestral instinct (dread or pride) — never a hardcoded rule. Dread also
  raises inference temperature for that turn. The only thing that currently generates a
  *negative* event is the hunting hazard below; without it this layer would only ever
  see pride.
- **`backend/memory.py`** — each tribe keeps an episodic memory (hash-based pseudo-embedding
  + cosine similarity, no extra model download required) that gets queried each turn and
  periodically consolidated into permanent cultural taboos, so context doesn't grow forever.
- **`backend/prompts.py`** — forces each tribe to reason privately in English but broadcast
  only in an invented phonetic language, so tribes have to develop and (mis)interpret their
  own shared vocabulary over time. Uses heavy structural delimiters and a restated JSON
  schema on every turn — small quantized local models drift out of strict JSON mode more
  easily on long contexts, and this measurably reduces that.
  Two schema bugs were found and fixed here: the `target_vector` example used to literally
  interpolate the tribe's own current coordinates, which models reliably echoed back
  instead of choosing anywhere else — tribes were never actually exploring. And the
  `visual_action` example was the instruction text itself ("SELECT STRICTLY ONE: [...]")
  sitting in the JSON value position, which models would copy verbatim, failing era-gating
  validation and silently falling back to `IDLE` on an unknown fraction of turns. Both are
  now generic placeholders with the real guidance moved into surrounding prose.

  Fixing those two bugs still wasn't enough -- several follow-on attempts (`Tribe.last_target`
  persistence across turns, `config.MOVEMENT_SPEED` moving several tiles per cycle instead of
  1) didn't close the gap either. A live run showed a model oscillating between two points
  exactly `MOVEMENT_SPEED` tiles apart, net displacement zero, forever.

  The actual root cause: every action moved the tribe toward `target_vector` regardless of
  what that action was, every single cycle. A model choosing `GATHER_WOOD` had no reason to
  also specify a distant destination -- gathering happens where you stand -- so `target_vector`
  defaulted to "wherever I already am," and the tribe drifted at most a few tiles before
  snapping back to routine. The fix (see `backend/actions.py`) was architectural, not another
  prompt patch: only a new `RELOCATE` action moves the tribe at all. Every other action
  (gathering, hunting, building, a new `SCOUT` that looks at a distant tile and reports back
  without moving anyone) happens at the tribe's current camp and leaves its position alone.
  Confirmed live: a model committed to `RELOCATE` for 8 consecutive cycles, actually
  traveling the full 30 tiles to its stated destination, then scouted, reconsidered, and
  relocated again toward a new one -- a real explore-then-commit pattern, not oscillation.
  `RELOCATE` costs stamina (`config.RELOCATE_FOOD_COST`/`RELOCATE_WATER_COST`, on top of
  ordinary upkeep) so marching isn't strictly free compared to every action that does cost
  something.
- **`backend/scheduler.py`** — groups a tick's turns by which model they target and runs
  each model's group concurrently via `asyncio.gather`, so a tick with N tribes across M
  models costs at most M model swaps in Ollama, not N. Also sets `keep_alive` on every
  request so Ollama doesn't evict a model between ticks.
- **`backend/vram_guard.py`** — a one-time-per-tribe sanity check (not a live enforcement
  layer) against a model's real on-disk size from `/api/tags`. Runs once when a simulation
  starts (`Simulation.create`, the async factory `app.py` actually calls); an oversized
  model gets a warning in that tribe's chronicle rather than being silently excluded from
  ever taking a turn again.
- **`backend/translation_matrix.py`** — tracks whether two tribes are converging on shared
  vocabulary. There's no ground-truth "correct guess" signal available (nobody, including
  a tribe itself, knows what its own invented token "really" means), so convergence is
  measured empirically: two tribes independently broadcasting the *same* phrase for the
  *same* action is treated as evidence of shared meaning, and it decays if not reinforced.
  This only produces a real signal because tribes can now actually hear each other —
  `Simulation._prepare_turn` feeds every tribe its neighbors' most recent broadcast +
  action every turn, but only within `config.BROADCAST_HEARING_RADIUS` (Euclidean
  distance). This used to be audible map-wide regardless of distance; that gave away free
  information and removed any incentive to actually travel toward another tribe, so it's
  now proximity-gated -- closing distance with another tribe is what unlocks understanding
  them at all. Confirmed live: two different local models (`gemma2:2b`, `qwen2.5:3b`)
  converged on a shared token for the same action within 4 cycles once in range. Surfaced
  in the sidebar as a "Linguistic Consensus" panel per tribe pair.
- **`backend/eras.py`** — the progression ladder (Stone Age → Bronze Age → Classical Age),
  as an ordered, data-driven table rather than a hardcoded if/elif chain: each era declares
  a population + resource threshold to reach it, a resource cost paid on advancing, and
  which actions it unlocks. Advancement is automatic once a tribe clears the threshold —
  deliberately *not* gated behind the model choosing a special "advance" action, since
  relying on a small quantized model to correctly reason its way to a meta-progression
  action would make the payoff moment unreliable. Reaching Classical Age is what "founds a
  city" now (`Era.founds_city`), replacing the old flat population-threshold flag.
- **`backend/actions.py`** — the action registry: each action name maps to a handler
  function (`ACTION_REGISTRY`), looked up rather than dispatched through `if/elif`. This is
  the Registry Factory pattern — adding an action means registering a handler here, not
  extending a branch chain in `Simulation`. Includes `GATHER_WATER` (river tiles yield far
  more than elsewhere, `config.WATER_YIELD_RIVER` vs `WATER_YIELD_OFF_RIVER`), the first
  resource requirement gating advancement into the Bronze Age. `BUILD_FIRE`/`CONSTRUCT_WALL`
  are no-ops at a tile that already has that structure -- a live run showed a model
  choosing `BUILD_FIRE` for 7 straight cycles at the same spot, each one radiating *more*
  ancestral pride at zero additional benefit, a self-reinforcing loop that made staying put
  look increasingly attractive. That was a real mechanical gap, not model behavior to work
  around. Also includes `SCOUT` (looks at a distant tile and writes what's found -- terrain
  and any structures observed -- into memory, without moving the tribe there) and
  `RELOCATE` (the only action that actually moves the tribe; costs stamina). See the
  `backend/prompts.py` entry above for why this action/movement split exists.
- **`backend/instincts.py`** — physiological survival pressure, distinct from
  `ancestral_matrix.py`'s location-based bias: this is about a tribe's *current* condition
  (starving or dehydrated right now), not the history of the ground it's standing on.
  Surfaced as its own "SURVIVAL INSTINCT LAYER" in the prompt, and a critical state raises
  inference temperature the same way ancestral dread does. Population upkeep
  (`Simulation._apply_upkeep`, scales with population) is what makes this real: food and
  water previously only ever went up, so hunger and thirst had no actual stakes. Failing to
  pay upkeep costs a life (starvation/dehydration), radiates dread at that location, and
  the tribe's own reasoning visibly responds to the pressure before it even turns critical
  (confirmed live — a model started saying "resource scarcity is evident" well before the
  hardcoded warning threshold). `GATHER_WATER` on a river tile also carries a drowning risk
  (`config.DROWNING_HAZARD_CHANCE`), mirroring the forest hunting hazard — the best water
  source isn't a free lunch.

  Population used to be floored at 1 forever (a permanent "walking dead" state, never real
  death) — a tribe can now actually go extinct (`Tribe.extinct`, `Simulation._lose_population`),
  which stops it taking turns, announces itself with a banner, and marks a grave on the map
  instead of quietly capping out.

  One design reversal worth being explicit about: an earlier version of the critical
  hunger/thirst message named the exact fix directly ("Choose visual_action HUNT_DEER this
  cycle") after watching models correctly narrate "we are starving" in their own rationale
  and then still choose `GATHER_WOOD` (to the point of stockpiling 2,000+ wood at population
  1 with zero food). That measurably worked, but it's scripting the outcome from outside the
  simulation, not letting the model reason its way there — reverted back to purely
  descriptive text ("Your people are starving.") on the position that an honest failure is
  more informative than a papered-over success. See `backend/leadership.py` below for the
  approach taken instead: motivation generated *inside* the simulated world, not injected
  from outside it.
- **`backend/leadership.py`** — a one-time, in-fiction leadership contest run when a tribe
  is created: one extra LLM call asks the model to imagine 2-3 individuals competing for
  chief through whatever trial fits (strength, wisdom, oratory, its choice), then declare a
  winner and a one-sentence governing philosophy. That philosophy becomes standing context
  in every future turn (`get_prime_consciousness_prompt`), explicitly framed as *context
  about who leads you, not a command* — what the tribe actually does with it each cycle is
  still its own reasoning. This exists specifically as the alternative to the reverted
  hard-coded survival directive above: motivation generated inside the simulated world
  by the model itself, rather than injected from the engineering layer around it. Confirmed
  live: two tribes produced genuinely distinct, in-character elections on their own (one
  chief won by healing the previous leader with herbal medicine and adopted a
  "harmony with nature" philosophy; another won through demonstrated wisdom and prioritized
  "resource sharing").
- **`backend/world.py`** — local resource depletion: harvesting wood/stone/water/game at a
  tile raises that tile's scarcity (`config.DEPLETION_PER_HARVEST`), which scales down yield
  there on subsequent harvests, capped below total depletion (`config.MAX_SCARCITY`) so
  staying put is costly rather than a hard lock. Regenerates globally and constantly
  (`Simulation.step` calls `Landscape.regenerate` once per tick), independent of whether a
  tribe is currently there. Scarcity is surfaced to a tribe as plain telemetry ("local wood
  scarcity here: 45%"), not a suggestion to move — same principle as the leadership system,
  real environmental pressure instead of a scripted nudge.
- **`backend/genetics.py`** — an optional crossover step that splices two tribes' ideology +
  lexicon into a descendant profile via the model itself.
- **`backend/self_mod.py`** — an opt-in (off by default) engine that lets a model rewrite
  `backend/physics.py` when turns get slow, validated with `ast.parse` and a cooldown lockout
  if the patch is broken. Flip on with `ENABLE_SELF_MODIFICATION = True` in `backend/config.py`
  if you want to see it. It's still model-authored code landing on your disk, sandboxed to
  one small file, so know what you're turning on.
- **`backend/simulation.py`** — the orchestrator: builds every tribe's prompt (only showing
  the actions its current era has unlocked), hands them to the scheduler as one batch,
  applies the results through the action registry, moves avatars, grows population on food
  surplus, rolls a hunting hazard (wolves, forest biome only — the thing that actually
  produces ancestral dread), and checks era advancement every tick.
- **`backend/app.py`** — an `aiohttp` server serving the frontend, a `/api/models` endpoint
  that proxies Ollama's model list for the picker, and a `/ws` websocket streaming state.
- **`frontend/index.html`** — the tribe picker (choose a local model per tribe, up to 4) and
  the canvas theater: a header bar (status + pause), a tabbed sidebar (Tribes / Consensus /
  Chronicle, so the three panels don't all fight for the same vertical space), and a canvas
  that resizes to fill available space (terrain is cached to an offscreen canvas and
  blitted each frame rather than redrawing 10,000 tiles every frame). Each tribe renders as
  a cluster of individuals wandering near its actual grid position — one per population,
  capped at 24 — with deterministic per-individual orbits, not one square standing in for
  the whole tribe.

Spawn points are one-per-biome (`SPAWN_POINTS` in `backend/simulation.py`) so the default
picker order (Forest Tribe, Mountain Tribe, ...) actually starts each tribe in the biome
its name implies. A tribe config (in `START` or `ADD_TRIBE`) can include explicit `x`/`y` to
override this -- used to set up two tribes starting near each other for a demo. This is an
initial condition, same category as `SPAWN_POINTS` itself; it says nothing about what either
tribe then chooses to do about being close (there's no UI for it in the picker yet, only the
underlying support).

## Multiple simulations at once

Each websocket connection owns its own `Simulation` (`backend/app.py`), so opening a
second browser tab starts a second, fully independent world rather than replacing the
first one's run — this was previously a real limitation (documented below, now fixed).
`run.py` also takes `--port` if you'd rather run fully separate OS processes:

```bash
python run.py --port 8766
```

## Running tests

```bash
python -m pytest tests/
```

No `pytest-asyncio` dependency — async tests use a small `asyncio.run()` wrapper in
`tests/conftest.py` instead. Coverage is deliberately scoped to the deterministic pieces
(falloff math, decay, hazard rolls under a seeded RNG, scheduler grouping, VRAM threshold
logic) — anything that requires a live Ollama call is still verified by hand against the
real server, the way every change in this repo so far has been.

## Running it

```bash
pip install -r requirements.txt
ollama pull llama3
ollama pull mistral
python run.py
```

Then open `http://localhost:8765` in a browser, pick a model for each tribe, and press
**Begin Simulation**.

## Known limitations / next steps

- Memory recall uses a deterministic hash-based pseudo-embedding, not real semantics. Swap
  `TribeMemory._embed` in `backend/memory.py` for a call to Ollama's `/api/embeddings` (e.g.
  with `nomic-embed-text`) for genuine similarity search.
- `genetics.breed()` is wired but not yet triggered automatically — now that era transitions
  exist as a concrete event (`Simulation._advance_era_if_ready`), that's the natural place
  to call it, e.g. on reaching Bronze Age.
- Ancestral trauma currently only comes from individual mishaps (wolf attack, drowning,
  starvation, dehydration) or an individual tribe's own milestones (building a fire,
  advancing an era) — nothing yet comes from inter-tribe conflict or environmental events
  (a harsh winter, a plague). Same mechanism either way: call
  `self.trauma.radiate_event_wave(x, y, magnitude, radius)` from `simulation.py`.
- No persistence: closing the server drops the run. Add a snapshot dump if you want to
  resume or analyze runs later.
- Models defaulting to `GATHER_WOOD` regardless of what's actually needed (confirmed live,
  including 2,000+ wood stockpiled at population 1 with zero food) is a real, unaddressed
  bias. A directive fix that named the correct action explicitly measurably worked but was
  reverted on principle (see `backend/instincts.py`) in favor of the leadership system,
  which as of this writing has not yet been observed to fix it in practice -- the two live
  runs since adding chiefs still showed heavy wood accumulation. Whether an emergent,
  in-fiction chief philosophy can actually out-compete this bias, versus a hard-coded
  directive being the only thing that reliably does, is an open, unresolved question.
- The era ladder only has 3 rungs and no branching — a linear Stone → Bronze → Classical
  path. Seasons/weather, an "inspiration" mechanic, and inter-tribe interaction (trade,
  conflict, travel) are all planned to hang off it next (see `backend/eras.py`'s docstring),
  rather than being bolted on independently.
- Drowning risk is tied specifically to the `GATHER_WATER` action on a river tile, not to
  movement — a tribe can path directly across a river tile with no risk as long as it
  doesn't choose to gather there. Full "crossing dangerous terrain" risk is a natural fit
  for whenever travel/movement gets built out more deliberately (inter-tribe interaction).
- Leadership (`backend/leadership.py`) is a one-time election at tribe creation with a
  static standing philosophy, not an ongoing role. Several natural extensions, none started:
  - A chief that periodically re-issues a fresh directive reflecting current tribe state
    (a second, less-frequent LLM call, to avoid doubling inference cost every tick).
  - The election deciding on an initial relocation, not just a philosophy -- e.g. "the
    newly elected chief decides the tribe should migrate toward reliable water." The
    *fact* of where the nearest water is would be legitimate map knowledge for the
    simulation to supply (a game master telling players what's nearby), while whether to
    act on it stays the model's own decision -- same category as the leadership system
    itself, not a scripted directive.
  - Leader-to-leader contact when two tribes' chiefs come into proximity -- today,
    proximity only means their people can overhear each other via the broadcast
    mechanism; nothing about chiefs specifically triggers a summit, alliance, or rivalry.
  - An "education" system unlocked once a tribe is established enough (a
    Classical-Age-or-later structure), which could be what finally triggers
    `genetics.py`'s still-unwired crossover -- a real generational knowledge-transfer
    event rather than just prompt splicing.
- Multiple individuals per tribe are a rendering layer only (deterministic wander animation
  keyed off population count) — there's still one authoritative (x, y) and one LLM call per
  tribe per tick, not per individual. Simulating individual members with their own
  micro-behavior would multiply inference cost by population size and isn't planned.
- Resource depletion only applies to the four harvest actions (wood/stone/water/game) --
  `BUILD_FIRE`/`CONSTRUCT_WALL` consume resources but don't deplete anything themselves
  (the fix there was the already-built no-op guard, a different mechanism). It also doesn't
  yet appear to be strong enough, on its own, to force relocation within the timeframe of a
  short observed run -- whether it does over a genuinely long run is untested.
- There's no sense of what a tribe's invented tokens actually *mean*, even to itself.
  `translation_matrix.py` tracks whether two tribes converge on the *same* token for the
  same action, but nothing tallies a single tribe's own token-to-action history to infer
  "this word is used with GATHER_WOOD 80% of the time." Would reuse the same co-occurrence
  idea already proven out in that module.
- `RELOCATE` currently moves at a flat `config.MOVEMENT_SPEED` regardless of context.
  Raised in conversation but not built: some notion of velocity/momentum (sustained travel
  in one direction building up speed, matching how a model committing to a multi-cycle
  RELOCATE run was the actual fix for the oscillation bug above) and of exposure while away
  from an established camp (a temporary camp mid-journey being more vulnerable than a
  built fire/wall) -- a real hazard-rate difference, not a scripted warning.
- No biome-specific fauna beyond forest deer -- rivers, oceans, and other biomes have no
  wildlife of their own yet, and `HUNT_DEER` succeeds the same way regardless of biome
  (including, oddly, in the ocean). Also unbuilt: a driftwood/raft mechanic -- floating
  material spotted along the river inspiring a boat, which would give the large wood
  stockpiles tribes tend to accumulate an actual use, and would be a concrete first
  instance of the "inspiration" mechanic that's existed only as a concept so far
  (observing something ordinary from a new angle and building something nobody
  specifically asked for).
