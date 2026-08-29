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

- **`backend/world.py`** — a 100x100 tile landscape with four biomes (forest, mountains,
  river, plains) and the structures tribes build on it.
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
  action every turn (broadcasts are audible regardless of distance, a deliberate
  simplification for a handful of tribes). Confirmed live: two different local models
  (`gemma2:2b`, `qwen2.5:3b`) converged on a shared token for the same action within 4
  cycles in testing. Surfaced in the sidebar as a "Linguistic Consensus" panel per tribe
  pair.
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
  resource requirement gating advancement into the Bronze Age.
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
  the canvas theater that renders the grid, avatars, speech bubbles, fires, and a live stats
  sidebar.

Spawn points are one-per-biome (`SPAWN_POINTS` in `backend/simulation.py`) so the default
picker order (Forest Tribe, Mountain Tribe, ...) actually starts each tribe in the biome
its name implies.

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
- The only source of ancestral trauma is the forest hunting hazard
  (`config.HUNT_HAZARD_CHANCE`). If you want dread from other causes (inter-tribe conflict,
  starvation, a harsh winter), add the event and call
  `self.trauma.radiate_event_wave(x, y, negative_magnitude, radius)` from `simulation.py`.
- No persistence: closing the server drops the run. Add a snapshot dump if you want to
  resume or analyze runs later.
- The era ladder only has 3 rungs and no branching — a linear Stone → Bronze → Classical
  path. Seasons/weather, an "inspiration" mechanic, and inter-tribe interaction (trade,
  conflict, travel) are all planned to hang off it next (see `backend/eras.py`'s docstring),
  rather than being bolted on independently.
- Water only has one consumer right now (era advancement cost/requirement) — there's no
  ongoing upkeep or drought mechanic yet, so a tribe that stops gathering water never
  actually suffers for it outside of failing to advance eras.
