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
- **`backend/genetics.py`** — an optional crossover step that splices two tribes' ideology +
  lexicon into a descendant profile via the model itself.
- **`backend/self_mod.py`** — an opt-in (off by default) engine that lets a model rewrite
  `backend/physics.py` when turns get slow, validated with `ast.parse` and a cooldown lockout
  if the patch is broken. Flip on with `ENABLE_SELF_MODIFICATION = True` in `backend/config.py`
  if you want to see it. It's still model-authored code landing on your disk, sandboxed to
  one small file, so know what you're turning on.
- **`backend/simulation.py`** — the orchestrator: builds every tribe's prompt (no network
  calls), hands them to the scheduler as one batch, applies the results, moves avatars,
  grows population on food surplus, rolls a hunting hazard (wolves, forest biome only —
  the thing that actually produces ancestral dread), and flags when a tribe crosses the
  population threshold to "found a city".
- **`backend/app.py`** — an `aiohttp` server serving the frontend, a `/api/models` endpoint
  that proxies Ollama's model list for the picker, and a `/ws` websocket streaming state.
- **`frontend/index.html`** — the tribe picker (choose a local model per tribe, up to 4) and
  the canvas theater that renders the grid, avatars, speech bubbles, fires, and a live stats
  sidebar.

Spawn points are one-per-biome (`SPAWN_POINTS` in `backend/simulation.py`) so the default
picker order (Forest Tribe, Mountain Tribe, ...) actually starts each tribe in the biome
its name implies.

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
- `genetics.breed()` is wired but not yet triggered automatically on any era/generation
  transition — call it from `simulation.py` when you decide what should trigger a handoff.
- The only source of ancestral trauma is the forest hunting hazard
  (`config.HUNT_HAZARD_CHANCE`). If you want dread from other causes (inter-tribe conflict,
  starvation, a harsh winter), add the event and call
  `self.trauma.radiate_event_wave(x, y, negative_magnitude, radius)` from `simulation.py`.
- Only one simulation runs at a time, shared across every connected websocket client — a
  second browser tab sending `START` replaces the first tab's run. Fine for solo spectating,
  not for multiple simultaneous viewers.
- No persistence: closing the server drops the run. Add a snapshot dump if you want to
  resume or analyze runs later.
