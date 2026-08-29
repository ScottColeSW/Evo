# Evolution2Civ

A local, spectator-mode sandbox where LLM-driven tribes (via [Ollama](https://ollama.com))
gather resources, invent their own language, and try to grow from a handful of survivors
into a founded settlement — entirely on their own. You pick which local model runs each
tribe from a picker screen, hit start, and watch.

This project started as a rambling voice-to-text brainstorm about factory patterns,
multi-agent adversarial games, and evolutionary civilization sims. That transcript is not
included here — it was full of search-engine UI chrome and citation ads, and its final code
dump didn't actually run (broken string ops, mismatched method signatures, an async loop
calling a synchronous HTTP client). This repo is a clean rewrite of the design that
transcript converged on.

## What's actually happening

- **`backend/world.py`** — a 100x100 tile landscape with four biomes (forest, mountains,
  river, plains) and a "ghost matrix": triumphant or traumatic events radiate outward and
  get quietly folded into future prompts near that coordinate as an unexplained ancestral
  instinct (dread or pride), without ever being a hardcoded rule.
- **`backend/memory.py`** — each tribe keeps an episodic memory (hash-based pseudo-embedding
  + cosine similarity, no extra model download required) that gets queried each turn and
  periodically consolidated into permanent cultural taboos, so context doesn't grow forever.
- **`backend/prompts.py`** — forces each tribe to reason privately in English but broadcast
  only in an invented phonetic language, so tribes have to develop and (mis)interpret their
  own shared vocabulary over time.
- **`backend/genetics.py`** — an optional crossover step that splices two tribes' ideology +
  lexicon into a descendant profile via the model itself.
- **`backend/self_mod.py`** — an opt-in (off by default) engine that lets a model rewrite
  `backend/physics.py` when turns get slow, validated with `ast.parse` and a cooldown lockout
  if the patch is broken. Flip on with `ENABLE_SELF_MODIFICATION = True` in `backend/config.py`
  if you want to see it. It's still model-authored code landing on your disk, sandboxed to
  one small file, so know what you're turning on.
- **`backend/simulation.py`** — the orchestrator: runs one turn per tribe per tick, applies
  actions, moves avatars, grows population on food surplus, and flags when a tribe crosses
  the population threshold to "found a city".
- **`backend/app.py`** — an `aiohttp` server serving the frontend, a `/api/models` endpoint
  that proxies Ollama's model list for the picker, and a `/ws` websocket streaming state.
- **`frontend/index.html`** — the tribe picker (choose a local model per tribe, up to 4) and
  the canvas theater that renders the grid, avatars, speech bubbles, fires, and a live stats
  sidebar.

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
- No persistence: closing the server drops the run. Add a snapshot dump if you want to
  resume or analyze runs later.
