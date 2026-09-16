# Evolution2Civ — System Design

**Status:** As-built design document
**Scope:** Current implementation as of 2026-09-16; no proposed changes
**Primary purpose:** A local, spectator-mode environment for observing how LLM-driven tribes make survival, settlement, social, and strategic decisions over time.

## 1. Product definition

Evolution2Civ (also called Project Chronos) is a browser-based sandbox in which up to four tribes are each controlled by a locally running Ollama model. The player selects the models and watches; they do not direct the tribes turn by turn.

The design deliberately separates **agent judgment** from **world enforcement**:

- The LLM chooses an allowed action, a target, private rationale, and public invented-language broadcast.
- The simulation deterministically validates and applies that choice.
- The world supplies scarcity, terrain, hazards, progression, infrastructure, diplomacy, and consequences.
- The UI visualizes the resulting state and chronicle in real time.

This makes the application both a game-like spectator experience and a practical testbed for comparing local models under the same evolving constraints.

## 2. Goals and non-goals

### Goals

- Observe autonomous LLM decision-making under persistent resource pressure.
- Make exploration, settlement, cooperation, conflict, and advancement emerge from mechanics rather than scripted agent behavior.
- Support direct, repeatable comparison of local Ollama models, both live (spectator) and headless (benchmark harness).
- Keep each browser run isolated, inspectable, and durable enough for later analysis.
- Make the LLM's inputs and raw responses visible through a dedicated debug view.
- Recover gracefully from a genuinely broken model rather than silently masking it or grinding to a halt.

### Non-goals

- It is not an individual-agent simulation: a tribe has one authoritative position and one LLM turn per cycle, even though the renderer depicts multiple people.
- It is not a globally shared multiplayer world: each normal WebSocket session owns an independent `Simulation`.
- It is not a semantic language-research system: the translation-confidence matrix is a deliberately lightweight empirical approximation (exact-phrase, same-action coincidence), not a real language model.
- Autonomous source rewriting is not part of normal operation; `SelfModEngine` (`backend/self_mod.py`) is opt-in and disabled by default (`config.ENABLE_SELF_MODIFICATION = False`).

## 3. Architecture overview

```text
Browser board / picker                 Debug browser page
        |                                      |
        | WebSocket START, controls             | WebSocket OBSERVE
        v                                      v
                    aiohttp application
       HTTP: /, /debug, /about.html, /books.html,
             /api/models, /api/scoreboard, /api/experiments
                         |
              one Simulation per owning socket
                         |
     +-------------------+-------------------+
     |                   |                   |
world + mechanics   prompt/LLM pipeline   persistence/analytics
     |                   |                   |
Landscape, hazards,    Ollama client,       JSONL chronicle,
eras, actions,         model-batch          SQLite board history,
memory, diplomacy      scheduler            scoreboard/experiments/benchmarks
                         |
                    local Ollama API

Headless path (no browser, no WebSocket):
run_benchmark.py -> Simulation.create(...) -> Simulation.step() loop -> benchmark_db.db
```

The server is intentionally small. `backend/app.py` hosts static HTML, exposes read-only API endpoints, owns WebSocket sessions, and runs a timed broadcast loop. `backend/simulation.py` is the aggregate root for an active run; it coordinates the subsystems but delegates individual action behavior to the registry in `backend/actions.py`. A second, entirely separate entry point (`run_benchmark.py`) drives the same `Simulation` headlessly for reproducible model comparison, with no websocket or frontend involved at all.

## 4. Major components

| Area | Main modules | Responsibility |
|---|---|---|
| Web delivery | `backend/app.py`, `run.py`, `frontend/index.html`, `frontend/debug.html` | Serve UI, own WebSocket sessions, stream snapshots, provide spectator controls and debug visibility. |
| Simulation orchestration | `backend/simulation.py` | Create tribes, prepare turns, advance the world, apply choices, progression, population, expeditions, game endings, and snapshots. |
| Agent interface | `backend/prompts.py`, `backend/ollama_client.py`, `backend/scheduler.py` | Build constrained prompts, call Ollama, parse responses, group same-model requests to reduce model swaps. |
| World model | `backend/world.py`, `backend/world_hydrology_data.py`, `backend/physics.py` | Deterministic 100×100 terrain (a pure function of coordinates, not seeded per instance), hydrology, fixed resource sites, constructions, resource depletion, trails, roads, and terrain-aware movement. |
| Action system | `backend/actions.py`, `backend/architect.py`, `backend/city_layout.py` | Registered action handlers, resource effects, construction placement, walls, expeditions, military, trade, diplomacy, and conquest. |
| Social/cultural state | `backend/memory.py`, `backend/leadership.py`, `backend/reflection.py`, `backend/translation_matrix.py`, `backend/ancestral_matrix.py`, `backend/instincts.py`, `backend/wellbeing.py` | Episodic memory, elected leadership, periodic self-review/dreams, empirical language convergence, location-based pride/dread, moment-to-moment survival alarms, and a five-tier Maslow read on overall condition. |
| Genetics & continuity | `backend/genetics.py`, `backend/breeding.py`, `backend/might.py` | Individual-level breeding/hatching (real, non-scripted LLM calls, not dice rolls), and a tribe's combat-readiness score from equipment/training/wellbeing. |
| Resilience | `backend/vram_guard.py`, `backend/self_mod.py` | Pre-flight VRAM sizing checks before assigning a model, and the (default-off) sandboxed self-modification engine. |
| Persistence & analytics | `backend/event_log.py`, `backend/board_history.py`, `backend/decision_log.py`, `backend/scoreboard.py`, `backend/experiment_log.py`, `backend/benchmark_db.py`, `backend/benchmark_scenarios.py`, `backend/benchmark_scoring.py`, `backend/tribe_fixtures.py` | Append-only chronicle, full per-cycle board snapshots, a derived flat decision table, cross-run tribe outcomes, A/B experiment results, and the headless benchmark harness (scenarios, raw-fact scoring, curated late-game starting fixtures). |

## 5. World model

The map is a deterministic 100×100 grid (`backend.world.biome_at(x, y)` takes no seed — every `Simulation` instance sees the same terrain), with an island frame/ocean, mountains, forests, plains, desert, volcano, river, lake, cliffs, and shoals. Hydrology and resource-site placement are repeatable rather than being re-rolled during play.

The landscape also holds mutable shared state:

- constructions and city/wall geometry;
- per-resource, per-tile scarcity that increases on harvest and regenerates globally;
- trails, whose wear gives movement benefit;
- permanent toll roads after repeated crossings, with first-trailblazer ownership.

### Era ladder

Advancement is automatic once real population and resource thresholds are met; it is not an LLM action. The current ladder (`backend/eras.py`, `ERAS`) has seven rungs:

1. Primitive Dawn
2. Cognitive Horizon
3. Tribal Synapse — unlocks `DECLARE_ALLIANCE`/`DECLARE_WAR`/`SPY`/Barracks
4. Monolithic Era
5. Object Creator Era (Dream Manifestation Machine)
6. War and World Domination — `DECLARE_CONQUEST`, Joint Castle
7. Beyond the Horizon — departure ending (`BUILD_VESSEL`)

Each era is data-driven: requirements, advancement costs, unlocked actions, announcement text, and city-founding eligibility are defined as data, not scattered conditionals. Action availability is further constrained by genuine state prerequisites (e.g. nothing diplomatic or military is offered before a Barracks exists; several early actions are unavailable before a tribe settles near confirmed water).

## 6. Agent-decision contract

Each tribe is prompted to reason privately in English while broadcasting publicly in a phonetic, invented language — deliberately drawn from a per-tribe example pool (`prompts.language_examples_for`), not one fixed set of literal example words, so different tribes in the same game don't all echo the same seed vocabulary back verbatim. The prompt supplies strict structure and JSON-oriented expectations because the intended local models are often small quantized models that can drift from a schema under long contexts.

The LLM is allowed to choose only from a context-specific action menu. Its response is not trusted as executable state:

- text is normalized and matched against available actions (`Simulation._resolve_action`);
- an unrecognized action falls back to a legible substitute and creates a correction fact for the next prompt, rather than a no-op (`IDLE` was deliberately removed);
- a *sustained* streak of unrecognizable decisions (not a one-off) is treated as a sign the model itself is stuck, not the prompt — `Simulation._handle_model_failure` escalates by swapping to a different locally available model, and if none work, lets the tribe fail for real (the run continues with any survivor);
- target vectors are meaningful only to actions that need them;
- mechanics, affordability, terrain, hazards, and action effects are enforced server-side.

This is the key trust boundary: **model output is a proposal; the simulation remains authoritative.**

### Informational nudges vs. mechanical enforcement

Two distinct systems shape behavior, and only one of them the model can safely ignore:

- **Nudges** — plain facts appended to the prompt (e.g. "the mine has produced ore, a forge would let it be worked into tools") meant to make an unlocked capability legible. These never change `available_actions` or force an outcome. `config.DISABLED_NUDGE_TAGS` is an opt-in, empty-by-default set for experimentally silencing one category at a time to measure whether a given nudge actually changes behavior (the standing project finding across many of these is that a fact alone often does not reliably redirect a small model's choice).
- **Mechanics** — real menu changes or automatic systems, used whenever a nudge alone proved insufficient. Examples: a survival crisis narrows the action menu to only plausibly-helpful actions; the Forge crafts items automatically once ore and wood are on hand (no manual action exists for this any more); a flock's Coop+Hatchery incubation runs off its egg stockpile regardless of the flock's current size, so it can recover from zero instead of being permanently stuck there.

## 7. Mechanics and feedback loops

### Survival and settlement

Food and water upkeep scale with population. Scarcity, harvesting hazards, drowning risk, and local depletion create pressure to scout, relocate, diversify food production, and establish resilient infrastructure. A tribe that stays near confirmed water long enough becomes settled, unlocks broader possibilities, and can acquire passive water/food systems. "Food secure" specifically requires a Kitchen *and* a genuinely proven passive source — fishing actually learned (not merely a Fishery built; the passive catch starts the moment fishing is learned, a Fishery only multiplies it) or at least one real harvest ever brought in.

### Exploration and geography

`SCOUT`, exploration parties, hunting parties, and `RELOCATE` are distinct. Scouting discovers terrain and sites without relocating the tribe; expeditions are multi-cycle and findings are only realized when they return. Relocation is the ordinary tribe movement mechanism and costs food and water. Terrain-aware movement, trails, roads, and tolls make routes part of the strategic world state.

### Economy, city-building, and defense

The action registry covers gathering, hunting, farming, eggs/fishing, crafting, storage, walls and rings, houses, docks/fisheries, workshops, mines, military facilities, libraries, and object creation. Construction has placement, affordability, and precondition checks. City layout, rather than an abstract defense value, models real wall sections and natural barriers. Turn resolution order within a cycle rotates round-robin by cycle number (`Simulation._round_robin_order`) rather than always favoring the same tribe, so no side keeps a permanent first-mover edge in a same-cycle direct conflict.

### Diplomacy and conflict

`DECLARE_ALLIANCE`/`DECLARE_WAR` set a persistent, symmetric stance. An alliance overture is a real gamble, not a guaranteed success: the initiating tribe's own physiological wellbeing (the most volatile of the five Maslow tiers, unlike the others which only ratchet upward) drives a backfire chance — secure and it basically always lands, starving and it can blow up into an immediate, single-round clash and a WAR stance instead of the peace that was asked for. `DECLARE_CONQUEST` (War and World Domination era) is the decisive, higher-stakes version: a bounded multi-round war of attrition with real population loss each round, ending in one side's absorption (`Simulation._merge_tribes`), a surrender, or a costly stalemate. Both battle types render through the same "Die of Battle" presentation layer in the frontend — each round's real, already-decided win chance is translated into a die size and target number purely for visual/narrative effect; the die never itself decides anything.

### Culture and cognition

Memory uses compact hash-based pseudo-embeddings and cosine similarity to retrieve relevant episodes, then periodically condenses them into longer-lived cultural lessons. The emotional matrix independently imprints pride or dread around locations and can increase inference temperature during distress. Leadership begins with an LLM-generated contest and philosophy; a periodic night cycle can reflect on and evolve guidance (collapsing repeated event types to the single latest instance, so an automatic, frequent event like an egg hatch never crowds out everything else worth reflecting on).

### Inter-tribe interaction

Tribes only hear broadcasts within a configured distance, wide enough to cover every default spawn pairing once both are settled (a past radius was tighter than even the closest possible spawn pairing, making convergence structurally unreachable). The translation matrix treats matching phrases used for the same action as empirical evidence of shared vocabulary, with decay when not reinforced. Trade, alliances, war, raids, espionage, conquest, and minor settlements provide the material interaction layer.

## 8. External interfaces

| Interface | Contract |
|---|---|
| `GET /` | Main picker and spectator board. |
| `GET /debug` | Separate raw prompt/response observer view (attaches to the most recently started session via `OBSERVE`, doesn't own its own `Simulation`). |
| `GET /about.html`, `GET /books.html` | Static pages, linked from the picker footer. |
| `GET /api/models` | JSON list of models reported by local Ollama; empty if unavailable. |
| `GET /api/scoreboard` | Cross-run tribe outcomes and per-model aggregation. |
| `GET /api/experiments` | Recorded A/B experiment runs and summaries. |
| `GET /ws` | Commands: `START`, `OBSERVE`, `TOGGLE_PAUSE`, `ADD_TRIBE`, and `STOP`; server streams state snapshots. |
| Ollama API | Local HTTP service, normally `http://localhost:11434`, used for model listing, inference, and model lifecycle operations. |
| `run_benchmark.py` (CLI, not HTTP) | `--scenario {survival,settlement,cooperation,conflict,war_ready_5000,all}`, `--models`, `--trials`, `--seed-base`, `--report` — headless trials against the same `Simulation`, no browser involved. |

## 9. State durability and observability

The live UI only needs a limited narrative window (`Tribe.history` keeps the last 6 entries for display), so the system persists more durable records separately:

- `logs/run_<timestamp>.jsonl` — the complete narrative chronicle, one line per event, for after-the-fact analysis a truncated in-memory list can't support.
- `logs/board_history.db` — a full per-cycle board snapshot (every tribe's complete state), the primary source for "what was true at cycle N."
- `backend/decision_log.py` — a derived, flat, queryable decisions table materialized from board snapshots on demand (not written live).
- `backend/scoreboard.py`, `backend/experiment_log.py` — lifetime tribe outcomes and A/B hypothesis-test results, both append-only JSONL.
- `logs/benchmark_results.db` (`backend/benchmark_db.py`) — headless trial results (raw facts only; scores are computed at report time from a versioned formula, never stored, so a formula change never silently invalidates historical data).
- `backend/fixtures/*.json` (`backend/tribe_fixtures.py`) — curated, real tribe snapshots (e.g. `war_ready_5000`'s population-~5,000, Barracks-and-Battalion-already-built pair) that a benchmark trial can start from instead of the usual population-8 grind. Deliberately narrower than a live snapshot: identity and relationship state are never copied, only a tribe's own earned capability.

## 10. Analytics beyond this repo

A "Developments" page — population, era progression, well-being, civilization-building, mutual understanding, and reliance-on-nudges curves across real runs, plus a cross-run project-progression view — exists today as a published, externally-hosted artifact, linked from the game-over splash (`frontend/index.html`). Building it as a real in-app page (a `run_benchmark.py`/`board_history.db`-backed endpoint and a dedicated frontend page) was scoped but explicitly deferred as unnecessary standing infrastructure for now; the artifact link is the current, lighter-weight substitute.

## 11. Known limits

- Single-machine, single-Ollama-instance assumption throughout (VRAM guard, model unload/reload lifecycle).
- No authentication or multi-tenant isolation — this is a local spectator tool, not a hosted service.
- The translation-confidence matrix is an empirical proxy for language convergence, not a semantic model; it can plausibly stay at zero for an entire run even with real, sustained contact if the two tribes never happen to phrase the same action identically.
- Benchmark trial reproducibility covers every gameplay-relevant roll (`random.seed`), but not the LLM's own sampling — multiple trials per (scenario, model) matter more than seeking bit-identical runs.
