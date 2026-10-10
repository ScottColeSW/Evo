# Measuring the app, not the models (2026-10-09)

The owner's report: the animation halts and nothing moves while the models are "thinking", and logging and double-checking are getting in the way of measuring the app.
This records what was measured, what causes the freezing, and the one switch that was built (`LEAN_RUN`). The optimizations below are **measured but not yet applied**.

## How it was measured

- **The app alone:** `scripts/app_profile.py` runs the real simulation with a stub in place of Ollama that answers at once (or after a fixed pretend delay), and times a tick the
  way `backend/app.py` `_tick_session` does it: step, snapshot, the SQLite board-history write, `json.dumps` for the websocket. It also records how late a 5 ms sleep wakes up
  (event-loop lag) and can run under cProfile. Fresh start (120 cycles) and mid-game fixtures (100 cycles from the 15,000-person states).
- **The real cadence:** the per-cycle timestamps in `logs/board_history.db` for two real browser runs (`run_20261008_130534`, 580 cycles; `run_20261008_141708`, 329 cycles).
- **The page:** read from `frontend/index.html` (`socket.onmessage`, `interpolatedTilePos`).

## What the player sees

| | run 130534 | run 141708 |
|---|---|---|
| gap between server updates (median / 90th percentile / longest) | 5.2 / 8.1 / 51.6 s | 5.0 / 8.1 / 133 s |
| updates later than the last gap by 15% or more | 20% | 18% |
| time the page stands still after a glide ended (share of all time) | 24% | 29% |

- **Why it stands still.** The page glides each moving thing from its previous position to its current one over `stateIntervalMs`, the gap between the last two updates, and
  `interpolatedTilePos` clamps progress at 1. When a model is slow the next update is late, the glide has finished, and everything holds still until it arrives. The sun and moon
  (`interpolatedCycle`) clamp the same way.
- **Long stalls dominate:** in run 130534, 41 of 579 cycles took over 15 seconds, and together they were **30% of the whole run's wall-clock time** (1,227 of 4,156 s). 31 of the 41 fall on
  cycle 0 or 1 of a 30-cycle day: the **night cycle**, about 40 seconds each, during which the game and the page both wait on the chiefs' reflections. The others: a chief's death and
  election (51 s at cycle 197), naming a settlement (48 s at cycle 21), and two of about 48 s with no logged event (consistent with a model loop-abort and its retry, which took 40 to 47 s each in the logged cases; not confirmed for these two).

## What the app itself costs (models removed)

Median / 95th percentile / worst, milliseconds per tick, default settings:

| | step | board-history write | tick total | event-loop lag over 100 ms |
|---|---|---|---|---|
| fresh start | 30 / 104 / 558 | 117 / 159 / 249 | 151 / 234 / 661 | 121 |
| mid-game | 44 / 494 / 666 | 128 / 172 / 256 | 183 / 634 / 791 | 119 |

- **The board-history write is the largest steady cost** (about 120 ms a cycle, 78% of a fresh-start tick). Each cycle opens a new SQLite connection, sets WAL mode, creates the
  table if missing, writes the whole snapshot as JSON, commits and closes, all on the thread that serves the page. Under cProfile, the commit (`__exit__`) and `close` are the cost: file-system
  flushing, not computation. The snapshot is 10 to 92 KB and `json.dumps` of it costs under 1 ms.
- **The step has spikes of 400 to 700 ms in mid-game** that block the event loop. The profile puts 65% of all time in `_prepare_turn` building each tribe's menu, almost all in the "is there
  room to build this" check (`_can_place` to `architect.find_free_slot`), which scans tiles and calls the uncached terrain function `world.biome_at` 2.5 million times in 200 turns (it evaluates
  sine waves, 45 million `math.sin` calls). `locked_building_facts` (extended on 2026-10-08) is 21% of the profiled time because it reaches the same check.
- **With a pretend 3.0 s model delay** a cycle took 3.62 s by default and 3.28 s with logging and checking off: about **0.34 s (9%) is the app's logging and checking**. The loop was blocked
  about 190 ms a cycle in total (up to 0.5 s at once) in the lean run and 400 ms in the default one.

## The page itself (measured in the browser pane, 2026-10-09)

Two real consecutive late-game snapshots from run 130534 (cycles 299 and 300, 139 KB and 142 KB) were loaded into the page. Receiving and drawing are cheap on this machine:

| | median | 95th percentile |
|---|---|---|
| `JSON.parse` of one update | 0.3 to 0.4 ms | |
| `updateSidebar` | 2.8 ms | 4.7 ms |
| `drawFrame` on the 625 x 625 board | 2.8 ms | 4.2 ms |

So the page is not slow, and a 140 KB message every few seconds is not a problem. The freezing is the glide design and the server cadence above, not rendering. A frame rate was not measured: the pane throttles an
unfocused tab to about 1 to 2 requestAnimationFrame calls a second, which is an artifact of the pane and not of the page.

## Measured effect of the changes (mid-game, 100 cycles, no model delay)

| | step median / p95 / worst | tick total median | event-loop lag over 100 ms | wall time |
|---|---|---|---|---|
| as it was, default | 44 / 494 / 666 | 183 | 119 | 26.4 s |
| as it was, lean (switched off by patches in the test script) | 215 / 400 / 704 | 216 | 58 | 16.1 s |
| `biome_at` cached, default | 34 / 75 / 196 | 159 | 100 | 16.8 s |
| `biome_at` cached, **lean** (the `LEAN_RUN` switch itself) | **4.5 / 50 / 219** | **5.5** | **1** | **1.7 s** |

The cached rows were measured with `functools.lru_cache` applied to `world.biome_at` inside the test script only; **no cache has been added to the code**.

## Built: `LEAN_RUN` (off by default; `tests/test_lean_run.py`)

`LEAN_RUN=on python run.py --port 8766` (or `config.LEAN_RUN = "on"`) turns off everything that exists to record or to double-check: the run-data records and the chronicle file mirror
(`event_log.py`), the board-history write (`board_history.py`), the decision journal and its read-back, the reflection judge and the Library shadow judging (the Palimpsest checks), and the raw
model transcript kept for the debug page. Gameplay, the prompt and every mechanic are untouched; a test checks the same turn gives the same result. The server prints a `[lean run]` line at
startup. In a lean run nothing is written to `logs/` (verified: 100 mid-game cycles wrote no file; the default wrote a database and a log).

Not switched off, deliberately: the model-reply handling that exists for robustness (salvaging a cut-off reply, one retry of an empty one, the token-repeat retry) and the failover, since those
change what the models' answers do rather than record anything.

## Applied 2026-10-10 (items 1, 2 and 4 of the earlier list)

Measured with `scripts/app_profile.py` on the real code, no model delay, median / 95th percentile milliseconds per tick (mid-game unless stated):

| | before | after 1 (terrain cache) | after 1 and 4 (and the cheap write) |
|---|---|---|---|
| step | 44 / 494 | 27 / 69 | 30 / 74 |
| board-history write (default settings) | 128 / 172 | 139 / 175 | **0.9 / 2.0** |
| whole tick (default settings) | 183 / 634 | 175 / 225 | **33 / 95** |
| event-loop lag samples over 100 ms | 119 | 100 | 5 |
| 100 cycles, wall time | 26.4 s | 18.1 s | **5.0 s** |
| fresh start, whole tick | 151 / 234 | 150 / 195 | 22 / 49 |

With `LEAN_RUN=on` as well, a mid-game tick is about 5 ms. The app's own work is now a small part of a cycle that the models make 3 to 5 seconds long.

1. **`world.biome_at` is cached** (`functools.lru_cache`, bounded at 131,072 tiles). A test checks it gives the same answer as working it out for every tile on and around the map, and `tests/conftest.py` clears it around
   every test so a test that changes the terrain inputs cannot see a stale answer.
4. **The board-history write is cheap.** One writer connection stays open per database file (WAL with `synchronous=NORMAL`), `board_history.close_all()` releases them (server shutdown and between tests), the snapshot is
   serialized once for the page and the history, and the page is sent first so a slow write cannot delay it. The cost is that a power loss can lose the last few cycles of history.
2. **The page, reshaped by the data.** The planned change (a longer glide that keeps moving past the expected arrival) turned out to address little: replaying the two real runs' timing, **81 to 88% of the standing-still time
   comes from waits over 15 seconds** (the night cycle, elections, loop-abort retries), which no glide covers, and a longer window cut ordinary standing still only from 29% to about 21% while adding lag. What was built instead is
   three small things: the gap is a **smoothed estimate** (after a 40 s stall the next glide window is 6.5 s, not 8 s, so one long wait no longer makes the next glide crawl and then hop), the glide lasts **1.3 times that estimate**
   so ordinary late updates no longer leave things stopped, and a **ring pulses around each living settlement while the page is waiting** (longer than 1.6 times the usual gap, never when paused or over) so a pause reads as waiting.
   Verified in the page: after a 40 s stall the estimate is 6.5 s and recovers to 5.7 s; at 5 s a 4-tile move is at 3.08 tiles (the old rule would be done); the waiting test is false when paused or over; real late-game states
   draw with no error, a frame costs 5 ms in either state, and the ring band around a settlement measurably changes color in the waiting state.
   The long stalls themselves are untouched: that is item 3 below.

## Still open, in order

3. **Take the night cycle off the critical path.** About 19 nights of 40 s in one run (roughly 18% of its wall time), plus elections and naming (about 48 s each). The game loop has no way to hand a reflection off and collect it
   later (no publish and subscribe), so this needs a real design: start the reflection calls as background tasks, keep stepping the day, and apply each result at a safe point in a later cycle. Not started.
5. **The other long waits:** a chief's election and a settlement's naming (about 48 s each) and model loop-aborts (about 40 s each) could run in the background or time out.
6. **Cut the scan itself:** with `biome_at` cached, `find_free_slot` is still about 18 ms a call; a smaller scan area or caching the result per territory state would remove most of it.

## Limits

The stubbed runs use random action choices, so the game states differ a little between runs and the step spikes depend on which actions were chosen; the medians are steady across runs, the
worst cases less so. The board-history write was measured on the H: drive where the project's `logs/` lives; another disk will differ. The real cadence comes from two runs of two models each.
The browser-side numbers are for one machine and one pane; the frame rate itself was not measurable there.
