# Design: the night reflection as a background task (proposal, nothing built; 2026-10-10)

Written after the performance measurements in `docs/APP-PERFORMANCE-2026-10-09.md`. Nothing in the code has been changed. The owner's words: "run the reflection as a background task and apply the result
a few cycles later" is an idea that "could work", provided it still gives the tribes their values and the observers (us) the record.

## The problem, measured

In run 130534, 41 of 579 cycles took over 15 seconds and together were 30% of the run's wall-clock time; 31 of those 41 are the night cycle (cycle 0 or 1 of each 30-cycle day). Inside two real nights:

- cycle 60: the first tribe's reflection record appeared **36.7 s** after the previous decision, the second tribe's **3.4 s** later, and the next day's first decision **22.6 s** after that (63 s in all);
- cycle 150: 27.7 s, then 2.9 s, then 15.2 s (46 s in all).

The second tribe's reflection is fast, so the model is not slow at reflecting. The long waits are the first reflection (the reflection model, gemma2:2b, being loaded) and the first decision after it (the tribes' models being loaded
again). On the 8 GB card two different tribe models, the reflection model and the embedder do not all fit, so the night is mostly model loading and unloading, not thinking.

## Check this first (a pre-registered decision rule)

One tribe model plus the reflection model should fit in memory together, so the swapping should mostly disappear. The owner is running both tribes on one model (first qwen2.5:3b, then possibly gemma2:2b) with lean off, which
records timestamps for every cycle.

- **If the median gap at cycles 30, 60, 90 and so on is under 10 s** in those runs: the stall was swapping, and this design is not needed. Record that and stop.
- **If it stays long:** build this.
- The same runs also show whether concurrency between a background reflection and the day's turns would fit in memory, which this design depends on.

## What the night cycle does today (`Simulation._run_night_cycle`, called from `step()` for each tribe in turn)

1. **Quick state updates** (no model): population pressure, the peace gate, the Library shadow judging, the night watch record.
2. **The model work:** gather the recent history and an inventory, call `reflect_on_history` on the reflection model, then (if there is a private thought) embed it and run the reflection judge (a CPU model, off the event loop).
3. **Apply the result:** the reflection counter, the stored memory and its reinforcement, promotion of a repeated thought to a decree, the new philosophy, a proposed award, decree and dream (including the departure dream), and a random chance to start a family.

All tribes run one after another, so the game waits for the sum.

## Proposed design

Keep step 1 as it is. Split steps 2 and 3:

- **Start:** at the night, for each living tribe with a chief, copy exactly the inputs step 2 reads (recent events, inventory, philosophy, decree, DMM and departure flags, the chief's name and the cycle) and start one `asyncio` task per tribe that makes
  the model call and the embedding, touches no tribe state, and returns a result bundle. Remember it in `self._pending_reflections[tribe_id]`.
- **Apply:** at one fixed safe point, the top of `step()` before turns are prepared, look at each pending task. A finished one is applied by `_apply_reflection(tribe, bundle)`, which is the existing step 3 code moved into its own method unchanged. The judge still runs in
  a thread at that point.
- **How late:** applied at the first step after it finishes, never in the same cycle it started, and **never later than a deadline of K cycles** (a config value, suggested 3): at the deadline the step waits for the task. That keeps the lag bounded and the science reproducible: a result lands at
  night + (1 to K) cycles, and the record says which.
- **Policies:** at most one pending reflection per tribe (a second night while one is still running is skipped and logged); a task is discarded, with a logged reason, if the tribe went extinct, was conquered, or its chief changed since the start (a reflection by a dead chief
  must not become the new chief's decree); a timeout (suggested 120 s) discards a hung call; game over and `shutdown()` cancel every pending task and unload the reflection model.
- **What the tribes get:** the same philosophy, decree, dream, award and memory, applied a few cycles later. `tribe.last_reflection_cycle` keeps the night's cycle so the thought bubble shows at night as it does now, and the applied cycle is recorded alongside.
- **What the observers get:** a `night_reflection` record in the run log with `started_cycle`, `applied_cycle`, `latency_s`, `model` and any `discarded` reason, and a "reflecting" marker the page can show until the thought arrives. Existing records (`reflection_memory` and the rest) are unchanged apart from their cycle.
- **A mode switch:** `config.NIGHT_REFLECTION_MODE = "inline"` (today's behavior, the default at first, and what the benchmark harness and every existing test keep using) or `"background"`. The first live comparison is a straight A/B of the two modes.

## Risks, stated plainly

- **Memory contention can make it worse.** A background reflection loads the reflection model while the day's turns run. If it does not fit beside the tribes' model, the two compete and every turn gets slower. This is why the check above comes first, and why inline stays available.
- **The random family chance** (`NIGHT_CYCLE_RANDOM_BREED_CHANCE`) is rolled in the apply step, so seeded runs draw it at a different point than today. Seeded benchmark runs stay in inline mode for that reason.
- **A decree or philosophy that arrives later** changes what a tribe is told a few turns after the night, not at it. That is the point of the design and the thing to watch.
- **Concurrency bugs** are the main engineering risk: nothing in a background task may read or write tribe state, and everything it needs is copied before it starts.

## A cheaper step that changes no meaning

Today the tribes' reflections run one after another. Running them together (`asyncio.gather` over the model calls, applying the results in tribe order) would halve the model part when both tribes share a model, with no change in what any tribe receives.
It does not help the model-loading cost, so it is not a substitute for the check above.

## Tests it would need

Cycles keep advancing while a reflection is pending; the applied result equals what inline mode gives for the same model reply; discard on extinct, conquered and changed chief; one pending per tribe; a deadline forces the wait; a hung call times out; game over cancels the tasks;
`inline` mode is byte-for-byte today's behavior; the records carry started and applied cycles. A fake client with controllable latency makes all of these deterministic.

## Questions for the owner

1. Is "applied at the first step after it finishes, never later than 3 cycles" the right meaning of "a few cycles later", or should it be a fixed number of cycles for cleaner science?
2. Should the page show a "reflecting" marker until the thought arrives?
3. If memory contention turns out to be the limit, would you accept the reflection reusing the tribes' own model when only one model is in play (no loading at all), at the cost of the dedicated reviewer that pushes back (the choice made on 2026-09-18)?
