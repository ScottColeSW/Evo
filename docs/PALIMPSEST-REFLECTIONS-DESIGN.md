# Design: Palimpsest judging a chief's reflections (proposal, nothing built)

Written 2026-10-03 after reading `backend/memory.py` and its call site. No code in this repository has been changed. The
evidence check lives in the Palimpsest repository (`bench/integration/`, committed `ec8e034` and `a85d442`).

## What is wrong today (checked)

`TribeMemory.remember_reflection` strengthens an earlier reflection when a new one is similar enough (embedding cosine of at least
0.75; token overlap of at least 0.3 as the fallback). On 24 hand-written pairs, in each group the number reinforced was:

| Pair kind | Right behavior | Token path | Embedding path |
|---|---|---|---|
| restatement | reinforce | 4 of 6 | 6 of 6 |
| **reversal** ("trust the river clan" then "never trust the river clan") | do not reinforce | **6 of 6** | **6 of 6** |
| refinement ("only in good harvest years") | do not reinforce | 3 of 6 | **6 of 6** |
| unrelated | do not reinforce | 0 of 6 | 0 of 6 |

On reinforcement only the weight, cycle and timestamp change; the entry keeps its **first text**. After
`REFLECTION_STABILIZED_REINFORCEMENT_COUNT` (2) reinforcements, and only into an empty slot, that text becomes the standing
decree (`simulation.py`, the night cycle, about lines 2921 to 2945). So a chief who flips from trusting a clan to distrusting it
can end up with the old, trusting text as a decree.

**Not yet seen in real play.** The 498 saved run logs contain no promotion event (the mechanism is from 2026-09-18 and 19; the
only "standing decree" match is a chief's biography), and I found no reflection text in the logs I checked. The flaw is shown in the code and on constructed pairs. How often it happens in a real
run is unknown, which is why phase 0 below is logging.

## Goal and non-goals

Goal: a new reflection is judged against the held ones as a restatement (reinforce), a reversal (a visible disagreement), a
refinement or exception (its own entry), or unrelated; and nothing is promoted to a decree while it is in open disagreement.

Not goals: changing the per-turn `recall()` (it stays on token overlap, for the reason in its docstring); a gate against
untrusted claims (tribes interact through stances, not free-text claims, so there is nothing to guard); automatic resolution of
a disagreement (the chief resolves it, never the memory).

## The seam

`remember_reflection(text, cycle, weight, embedding)` is the one place that decides. Keep its signature and add an optional
`judge` to `TribeMemory`: a callable `(new_text, held_reflections) -> relation`. With no judge, or on any judge failure, behavior
is exactly today's. That matches the file's own rule that an embedding failure must degrade to the old path, never fail.

| Relation from the judge | What `remember_reflection` does |
|---|---|
| reinforces | reinforce as today (one entry, weight and cycle updated) |
| collides (a reversal) | store as its **own** entry; mark both `conflicts_with` each other; reinforce neither |
| exception / refinement (compatible) | store as its own entry; no reinforcement |
| new | store as today |

Promotion to a standing decree (the existing check) additionally requires that the entry has **no open conflict**.

An open conflict is shown to the chief at the next night cycle: both texts, in the chief's own prompt, as an inner
disagreement to settle ("you have held two opposing convictions about the river clan"). Settling it is the chief's choice (keep
one, merge, or leave it open); the memory records the choice and its reason and never decides. How to feed an unresolved
conflict into the tribe's emotional matrix (dissonance) is a later question, not part of the first version.

## Data model additions

Each reflection entry gets a stable `id`, `conflicts_with: list[id]`, and `resolution: None | {kind, reason, cycle}`. Entries that are
not reflections are untouched.

## Dependency

Palimpsest's judge needs `torch` and `transformers` below 5 (CPU only, about 0.4 s per judgment in Void Marauders), which Evo
does not carry today (`aiohttp`, `httpx`, `numpy`). So it is **optional**, as in Void Marauders (`MEMORY_JUDGE=nli`): a
`REFLECTION_JUDGE` setting, off by default, with the extra documented, not added to `requirements.txt`. The judge call is
synchronous, so the night cycle calls it through `asyncio.to_thread`. It runs once per night cycle per tribe (already off the
per-turn path), so latency is not a concern.

## Phases

0. **Log first (no behavior change).** Write each reflection, its nearest held reflection, the embedding similarity, and the
   reinforce decision to the run log, so the flaw's real frequency and the later effect can both be measured. This is useful even
   if nothing else is built.
1. **Judge behind a switch.** The seam, the table above, the no-open-conflict promotion rule, the conflict shown to the chief.
   Off by default.
2. **Taboos.** A permanent taboo contradicted by later experience (the tribe crosses the "deadly" ground safely) opens a conflict
   instead of staying permanent (Evo's own notes complain taboos accumulate for a tribe's whole life). Separate design; only
   after phase 1 is measured.

## How it would be evaluated (to be pre-registered before phase 1 runs; thresholds fixed then)

- **Offline, deterministic:** the 24-pair harness run against the real `TribeMemory` with the judge on. Expected: no reversal
  reinforced, at least 4 of 6 restatements reinforced.
- **From logs (needs phase 0 data):** the share of reflections that reverse a held one and were reinforced; decrees whose text
  contradicts the chief's latest reflection on the same subject. Before and after, same scenarios.
- **Not a primary measure:** game outcome (scores, survival). Void Marauders found no measurable benefit from memory on a
  scenario score, so a gameplay claim is not made unless the data shows it.

## Risks and limits

- A new dependency for one feature; judged worthwhile only if phase 0 shows the flaw is common.
- The judge is about 72 to 77% accurate on held-out cases and is weaker on small nuance; the safe error is filing a restatement as
  separate (it builds more slowly), not the reverse. A false collision costs the chief one extra decision at night.
- Chiefs' real reflections are model-written, longer and messier than the test pairs.
- Whoever resolves a conflict decides what the tribe believes; here that is the chief model, which can be wrong.

## Decisions

1. **Phase 0 (logging) first.** Decided 2026-10-03. Built the same day (see below).
2. **Judge off by default.** Decided 2026-10-03; applies to phase 1.
3. **No coupling to the emotional matrix, and no new prompt text.** Decided 2026-10-03. Evo is deliberately nudge-free and its own
   tests showed specific nudges did not work. The emotional matrix is driven by physical events at map locations (fires, raids,
   celebrations, era changes), never by a chief's thoughts, so there is no natural link, and wiring one would be an invented
   nudge. The judge's only effects are that a contradicting reflection is stored on its own instead of being merged, and that
   a belief in open conflict is not promoted to a standing decree.

## Phase 0, built (2026-10-03)

Logging only; no decision changes. `TribeMemory.last_reflection_trace` records, for each reflection, the method (`token` or
`embedding`), whether it reinforced, the similarity to the nearest held reflection, and that reflection's text.
`RunEventLog.record_data` writes it, with the reflection text, as a `reflection_memory` line in `logs/run_*.jsonl` (the line
keeps the usual `message`, so anything reading the log by that key still works). The night cycle writes it right after the
decision, best-effort: a logging failure cannot affect or interrupt the cycle. Four new tests (the full suite is 1,544 passing),
including one that runs the night cycle twice with a reversal and checks the log.

**How to read it after a few runs:** the share of `reflection_memory` lines with `reinforced: true` whose text is a reversal of
the `nearest` text is the flaw's real frequency; a similarity distribution on `reinforced: false` lines shows how close misses
come to the threshold. That tells us whether phase 1 is worth its dependency.

## Phase 1, built (2026-10-03), off by default

`config.REFLECTION_JUDGE = "off"` (the `REFLECTION_JUDGE` environment variable overrides it; `"nli"` turns it on). With it off,
nothing changes: the full suite passes unchanged. With `"nli"`, `backend/reflection_judge.py` builds Palimpsest's hybrid judge
(an NLI model plus the embedding model Evo already uses, CPU only, no language model) once per simulation, off the event loop.
`TribeMemory.remember_reflection` then asks it what the new reflection is to the held ones:

- **restatement:** reinforce, as today.
- **contradiction:** stored as its own entry, linked both ways in `conflicts_with`, neither reinforced.
- **anything else:** stored as its own entry.

The night cycle promotes a reinforced belief to a standing decree only if it has no open conflict. If the judge is missing or
errors, reflections use the built-in rules. There is no resolution step: a conflicted belief stays blocked until the other entry
is pruned by ordinary consolidation (a limitation: a chief who truly changes their mind and holds the new view must wait for the
old entry to age out). The judge's decision, relation and reason are written to the run log (the phase 0 line).

**Verified:** seven new tests (a fake judge, so they check Evo's wiring only: storage, conflicts, promotion, fallback, off by
default), the full suite at 1,551 passing, and the 24 pairs from `bench/integration/` run through Evo's real `TribeMemory` with
the real judge on: 0 of 6 reversals and 0 of 6 refinements merged, 5 of 6 restatements reinforced, 0 of 6 unrelated.
**Not verified:** a real run with the judge on. To try one: install Palimpsest with its NLI extra
(`pip install -e <path to Palimpsest>[nli]`), set `REFLECTION_JUDGE=nli`, and play as usual; the `reflection_memory` lines in
`logs/run_*.jsonl` then show what the judge decided.
