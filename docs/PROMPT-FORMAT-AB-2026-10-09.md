# Prompt size, the compact format A/B, and what two days of testing found (2026-10-08 and 2026-10-09)

This records what was measured, what was found, what turned out to be nothing, and what is not yet known. Numbers come from the files named under each
heading. Where an earlier claim of mine turned out wrong, it is corrected in the last section.

## Short version

- The A/B asked whether a shorter prompt (the "compact" format) changes what tribes do. **It does not make a turn faster** (3,283 ms against 3,291 ms) and it
  **shortens the prompt by 10 to 18%**. Outcomes were the same except one: tribes in the compact arm built the Dream Manifestation Machine in 1 of 6
  tribe-runs against 5 of 6 in the full arm. That could be chance (about an 8% chance, 6 against 6) and has not been tested further. **The compact format is
  not recommended for play yet.** It is committed and off by default.
- The more valuable results were defects the testing exposed, all fixed (list below): replies cut off partway through the JSON (18% of one model's turns),
  stale or incomplete action descriptions, an action offered when it could do nothing, and a silent model switch that invalidated one run.
- A prompt is 11,000 to 14,000 characters, about 4.5 characters per token, at most 3,117 tokens against the 4,096 context in the largest case tested. **Nothing
  is cut today.** The action descriptions are about 30% of a prompt and the entity list about 21%.
- Palimpsest is already wired into the reflection memory (phase 1). In two long runs it judged 58 reflections and reinforced 3; no reflection ever reached
  a decree. Nothing found here depends on memory. See the section on it.

## Defects found and fixed

| Commit | Defect | Evidence |
|---|---|---|
| `78ae314` | A model reply that stopped partway through the JSON was thrown away whole, and the game ran the first menu item instead. | gemma3:4b: 37 of 201 turns (18%) against 0 for llama3.2. Two raw replies still in the debug view showed `GATHER_STONE` chosen and `TRADE` and `EXPLORATION_PARTY` run. The client now keeps the finished key and value pairs and asks once more when none finished. |
| `0b5e6b5` | A run stalled in the Monolithic era: the mine and forge chain needs a first gather that small models never chose. | A scout's visit to a site now brings a small sample (5 wood, stone or food; 3 of a vein's own ore) and sets the first-gather flag. Confirmed live: homecomings in the next run named the amounts (a full store marks the site and adds nothing). |
| `ff82a66` | BUILD_BARRACKS did not mention it needs a kitchen. BUILD_LIBRARY and RESEARCH still said they shorten the path to the next era (untrue since 2026-10-04). FORGE_ITEM was on the menu before a forge existed, where it does nothing. | Read from the code while mapping the keep, barracks, library, mine and forge chain. |
| `89540bc` | A tribe was told "still short on: a barracks, a keep" for 340 turns and nothing said what a keep needs. | Every building of the chain now says what it lacks, what the tribe has and what it would open. A test walks over 1,000 tribe states and checks the explanation always agrees with the real menu rule. |
| `2032182` | The same water site appeared up to three times in a prompt (memory, taboo and the confirmed-source line). | Measured on a live prompt. |
| `6d56411` | (Not a defect: the compact format itself, off by default.) | See the A/B below. |

Also found, not a code defect: the game's failover switches a tribe to another local model after 10 unusable turns in a row. During the early-game batch
that silently turned one run into a test of hermes3:3b (see the invalid run below). The test script now stops such a run at once and marks it invalid.

## Prompt size, measured

Measured on saved late-game states (run `run_20261008_130534` at cycles 150 and 500) and one live early turn.

- **Size:** 12,356 characters for the early turn, 11,000 to 14,000 late. A cache-busting probe against phi4-mini (a unique salt on the front, one token asked for)
  gave 2,436 to 3,117 tokens, so about 4.5 characters per token. The context is 4,096. If a prompt went past it, Ollama would drop the start of the
  prompt without an error; today there is about 25% headroom. A permanent token counter in the client was rejected because Ollama counts only newly evaluated
  tokens when it reuses a cached prefix, so it would under-report.
- **Where the characters go (early turn):** action descriptions 3,654 (30%), entity list 2,585 (21%), role text 1,010, JSON schema 1,073, the movement notes
  1,126, the rest in short layers. Late in a run the entity list is 3,254 of 13,559 (24%). The entity list is mostly the same sentence frame repeated for every
  site: timber groves 15%, danger spots 11%, veins 10%, game sites 9%, stone 8%, water 6%.
- **Memory is a small part of it:** two memory lines, three taboos and one recalled private thought came to about 400 characters of the early prompt (about 3%),
  and none appeared in the late-game turn. Making memory smarter cannot make a prompt much smaller.

## The compact format (`config.PROMPT_FORMAT`, default "full")

Same facts, shorter wording. Short subject-verb-object sentences in a fixed order: what it does, limits and risks, cost, what it opens, which eras require it.

- 52 action descriptions rewritten (the era-chain buildings included), 22% shorter than the full text for the same actions (12,624 against 16,253 characters),
  plus one shared 315-character line for what SCOUT, EXPLORATION_PARTY and HUNTING_PARTY have in common.
- The entity list groups each kind of repeated site line into one line per kind, with every coordinate kept, and the settled, wall and fishing sentences are
  plainer and shorter. On saved late-game states the whole prompt came out 18% shorter on average (13,048 to 10,609 characters). In the A/B the saving was
  9.7% mid-game (15,470 to 13,964) and about 12% early game (14,155 to 12,370).
- `tests/test_prompt_format.py` checks that no number, number word or named action of the full text is lost, that every coordinate survives grouping, and that the
  default prompt is unchanged. It caught two real losses while the text was written ("a second layer" on the moat, "not a one-time exchange" on alliances).

## The A/B

Harness: `scripts/ab_test_nudges_off.py --knob prompt_format` (also `--model`), two tribes on one model, the same seeds in both arms, arms interleaved, results in
`scripts/ab_test_prompt_format_results.json`. `scripts/ab_compare.py <results file> <mode>` prints the comparison. Per run it records mean prompt length and turn
latency from each tribe's transcript.

### Mid-game: qwen2.5:3b, 100 cycles from the 15,000-person fixtures, 3 seeds per arm, 6 valid runs

| | full | compact |
|---|---|---|
| mean prompt length | 15,470 | 13,964 (9.7% shorter) |
| mean turn latency | 3,283 ms | 3,291 ms |
| final population (mean, min) | 27,841 (19,791) | 29,723 (28,695) |
| final eras (6 tribe-runs) | 3 War and World Domination, 3 Dream Manifestation | 1 War and World Domination, 5 Dream Manifestation |
| first Dream Manifestation entry (mean cycle) | 329 | 318 |
| forge, mine, fortress, castle built | all 6 | all 6 |
| **Dream Manifestation Machine built** | **5 of 6** | **1 of 6** |
| decisions that changed nothing | 23.7% | 25.7% |
| extinctions | 0 | 0 |

- **Speed:** per-run latency was 3.2 to 3.3 seconds in every run of both arms. A turn is dominated by the model writing its answer, not by reading the prompt.
- **The Dream Manifestation Machine:** when BUILD_DMM was on the menu, the full arm picked it on 5 of 56 offered turns and the compact arm on 1 of 153. Turns within
  a run are correlated, so the honest unit is the tribe: 5 of 6 against 1 of 6 (exact two-sided p of about 0.08). BUILD_DMM's own text is not rewritten, so
  the shift comes from the text around it. The first, less-trimmed compact wording built it in 5 of 6 (same as full), so something in the second version moved
  it. Which part (descriptions, the grouped entity list or the shorter notes) was not isolated.
- The first compact wording (`scripts/ab_test_prompt_format_results_v1_first_compact.json`): prompt 13,987 against 14,598, latency 3,334 against 3,700 ms (the
  gap came from one slow full-arm run, not from the format), the machine built in 5 of 6 in both arms.

### Early game: phi4-mini:3.8b, 250 cycles, 2 seeds per arm, **3 of 4 runs valid**

Valid: seed 1000 full, seed 1000 compact, seed 1001 compact. Too few to compare. All tribes survived and grew; every one reached the Tribal Synapse or Monolithic
era; compact tribes entered Cognitive Horizon at cycle 58 and Tribal Synapse at 110 against 70 and 124 for the one full run; the weaker tribe of each pair was
6,155 and 9,190 people in the compact runs against 16,455 in the full run. Latencies (3.6 to 6.6 seconds) are not usable, because the card was shared with
extra models for part of the batch.

**Invalid run (seed 1001, full arm), and why it was stopped.** From its first turn phi4-mini's requests were failing: Ollama returned HTTP 500 after 11 to 47 seconds
(41 of them in one hour, against 3 to 6 in other hours), which the client reads as a token-repeat loop abort; the retry failed too. After 10 unusable turns in a
row the game's failover switched both tribes to other local models (hermes3:3b, then a digest-named entry), which loaded two more models on the 8 GB card and
slowed every later turn to about 23 seconds a cycle. The chronicle line "falls silent mid-thought" marks a switch; none of the other nine runs has one. The run
was stopped by hand and left out. The script now stops and marks any run where a tribe's model changes or a tribe has 5 unusable turns in a row, and records
the loop-abort, retry and truncation counts per run. phi4-mini's loop rate looks high enough to matter on its own and was not measured per arm.

### What this does and does not show

Shows: shorter wording leaves growth, buildings and era timing the same on qwen2.5:3b mid-game, saves no time, and moves at least one late-game choice. Does not
show: anything for other models, other starting points, or the early-game choices the compact descriptions mostly cover (the early batch is incomplete). Six
tribe-runs per arm, one model, an unseeded model sampler.

## Model results logged (not patched)

Decision (2026-10-08): these are honest results about each model, not something to script around. Nothing builds for a tribe or chooses for it.

| Model | First-era food (PLANT_CROP, COOK_FOOD, CATCH_FISH) |
|---|---|
| gemma2:2b | never chose any in 323 turns, all offered 307 times; stayed at 15 to 30 people on a foraging ceiling of 1.8 food per gather |
| hermes3:3b | 1-food gathers for about 150 cycles; cooked at cycle 158, first crop at 216 |
| phi4-mini:3.8b | first crop at cycle 141, cooked at 153, fished at 155, then grew past 15,000 |
| qwen2.5:3b, llama3.2, gemma3:4b | farmed early and grew fast |

Late gate buildings (`run_20261008_130534`): gemma3:4b was offered BUILD_KEEP 340 times from cycle 131 and never chose it, so it stayed in Tribal Synapse at 20,000 to
30,000 people. llama3.2 reached the Monolithic era at cycle 356, built its mine at 399, then was offered BUILD_FORGE 108 times with 281 of the mine's ore in
stock and never chose it. The prompt already told gemma3:4b "still short on: a barracks, a keep". A fact on the menu did not move these models.

## Palimpsest

Palimpsest (`H:\pet_projects\Palimpsest`) is a belief store that judges whether a new claim reinforces, collides with or coexists with what is held. The
question was whether it would help here.

- **It is already in.** Phase 1 of `docs/PALIMPSEST-REFLECTIONS-DESIGN.md` is built (`backend/reflection_judge.py`, `config.REFLECTION_JUDGE`, on in the two browser
  runs). The offline check that motivated it is in the Palimpsest repository (`bench/integration/`): on 24 hand-written pairs Evo's old reinforcement
  reinforced all 6 reversals on both paths and all 6 refinements on the embedding path, and Palimpsest reinforced none of them and 5 of 6 restatements.
- **How often it matters in real runs:** across `run_20261008_130534` and `run_20261008_141708` the judge made 58 decisions: 18 new, 29 coexists, 8 collides and 3
  reinforces. **No reflection reached a decree** (it takes 2 reinforcements). The decrees seen in those runs were written directly by the chief model.
- **Memory is about 3% of the early prompt** and absent from the late one, so it cannot shrink the prompt either.
- **None of the problems found here is a memory problem:** cut-off replies, repeat loops, the model failover, the first-era food trap, the unchosen gate
  buildings and the compact format's shift on one action are all model or prompt behavior. Palimpsest's own README says what is not yet measured is whether having
  the memory improves an agent's answers at all.
- **Where an expansion could be tested, in order (proposed 2026-10-09, nothing run):**
  1. **The journal read-back, already built and never compared** (`docs/CHIEF-EVIDENCE-MEMORY-DESIGN.md`, Step 2; `JOURNAL_READBACK`, off by default). It tells the
     chief plain facts such as "In your last 10 choices you picked GATHER_STONE 6 times: stone +600; nothing was built." That is the observed problem (gemma2:2b
     gathering food at 1.8 per gather for 250 cycles, gemma3:4b repeating TRADE and RAID with a keep on the menu). Run it off and on with the harness on gemma2:2b
     and gemma3:4b and count repeats and gate-building adoption. This is also the question Palimpsest's own README lists as unmeasured: does stored material change a
     judgment compared with its absence.
  2. **Only if that moves anything:** Palimpsest judging a chief's stated philosophy or decree against the journal's evidence, so a mismatch ("diversify resources"
     while 12 of the last 12 choices were stone) is a visible collision the chief can see. Palimpsest's judge exists for claim against evidence; Evo does not yet
     use it that way (a proposed decree is written without being judged against the philosophy).
  3. **A fidelity check for prompt compaction:** Palimpsest's CPU NLI (`NLI.compare`) can test whether a compact description still implies each sentence of the full
     one, which covers what the number and name check in `tests/test_prompt_format.py` cannot.
  The project's own finding (four instances so far) is that a fact in the prompt does not move small models, so 1 may well show nothing; that would be a result.
- **Conclusion:** no evidence from this work that more Palimpsest would help. Where it was designed to (reflection reinforcement) it is installed and rarely
  fires. If its effect is to be measured, it needs runs long enough for reflections to recur, and with three reinforcements in 58 judgments the expected
  effect on play is small.

## Corrections to what I said earlier in these sessions

- I called the memory judge's cycle-90 "reinforces" (NLI 1.00) an over-credit of an unrelated thought. That is arguable: Palimpsest deliberately treats "the held
  claim already implies the new one" as a restatement (its README documents the choice and its cost), and the real rate was 3 in 58 with no promotions.
- I described the standing decrees as promoted from reinforced thoughts. None was; the chief wrote them. Decrees that read as commands ("practice raiding, with a
  focus on defensive tactics") and the self-repeating philosophy text are still real observations, but not through the promotion path.
- I first read a 10% latency gain into the first A/B. It came from one slow run; the second A/B shows no difference.
- I told you the extra model on the GPU was probably your game. It was my own failover (above).
- I wrote "no era I read requires" a castle; Beyond the Horizon does.

## Appendix: every valid run

The raw results (`scripts/ab_test_prompt_format_results.json`, `scripts/ab_test_prompt_format_results_v1_first_compact.json`) and the batch logs stay local, as the
other A/B results do (`.gitignore`: `scripts/*_results.json`, `scripts/*.log`). This table is the record. Prompt length is each tribe's mean in characters;
latency is the mean turn time for the run (both tribes share one batch).

Mid-game, qwen2.5:3b, 100 cycles:

| Seed | Arm | Latency | Prompt length (T1, T2) | Final population (T1, T2) | Seconds |
|---|---|---|---|---|---|
| 1000 | full | 3,289 ms | 15,132; 16,257 | 30,156; 30,305 | 385 |
| 1000 | compact | 3,303 ms | 13,655; 14,377 | 29,751; 28,695 | 378 |
| 1001 | full | 3,245 ms | 15,241; 15,469 | 19,791; 27,663 | 366 |
| 1001 | compact | 3,271 ms | 13,655; 14,389 | 29,953; 30,643 | 372 |
| 1002 | full | 3,314 ms | 15,277; 15,445 | 29,468; 29,663 | 384 |
| 1002 | compact | 3,298 ms | 13,653; 14,056 | 29,946; 29,351 | 383 |

Early game, phi4-mini:3.8b, 250 cycles (latency affected by extra resident models, see above):

| Seed | Arm | Latency | Prompt length (T1, T2) | Final population (T1, T2) | Seconds |
|---|---|---|---|---|---|
| 1000 | full | 5,646 ms | 14,280; 14,030 | 28,787; 16,455 | 1,562 |
| 1000 | compact | 3,646 ms | 12,567; 11,938 | 31,185; 6,155 | 1,045 |
| 1001 | compact | 6,594 ms | 12,465; 12,510 | 29,997; 9,190 | 1,962 |
| 1001 | full | invalid, stopped at about cycle 75 | | | |

## Not yet done

- Early-game batch: one valid run missing (seed 1001, full arm), and phi4-mini loops on JSON need counting per arm.
- Three more mid-game seeds per arm to see whether the Dream Manifestation Machine gap holds; if it does, test the descriptions, the entity grouping and the
  notes separately.
- The journal read-back off and on, on gemma2:2b and gemma3:4b (see the Palimpsest section); needs a `journal_readback` knob in the harness.
- The one-line "available now, needs X" format for every action (the near-miss lines for the era chain are done).
- Reflection findings still open: decrees written as commands, the repeating philosophy text, and the placeholder chief name "Elder of Tribe 1" that appears
  when a succession gets no usable reply.
- The two tribes' different population targets in the night watch (6,675 against 2,670), not yet explained.
