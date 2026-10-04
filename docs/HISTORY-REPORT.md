# Evo development history, distilled

Generated 2026-10-04 from the chronicle logs (502), `scoreboard.jsonl` (145 rows) and the board-snapshot database (76,000 snapshots, 240 runs, 5 GB), so the raw data can be deleted once this is kept. The code changed constantly from 2026-08-30 to 2026-10-04, so rows pool runs made under different rules; tables split by period where it matters. Method: `scripts/history_report.py` and `scripts/history_report_db.py` (they need the raw data, which is gone after cleanup; they stay as a record of how each number was made).

## The corpus

503 chronicle logs from 20260829 to 20261004. Cycles per run: median 80, 75th percentile 182, longest 1733. 224 runs reached 100 cycles, 52 reached 400, 150 ended under 30 (mostly test and start-up runs).

| Period | Runs | Median cycles | Longest |
|---|---|---|---|
| Aug 30 to Sep 10 | 360 | 62.5 | 1724 |
| Sep 11 to Sep 18 | 113 | 150 | 1733 |
| Sep 19 to Oct 3 midday | 21 | 257 | 1201 |
| Oct 3 evening on (50-cycle floor in place) | 4 | 335.0 | 455 |

## How tribes ended (scoreboard.jsonl)

103 tribe results (test tribes excluded). Each is a tribe's whole life summary, recorded when it ended. Only tribes that ended are recorded, so this skews toward short lives (a tribe still alive when a run stopped is missing). That is why most of these never traded, while in the database sections below most tribe lives of 100 cycles or more did.

| Cause of death | Tribes | Share |
|---|---|---|
| starvation | 59 | 57% |
| thirst | 26 | 25% |
| absorbed | 13 | 13% |
| volcano | 4 | 4% |
| raider_ambush | 1 | 1% |

| Model | Tribes | Median cycles survived | Longest | Median max population | Most common cause |
|---|---|---|---|---|---|
| gemma2:2b | 51 | 32 | 734 | 10 | starvation |
| qwen2.5:3b | 35 | 43 | 588 | 10 | starvation |
| llama3.2:1b | 4 | 36.5 | 55 | 11.5 | thirst |
| llama3.2:latest | 4 | 315.5 | 687 | 8040.0 | absorbed |
| mistral:7b | 3 | 54 | 60 | 11 | thirst |
| phi4-mini:latest | 3 | 53 | 334 | 12 | starvation |
| qwen2.5:1.5b | 2 | 32.5 | 46 | 11.0 | starvation |
| qwen2.5-coder:7b | 1 | 53 | 53 | 10 | thirst |

Era reached at extinction: stone_age 76, monolithic_era 6, primitive_dawn 5, cognitive_horizon 5, tribal_synapse 4, object_creator_era 3, war_and_world_domination_era 2, bronze_age 1, cosmic_post_human 1.

Trades completed per tribe life: median 0, 89 of 103 tribes never traded. Raids won 3, lost 13, defended 54.

## When things first happen (runs of at least 100 cycles)

Cycle of the first matching chronicle line per tribe, so a run with two tribes contributes two. Medians hide the code changes of the period; the three columns split them.

| First occurrence | Aug 30 to Sep 10 | Sep 11 to Sep 18 | Sep 19 to Oct 3 midday | Oct 3 evening on (50-cycle floor in place) |
|---|---|---|---|---|
| discovers fire | 8 (n=97) | 4 (n=86) | 4 (n=32) | 4 (n=7) |
| first raiders sighted | 46 (n=130) | none | none | none |
| first raid repelled | 85 (n=125) | 84 (n=102) | 74 (n=30) | 149 (n=7) |
| first raid struck | 65 (n=132) | 64 (n=80) | 48 (n=22) | 71 (n=7) |
| celebrates finding water | 4 (n=169) | 4 (n=107) | 8 (n=32) | 8 (n=7) |
| a road takes shape | 94 (n=26) | 330 (n=6) | 260 (n=3) | 280 (n=3) |
| forges the Tribal Synapse | 90 (n=107) | 141 (n=77) | 156 (n=25) | 262 (n=6) |
| crosses into the Cognitive Horizon | 15 (n=144) | 10 (n=98) | 12 (n=32) | 126 (n=6) |
| first extinction | 40 (n=28) | none | none | none |

Reading the 'none' cells (checked against the raw lines): 'first raiders sighted' is a wording artifact. The approach-warning line ('raiders have been spotted riding in from...') appears only in the first period's logs; raids themselves kept happening (the repelled and struck lines continue in every period). 'First extinction' is a real absence in the chronicle logs: all 134 'has gone extinct' lines are from before Sep 11, and tribes in later runs ended mostly by being absorbed or stayed alive when the run stopped.

## What kills and what threatens (all runs)

Chronicle line counts, per 1,000 tribe-cycles so runs of different length compare. Tribe-cycles are approximated as run cycles times two (most runs had two tribes). These are lines, not distinct events: a tribe in a starvation spiral writes the starvation line every cycle, so the starvation and thirst rows measure how long tribes spent starving, not how many died.

| Event | Lines | Per 1,000 tribe-cycles |
|---|---|---|
| starvation deaths | 3622 | 23.9 |
| thirst deaths | 1447 | 9.6 |
| river drownings | 622 | 4.1 |
| volcano deaths | 354 | 2.3 |
| rip current deaths | 1653 | 10.9 |
| cliff deaths | 348 | 2.3 |
| raids repelled | 1127 | 7.4 |
| raids struck | 635 | 4.2 |
| raider camps destroyed | 401 | 2.6 |
| old per-cycle overcrowding culls | 3023 | 20.0 |
| flock lost for lack of feed | 670 | 4.4 |

Raid defense: 1127 repelled against 635 that struck, a 64% repel rate.

## Repetition and wasted choices

The repetition guard (the 'Historian insists on a different choice' line) fired 6231 times, 41.1 per 1,000 tribe-cycles. By action:

| Action repeated | Times the guard fired |
|---|---|
| GATHER_STONE | 1215 |
| HUNT_DEER | 898 |
| SCOUT | 788 |
| GATHER_FOOD | 666 |
| EXPLORATION_PARTY | 642 |
| GATHER_WOOD | 587 |
| GATHER_EGGS | 388 |
| TRAIN_BATTALION | 204 |

| Waste or noise | Lines |
|---|---|
| item stores already full (an item action with nowhere to put it) | 2497 |
| model reply the game could not read as a decision | 568 |
| celebration reason that is really a routine action log | 3038 |

The last row is a real finding: a celebration's stated reason was a per-turn action log (for example 'At (x,y) in river, chose SCOUT'), because those logs were stored at high weight as memories. They are no longer filed as Library candidates (2026-10-04), but the celebration wording still draws on them.

## Language: what tribes shouted

Celebration shouts in the chronicle (the quoted word before the exclamation mark). On 2026-09-16 the example words in the prompt were rotated per tribe to stop every tribe echoing KRA-ZUL, MEE-LO and VASH-TA. This checks whether it worked.

| Period | Words shouted | Distinct words | Share that are the three old seed words | Most common |
|---|---|---|---|---|
| before the rotation | 4606 | 453 | 88% | KRA-ZUL 2603, VASH-TA 1145, MEE-LO 291, KRA-VASH-TA 16, KRA-VA-LO 8 |
| after the rotation | 2428 | 464 | 41% | KRA-ZUL 442, VASH-TA 415, TIB-RAN 389, ZUR-NEV 241, DOL-KASH 158 |

# Sections from the board-snapshot database

_Database: 503 tribe lives in all, 271 of at least 100 cycles used below._

## Eras, from the snapshots (tribe lives of at least 100 cycles)

Exact cycle of each era change. Dwell counts only eras the tribe left (the era a run ended in is unfinished). 'Reached' is the share of tribe lives that ever entered the era.

**Aug 30 to Sep 10** (161 tribe lives)

| Era | Reached | Median first cycle | Median dwell | Shortest dwell | Completed stays |
|---|---|---|---|---|---|
| primitive_dawn | 87% | 1 | 15 | 6 | 137 |
| cognitive_horizon | 85% | 16 | 75 | 4 | 103 |
| tribal_synapse | 64% | 94 | 67 | 1 | 56 |
| monolithic_era | 35% | 205 | 16 | 1 | 41 |
| object_creator_era | 12% | 188 | 22 | 1 | 20 |
| war_and_world_domination_era | 12% | 224 | n/a | n/a | 0 |

**Sep 11 to Sep 18** (70 tribe lives)

| Era | Reached | Median first cycle | Median dwell | Shortest dwell | Completed stays |
|---|---|---|---|---|---|
| primitive_dawn | 91% | 1 | 8 | 2 | 64 |
| cognitive_horizon | 91% | 9 | 130 | 54 | 55 |
| tribal_synapse | 79% | 140 | 61 | 3 | 41 |
| monolithic_era | 59% | 222 | 21 | 1 | 39 |
| object_creator_era | 54% | 268 | 20 | 1 | 33 |
| war_and_world_domination_era | 53% | 297 | 8 | 1 | 19 |
| departure_era | 31% | 298 | n/a | n/a | 0 |

**Sep 19 to Oct 3 midday** (33 tribe lives)

| Era | Reached | Median first cycle | Median dwell | Shortest dwell | Completed stays |
|---|---|---|---|---|---|
| primitive_dawn | 94% | 1 | 11 | 6 | 31 |
| cognitive_horizon | 94% | 12 | 141 | 79 | 25 |
| tribal_synapse | 76% | 156 | 88 | 1 | 21 |
| monolithic_era | 64% | 251 | 80 | 2 | 17 |
| object_creator_era | 52% | 352 | 100 | 1 | 16 |
| war_and_world_domination_era | 52% | 521 | 43 | 1 | 9 |
| departure_era | 30% | 324 | n/a | n/a | 0 |

**Oct 3 evening on (50-cycle floor in place)** (7 tribe lives)

| Era | Reached | Median first cycle | Median dwell | Shortest dwell | Completed stays |
|---|---|---|---|---|---|
| primitive_dawn | 100% | 1 | 92 | 68 | 6 |
| cognitive_horizon | 86% | 126 | 129 | 83 | 6 |
| tribal_synapse | 86% | 262 | 53 | 50 | 5 |
| monolithic_era | 71% | 361 | 54 | 50 | 2 |
| object_creator_era | 29% | 316 | 71 | 50 | 2 |
| war_and_world_domination_era | 29% | 387 | n/a | n/a | 0 |

## Population over time

Median population at fixed cycles, among tribe lives that were still going at that cycle (n in brackets), and the largest population each life reached.

| Period | cycle 25 | cycle 50 | cycle 100 | cycle 200 | cycle 300 | cycle 400 | cycle 600 | Peak population |
|---|---|---|---|---|---|---|---|---|
| Aug 30 to Sep 10 | 14 (160) | 20 (161) | 39 (161) | 98 (100) | 90 (60) | 142 (48) | 241 (20) | median 143, 90th pct 8645, max 605900 |
| Sep 11 to Sep 18 | 18 (64) | 28 (64) | 218 (64) | 2360 (52) | 7632 (39) | 14984 (19) | 20926 (11) | median 10868, 90th pct 35729, max 210242 |
| Sep 19 to Oct 3 midday | 18 (30) | 48 (30) | 548 (30) | 5190 (22) | 16656 (13) | 28344 (11) | 39725 (7) | median 14365, 90th pct 45762, max 65563 |
| Oct 3 evening on (50-cycle floor in place) | 17 (6) | 22 (6) | 54 (6) | 321 (7) | 4919 (5) | 7725 (1) | n/a | median 15919, 90th pct 25645, max 32359 |

## When structures first appear

Median first cycle at which each structure existed, and the share of tribe lives that ever built it (all periods pooled; split would be too thin for the rarer ones).

| Structure | Median first cycle | Share of lives that built it |
|---|---|---|
| fire ever | 8 | 87% |
| well | 88 | 59% |
| warehouses | 142 | 58% |
| dock | 170 | 50% |
| quarry | 143 | 50% |
| bath house | 162 | 49% |
| boat | 170 | 48% |
| tannery | 132 | 47% |
| long houses | 75 | 43% |
| sawmill | 223 | 42% |
| kitchen | 163 | 39% |
| hatchery | 172 | 38% |
| fishery | 232 | 37% |
| road | 260 | 32% |
| library | 266 | 26% |
| mine | 292 | 25% |
| keep | 270 | 25% |
| barracks | 269 | 23% |
| coop | 160 | 23% |
| deer pen | 165 | 23% |
| forge | 359 | 17% |
| castle | 364 | 12% |
| moat | 370 | 11% |
| dmm | 378 | 10% |
| fortress | 392 | 9% |
| object creator | 339 | 4% |

## Trade, war and alliance

| Period | Tribe lives | First trade (median cycle) | First war stance | First alliance |
|---|---|---|---|---|
| Aug 30 to Sep 10 | 161 | 114 (1% ever) | 318 (4% ever) | 138 (11% ever) |
| Sep 11 to Sep 18 | 70 | 64 (81% ever) | 346 (17% ever) | 302 (53% ever) |
| Sep 19 to Oct 3 midday | 33 | 71 (82% ever) | 424 (12% ever) | 244 (27% ever) |
| Oct 3 evening on (50-cycle floor in place) | 7 | 207 (100% ever) | n/a (0% ever) | n/a (0% ever) |

## What tribes chose, by era

Share of tribe-cycles spent on each action (the action chosen that cycle), top six per era, all periods pooled.

- **primitive_dawn** (5894 tribe-cycles): SCOUT 27%, RELOCATE 23%, GATHER_FOOD 16%, GATHER_STONE 7%, GATHER_WOOD 7%, EXPLORATION_PARTY 5%
- **cognitive_horizon** (30082 tribe-cycles): GATHER_STONE 20%, GATHER_FOOD 16%, SCOUT 16%, GATHER_WOOD 15%, HUNT_DEER 7%, EXPLORATION_PARTY 6%
- **tribal_synapse** (22782 tribe-cycles): GATHER_STONE 15%, HUNT_DEER 11%, SCOUT 10%, GATHER_WOOD 8%, BUILD_LONG_HOUSE 8%, GATHER_EGGS 8%
- **monolithic_era** (9874 tribe-cycles): SCOUT 17%, GATHER_STONE 13%, HUNT_DEER 11%, EXPLORATION_PARTY 7%, CONSTRUCT_WALL 7%, GATHER_WOOD 6%
- **object_creator_era** (4422 tribe-cycles): GATHER_STONE 18%, GATHER_WOOD 11%, EXPLORATION_PARTY 7%, SCOUT 7%, CONSTRUCT_WALL 6%, RESEARCH 6%
- **war_and_world_domination_era** (4219 tribe-cycles): GATHER_STONE 12%, HUNT_DEER 10%, GATHER_EGGS 7%, TRAIN_BATTALION 7%, SCOUT 7%, RESEARCH 7%
- **departure_era** (4128 tribe-cycles): UPGRADE_WAREHOUSE 17%, TRAIN_BATTALION 13%, GATHER_STONE 11%, GATHER_WOOD 9%, DECLARE_CONQUEST 8%, SCOUT 6%

## Early stall by slot and model

Time spent in the first era (primitive_dawn) before the tribe left it, by tribe name and model, tribe lives that did leave it. Tribes that never left are counted separately.

| Tribe | Model | Left the first era | Median cycles in it | Never left |
|---|---|---|---|---|
| Tribe 2 | qwen2.5:3b | 71 | 11 | 1 |
| Tribe 1 | gemma2:2b | 65 | 8 | 2 |
| other names | gemma2:2b | 27 | 17 | 0 |
| other names | qwen2.5:3b | 27 | 10 | 0 |
| Tribe 1 | qwen2.5:3b | 8 | 6 | 0 |
| Tribe 1 | phi4-mini:latest | 7 | 14 | 0 |
| Tribe 2 | llama3.2:latest | 7 | 9 | 0 |
| Tribe 1 | llama3.2:latest | 7 | 19 | 0 |
| Tribe 2 | gemma2:2b | 6 | 6 | 0 |
| other names | llama3.2:1b | 3 | 39 | 1 |
| other names | qwen2.5:1.5b | 4 | 16 | 0 |
| Tribe 2 | phi4-mini:latest | 4 | 15 | 0 |

# The last two runs and what changed because of the data

## The last two runs (2026-10-04), read closely

The only runs made with the structured logging in place, so they are the best evidence for the current rules. Both had the reflection judge and the
journal read-back switched on, so their effects cannot be separated.

- **Run 1:** 266 cycles, two tribes. **Run 2:** 455 cycles, three tribes (a third was injected at cycle 153). Tribe 2 absorbed Tribe 1 at cycle 395
  and the injected tribe at 455.

**Era dwell, exact to the cycle (run 2).** Every completed era lasted at least 50 cycles, so the floor works. From tribal_synapse on, the floor itself
sets the pace (stays of 50, 52, 53), which means the population lines in the later eras are crossed long before it ends.

| Tribe | primitive_dawn | cognitive_horizon | tribal_synapse | monolithic | object_creator | war era |
|---|---|---|---|---|---|---|
| Tribe 1 | 235 | 83 | 52 | 24+ | | |
| Tribe 2 | 72 | 106 | 53 | 57 | 92 | 14, then absorbed Tribe 1 |
| Injected | 68 | 140 | 50 | 44+ | | |

**Tribe 1's long first era.** Tribe 1 (model `gemma2:2b`) stayed in primitive_dawn for 235 cycles in run 2 and never left it in run 1's 266. Its food
sat at 1 to 10 for about 200 cycles, it had no farm plots and had not learned fishing, and it kept choosing GATHER_WOOD. Population stayed at 16 to 28
while Tribe 2 (`qwen2.5:3b`) reached 323 by cycle 101. Cause: chief decision quality, made worse by raising the first population line to 60 on
2026-10-03 (the earlier line, 12, would have let it advance at once). Not changed; the choice is between lowering that line and leaving the stall. For scale: across the 271 tribe lives of 100 cycles or more, the median first-era stay was 8 to 15 cycles before 2026-10-03 (population line 12); in the seven lives since the floor, it is 92 (shortest 68).

**Population and culls.** Final populations were 31,853 (Tribe 2), 15,395 (injected) and 11,930 (Tribe 1). No overcrowding cull happened in either run;
the closest any tribe came to the cull line was 1,472 people of headroom. So the night cull and the cull half of the trade gate have never been
exercised in a real run.

**TRADE was mostly wasted.** 73 TRADE choices, 8 with an effect. Tribe 2 alone made 59 failed attempts ("found no rival encampment there to trade
with"). Measured against the stored targets, the Chief had named no target, so the game used its own camp, 36 to 38 tiles from the rival; only 9 of 78
TRADE turns were within 3 tiles. 22 trades happened in the run, the first at cycle 164. Changed on 2026-10-04: a TRADE with no rival at the target now
goes to the nearest rival already found, once the first wall ring stands.

**Trade gate (logging only).** It would have blocked 0 of the 22 trades: the contact path (3 nights with new outside contact) opens the tier by
cycle 90 for every tribe, long before the first trade. Not tuned; with the TRADE fix trades may come earlier, which is the thing to re-measure.

**Library shadow.** 17 filings. The NLI judge called 34 of 36 evidence items collisions (different buildings in the same sentence template read as
contradictions, score 1.00). Beliefs: 3 new, 3 collides, 2 coexists (the 0.98 contradiction looked real, the two at 0.63 to 0.65 looked doubtful).
RESEARCH filed 15 times in 150 cycles, in bursts (for example cycles 271 to 274), and reached the 13-research cap on the era discount near cycle 355,
so the new "only when something new" rule slowed but did not stop the discount filling. Changed: evidence is no longer sent to the judge. Not changed:
a per-era cap on research counting toward the discount.

**Tannery.** Fur a day had a median of 6 to 8 and a maximum of 13 (the old ceiling was 6). The herd settles around 21 to 23 (harvest balances
breeding), and Fur stock grew to 22, 33 and 53, so it accumulates slowly because little spends it.

**Language.** Mutual information between a tribe's word and the action it was doing, above the shuffled 95th percentile: run 1 Tribe 1 +0.05 bits,
Tribe 2 +0.31; run 2 injected +0.26, Tribe 1 +0.13, Tribe 2 +0.19, Tribe 2 (Advanced) 0.00. Faint, with two cases just over the 0.25-bit line. Tribes
borrow each other's seed words at 0.3% to 5.3% of their use (the injected newcomer most). Run 2 witnessed 16 raid cries and 90 celebration shouts.

**Choices that changed nothing.** Run 2: 199 of 1,152 decisions had no measurable effect (SCOUT 123, which is an expedition whose result comes later;
TRADE 65; the rest a handful each). Gathers into full storage: 0 (fixed the same day; the previous run had 8).

## What the data led us to change (2026-10-03 and 2026-10-04)

| Finding | Source | Change |
|---|---|---|
| Eras lasted 10 to 21 cycles in late game | era dwell curve, 19 recorded tribes | 50-cycle floor, wider population lines, more timber groves |
| Overcrowding was thousands of small culls (3,023 lines) | chronicle counts | one big cull at the start of the night |
| RESEARCH repeated 83% (and filed action logs) | first run with the shadow log | offered only when it files something new; candidates are reflections and evidence, counted per pattern |
| Tannery Fur capped at 6 a day whatever the herd | code and run | Fur scales with the herd (20% of the surplus a day) |
| Gather offered with stores full | decision journal | not offered when the stockpile is at the cap |
| TRADE offered before any rival was met | owner report | five rival actions need a rival found |
| TRADE aimed at the tribe's own camp | stored targets | falls back to the nearest rival found |
| Judge flagged evidence as collisions | shadow log | evidence is not judged |
| Contact count opened the trade gate at cycle 30 | first gate log | counts nights of contact, not words |
| Every word was one of three seed words | chronicle shouts | per-tribe example words (2026-09-16); 88% to 41% seed share after |

## Open questions the data raised

- **Trade began about 140 cycles later after the pacing changes.** Tribe lives from Sep 11 to Oct 3 midday traded first at a median of 64 to 71 cycles
  (81% to 82% ever traded). The seven lives since the floor traded first at cycle 207. Cause not determined. Candidates: slower early development (the
  first era now lasts about 92 cycles, not 11), the read-back and judge being on in those runs, and the TRADE aiming problem fixed afterwards. Re-measure
  after the TRADE fixes.
- **The cull half of the trade gate and the night cull are unexercised.** No cull in either of the last two runs.
- **First-era stall is model-dependent.** Whether to lower the first population line (60) or leave Tribe 1's stall is the owner's call.

## Limits of this report

- Code and rules changed continuously from 2026-08-30, so pooled rows mix rule sets; the period split is the guard, and the last period is small
  (seven tribe lives).
- Tribes created by a conquest merge appear as new lives (for example "Tribe 2 (Advanced)"), and their inherited structures show as first built at the
  merge cycle. A few lives, small effect.
- The scoreboard records only tribes that ended, so it skews to short lives; the database sections are the better guide for lives of 100 cycles or more.
- Chronicle counts are lines, not events (a starving tribe writes the same line every cycle).

## Cleanup record and decisions after this report (2026-10-04)

- **Raw data removed.** The owner ran `scripts/cleanup_history_raw.py --yes`: `logs/board_history.db` (5.02 GB) and the 499 chronicle logs older than
  the evening of 2026-10-03 (62.8 MB) are gone; `logs` is now 9.3 MB. Kept: the four newest run logs, `scoreboard.jsonl`, `experiments.jsonl`,
  `benchmark_results.db`. The numbers above cannot be recomputed from raw data any more; the scripts remain as the record of method.
- **Research discount capped per era** (3 per era). **Tribe 1's long first era left as it is,** with a stated trigger to revisit. **Trade-gate contact
  path raised to 5 nights; lock not built,** with a stated trigger. Details and triggers: `docs/EVO-BACKLOG.md`.
