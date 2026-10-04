# Evo development history, distilled

Generated 2026-10-04 from the chronicle logs and scoreboard before they were deleted. The code changed constantly from 2026-08-30 to 2026-10-04, so rows pool runs made under different rules; tables split by period where it matters. Method: `scripts/history_report.py` and `scripts/history_report_db.py`.

## The corpus

502 chronicle logs from 20260829 to 20261004. Cycles per run: median 80.0, 75th percentile 181, longest 1733. 223 runs reached 100 cycles, 51 reached 400, 150 ended under 30 (mostly test and start-up runs).

| Period | Runs | Median cycles | Longest |
|---|---|---|---|
| Aug 30 to Sep 10 | 360 | 62.5 | 1724 |
| Sep 11 to Sep 18 | 113 | 150 | 1733 |
| Sep 19 to Oct 4 | 24 | 248.5 | 1201 |

## How tribes ended (scoreboard.jsonl)

101 tribe results (test tribes excluded). Each is a tribe's whole life summary, recorded at extinction.

| Cause of death | Tribes | Share |
|---|---|---|
| starvation | 59 | 58% |
| thirst | 26 | 26% |
| absorbed | 11 | 11% |
| volcano | 4 | 4% |
| raider_ambush | 1 | 1% |

| Model | Tribes | Median cycles survived | Longest | Median max population | Most common cause |
|---|---|---|---|---|---|
| gemma2:2b | 49 | 31 | 734 | 10 | starvation |
| qwen2.5:3b | 35 | 43 | 588 | 10 | starvation |
| llama3.2:1b | 4 | 36.5 | 55 | 11.5 | thirst |
| llama3.2:latest | 4 | 315.5 | 687 | 8040.0 | absorbed |
| mistral:7b | 3 | 54 | 60 | 11 | thirst |
| phi4-mini:latest | 3 | 53 | 334 | 12 | starvation |
| qwen2.5:1.5b | 2 | 32.5 | 46 | 11.0 | starvation |
| qwen2.5-coder:7b | 1 | 53 | 53 | 10 | thirst |

Era reached at extinction: stone_age 76, primitive_dawn 5, cognitive_horizon 5, tribal_synapse 4, monolithic_era 4, object_creator_era 3, war_and_world_domination_era 2, bronze_age 1, cosmic_post_human 1.

Trades completed per tribe life: median 0, 89 of 101 tribes never traded. Raids won 3, lost 13, defended 48.

## When things first happen (runs of at least 100 cycles)

Cycle of the first matching chronicle line per tribe, so a run with two tribes contributes two. Medians hide the code changes of the period; the three columns split them.

| First occurrence | Aug 30 to Sep 10 | Sep 11 to Sep 18 | Sep 19 to Oct 4 |
|---|---|---|---|
| discovers fire | 8 (n=97) | 4 (n=86) | 4 (n=36) |
| first raiders sighted | 46 (n=130) | none | none |
| first raid repelled | 85 (n=125) | 84 (n=102) | 80 (n=34) |
| first raid struck | 65 (n=132) | 64 (n=80) | 48 (n=26) |
| celebrates finding water | 4 (n=169) | 4 (n=107) | 8 (n=36) |
| a road takes shape | 94 (n=26) | 330 (n=6) | 280 (n=5) |
| forges the Tribal Synapse | 90 (n=107) | 141 (n=77) | 164 (n=28) |
| crosses into the Cognitive Horizon | 15 (n=144) | 10 (n=98) | 12 (n=35) |
| first extinction | 40 (n=28) | none | none |

Reading the 'none' cells (checked against the raw lines): 'first raiders sighted' is a wording artifact. The approach-warning line ('raiders have been spotted riding in from...') appears only in the first period's logs; raids themselves kept happening (the repelled and struck lines continue in every period). 'First extinction' is a real absence in the chronicle logs: all 134 'has gone extinct' lines are from before Sep 11, and tribes in later runs ended mostly by being absorbed or stayed alive when the run stopped.

## What kills and what threatens (all runs)

Chronicle line counts, per 1,000 tribe-cycles so runs of different length compare. Tribe-cycles are approximated as run cycles times two (most runs had two tribes). These are lines, not distinct events: a tribe in a starvation spiral writes the starvation line every cycle, so the starvation and thirst rows measure how long tribes spent starving, not how many died.

| Event | Lines | Per 1,000 tribe-cycles |
|---|---|---|
| starvation deaths | 3605 | 24.0 |
| thirst deaths | 1447 | 9.6 |
| river drownings | 621 | 4.1 |
| volcano deaths | 350 | 2.3 |
| rip current deaths | 1600 | 10.6 |
| cliff deaths | 337 | 2.2 |
| raids repelled | 1119 | 7.4 |
| raids struck | 627 | 4.2 |
| raider camps destroyed | 393 | 2.6 |
| old per-cycle overcrowding culls | 3023 | 20.1 |
| flock lost for lack of feed | 670 | 4.5 |

Raid defense: 1119 repelled against 627 that struck, a 64% repel rate.

## Repetition and wasted choices

The repetition guard (the 'Historian insists on a different choice' line) fired 6158 times, 40.9 per 1,000 tribe-cycles. By action:

| Action repeated | Times the guard fired |
|---|---|
| GATHER_STONE | 1203 |
| HUNT_DEER | 890 |
| SCOUT | 786 |
| GATHER_FOOD | 648 |
| EXPLORATION_PARTY | 642 |
| GATHER_WOOD | 565 |
| GATHER_EGGS | 388 |
| TRAIN_BATTALION | 204 |

| Waste or noise | Lines |
|---|---|
| item stores already full (an item action with nowhere to put it) | 2496 |
| model reply the game could not read as a decision | 566 |
| celebration reason that is really a routine action log | 2996 |

The last row is a real finding: a celebration's stated reason was a per-turn action log (for example 'At (x,y) in river, chose SCOUT'), because those logs were stored at high weight as memories. They are no longer filed as Library candidates (2026-10-04), but the celebration wording still draws on them.

## Language: what tribes shouted

Celebration shouts in the chronicle (the quoted word before the exclamation mark). On 2026-09-16 the example words in the prompt were rotated per tribe to stop every tribe echoing KRA-ZUL, MEE-LO and VASH-TA. This checks whether it worked.

| Period | Words shouted | Distinct words | Share that are the three old seed words | Most common |
|---|---|---|---|---|
| before the rotation | 4606 | 453 | 88% | KRA-ZUL 2603, VASH-TA 1145, MEE-LO 291, KRA-VASH-TA 16, KRA-VA-LO 8 |
| after the rotation | 2353 | 446 | 41% | KRA-ZUL 429, VASH-TA 406, TIB-RAN 382, ZUR-NEV 239, DOL-KASH 158 |


## Database sections

Pending: era dwell, population curves, action mix and stance timing come from logs/board_history.db, which is being written by a live run. They will be appended once that run ends, then the raw data can go.
