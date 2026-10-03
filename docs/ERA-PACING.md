# Era pacing: a 50-cycle floor, wider population lines, more timber groves, a bigger warehouse (2026-10-03)

Decided with the project owner after reading how long tribes actually spend in each era.

## What the data showed

From 107 recorded runs (`logs/board_history.db`, runs of 150 cycles or more; "median" is cycles spent in the era by tribes that moved on):
Primitive Dawn 11, Cognitive Horizon 111, Tribal Synapse 65, Monolithic 21, Dream Manifestation 20, War and World Domination 10. The
early game is the grind; the late eras fly. In the war-ready benchmark the war era lasted one cycle, so nothing could happen in it.
The owner's guide: no less than 50 cycles in any era.

On the cycle before a tribe advanced, it held 5 to 9 times the next era's population line, 15 to 23 times its water and 4 to 6 times
its stone, but only 0.7 to 1.0 times its wood. So wood is the real gate, and population, water and stone were loose.

## What changed

1. **`config.ERA_MIN_CYCLES = 50`.** A tribe must live 50 cycles in an era before the next opens (`Simulation._advance_era_if_ready`,
   `Tribe.era_entered_cycle`). Ordinary advancement only; a conquest that hands a tribe a higher era restarts the clock. The research
   discount does not shorten it. This is a floor no population jump can skip, which thresholds alone cannot give.
2. **Population lines widened** (`backend/eras.py`): 60, 600, 2,200, 7,500, 14,000, 20,000 (were 12, 50, 200, 800, 1,500, 3,000). A replay
   of 19 recorded tribes under these lines gave median dwells of about 57, 71, 101, 80, 120 and 27 cycles. The top line stays near 20,000
   because most tribes level off around 27,000. Resource lines and costs are unchanged.
3. **More Timber Groves** (`world.SITE_DENSITY_BY_TYPE["lumber"]`, spacing 8/18 to 7/15): 39 to 54 lumber sites on the 100 by 100 map (+38%).
   Wood is the binding gate, but earlier data showed lumber sites were found in 85% of runs, so more groves may help less than hoped.
4. **Warehouses hold a vessel** (`WAREHOUSE_STORAGE_BONUS_PER_BUILDING` 400 to 450): five warehouses now hold 2,550 against the vessel's
   2,500 cost, with no upgrade needed.
5. **A soft-lock fixed** in the gather removal (the vessel was unreachable); see the commit `e5be523`.

## What to check in the next run (predictions, to be read against the logs)

- Every tribe spends at least 50 cycles in every era it leaves (this is guaranteed by the floor; a shorter dwell would be a bug).
- The median dwell is near or above 50 in every era, and the early eras are not made unreasonably long: the replay predicts roughly
  57, 71, 101, 80, 120 and 27.
- The tribes still reach Departure (the replay: 16 of 19). If they do not, the likely cause is wood, or the 20,000 line against populations
  that level off near 27,000.
- The war era lasts long enough for conflict to happen in it; whether tribes then choose war is a separate question.

## Limits

The replay keeps each tribe's other gates as recorded and ignores how behavior changes when eras last longer (feedback), the 19 tribes
come from runs under different versions of the game, and the first rungs lengthen the game a lot (a median tribe now reaches the war
era near cycle 350, not 160). One real run is the test.
