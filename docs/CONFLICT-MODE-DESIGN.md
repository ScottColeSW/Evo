# Design: a conflict mode that narrows the menu (proposal, nothing built)

Written 2026-10-03 with the project owner. No code changed for this.

## The observation

In conflict (raiders, a skirmish, a war stance) the models are still offered a full menu of easy, unrelated choices (gathers, builds,
research) and tend to pick them. The idea: a "conflict mode" that narrows the menu to the actions that answer the conflict for a
cooldown period. The risk the owner named: the mode could keep forcing tribes into further conflict.

## What Evo already has

`_prepare_turn` already narrows the menu in five situations, each with the same shape, which conflict mode should copy:

| Existing lock | Entry (world state) | Exit | Safeguard |
|---|---|---|---|
| survival crisis | food or water crisis active | the crisis ends | fail-open: never an empty menu |
| wall commitment | a wall section is under construction | it is finished | fail-open |
| territory threatened | a raider camp near the boundary | the camp is cleared | fail-open; `CLEAR_TERRITORY` pops to the top |
| endgame | top era, a living rival, no mutual alliance | mutual alliance or a resolution | **keeps `DECLARE_ALLIANCE` on the menu**, "despite its name suggesting peace always stays an option" |
| battle-ready | both sides fully committed to the military branch | resolved | narrows to `DECLARE_CONQUEST` and related actions |

Every one is tied to a state of the world, not only a timer, and survival crises outrank the rest.

## Proposed shape

1. **Entry:** a real conflict event, not a mood: a skirmish or raid resolved against this tribe or by it, or a WAR stance newly in
   force. (Overcrowding unrest, from the rebellion design, can be a source of that event.)
2. **While active (a cooldown of N cycles):** the menu keeps the actions that answer the conflict (training and upgrading the
   battalion, clearing territory, scouting the rival, `RAID`, `DECLARE_CONQUEST`) **and peace** (`DECLARE_ALLIANCE`,
   `SEND_TRADE_EMISSARY`), plus survival actions. The mode never means "war only": peace stays available, as in the endgame lock.
3. **Exit:** the cooldown ends, a peace is made, the rival is gone, or a survival crisis outranks it (the existing priority).
4. **The lock-in guard (the owner's concern):**
   - a **refractory period** after the mode ends (M cycles during which the same event cannot start it again), so a tribe cannot
     chain from one conflict into the next without a normal menu in between;
   - the mode is entered by an event but **extended by nothing**: no event while active renews the cooldown;
   - **the log records the share of cycles each tribe spends in the mode**, so any lock-in shows in the data.
5. **Fail-open:** if narrowing would leave an empty menu, the full menu stays.

## Phases

0. **Log first (no behavior change):** record each conflict event and, for the following N cycles, the actions the tribe chose and
   those it was offered. That shows how much of conflict-time play is "easy unrelated choices" today, which is the observation this
   rests on. (I have not measured it; it comes from watching the LLM monitor.)
1. **The mode** behind a setting, off by default, with a banner and log line when active.

## Questions to settle

1. Cooldown N and refractory M (starting points to test, not conclusions: N = 10 cycles, M = 20).
2. Which events count as conflict (a raid by NPC raiders, or only tribe against tribe)?
3. Should peace actions stay on the menu for the whole cooldown (recommended), or only after a minimum number of cycles?
4. Build phase 0 (logging) first, as with the other proposals?

## Limits

The whole idea rests on an observation from watching, not on measured data; phase 0 exists to measure it. A narrower menu is
itself a strong nudge, so Evo's nudge-free principle needs a decision: this changes what is offered, not what a prompt says,
the same way the existing locks do.

## Phase 0, built (2026-10-03)

Logging only; no outcome, menu or prompt changes. A conflict event is any combat outcome (every one goes through
`actions._record_combat`: raids, raid defense, raider-camp strikes, expelling raiders, conquest, alliance backfire, ambush, home
defense) or a declared war. Each writes a `conflict_event` line to `logs/run_*.jsonl` and opens a **10-cycle window**
(`config.CONFLICT_LOG_WINDOW_CYCLES`) for that tribe. A further event inside an open window is logged but does not extend it.
While a window is open:
- `conflict_turn`: the menu the tribe was offered each cycle (`offered`, `offered_count`), which of those answer the conflict
  or reach for peace (`answering`, `answering_share`; the list is `actions.CONFLICT_ANSWER_ACTIONS`), and `cycles_since` the event;
- `conflict_choice`: the action it chose.

Join `conflict_turn` and `conflict_choice` on tribe and cycle. Three new tests; the full suite is 1,560 passing.

**How to read it after a run:** the average `answering_share` is how much of the offered menu was about the conflict; the
`conflict_choice` actions that are in `CONFLICT_ANSWER_ACTIONS` against those that are not is the owner's observation (too many
easy, unrelated choices) as a number. How often windows start, and how often a new event lands inside an open one, shows how
sticky a mode with this window would be.

One slip during the build, caught before it shipped: the first patch skipped the `DECLARE_WAR` hook silently because its anchor
text appears twice. It now has its own anchor and a test.
