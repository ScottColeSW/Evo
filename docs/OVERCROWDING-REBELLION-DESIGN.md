# Design: overcrowding as the rebellion trigger (proposal, nothing built)

Written 2026-10-03 with the project owner. No code in this repository has been changed for this.

## What the last real run showed

A 706-cycle run, two tribes (checked in `logs/run_20261003_111611.jsonl` and `logs/board_history.db`):

- The tribes settled **38 tiles apart** by cycle 50 and never moved. Each had discovered the other (by cycle 50 and by cycle 300).
- In 706 cycles neither chose a single `DECLARE_*`, `RAID` toward the other, or `SPY`. Evo's own notes from 2026-09-09 record the same
  pattern in an earlier run, so a larger contact radius is not the lever.
- Overcrowding culls (`Simulation`'s population-pressure rule: 5% of the excess over 89% of the sustainable population, per cycle)
  began at **cycle 506** for both tribes: 79 culls and 1,870 people lost for Tribe 1, 102 culls and 3,239 for Tribe 2, at most 52 in
  one cycle. The people simply vanish ("lost to overcrowding"); nothing else happens.

## The idea

Overcrowding is the trigger for unrest, and unrest has consequences the minds did not choose, so neither side is nudged:

1. **Defection (first).** Part of each overcrowding cull becomes migrants instead of losses. Eligible destination: a rival that has
   been discovered and is itself under its own target with some margin (people move where the land supports them). The migrants
   leave one tribe and join the other, and both tribes' chronicles say so. If no rival is eligible, the cull is exactly as today.
2. **Conflict (second), with the chief keeping the choice.** Peace is the natural impulse, and circumstances can prevent or
   override it. Evo already works this way for alliances: a starving tribe's overture backfires more often (up to a 60% chance,
   `ALLIANCE_BACKFIRE_MAX_CHANCE`), tipping into a skirmish and a WAR stance (`_alliance_backfire_skirmish`). The rebellion extends
   that same family: persistent overcrowding raises the backfire chance of a peace attempt and makes a failed one more likely to
   end in war, so unrest can undo peace without anyone being forced into a fight. The chief still decides whether to seek peace,
   trade, or raid. What the chief sees is facts, not instructions: its own chronicle lines ("a crowd of people left for Tribe 2",
   "Tribe 2 has room"), the way other consequences already reach it.

   An important gap: in the last run **neither tribe ever attempted peace either** (no `DECLARE_ALLIANCE`), so there is no peace
   impulse yet for circumstances to override. The first thing to watch after defection exists is whether visible consequences
   (people leaving) make the models reach for diplomacy at all.

No prompt text changes. Evo is nudge-free by design, and its own tests showed specific nudges did not work; these are world rules,
like the existing cull.

## Questions to settle before building

1. **Order:** defection first, conflict after we see it work? (Recommended: yes.)
2. **Defection share:** what part of the cull migrates, and how much room must the destination have? (Starting point to test, not a
   conclusion: half the cull; destination below 85% of its own target.)
3. **Conflict:** how much overcrowding adds to the backfire chance (a cap, by analogy with the existing 0.6), and whether it applies only
   to a peace attempt or also to trade. No forced skirmish: the chief keeps agency, and circumstances only change the odds.
4. **Optional link to language:** migrants could carry their tribe's invented vocabulary to the destination (see
   `docs/LANGUAGE-CONVERGENCE-OPTIONS.md`), which would make contact produce shared words without seeding anything.

## Phases

0. **Log first (no behavior change):** a structured `overcrowding` line per cull with the tribe's population, its sustainable
   target, the excess, the lost count, and for each discovered rival its room (population against its own target). That shows how
   often a defection would have been possible before anything is built.
1. **Defection**, behind a setting (off by default, like the reflection judge), with a banner and a log line when on.
2. **Conflict**, only after phase 1 has been run and read.

## Risks and limits

- Two tribes both overcrowded means no eligible destination and no change; with many tribes the effect is larger.
- A defector stream could push the destination over its own line and start its culls (back and forth); the room margin guards
  this but the right value is not known.
- Populations in the last run were in the tens of thousands; a share of at most 52 per cycle is small against that, so effects on
  war may be slow. Phase 0 will show the real size.
- One run; the numbers above describe it, not Evo in general.

## Phase 0, built (2026-10-03)

Logging only; the cull is unchanged. Each overcrowding cull now also writes an `overcrowding` line to `logs/run_*.jsonl` (with the
usual `message`): the tribe's population, its target, the excess, the number lost, and for every other living tribe its
id, whether it has been discovered, its population, its own target and its **room** (target minus population), the stance toward
it, and the distance. Two new tests; the full suite is 1,557 passing. After a run, the share of `overcrowding` lines where a
discovered rival has room (positive `room`) is how often a defection would have been possible.

## What the second real run showed (2026-10-03): the trigger mostly does not fire

A 526-cycle run, journal and logging in place, stopped by the owner because "population control isn't happening". Both tribes were in
the war era from cycle 521 only (5 cycles of play there). The overcrowding log shows:

- **6 culls in the whole run**, each losing 1 to 3 people. At cycle 498 Tribe 2 had a population of 38,280 against a target of
  38,270 (excess 10); Tribe 1 had 34,725 against 34,710. The previous run had 181 culls from cycle 506; this one barely culled.
- **Population is not being controlled by culling; it is pinned at the carrying capacity.** Growth tapers to zero as a tribe
  approaches its sustainable population (`_grow_population`), so a tribe settles on the line. The cull is only a backstop for
  jumps, such as a conquest absorbing a rival's people (its own docstring says so).
- **Both tribes were at capacity, so there was nowhere to go.** At the first cull, the rival's room was +39 for one tribe and -65
  for the other, out of populations near 35,000 to 40,000. A defection stream would move almost nobody.

What this changes: **a persistent excess is not what overcrowding looks like in Evo; being full is.** The phase 0 logging captures
only culls, so it sees almost none of the real condition. If overcrowding is to trigger rebellion, the trigger has to be "at or near
capacity for a sustained stretch" (a ratio of population to target, held for N cycles), not "culled". That needs the ratio logged
every cycle (it is cheap: population, target, and each rival's room), and it makes the defection idea weaker than it looked, since
two full tribes have no room for migrants. The conflict path (full tribes with nowhere to expand and no one to absorb) is the part
the data supports; the defection path needs one tribe well under its line, which this run did not have.
