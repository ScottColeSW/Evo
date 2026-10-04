# Design: trade and alliance are earned (proposal, nothing built)

Written 2026-10-03 from the owner's notes: tribes should not be able to trade or ally before they have learned what war or poor
management costs, and a tribe that just keeps culling, even under new chiefs, must not get there. "They have to evolve."

## What the data says today

In the era-pacing run, trade began at cycle 31, long before any war (declarations started near cycle 644). Trade and alliance are
open from the start, so nothing in the world teaches the cost of the alternative.

## The rule, in world terms (no prompt nudge)

Two things must both be true before the **peace tier** opens (`TRADE`, `SEND_TRADE_EMISSARY`, `DECLARE_ALLIANCE`):

1. **A cost was felt.** At least one *cost event* in the tribe's record: a night cull, a battle or raid that lost people, a war
   declared against it, or a famine loss.
2. **The lesson was taken.** After the latest cost event, the tribe goes `TRADE_GATE_CLEAN_NIGHTS` consecutive nights (starting
   point 3, about 90 cycles) with **no cull**, with population under the sustainable line.

"Taken" means the tribe changed what it does, not that it waited: with the night cull in place, the only way to a clean night is
to have grown capacity (warehouses, farms, wells, relocation, expeditions that settle) or to have stopped over-growing. That is
the "evolve" the owner describes, and it is checked by what the world shows.

## Punishment for the cull habit, across chiefs

State lives on the **tribe**, not the chief, so a new chief inherits it. Each night cull adds to a *scar count* that decays slowly.
While it is above zero:

- the clean-night requirement multiplies (3, 6, 9, ... capped at 12), so a tribe that culls every other night never reaches it;
- every cull resets the clean-night counter to zero;
- the scar count itself is a plain fact in the chief's view ("culls in the last 10 nights: 4"), numbers only.

An optional second consequence, to decide after the first run: rivals' acceptance of a trade from a tribe with a high scar count is
lower (a world rule about reputation, not a prompt).

## What stays open regardless

- `DECLARE_ALLIANCE` in the departure-era endgame stays on the menu (the existing endgame lock), so the ending stays reachable.
  The gate must not strand a tribe with no way to the ending: **fail-open** if the gate would leave a tribe in the top era with
  no peace action for more than `TRADE_GATE_ENDGAME_GRACE` cycles.
- Raiders and war are untouched; the gate only closes the peace tier.

## Decisions for the owner

1. **A tribe that never suffers a cost event never trades.** That follows from "learn the damage first", but a perfectly managed
   tribe is then locked out of trade. Raids by the NPC raiders make a cost event likely in time, but not certain. Options: accept
   it, or open the tier after a long stretch (say 600 cycles) of stability as its own way to earn it. I recommend the second as a
   fallback, so excellent management is not punished.
2. Does a raid that was **repelled** count as a cost? I recommend yes only if people were lost.
3. Starting numbers (3 clean nights, scar cap 12, decay) are guesses to test, not conclusions.

## Build plan

Phase 0: log `cost_event`, `clean_nights` and the gate state per tribe (no behavior change). Phase 1: the gate behind a setting,
off by default, with a banner. Measure: when does trade now begin, how many wars start, and do any tribes end up in the "always
cull" trap. A pre-registration (prediction and falsifier) is committed before the run.
