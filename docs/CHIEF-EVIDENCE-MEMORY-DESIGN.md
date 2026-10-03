# Design: a memory that lets the Chief weigh what actually followed (proposal, nothing built)

Written 2026-10-03 with the project owner. No code changed for this.

## The idea

The Chief's memory today holds what happened (episodes, hazards) and what the Chief thought (reflections). It does not hold what
followed from a choice. The idea, from the project owner: a memory that helps the Chief decide better for the future by linking
choices to consequences, as plain facts the Chief reads and weighs: "last time you chose X, here is what followed". Not an
instruction, not a judge of similar text. This is the Palimpsest idea of weight moving on evidence, applied to decisions.

## What exists to build on

- Every cycle's chosen action and its short rationale are already in the run log, and the action's result message ("153 stone
  gathered", "the strike on the raider camp failed").
- Phase 0 conflict logging already pairs the menu offered with the action chosen inside a conflict window.
- `TribeMemory` holds weighted episodes; Palimpsest holds claims with weights that move on evidence and can open a visible
  disagreement.

## The warning from Evo's own history

Evo's owner has shown that specific nudges did not change behavior. A line of facts in the prompt is also a nudge of a kind, so this
feature must be measured, not assumed: off by default, with a baseline (the logs above) and a before/after on behaviors that can
be counted (the repetition rate of one action, the share of conflict choices that answer the conflict).

## Proposed shape, in three steps

**Step 1: a decision journal (no behavior change).** For each chosen action, record what changed: the resource deltas
(wood, stone, food, water), population change, any structure started or finished, and whether a conflict event followed within a
few cycles. Logged to the run log and kept per tribe in memory. This alone answers "what does each choice actually produce".

**Step 2: reading it back (behind a setting, off by default).** One short line in the Chief's visible facts, only when the journal
has something notable, in the same plain-fact register as the rest. Candidates, to be chosen:
- a repeated choice and its yield: "In the last 10 cycles you chose GATHER_STONE 6 times: stone rose from 120 to 880, no structure
  was built";
- the last high-stakes choice and what followed: "Last RAID (cycle 120) failed; 8 people were lost".
Facts and numbers only; no "you should".

**Step 3: beliefs against evidence (Palimpsest).** A Chief's stated conviction ("gathering stone secures our future") is a claim; a
run of choices on it that produced no progress is evidence against it. Palimpsest's collision, weight and `resolve` semantics fit:
the belief stays, the disagreement is visible, and the Chief, not the memory, settles it. This is the part that uses the library for
what it is for, and it is deliberately last because it depends on steps 1 and 2 showing there is something to find.

## Questions to settle

1. What counts as "what followed": immediate resource and population deltas plus a conflict event within 3 cycles? (A starting point.)
2. Step 2: which facts to surface, and how often (every cycle, or only on a repeated choice or a failed high-stakes one)?
3. Is Step 1 alone, built and run first, the right first stop? (Recommended.)

## Limits

- Small models often ignore facts in a prompt (the owner's own finding), so Step 2 may change nothing; the journal still has value.
- The consequence of a choice is rarely immediate or single-cause; deltas over a few cycles are a rough proxy.
- One run's numbers (above) are suggestive, not a rate.

## Step 1, built (2026-10-03)

A decision journal, recording only. `config.DECISION_JOURNAL` is `"on"` by default (`"off"` records nothing, for a with and without
comparison); `DECISION_JOURNAL_LENGTH` (60) caps how many entries each tribe keeps. For every chosen action, `_apply_turn` records
the cycle, the action, what changed (stockpile and population deltas), any structure counter that changed, whether the tribe moved,
the stock before, and the action's result note. Entries are kept on the tribe (`tribe.decision_journal`, newest last) and written to
`logs/run_*.jsonl` as `decision` lines. A `conflict_event` line now also lists `followed_decisions`: the cycles and actions within
the previous 3 cycles (`DECISION_JOURNAL_CONFLICT_LOOKBACK`), so a conflict can be traced back to what the tribe had just chosen.
Nothing reads the journal into a prompt; that is Step 2, unbuilt, and not turned on by this. Three new tests; the full suite is
1,563 passing.

**How to read it after a run:** group `decision` lines by action and average the deltas to see what each choice actually produces
(for example what GATHER_STONE yields in a run where it was chosen hundreds of times, and whether anything was built meanwhile);
`followed_decisions` on conflict events shows which choices tend to precede a raid or ambush.
