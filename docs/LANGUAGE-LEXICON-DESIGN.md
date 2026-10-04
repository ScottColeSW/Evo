# Design: words that come to mean something (proposal, nothing built)

Written 2026-10-04 from the owner's idea: a scout reports "I overheard a group at [x,y] saying 'Tik'", the Chief sees a timber
grove there and may conclude 'Tik' means wood. Even a wrong conclusion would be interesting.

## What is true today (measured)

- **The vocabulary is mostly prompt echo.** Each tribe is shown three example words (`prompts.LANGUAGE_EXAMPLE_POOLS`, by tribe). In the
  2026-10-03 run the most-heard words were exactly those: Tribe 2 `ZUR-NEV`, `DOL-KASH`, `TIB-RAN`; Tribe 1 `KRA-ZUL`, `MEE-LO`,
  `VASH-TA`, recombined. The `prompts.py` header already records finding this once and rotating the pools.
- **Words do not track actions.** Mutual information between token and the action performed, against 200 shuffles of the same data
  (`scripts/token_signal.py`): Tribe 1 2.49 bits against a shuffled 95th percentile of 2.45; Tribe 2 2.91 against 2.87. With few
  samples and many distinct tokens the figure is inflated by sparsity, so the honest reading is "indistinguishable from chance".
- So a Chief interpreting a heard word today would be reading noise. Step 1 (done, 2026-10-04) makes the report grounded anyway: where
  it was heard, the terrain, and the resource sites within `SITE_DISCOVERY_RADIUS`. That is the observation; it is only worth
  something once the speakers' words mean something.

## Goal

A word a tribe uses is tied to something it does, so that hearing it is real (partial) evidence. A Chief can be right or wrong about
what it means; both are interesting. No prompt tells anyone what a word means (Evo's nudge-free rule).

## The mechanism: a lexicon per tribe

Each tribe keeps a small word list: `{word: {"action": ..., "uses": n, "first_cycle": c, "last_cycle": c}}`.

1. **Coining.** When a Chief broadcasts a word the tribe has not used, it is entered with the action performed at that moment.
   Words that echo a prompt example are entered too (they are the tribe's seed words; the first action wins).
2. **Use.** A broadcast word that is in the lexicon counts as another use. If the action performed differs from the word's action the
   entry records a mismatch count, so a word can drift: after enough mismatches its action changes to whichever action it was used
   with most (the word's meaning follows its use).
3. **Hearing.** A heard word is evidence about the speaker's lexicon: `(word, action, where, terrain, nearby sites)` goes to the
   hearer as a counted observation (same pattern as the Library's counted evidence: "heard 7 times, 5 of them while GATHER_WOOD").
4. **Inferring.** The Chief may write its own belief about a heard word in its night reflection. The belief and the counted evidence
   sit side by side (the Library shelf, `docs/LIBRARY-PALIMPSEST-SPEC.md`: beliefs against evidence). Nothing tells the Chief what
   the word means or whether it is right.

The open question for the owner: does coining need the Chief to choose a word for an action deliberately (a field in the turn JSON,
more load for a 2b model), or is it enough that the simulation records the association the broadcast already makes? The second needs
no change to the prompt and is the nudge-free choice. I recommend it.

## Where it plugs in

| Piece | Where |
|---|---|
| Lexicon | `Tribe.lexicon`, updated where `last_broadcast`/`last_action` are set (`simulation.py`, `_apply_turn`) |
| Heard words as counted observations | `_party_report_overheard`, grouped by `(word, action)`, reusing `actions._group` |
| Chief sees it | a read-back behind its own setting, off by default (as with the journal read-back) |
| Measurement | `scripts/token_signal.py` run on a log of the new design |

## The test before anything is claimed

Recording an association does not change what anyone says, so the phases are:

- **Phase 0 (no behavior change):** log `lexicon_update` (word, action, uses, mismatches) per broadcast, and run
  `scripts/token_signal.py` on the log. This is the baseline. Expected, from the 2026-10-03 run: indistinguishable from chance.
- **Phase 1 (behavior change, behind a setting, off by default):** show a Chief its own tribe's lexicon as plain facts ("your people
  have used ZUR-NEV 18 times, 15 of them while SCOUT"), so a word can be reused on purpose. Nothing says what a word should mean.
  - **Prediction:** with the lexicon shown, mutual information between token and action rises above the shuffled 95th percentile for
    both tribes in a run of at least 400 cycles.
  - **Falsifier:** it does not. Then the small models do not use a shown vocabulary to carry meaning, and the design would need words
    chosen deliberately (a field in the turn JSON), which is a heavier ask of a 2b model.

Honest limit: if the models keep drawing from the three seed words, the lexicon records that every word means "whatever was
happening", and the measurement will say so.
