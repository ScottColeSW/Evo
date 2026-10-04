# Design: words that come to mean something (proposal, nothing built)

Written 2026-10-04 from the owner's idea: a scout reports "I overheard a group at [x,y] saying 'Tik'", the Chief sees a timber
grove there and may conclude 'Tik' means wood. Even a wrong conclusion would be interesting.

## What is true today (measured)

- **The vocabulary is mostly prompt echo.** Each tribe is shown three example words (`prompts.LANGUAGE_EXAMPLE_POOLS`, by tribe). In the
  2026-10-03 run the most-heard words were exactly those: Tribe 2 `ZUR-NEV`, `DOL-KASH`, `TIB-RAN`; Tribe 1 `KRA-ZUL`, `MEE-LO`,
  `VASH-TA`, recombined. The `prompts.py` header already records finding this once and rotating the pools.
- **Words do not track actions.** Mutual information between token and the action performed, against 200 shuffles of the same data
  (`scripts/token_signal.py`): Tribe 1 2.49 bits against a shuffled 95th percentile of about 2.44; Tribe 2 2.91 against about 2.86.
  Both are technically above the shuffled line, by only about 0.05 bits out of roughly 2.5. With few samples and many distinct tokens
  the raw figure is inflated by sparsity. The honest reading: a faint trace at most (some words do favor an action, such as `ZUR-NEV`
  with SCOUT 15 times), nothing a Chief could rely on. (I first described this as "indistinguishable from chance", which overstated
  it: the real figure is above the 95th percentile, just barely.)
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

## Language that has a job: contexts that teach (the owner's idea, 2026-10-04)

The point of a language is to communicate. Today it communicates nothing: a war declaration sets both tribes' stance mechanically
(`actions._declare_war`), and the other Chief is told "Currently war with X" in plain words, so a tribe that shouts something while
declaring war has not told the other side anything. The idea: let meaning be learned from situations whose meaning is unmistakable,
and corrected by what happens when a guess is tried.

Two kinds of teaching context, both observations and outcomes, never an instruction:

1. **Witnessed contexts (a war cry).** When a tribe attacks (a raid, a strike, a conquest, a declared war), the words it was
   broadcasting are part of what the other tribe witnesses: "a rival attacked, shouting 'KRA-ZUL'". The victim's lexicon records the
   word with the context "attack". Hearing it again, before a raid arrives, is then a real, learned warning. Raiders (the NPCs) could do
   the same with a fixed cry, so a tribe learns what that cry means from being raided. The plain "Currently war with X" fact stays, so
   no tribe is ever blind to a war; the cry is added beside it, which is what teaches the pairing.
2. **Tested guesses (a trade probe).** A tribe that believes a word means something can try it: offer a trade naming a good by the
   word, and see what comes back. If the partner hands over wood for 'Tik', the guess (wood) gains evidence; if it hands over
   something else, the guess is contradicted, and the tribe now holds a corrected term. This needs trades to carry the proposer's word for
   the good (a small change to `TRADE`'s result and to what the partner sees) and a rule for how a partner interprets a word it holds a
   guess for. It is the largest piece and comes last.

**Where Palimpsest fits.** A translation guess is exactly a belief with evidence: "'Tik' means wood" (belief, with its reason), the
trade outcomes (evidence, counted), and a correction that supersedes the earlier term with the reason recorded. That is Palimpsest's
own flow (a claim superseded or released stays in history with its reason; a disagreement stays visible until resolved), applied per
tribe to its dictionary of other tribes' words. It is also a use of the Library shelf in `docs/LIBRARY-PALIMPSEST-SPEC.md`.

**Order, cheapest and most informative first:**

| Step | What | Risk |
|---|---|---|
| A | The lexicon above (record word to action, per tribe), logged | none; no behavior change |
| B | Witnessed war cries: add the broadcast to the attack/raid facts the victim sees | low; adds a fact, removes none |
| C | Show the Chief its lexicon, including heard words with their contexts | a nudge risk if worded as advice; facts only |
| D | Trade probes and corrections, with Palimpsest holding the dictionary | largest: changes trade, and small models may not use a word reliably |

**Honest limits.** The small models mostly echo three seed words, so before step D the contexts in B will teach a tribe that one of
its three words was shouted at it during attacks, which is a start but a thin vocabulary. Whether a model uses a shown word to mean
something is what the measurement in the next section is for; if it does not, D has nothing to build on.

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
  `scripts/token_signal.py` on the log. This is the baseline. Expected, from the 2026-10-03 run: a faint trace (about 0.05 bits above the shuffled line).
- **Phase 1 (behavior change, behind a setting, off by default):** show a Chief its own tribe's lexicon as plain facts ("your people
  have used ZUR-NEV 18 times, 15 of them while SCOUT"), so a word can be reused on purpose. Nothing says what a word should mean.
  - **Prediction:** with the lexicon shown, mutual information between token and action rises above the shuffled 95th percentile for
    both tribes in a run of at least 400 cycles.
  - **Falsifier:** it does not. Then the small models do not use a shown vocabulary to carry meaning, and the design would need words
    chosen deliberately (a field in the turn JSON), which is a heavier ask of a 2b model.

Honest limit: if the models keep drawing from the three seed words, the lexicon records that every word means "whatever was
happening", and the measurement will say so.
