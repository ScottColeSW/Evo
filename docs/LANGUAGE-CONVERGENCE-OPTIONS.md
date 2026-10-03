# Options: letting the tribes' languages converge (proposal, nothing built)

Written 2026-10-03 with the project owner.

## What is happening

`linguistic_consensus` stayed at zero tracked tokens for all 706 cycles of the last run. The tribes were 38 tiles apart, inside the
broadcast hearing range, and each tribe heard the other's broadcast (the prompt shows "overheard: ... broadcasted '...'"). They did
not converge because each tribe's tokens differ ("KRA-ZUL VASH-TA" against "TIB-RAN") and convergence is only counted when both use
the **exact same phrase** for the **same action** (`TranslationConfidenceMatrix.record_broadcast`). Nothing makes a small model copy
a heard token.

## Why the tribes start with different words

On 2026-09-16 each tribe was given a different set of three example tokens (`prompts.LANGUAGE_EXAMPLE_POOLS`). Before that every
tribe saw the same literal examples, and real logs showed all of them echoing "KRA-ZUL", "MEE-LO" and "VASH-TA": a prompt leak, which
made convergence trivial and meaningless. Overlap at the start would partly undo that, so any option has to keep "converged"
meaning something.

## Options

A. **One shared word.** Each tribe's three examples include one common word. Cheap and natural ("a shared root"), but that word is
   echoed by construction, so it must be excluded from the convergence score (count only tokens outside the seeds as emergent).
B. **Shared syllables, different words.** The pools overlap in syllables ("ZUL", "KRA", "TA") but no whole word. Related languages
   that never match exactly, so the exact-match measure stays at zero unless it is changed to score similar tokens (edit distance or
   shared syllables). A larger change, and a new metric to defend.
C. **Do nothing.** Zero convergence remains the baseline.
D. **Contact brings words: migrants carry vocabulary.** With the rebellion design, tribes that exchange defectors exchange some of
   their invented tokens (a fraction of the origin tribe's vocabulary is added to the destination's heard tokens, or gives a small
   convergence credit). No seeding changes and nothing is added to a prompt. It depends on the defection mechanism existing.

Recommendation: D, since it makes language follow from the same contact that creates defection and conflict, and keeps the
2026-09-16 fix intact. If you want language to move before defection exists, A with the seed words excluded from the score is the
smallest honest step.

## What would count as success

`linguistic_consensus` tracked and stabilized tokens above zero in a run, from tokens that were not in either tribe's seeds.
