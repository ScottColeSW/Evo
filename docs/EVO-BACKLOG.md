# Evo backlog, and the Palimpsest experiments gated on data

Written 2026-10-04. Two lists, kept apart on purpose: Evo's own game work (worth doing, not integration progress), and the
Palimpsest integration experiments (each waits for data we have not read yet). `scripts/score_run.py` scores a run against every
prediction written so far in one pass; read it before choosing anything from either list.

## Evo's own work (not Palimpsest integration)

Not started:
- **Fur has few uses.** Fur is spent only on the Long House discount and on advancing eras, so it can pile up between eras. Options:
  clothing, trade goods, a use in the Library or the barracks. Decide after `tannery_day` shows whether it piles up.
- **Raider ambush has no war cry.** Home raids carry `RAIDER_WAR_CRY`; the expedition ambush (`simulation.py`, the `Ambush` combat
  records) does not. One extra call.
- **Targeting duds.** About eight RAID, TRADE and STRIKE_RAIDER_CAMP choices in the last run named places with nothing there. That
  is the Chief's targeting, not the menu; a fact about whether a rival or raider camp is known near the chosen tile may help.
- **Population growth.** Tribes went from about 1,300 to 25,000 in 200 cycles with no cull. Watch `night_watch` headroom; the
  population lines and the sustainable-population formula may need another look.
- **Trade gate, build.** Logging exists; the lock waits on the run. See `docs/TRADE-GATE-DESIGN.md`.
- **Rebellion and defection.** Design only (`docs/OVERCROWDING-REBELLION-DESIGN.md`); the trigger should be time near capacity.
- **Conflict mode.** Shelved (`docs/CONFLICT-MODE-DESIGN.md`); the era changes produced wars without it.
- **Language, steps C and D.** Show a Chief its lexicon; trade probes and corrections (`docs/LANGUAGE-LEXICON-DESIGN.md`).
  Held at B by the owner.

## Palimpsest integration (gated on data)

Each is an experiment, not a feature, and none is started. The owner's call, 2026-10-04: too early for a controlled experiment; more
data first, including data that is not yet understood.

1. **Library shelf, phase 1.** File reflections and evidence through Palimpsest, repeats reinforce instead of duplicating. Needs the
   shadow log to show the judge behaving on beliefs and evidence (it called action logs contradictions at 1.00).
2. **Beliefs against evidence read-back.** Does showing a Chief its own record change its choices? The sharpest experiment, with fixed
   seeds and an A/B. Costly and stochastic; state the cost before running.
3. **A misinformation adversary** (the owner's idea): a counter-spy campaign, a rival that plants false reports through the spy and
   overheard channels. This is where Palimpsest's external-source rule (a claim from outside can never supersede a held one) gets a real
   test in Evo, and it is the only place Evo has an adversary. Needs the lexicon or the spy-intel channel to carry claims first.
4. **Translation guesses as beliefs** (language step D): a guess with counted trade outcomes as evidence and a correction that
   supersedes it with the reason recorded. Depends on 3's channel and on whether the models use a word reliably.
