# Nudge audit and the project-wide switch (2026-10-04)

The project's rule (`DESIGN.md`, section 6; the design-philosophy memory): a tribe is told what is true, never what to do. A **nudge** is
prompt text whose job is to steer the tribe toward an outcome rather than report a fact. The standing finding is that a nudge alone rarely
moves a small model, so each one is a candidate to remove. This file lists every place prompt text is added, says which are nudges, and records
what is switched off by the new variable and what is not covered yet.

## The switch

- `config.NUDGES = "on"` (default, today's behavior). The `NUDGES` environment variable overrides it at each call, like `RUN_DATA_LOG`.
  `NUDGES=off python run.py --port 8790` silences every gated site below.
- `config.DISABLED_NUDGE_TAGS` (older, from 2026-09-16) still silences one category at a time with the switch on.
- `config.nudge_active(tag)` is the one test. `Simulation._nudge(visible_entities, tag, text)` is the helper for the turn builder.
- Off touches prompt text only. The action menu, every gate and every mechanic ignore it (`tests/test_nudge_switch.py` checks that the menu is
  identical on and off, and that the survival crisis flag still fires).

The default stays on until a headless A/B run (the project's standing practice) shows what the tribes do without the text. Running the same
scenario with `NUDGES=off` and `NUDGES=on` and comparing survival, era pace and action mix is the next step, not done yet.

## Gated now (41 sites in `_prepare_turn`, plus 2 elsewhere)

| Tag | What it says (paraphrased) | Sites |
|---|---|---|
| `build_hint` | "X would ...", "a Y is now worth building": kitchen, cooking, moat, long house, keep, fortress, castle, sawmill, tannery, coop, deer pen, dock, library, barracks, quarry, mine, forge, DMM (and "stands ready"), first wall | 26 |
| `farm_hint` | a farm plot "could" be planted, no flock yet, no one has fished, fishing pays out immediately versus a hunting party | 5 |
| `endgame_hint` | the castle "completes this tribe's legacy"; the vessel can be built; the vessel stands ready | 3 |
| `hunt_hint` | a hunting party sent to a known site "would likely fare better" | 1 |
| `war_hint` | a conquest "is a real, favorable bet now" | 1 |
| `water_hint` | "building a Well would secure it immediately" | 1 |
| `settle_hint` | water is found, "RELOCATE there to finally settle" | 1 |
| `inventory_advice` | "resupplying the lowest one first is usually the most efficient", the stockpile advice | 2 |
| `repetition` | the Historian line: "Choose something genuinely different" (the menu removal it accompanies is a mechanic and stays) | 1 |
| `survival_warning` | the starving and thirst messages that name actions (`instincts.py`; existed before, now follows the switch) | 1 |
| `growth_pressure` | the "your people are grateful ... growing is how that fragility ends" framing (`prompts.py`); the era-gap fact stays | 1 |

## Facts, left alone

Lightning and fire, scarcity percentages, wildlife sightings, a rival's known position and distance, spy reports, overheard words, war cries
and celebrations, Might comparisons, discovery direction ("every confirmed discovery lies to the ..."), the wall's built sections and cost, the
Dream Machine and the Battalion resting ("N cycles left"), fishing mastered, farm plots growing and their progress, torches and moats now in
place, "crops, eggs and fishing need the tribe settled", the invalid-action feedback line, the era-gap fact ("still short on: ..."), the settled
and relocation status lines.

## Mixed: a fact with a steer inside, not gated yet (review one at a time)

Splitting these means rewriting them as the fact alone, which changes the wording, so each wants its own look.

- Raiders inside the territory: "EXPEL_RAIDERS_FROM_TERRITORY is the real answer to this right now". Raiders riding in: "This is real time to
  prepare, not a surprise."
- Not yet settled: "so only survival and exploration actions are available ... Settling properly will open up ...". The menu fact is real;
  the "will open up" half is the steer.
- RELOCATE not available: "Sending scouts out is how a real destination gets found."
- Ground qualifies, ground does not qualify, supports gathering but no water, already at the confirmed water site ("relocating again would
  accomplish nothing"): status plus advice.
- Barracks at their built limit: "UPGRADE_BARRACKS is the only way to train a larger one."
- Nowhere further to grow: "What remains is settling things with the known rival tribe ... war, alliance, or the training".
- Both tribes fully committed to war: "DECLARE_CONQUEST is the only real choice left".
- Alliance stands: "a Joint Castle can now be raised".
- Crisis and wall filters ("the crisis is severe enough that only actions which could directly help ... are being offered"): the first half
  explains a real menu change and is a fact; the "can wait until" half is advice.

## Not covered by the switch yet

- The base prompt (`prompts.py`, `get_prime_consciousness_prompt`): "Your goals: grow your population, expand and defend your territory, and
  advance through the ages toward a permanent Capital City" is an objective given from outside. The "WHAT EACH OF YOUR CURRENT ACTIONS DOES"
  glossary (`ACTION_DESCRIPTIONS`) is descriptive, but the project owner has objected to it before. The leadership "RESPONSIBILITY" line states
  what the role obligates.
- The night reflection prompts (`reflection.py`): the departure-dream paragraph and the DMM dream paragraph invite a particular kind of answer.
- `journal_readback` (off by default, a home-page switch): plain-fact lines about what a repeated choice produced. Facts, but an experiment
  on whether showing a Chief its own record changes its choices.
- Menu mechanics that steer without text (survival-crisis narrowing, the repetition removal, scout rotation, resource-gather withdrawal): these
  are mechanics by the project's own definition, not nudges, and the switch does not touch them.

## A/B result, first pass (2026-10-05)

`scripts/ab_test_nudges_off.py`, two `qwen2.5:3b` tribes, `NUDGES=on` against `off`, same seeds. Raw results are in `scripts/ab_test_nudges_off_results.json` and the run log beside it; both are gitignored, so they exist only on the machine that ran it.

- **Early game** (fresh start, 250 cycles, 2 seeds per arm, so 4 tribe-runs per arm). No tribe went extinct in either arm. Final population median
  8,827 with nudges on and 10,723 off (ranges 6,684 to 13,266 and 1,564 to 15,195). Structures built: median 13.5 on, 14.5 off. Trades: 10.5 on,
  10 off. First entry to the Cognitive Horizon: cycle 91 on, 64 off. Entry to tribal synapse: 180 on, 173 off. Tribes still built long houses, kitchens,
  tanneries, sawmills, quarries, wells and took up fishing with the build hints off. The spread inside one arm is larger than any gap between arms,
  so this shows no harm from turning the gated nudges off, not that they do nothing.
- **Mid game** (from the 15,000-population fixtures, 100 cycles, 3 seeds per arm): **not usable.** In both arms the tribes chose RELOCATE in about
  60% of decisions (117 of 200 in one run, none with an effect), while the original run over the same cycles never did. Starting from a saved
  state changes something that makes the model pick RELOCATE; the cause is not found. The fixture also does not copy `lumber_site`, so "wood
  mastered" is never true and a fixture tribe in tribal synapse can never meet the monolithic gate. Fix both before using the fixtures again.
- Next: more early-game seeds (each run is about 15 minutes), then repair the fixtures and repeat the mid-game batch.

## How to back them out

1. Run the A/B with `NUDGES=off` against `on` on the benchmark scenarios and one live run; read survival, era pace and action mix.
2. If a category does no harm off, change its default by adding its tag to `DISABLED_NUDGE_TAGS`, then delete the site in a later cleanup.
3. Take the mixed list one entry at a time: rewrite to the fact alone, then gate the remainder if one is left.
4. Decide with the owner whether the base-prompt objective and the glossary belong under the switch.
