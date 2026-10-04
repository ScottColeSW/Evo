# Spec: the Library as a Palimpsest shelf (proposal, nothing built)

Written 2026-10-03 at the owner's request. Builds on `docs/PALIMPSEST-REFLECTIONS-DESIGN.md` and
`docs/CHIEF-EVIDENCE-MEMORY-DESIGN.md`. The reflection judge from the first is already built (`REFLECTION_JUDGE`, off by default).

## What the Library does today (checked in the code)

`actions._research` takes the three highest-weight entries in `TribeMemory` (plus any taboos), joins their text with "; ", and appends
`{"summary", "cycle"}` to `tribe.library_entries`. Then `research_completed` goes up by one, which discounts the next era by 4%
per research (cap 50%).

Four consequences, each verifiable:

1. **No curation.** The same top-weight memories can be filed again and again; a tribe that researches five times in a row can
   file five near-copies.
2. **No conflict handling.** Two opposing convictions can sit on the shelf side by side.
3. **No provenance.** An entry does not say whether it records something that happened (evidence) or something the chief decided to
   believe (a belief).
4. **Never read back.** `library_entries` appears only in the UI tab (`frontend/index.html`) and the snapshot. Nothing in a chief's
   prompt reads the shelf. The only mechanical effect of the Library is the number.

## Goal

The Library becomes a real shelf the chief files into and consults: a small Palimpsest store per tribe, where each entry is a
claim with a reason, a source kind, a weight, and visible relations to other entries. Palimpsest stays independent and
agent-driven: the chief (or a librarian turn) decides what is filed and how entries relate; Palimpsest records, weighs and
refuses nothing silently.

Not goals: changing `TribeMemory` (the per-turn recall stays), changing the era discount's size, adding a nudge ("you should
study X"). The shelf shows facts about itself, never advice.

## Design

### 1. Filing (RESEARCH)

`RESEARCH` keeps its costs and its era discount. What it files changes:

- The candidate claims are the same top memories, but each is filed **one at a time** through Palimpsest's `consult` against the
  tribe's shelf, with a source kind:
  - `evidence`: derived from the decision journal or the conflict and overcrowding logs (a fact with a cycle and numbers, for
    example "the cull at cycle 612 cost 140").
  - `belief`: from a chief's reflection or decree.
- The judge (the existing NLI plus embeddings hybrid in Palimpsest) returns the relation to the nearest held entry:

| Relation | What happens on the shelf | Counts toward the era discount? |
|---|---|---|
| NEW | filed as a new entry | yes |
| REINFORCES | the held entry gains weight and a new date; no second copy | **no** (nothing new learned) |
| COLLIDES | filed as its own entry, both marked as in disagreement | yes, once (a real tension is worth studying) |
| COEXISTS / SCOPE_CHILD | filed as its own entry (an exception or a refinement) | yes |

So repeating the same research no longer farms the discount: the era discount rewards new knowledge, not repetition. That is the
one deliberate mechanical change, and it is a world rule, not a prompt.

### 2. Provenance and the injection boundary

Palimpsest's `source` field maps directly: an entry the tribe produced itself is a self-sourced claim. Anything that arrives from
outside (a trader's report, a spy's report, a rival's overheard word brought home by a party) is `source="external"` and **can
never supersede a held entry**, only sit beside it and be weighed. That is useful in Evo for a concrete reason: spy reports and
overheard words are exactly the second-hand claims the shelf should not let overwrite what the tribe saw.

### 3. Reading it back (the part that is missing today)

At the night cycle, behind its own checkbox (off by default, like the journal read-back), the chief's inventory gets at most
three plain lines from the shelf, numbers and labels only:

- the heaviest entry with its source kind and date,
- any open disagreement ("entry A (belief, cycle 120) and entry B (evidence, cycle 540) disagree; open since cycle 540"),
- the newest evidence entry.

No "should". The open disagreement is the useful one: a chief that believes "culling keeps us strong" next to an evidence entry
"the culls at 300, 330 and 360 each cost more than the one before" is looking at its own record. This is Step 3 in
`CHIEF-EVIDENCE-MEMORY-DESIGN.md` (beliefs against evidence), delivered through the Library instead of a new building.

### 4. Settling a disagreement

Only the chief settles one (keep one, merge, leave open), through the existing night reflection JSON. Palimpsest records the
choice, the reason and the author. The shelf never resolves anything itself.

## Where it plugs in

| Piece | Where | Change |
|---|---|---|
| Store | a new `backend/library.py` wrapping `palimpsest.Memory` (in-memory SQLite per tribe, saved with the run) | new file |
| Filing | `actions._research` | file through the shelf; keep the old `library_entries` list as a plain derived view, so the UI tab and saved fixtures keep working |
| Era discount | `Simulation._advance_era_if_ready` and its twin near line 8264 | count new knowledge instead of `research_completed` |
| Read-back | `_build_night_inventory` | up to three lines, behind a setting |
| Settling | night reflection JSON | one optional field, `library_resolution` |
| Dependency | `pyproject` extra | Palimpsest stays an optional extra, as with the NLI extra; **with it absent, RESEARCH behaves exactly as today** (fail-open) |

## Phases

0. **Log only (no behavior change).** On every RESEARCH, run the shelf in shadow: record what each candidate would have been judged
   (NEW, REINFORCES, ...) in `logs/run_*.jsonl`. This answers the question the spec rests on: how often is research a repeat? I
   have not measured it. Saved logs can be replayed for part of it (RESEARCH events and the memory contents at that cycle).
1. **File through the shelf**, era discount on new knowledge only. Behind a setting, off by default.
2. **Read-back**, behind its own setting, measured the way the journal read-back is.
3. **Settling**, last, because it adds a field a small model may misuse (watch the 2b model's JSON).

## Pre-registration for phase 0 (to commit before running)

- Prediction: more than half of RESEARCH filings after the first two are judged REINFORCES on a real run.
- Falsifier: fewer than a quarter are. Then repetition is not a real problem and phase 1 loses its main reason.

## Risks, stated plainly

- **Small-model cost.** A judge call per candidate, on a night cycle that already runs a reflection. The NLI path is local and
  cheap; the embedding path reuses `nomic-embed-text`. Not measured in this loop yet.
- **A shelf nobody reads is the same as today.** Phase 2 is where any benefit would show, and Evo's own finding is that specific
  nudges did not work. The read-back is facts about the shelf, not advice, but it can still turn out to do nothing; the log will
  say.
- **Gaming the discount** by filing many tiny distinct claims. A cap per research (three candidates, as now) bounds it.

## Phase 0, built (2026-10-03)

`RESEARCH` queues what it filed (`tribe.library_shadow_pending`); each night, when the judge is on (the home-page judge checkbox),
`Simulation._library_shadow_night` judges each candidate against a shadow shelf and writes a `library_shadow` record to
`logs/run_*.jsonl` (per candidate: relation, reason, whether it would count as new; per research: new and repeat counts). Repeats are
not added to the shadow shelf. The real Library and the era discount are untouched. Only beliefs (memories) are judged for now; the
evidence source from section 1 comes with phase 1.

Reading it after a run (the prediction and falsifier above are scored from these records, skipping each tribe's first two filings):

    python -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if '\"library_shadow\"' in l]; print(sum(x['data']['repeats'] for x in r), sum(x['data']['new'] for x in r))" logs/run_XXXX.jsonl

## Phase 0 result and a rule change (2026-10-04)

First run with the shadow on (one tribe researched 18 times in cycles 289 to 388): 40 of 48 filings after the first two were repeats, so
the prediction holds. The candidates were routine action logs, and the judge called some of them contradictions at 1.00 confidence
(different actions, same sentence shape), so the judge needs reflections and evidence as its input, not episodes. The same tribe
reached the 50% era-discount cap at 13 researches.

Rule change, independent of the shelf: `RESEARCH` is offered only when the tribe remembers something the Library has not filed
(`actions.research_candidates`, word overlap of at least `LIBRARY_REPEAT_JACCARD` = 0.6 counts as filed, and the Library's own
entries are never refiled), so every research that happens adds something and counts toward the discount. The principle (the owner's):
an action that adds nothing is not on the list.

## Candidates filtered to reflections and evidence (2026-10-04)

`actions.library_candidates` now builds what RESEARCH can file, each tagged with its source (the tag is also in the `library_shadow` log):

- **belief:** the chief's own reflections (`TribeMemory` entries of kind `reflection`), reinforced ones weighing more.
- **evidence:** a structure or upgrade that came up in the decision journal, a high-stakes choice with what it changed, a cost the
  tribe suffered (the trade-gate cost events), and the world facts already kept as episodes (hazards, discoveries).
- **left out:** routine per-turn action logs ("At (30,64) in lake, chose GATHER_STONE..."), which are neither.

A filing takes the heaviest belief first, then evidence, then whatever is left, at most three, minus anything already on the shelf.
Replaying the last run's log: Tribe 2 had 77 journal evidence entries, 10 cost events and 13 reflections; Tribe 1 had 34, 13 and 13. So
RESEARCH should stay on the menu for a long time. Not yet measured: how the judge does on these texts (the false "contradiction" calls
were on action logs), which the next run with the judge on will show.
