# Working in Evolution2Civ

This is an operating manual for whoever (whichever session) picks up this
project next. It won't tell you what the game mechanics do — the code's own
comments are dense and explain the *why* behind every design decision
better than a summary here could. This file is about *how to investigate
and change this project effectively*: where the data lives, how to query
it, and the workflow habits that have paid off over many sessions of
live-bug-fixing.

## The standing rule: ground it before you fix it

The user reports bugs from *watching a live run*, often vaguely ("scouts
are doing weird things," "walls didn't unlock"). Do not guess a fix from
the symptom alone. Pull the actual run data first, find the real mechanism,
then fix that. Every fix that skipped this step and went straight from
symptom to patch has needed rework later. Every fix grounded in real data
has held up. This is the single most important habit in this repo.

## Where the data lives

Two independent sources, both written by a live run, neither committed to
git (`logs/` is gitignored — this data only exists on whatever machine ran
the simulation):

### `logs/board_history.db` (SQLite)

- **`board_snapshots`** — one row per `(run_id, cycle)`, with a
  `snapshot_json` blob that's the *entire* board state at that cycle
  (every tribe's full dict, structures, trails, linguistic consensus —
  literally what the frontend receives over the websocket). This is your
  primary source for "what was true at cycle N" — resource levels, wall
  ring geometry, expedition positions, era, everything.
  - **Gotcha**: `tribe["history"]` inside a snapshot is only the **last 6
    entries** at that cycle (see `Tribe.to_dict()` — the frontend sidebar
    only ever needed that many). It is NOT the tribe's full narrative
    history. Don't conclude "X never happened" from its absence here —
    check the jsonl log (below) for the complete text record instead.
  - Read it directly with `sqlite3` via Python (see query patterns below)
    — there's no need to go through `backend/board_history.py`'s
    functions for read-only analysis, raw SQL is faster to iterate on.

- **`decisions`** — a flat, queryable table (run_id, cycle, tribe_id,
  resources, position, action, target) meant for exactly this kind of
  analysis. **It is NOT populated live.** It's a derived materialization
  built by `backend/decision_log.py::materialize()` from
  `board_snapshots`, and nobody calls that automatically. If you query it
  and get zero rows for a recent run, that doesn't mean anything is
  broken — it means nobody has materialized it yet. Either run
  `materialize(run_id="...")` yourself first, or just query
  `board_snapshots` directly (usually just as fast for a one-off
  investigation, and always up to date).

### `logs/run_<timestamp>.jsonl`

One JSON object per line: `{cycle, tribe, message, ts}` — this is a
straight mirror of every `tribe.history.append(...)` call for the whole
run (see `backend/event_log.py::TribeHistory`), so it's the **complete**
narrative record, unlike the truncated 6-entry window in a snapshot. Use
this when you need to search across an entire run for a specific message
pattern (e.g., "how many times did CONSTRUCT_WALL fail with this exact
rejection text").

**The two sources correlate exactly**: `Simulation.run_id` is literally
`self.event_log.path.stem` (`backend/simulation.py`), so
`run_20260906_155108.jsonl` and `run_id = "run_20260906_155108"` in the DB
are the same run, guaranteed. Pick the run you care about from either
source's filename/run_id and use it to filter the other.

### Finding the right run

```bash
ls -la logs/*.jsonl                      # sorted by mtime already; the newest file is usually what "I just ran this" means
```
or, from the DB, to see every run and how many cycles it covered:
```python
import sqlite3
conn = sqlite3.connect('logs/board_history.db')
for row in conn.execute("SELECT run_id, MIN(cycle), MAX(cycle), COUNT(*) FROM board_snapshots GROUP BY run_id ORDER BY run_id DESC LIMIT 10"):
    print(row)
```
A "quick run" the user mentions is usually the most recent, shortest one —
cross-check cycle count against how long they said they watched it.

## Query cookbook (things that have come up repeatedly)

**Windows console encoding will bite you.** Tribe history strings contain
emoji (🎉 📜 ☠️ etc.). A bare `print()` of a snapshot's history on this
machine's default `cp1252` console raises `UnicodeEncodeError` partway
through and kills your loop silently-ish (you get a traceback, not silent
data loss, but it's easy to lose the first N results scrolled past). Either
write results to a file with `io.open(path, 'w', encoding='utf-8')` instead
of printing, or wrap prints in try/except, or just don't print emoji-laden
strings directly.

**Use `python`, not `python3`.** This environment's `python3` alias isn't
registered; `python -m pytest`, `python -c "..."`, etc. is what works.

**Track a value's transitions, not every cycle.** Most useful analyses are
"when did X change" not "print X every cycle" — e.g., watching era
transitions, watching when a wall ring first appears, watching a scout's
position only when it actually differs from the previous snapshot. Iterate
over `board_snapshots` in cycle order and diff against the last-seen value
per tribe; it's cheap and the output stays readable.

**A frozen position across many snapshots is not automatically a bug.**
Settled tribes' scouts only move once per `config.DAY_LENGTH_CYCLES` (20)
cycles by design — twenty identical-position rows in a row is expected,
not stuck. What IS a bug: frozen across a day-boundary cycle (a multiple of
20) with no movement at all, or frozen for many multiples of the day length
with the expedition still mid-trip (not newly arrived). Check both the
position *and* whether a day-boundary was actually crossed before calling
something stuck.

**Reproduce numerically before touching code.** When a report implies a
geometric/threshold problem ("too close," "not backing away," "wasted
cycles"), compute the actual numbers first — e.g., grep every historical
run for the specific condition (how many wall rings ever had >1 natural
barrier; what was the peak wood value a tribe ever reached this run) rather
than reasoning from the code in the abstract. This project's own git log
is full of fixes that were correctly scoped only because the exact
magnitude was measured first.

## Verifying a fix

1. **`python -m pytest -q`** must be green before and after. This project
   is disciplined about test coverage — a fix without a new regression
   test covering the exact bug is incomplete. Match the style of the
   nearest existing test in the same file (there is almost always a
   sibling test doing something structurally similar — copy its
   fixture/mock/assertion style rather than inventing a new one).
2. **For a backend-logic fix**, a targeted unit test (see `tests/`
   conventions below) is usually sufficient — these bugs are rarely
   visible in the UI in a way a screenshot would catch faster than a test.
3. **For a frontend change**, use the `Claude_Browser` tools:
   `preview_start({name: "evo2civ"})` (config already in
   `.claude/launch.json`, serves on port 8766 — deliberately different
   from `run.py`'s own default of 8765, so a preview server never collides
   with a real run the user might have going). You do **not** need a real
   live simulation running to verify frontend rendering — inject a
   synthetic state directly via `javascript_tool`:
   ```js
   latestState = { /* hand-built tribe/game-over/whatever state */ };
   document.getElementById("picker").style.display = "none";   // hide the setup screen
   document.getElementById("theater").style.display = "block"; // reveal the board
   updateSidebar(latestState);   // or showGameOverSplash(latestState), or drawFrame(), etc.
   ```
   then screenshot or read the DOM. This has been the standard technique
   for verifying sidebar panels, game-over splashes, and building icons
   without waiting on a real Ollama-driven run.

## Test file conventions (`tests/`)

- `_bare_simulation()` (defined near the top of `test_actions.py` and
  `test_simulation.py`) builds a `Simulation` via `__new__` with just
  enough attributes wired up for action/turn logic — much faster than a
  real `await Simulation.create(...)`, and what nearly every unit test
  uses. If you need a `Simulation` for something and reach for a full
  constructor, check whether `_bare_simulation()` already covers it.
- `_settle(sim, tribe)` — registers the tribe and calls
  `sim._found_territory(tribe)` for real (territory_center, wall ring 0,
  Town Hall placement). Use this instead of hand-setting
  `has_ever_settled = True`, whenever a test needs a *real* wall ring to
  exist (city_layout functions read real ring structure, not just the
  flag).
- `_complete_ring0(sim, tribe, tier=0)` / `_unlock_all_ring0_sections(sim,
  tribe)` — shortcuts for "the wall already exists in whatever state this
  test actually cares about," bypassing the real CONSTRUCT_WALL/
  EXPAND_TERRITORY action loop (those loops have their own dedicated
  tests).
- `_NO_TARGET = (0, 0)` — most actions ignore their target argument; only
  RELOCATE/SCOUT/RAID-like ones use it meaningfully.
- Mocking randomness: patch `backend.actions.random.random` or
  `backend.simulation.random.random` (module-qualified, matching where the
  call site actually lives) to force a win/loss/roll outcome
  deterministically.
- Mocking movement: `mock.patch("backend.physics.terrain_aware_step",
  return_value=(x, y))` is the standard way to force a specific step
  outcome (including simulating "boxed in" by returning the same position
  as the input) without depending on real map geometry.
- `@run_async` (imported `from tests.conftest import run_async`) wraps an
  `async def test_...` in `asyncio.run(...)` — this suite has no
  `pytest-asyncio` dependency, so this decorator (not a pytest marker) is
  how every async test in the project runs.
- `tests/conftest.py` autouse-fixtures redirect `event_log`'s and
  `scoreboard`'s default write locations to pytest's `tmp_path` for every
  test — this is why running the suite never litters the real `logs/`
  directory. If you're debugging why a test's file output isn't where you
  expect, this is why.

## Git workflow

- **One commit per logical fix.** This project's history is disciplined
  about this. If a single investigation session uncovers 2-3 distinct
  root causes (even from one bug report, even touching the same file),
  they get separate commits.
- **When multiple fixes land in the same file before you've committed
  anything**, split them at commit time rather than mixing them:
  1. `git diff -- path/to/file.py > /tmp/full.diff` and find the `@@`
     hunk headers (`grep -n "^@@" /tmp/full.diff`) — map each hunk to
     which fix it belongs to by reading its content.
  2. Build a patch containing just the header (`diff --git`, `index`,
     `---`, `+++`) plus the hunks for one fix, then
     `git apply --cached that.patch` to stage exactly that slice.
  3. **Verify hunk line counts before applying**: a hunk header
     `@@ -A,B +C,D @@` means exactly `B` old-side lines
     (context+removed) and `D` new-side lines (context+added) must
     follow. Miscounting by one line (very easy when hand-slicing with
     `sed`) produces a `corrupt patch` error. Check with:
     `awk 'NR>N{if(substr($0,1,1)=="+") a++; else if(substr($0,1,1)=="-") r++; else c++} END{print c+r, c+a}' patch.file`
     (where `N` is the line number of the `@@` header itself, 1-indexed)
     and compare against the header's declared counts before trying to
     apply.
  4. If `git apply --cached` reports "does not apply" (not "corrupt") on
     a patch whose counts check out, check `git status`/`git diff
     --cached` first — it may already be staged from an earlier attempt
     in the same session. `--cached` applies against the **index**, not
     the working tree, which is easy to lose track of mid-troubleshooting.
  5. Repeat for the next fix's hunks, then commit each staged slice
     separately.
- Commit messages: lead with what changed, then a paragraph on the real
  root cause (not just the symptom) and how it was confirmed — matching
  every existing commit's own style. `Co-Authored-By: Claude Sonnet 5
  <noreply@anthropic.com>` at the end, per standing instruction.
- Push after committing — standing authorization for this repo across
  sessions; no need to re-ask.

## Where prior context lives

- **`C:\Users\scott\.claude\projects\...\memory\MEMORY.md`** — the index
  of this project's accumulated design decisions, resolved questions, and
  standing user preferences. Skim it at the start of a session; it answers
  "have we already decided this" far faster than re-deriving it. It's kept
  up to date as part of normal work — update it when you resolve something
  it lists as open, or learn something a future session would otherwise
  have to re-discover.
- **`C:\Users\scott\.claude\plans\*.md`** — full design docs for larger
  features (eras, territory/wall rework, etc.), left in place after
  implementation as the historical record of *why* something was built
  the way it was. The code and tests are the source of truth if a plan
  goes stale, but the plan explains intent the code alone won't.
- **`scratch/`** (untracked, not gitignored but never committed) — where
  ad-hoc investigation scripts and trace dumps belong. Treat anything in
  here as disposable.
