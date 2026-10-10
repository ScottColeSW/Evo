"""2026-10-10: the page draws every moving thing as a smooth follower of its latest known position, so nothing jumps when an update arrives or is late
(docs/APP-PERFORMANCE-2026-10-09.md). The follower's real code is taken out of frontend/index.html and run in Node. Skipped where Node is not installed."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")


def _follower_block() -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("const FOLLOW_K")
    end = text.index("function isWaitingOnServer(")
    return text[start:end]


def _run(scenario: str) -> dict:
    program = (
        "let animFrame = 0; let stateIntervalMs = 5000; let lastStateArrivedAt = 0; const MAX_INTERPOLATE_TILES = 20; const DAY_LENGTH_CYCLES = 30;\n"
        + _follower_block()
        + "\n" + scenario
    )
    done = subprocess.run(["node", "-e", program], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_thing_moves_smoothly_to_a_new_position_and_settles_there():
    out = _run('''
      const xs = []; let t = 0;
      xs.push(followTilePos("a", [0, 0], t)[0]);
      for (let i = 1; i <= 12 * 60; i++) { t = i * 1000 / 60; animFrame++; xs.push(followTilePos("a", [10, 0], t)[0]); }
      const steps = xs.slice(1).map((x, i) => x - xs[i]);
      console.log(JSON.stringify({final: xs[xs.length - 1], monotonic: steps.every(s => s >= -1e-9), maxStep: Math.max(...steps)}));
    ''')
    assert out["monotonic"] and out["final"] > 9.8 and out["maxStep"] < 0.5


def test_a_new_update_arriving_mid_move_changes_neither_the_position_nor_the_speed_abruptly():
    """The old glide restarted from the previous state at every update and hopped (a thing jumped at 39 to 66% of updates in a real run)."""
    out = _run('''
      const xs = []; let t = 0;
      xs.push(followTilePos("a", [0, 0], t)[0]);
      for (let i = 1; i <= 8 * 60; i++) { t = i * 1000 / 60; animFrame++; const target = t < 2500 ? 10 : 20; xs.push(followTilePos("a", [target, 0], t)[0]); }
      const steps = xs.slice(1).map((x, i) => x - xs[i]);
      const speedChange = steps.slice(1).map((s, i) => Math.abs(s - steps[i]));
      console.log(JSON.stringify({maxStep: Math.max(...steps), maxSpeedChange: Math.max(...speedChange), monotonic: steps.every(s => s >= -1e-9)}));
    ''')
    assert out["monotonic"]
    assert out["maxSpeedChange"] < 0.03  # per frame; a restart would show up as a step of several tiles


def test_a_late_update_leaves_a_thing_still_moving_toward_where_it_is_going_not_stopped():
    out = _run('''
      let t = 0; followTilePos("a", [0, 0], t);
      const pos = [];
      for (let i = 1; i <= 9 * 60; i++) { t = i * 1000 / 60; animFrame++; pos.push(followTilePos("a", [10, 0], t)[0]); }
      console.log(JSON.stringify({at5s: pos[5 * 60 - 1], at6s: pos[6 * 60 - 1], at9s: pos[9 * 60 - 1]}));
    ''')
    assert out["at5s"] < out["at6s"] < out["at9s"] < 10.0  # still creeping up long after the usual 5 s gap


def test_a_big_jump_a_long_gap_and_a_second_ask_in_one_frame_behave():
    out = _run('''
      const r = {};
      followTilePos("t", [0, 0], 0);
      animFrame++; r.snapBigJump = followTilePos("t", [50, 0], 16)[0];            // beyond MAX_INTERPOLATE_TILES: snaps
      followTilePos("h", [0, 0], 0);
      animFrame++; r.snapAfterHiddenTab = followTilePos("h", [10, 0], 5000)[0];   // 5 s since the last frame: snaps
      followTilePos("o", [0, 0], 0);
      animFrame++; const first = followTilePos("o", [10, 0], 16)[0]; const second = followTilePos("o", [10, 0], 17)[0];
      r.sameInAFrame = first === second;
      console.log(JSON.stringify(r));
    ''')
    assert out["snapBigJump"] == 50 and out["snapAfterHiddenTab"] == 10 and out["sameInAFrame"] is True


def test_the_sky_follower_never_goes_backwards_or_past_the_cycle_it_is_following():
    out = _run('''
      let t = 0; followScalar("cycle", 100, t); const v = [];
      for (let i = 1; i <= 10 * 60; i++) { t = i * 1000 / 60; animFrame++; v.push(followScalar("cycle", i < 300 ? 101 : 102, t)); }
      const steps = v.slice(1).map((x, i) => x - v[i]);
      console.log(JSON.stringify({monotonic: steps.every(s => s >= -1e-12), max: Math.max(...v), final: v[v.length - 1]}));
    ''')
    assert out["monotonic"] and out["max"] <= 102.0 and out["final"] > 101.7  # 5 s after the target moves a 1.7 s follower is about 0.2 short


def test_the_old_restart_glide_is_gone_and_every_moving_thing_uses_the_follower():
    text = PAGE.read_text(encoding="utf-8")
    assert "interpolatedTilePos(" not in text
    for key in ('followTilePos("tribe:"', 'followTilePos("exp:"', 'followTilePos("patrol:"', 'followTilePos("storm"', 'followTilePos("raiders:"'):
        assert key in text, key
    assert re.search(r'followScalar\("cycle"', text)
