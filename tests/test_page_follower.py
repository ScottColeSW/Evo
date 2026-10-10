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


def test_a_scout_that_steps_once_a_day_walks_all_day_and_rests_one_cycle_at_the_boundary():
    """2026-10-10: a settled scout sat still for a whole day and then jumped; now it is drawn walking toward where its next step lands for 29 of the stub's 30 cycles and resting on the boundary cycle
    (the page test harness uses a 30-cycle day)."""
    out = _run('''
      const exp = {pos: [10, 20], glide_to: [39, 20]};
      const at = c => expeditionGlidePos(exp, c);
      console.log(JSON.stringify({rest: at(60), first: at(61), mid: at(75), last: at(89), arrived: at(90), none: expeditionGlidePos({pos: [3, 4]}, 75), negative: at(-1)}));
    ''')
    assert out["rest"] == [10, 20]                       # the boundary cycle: the real step has landed, the party rests
    assert out["first"][0] > 10 and out["mid"][0] == 25 and out["last"] == [39, 20]   # walking every cycle after, arriving on the last cycle of the day
    assert out["arrived"] == [10, 20]                    # next day: pos is now where it arrived, glide_to is the next leg
    assert out["none"] == [3, 4]                         # a party that already moves every cycle is drawn where it is
    assert out["negative"][0] >= 10 and out["negative"][0] <= 39


def test_chief_bubbles_are_drawn_after_the_landmarks_and_resource_icons():
    """2026-10-10: a bubble flushed before drawLandmarks() had the resource icons and labels painted over its text."""
    text = PAGE.read_text(encoding="utf-8")
    frame = text[text.index("function drawFrame()"):text.index("/* ---------- Chronicle line styling")]
    flush = frame.index("pendingSpeechBubbles.forEach(draw => draw());")
    for later_must_not_follow in ("drawLandmarks();", "drawMinorSettlements();", "drawJointCastle(latestState.joint_castle);"):
        assert frame.index(later_must_not_follow) < flush, later_must_not_follow
    assert flush < frame.index("drawDayNightOverlay(interpolatedCycle());")  # the day and night tint still goes over everything

def _relations_rows(tribe, tribes):
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("function relationsRows(")
    end = text.index("function relationsPanel(")
    program = text[start:end] + f"\nconsole.log(JSON.stringify(relationsRows({json.dumps(tribe)}, {json.dumps(tribes)})));"
    done = subprocess.run(["node", "-e", program], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_sidebar_lists_each_rival_met_with_its_stance_and_the_raid_ledger():
    """2026-10-10: the stance was only shown in the Consensus tab."""
    tribes = {"t0": {"name": "Big"}, "t1": {"name": "Small"}, "t2": {"name": "Unmet"}, "t3": {"name": "Gone", "extinct": True}}
    me = {"id": "t1", "discovered_rivals": ["t0", "t3"], "stance_toward": {"t0": "WAR"},
          "raid_ledger": {"t0": {"made": 3, "won": 0, "repelled": 3, "suffered": 1, "held_off": 0, "people_lost": 13}}}
    rows = _relations_rows(me, tribes)
    assert [r["name"] for r in rows] == ["Big"]                      # the unmet and the extinct are not listed
    assert rows[0]["stance"] == "WAR"
    assert "raided you 1x, took 13 of your people" in rows[0]["detail"] and "your raids: 0 won, 3 repelled" in rows[0]["detail"]


def test_a_met_rival_with_no_stance_set_reads_neutral_and_no_one_met_lists_nothing():
    tribes = {"t0": {"name": "Big"}, "t1": {"name": "Small"}}
    assert _relations_rows({"id": "t1", "discovered_rivals": ["t0"]}, tribes) == [{"id": "t0", "name": "Big", "stance": "NEUTRAL", "detail": ""}]
    assert _relations_rows({"id": "t1"}, tribes) == []

def _warband_lines(tribe):
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("function warbandLines(")
    end = text.index("function relationsPanel(")
    program = text[start:end] + f"\nconsole.log(JSON.stringify(warbandLines({json.dumps(tribe)})));"
    done = subprocess.run(["node", "-e", program], capture_output=True, text=True, encoding="utf-8", timeout=60)  # the lines carry emoji
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_sidebar_shows_the_warbands_standing_and_a_takeover():
    """2026-10-10: nothing on the sidebar said a warband had taken a tribe over."""
    standing = _warband_lines({"warband": {"name": "Grim", "attacks": 3, "wins": 2, "size": 120}})
    assert standing == ["🏴 Grim Raiders: 3 attacks, 2 wins ahead, about 120 strong"]
    assert _warband_lines({"warband": {"name": "Grim", "attacks": 0, "wins": 0, "size": 50}}) == ["🏴 Grim Raiders have not struck yet"]
    taken = _warband_lines({"warband": None, "takeover": {"leader": "Ruthless Gnashdrex", "cycle": 210, "tribe_before": 200, "survivors": 1, "joined": 1900}})
    assert len(taken) == 1 and "Taken over by the Ruthless Gnashdrex Raiders at cycle 210" in taken[0] and "1 of 200 left standing, 1900 raiders joined" in taken[0]
    assert _warband_lines({}) == []
