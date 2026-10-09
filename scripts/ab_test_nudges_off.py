"""A/B test (2026-10-04): the project-wide nudge switch, NUDGES=on against NUDGES=off.

Also (2026-10-05, --knob menu_cap): the same harness with the MENU_CAP experiment, a full menu ("full") against a menu capped at 8 actions ("cap8"),
to test whether smaller action sets give better choices. Results go to scripts/ab_test_menu_cap_results.json. Extra numbers recorded for it: the mean
menu size and the share of decisions that changed nothing, read from the run's own decision log.

Also (2026-10-09, --knob prompt_format): "full" against "compact", the same facts in shorter wording (config.PROMPT_FORMAT, backend/actions.COMPACT_DESCRIPTIONS), with
NUDGES off as in play. Question: does a shorter prompt keep decisions and outcomes the same, and make a turn faster? Extra numbers recorded for every run: the mean prompt length
and the mean latency of a turn, read from each tribe's own transcript. --model picks the model both tribes use. Results go to scripts/ab_test_prompt_format_results.json.

Also (2026-10-09, --knob journal_readback): the chief's journal read-back (docs/CHIEF-EVIDENCE-MEMORY-DESIGN.md, Step 2; JOURNAL_READBACK) off against on, with the prompt
at its defaults (NUDGES off, PROMPT_FORMAT full). Built on 2026-10-03 and never compared. Question: when the chief is told plainly what its recent repeated choices produced
("In your last 10 choices you picked GATHER_FOOD 8 times: food +14; nothing was built."), does the tribe repeat less, and take the one-time actions (farm, cook, fish, keep, mine,
forge) it otherwise skips? Extra numbers recorded for every run, read from the run's own log: per tribe the longest run of one action, the top action's share, the first cycle each
of those actions was chosen, and how many read-back lines were shown. Results go to scripts/ab_test_journal_readback_results.json.

Question (the nudge switch): when the prompt lines whose job is to steer a tribe (build hints, farm hints, the Historian's "choose something different", the
survival-warning text, the growth framing; docs/NUDGE-AUDIT.md) are removed, do tribes still survive, advance and build? Only prompt text
differs between the arms: menus, gates and mechanics are identical (tests/test_nudge_switch.py).

Design: two tribes on the same model (qwen2.5:3b, as in the last live run), 250 cycles, the same seeds in both arms, arms interleaved
(seed A on, A off, B off, B on) so drift in the machine does not line up with one arm. random.seed(seed) fixes the game's own dice; the
model's sampling is not seeded, so runs differ even with the same seed. With two runs per arm the result is suggestive, not proof.

Two modes. "early" starts both tribes fresh (the settling, farming and first-building nudges). "mid" starts both from real states at about
15,000 population (backend/fixtures/mid_game_15k_a.json and _b.json, cut from run_20261004_130133 cycle 281: five long houses, a keep, a
library, a tannery, and no forge, mine, DMM, fortress or castle yet), so the build hints for the later buildings can be tested in 100 cycles
instead of 300; the fixture path is the one run_benchmark.py already uses.

Each run records, per tribe: the cycle of each era entry, population at fixed cycles, whether it survived, the action mix, and a few
structures at the end. Results go to scripts/ab_test_nudges_off_results.json after every run, so an interrupted batch keeps what it finished.

    python scripts/ab_test_nudges_off.py --mode mid --cycles 100 --seeds 3
    python scripts/ab_test_nudges_off.py --mode early --cycles 250 --seeds 2
    python scripts/ab_test_nudges_off.py --mode mid --cycles 6 --seeds 1   (a smoke test; delete the results file afterward)
"""
import argparse
import asyncio
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from backend.simulation import Simulation
from backend.tribe_fixtures import apply_tribe_fixture, load_fixture

MODEL = "qwen2.5:3b"  # --model overrides
RESULTS = "scripts/ab_test_nudges_off_results.json"
KNOBS = {"nudges": ("on", "off"), "menu_cap": ("full", "cap8"), "nodes": ("on", "off"), "prompt_format": ("full", "compact"), "journal_readback": ("off", "on")}
FIXTURES = ("mid_game_15k_a", "mid_game_15k_b")
SAMPLE_CYCLES = (50, 100, 150, 200, 250)
STRUCTURE_FLAGS = ("long_houses_built", "kitchen_built", "tannery_built", "library_built", "barracks_built", "forge_built",
                   "mine_built", "coop_built", "dock_built", "sawmill_built", "quarry_built", "well_built", "fishing_learned",
                   "keep_built", "fortress_built", "castle_built", "moat_built", "dmm_built", "deer_pen_built", "road_built",
                   "hatchery_built", "bath_house_built", "warehouses_built")


def _decision_stats(run_id: str) -> dict:
    sizes, no_effect, total = [], 0, 0
    try:
        for line in open(f"logs/{run_id}.jsonl", encoding="utf-8"):
            record = json.loads(line)
            if record.get("kind") == "decision":
                data = record["data"]
                total += 1
                if data.get("menu_size"):
                    sizes.append(data["menu_size"])
                if not data.get("built") and not any(data.get("delta", {}).values()) and not data.get("moved"):
                    no_effect += 1
    except OSError:
        pass
    return {"decisions": total, "mean_menu_size": round(sum(sizes) / len(sizes), 1) if sizes else None,
            "max_menu_size": max(sizes) if sizes else None, "no_effect_share": round(no_effect / total, 3) if total else None}


FIRST_ACTIONS = ("PLANT_CROP", "COOK_FOOD", "CATCH_FISH", "BUILD_LONG_HOUSE", "BUILD_KEEP", "BUILD_MINE", "BUILD_FORGE")


def _behavior_stats(run_id: str) -> dict:
    """Per tribe, from the run's decision log: the longest run of one action in a row, the top action and its share of all choices, the first cycle each of FIRST_ACTIONS
    was chosen (None if never), and how many journal read-back lines were shown."""
    chosen, readback = {}, {}
    try:
        for line in open(f"logs/{run_id}.jsonl", encoding="utf-8", errors="replace"):
            record = json.loads(line)
            if record.get("kind") == "decision":
                chosen.setdefault(record["tribe"], []).append((record["cycle"], record["data"]["action"]))
            elif record.get("kind") == "journal_readback":
                readback[record["tribe"]] = readback.get(record["tribe"], 0) + len(record["data"].get("lines", []))
    except OSError:
        pass
    stats = {}
    for tribe, rows in chosen.items():
        actions = [a for _, a in rows]
        streak = best = 0
        previous = None
        for a in actions:
            streak = streak + 1 if a == previous else 1
            best = max(best, streak)
            previous = a
        top, top_n = collections.Counter(actions).most_common(1)[0]
        stats[tribe] = {"decisions": len(actions), "longest_repeat": best, "top_action": top, "top_share": round(top_n / len(actions), 3),
                        "first_chosen": {a: next((c for c, x in rows if x == a), None) for a in FIRST_ACTIONS}, "readback_lines": readback.get(tribe, 0)}
    return stats


def _anomalies(run_id: str, knob: str, variant: str, client_counters: dict) -> list:
    """Everything in a finished run that was not expected, as plain sentences, so a batch is never averaged over something odd (2026-10-09). Read from the run's own log:
    decisions the game could not read as a choice (unrecognized text, or no action fields at all), a pick that was not on the menu it was offered, an action the game does not
    know, a tribe that went quiet, loop aborts and truncated replies counted by the client, and an arm that did not really differ from the other (read-back on that never showed a
    line, or off that did)."""
    from backend.actions import ACTION_REGISTRY
    found, counts = [], collections.Counter()
    readback_lines = 0
    try:
        for line in open(f"logs/{run_id}.jsonl", encoding="utf-8", errors="replace"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["unreadable log line"] += 1
                continue
            message = record.get("message", "")
            if "unrecognized decision text" in message:
                counts["decision with unrecognized text"] += 1
            if "didn't come through clearly" in message:
                counts["decision with no usable action fields"] += 1
            if "falls silent mid-thought" in message:
                counts["failover switched a tribe's model"] += 1
            if "Traceback" in message or record.get("kind") == "error":
                counts["error record in the log"] += 1
            if record.get("kind") == "journal_readback":
                readback_lines += len(record["data"].get("lines", []))
            if record.get("kind") == "decision":
                data = record["data"]
                action, menu = data.get("action"), data.get("menu") or []
                if action not in ACTION_REGISTRY:
                    counts[f"action the game does not know: {action}"] += 1
                if menu and action not in menu:
                    counts["pick that was not on the menu offered"] += 1
    except OSError:
        counts["run log missing"] += 1
    found += [f"{n} x {what}" for what, n in counts.items()]
    for kind, per_model in client_counters.items():
        for model, n in per_model.items():
            if n:
                found.append(f"{n} x {kind.replace('_', ' ')} on {model}")
    if knob == "journal_readback":
        if variant == "on" and readback_lines == 0:
            found.append("read-back was ON but showed no line, so this arm did not differ from off")
        if variant == "off" and readback_lines:
            found.append(f"read-back was OFF but {readback_lines} lines were shown")
    return found


async def run_once(variant: str, seed: int, cycles: int, mode: str, knob: str = "nudges") -> dict:
    if knob == "journal_readback":
        os.environ["JOURNAL_READBACK"] = variant
        os.environ["NUDGES"] = "off"
        os.environ["PROMPT_FORMAT"] = "full"
    elif knob == "prompt_format":
        os.environ["PROMPT_FORMAT"] = variant
        os.environ["NUDGES"] = "off"
    elif knob == "menu_cap":
        os.environ["MENU_CAP"] = "8" if variant == "cap8" else "0"
        os.environ["NUDGES"] = "on"
    elif knob == "nodes":
        os.environ["NODES"] = variant
        os.environ["NUDGES"] = "on"
    else:
        os.environ["NUDGES"] = variant
    random.seed(seed)
    sim = await Simulation.create([{"name": "Tribe 1", "model": MODEL}, {"name": "Tribe 2", "model": MODEL}])
    if mode == "mid":
        cycle = None
        for tribe, name in zip(sim.tribes.values(), FIXTURES):
            fixture = load_fixture(name)
            cycle = fixture["cycle"]
            apply_tribe_fixture(tribe, fixture, new_cycle=cycle)
        sim.cycle = cycle
    start_cycle = sim.cycle
    start_flags = {tid: {f: getattr(t, f, 0) for f in STRUCTURE_FLAGS} for tid, t in sim.tribes.items()}
    era_first = {tid: {t.era: 0} for tid, t in sim.tribes.items()}
    pop_at = {tid: {} for tid in sim.tribes}
    actions = {tid: collections.Counter() for tid in sim.tribes}
    prompt_chars = {tid: [] for tid in sim.tribes}
    latency_ms = {tid: [] for tid in sim.tribes}
    last_seen = {tid: -1 for tid in sim.tribes}
    invalid = None
    started = time.time()
    for _ in range(cycles):
        await sim.step()
        for tid, t in sim.tribes.items():
            era_first[tid].setdefault(t.era, sim.cycle)
            if sim.cycle in SAMPLE_CYCLES:
                pop_at[tid][sim.cycle] = t.population
            if t.last_action:
                actions[tid][t.last_action] += 1
            for entry in t.debug_transcript:  # one entry per turn actually sent to the model: its prompt and how long the answer took
                if entry["cycle"] > last_seen[tid]:
                    last_seen[tid] = entry["cycle"]
                    prompt_chars[tid].append(len(entry["prompt"]))
                    latency_ms[tid].append(entry["latency_ms"])
        if all(t.extinct for t in sim.tribes.values()):
            break
        # 2026-10-09: a tribe whose model keeps failing is switched to another local model after 10 unusable turns in a row (Simulation._handle_model_failure), and
        # from then on the run is no longer a test of MODEL (a live batch silently ran two tribes on hermes3:3b and an odd digest-named model). Stop such a run at
        # once and mark it invalid, rather than let it run for an hour and be averaged in; 5 unusable turns in a row is the early warning.
        if any(t.model != MODEL for t in sim.tribes.values()):
            invalid = "a tribe's model was switched by the failover: " + ", ".join(f"{t.name}={t.model}" for t in sim.tribes.values())
            break
        if any(t.consecutive_unresolved_turns >= 5 for t in sim.tribes.values()):
            invalid = "a tribe had 5 unusable turns in a row (the model is failing, the failover would switch it next)"
            break
        if (sim.cycle - start_cycle) % 25 == 0:
            print(f"    [{variant} seed {seed}] cycle {sim.cycle}, {int(time.time() - started)}s: "
                  + " | ".join(f"{t.name} pop={t.population} era={t.era}" for t in sim.tribes.values()), flush=True)
    tribes = {}
    for tid, t in sim.tribes.items():
        tribes[t.name] = {
            "era_first_cycle": era_first[tid], "final_era": t.era, "population_at": pop_at[tid], "final_population": t.population,
            "max_population": t.max_population, "extinct": t.extinct, "trades": t.trades_completed,
            "structures": {f: bool(getattr(t, f, False)) for f in STRUCTURE_FLAGS},
            "built_during_run": [f for f in STRUCTURE_FLAGS if getattr(t, f, 0) and not start_flags[tid][f]],
            "action_mix": dict(actions[tid].most_common()),
            "turns": len(prompt_chars[tid]),
            "mean_prompt_chars": round(sum(prompt_chars[tid]) / len(prompt_chars[tid])) if prompt_chars[tid] else None,
            "mean_latency_ms": round(sum(latency_ms[tid]) / len(latency_ms[tid])) if latency_ms[tid] else None,
        }
    return {"knob": knob, "decision_stats": _decision_stats(sim.run_id), "behavior": _behavior_stats(sim.run_id),
            "anomalies": _anomalies(sim.run_id, knob, variant, {"repeat_retries": dict(sim.client.repeat_retries), "truncated_replies": dict(sim.client.truncated_replies),
                                                                 "empty_reply_retries": dict(sim.client.empty_reply_retries)}), "variant": variant, "seed": seed, "model": MODEL, "invalid": invalid,
            "client_counters": {"repeat_retries": dict(sim.client.repeat_retries), "truncated_replies": dict(sim.client.truncated_replies),
                                "empty_reply_retries": dict(sim.client.empty_reply_retries)},
            "mode": mode, "start_cycle": start_cycle, "cycles": sim.cycle - start_cycle, "seconds": int(time.time() - started),
            "run_id": sim.run_id, "tribes": tribes}


def load() -> list:
    try:
        return json.load(open(RESULTS, encoding="utf-8"))
    except Exception:
        return []


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--mode", choices=("early", "mid"), default="early")
    parser.add_argument("--knob", choices=tuple(KNOBS), default="nudges")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    global RESULTS, MODEL
    if args.model:
        MODEL = args.model
    if args.knob == "journal_readback":
        RESULTS = "scripts/ab_test_journal_readback_results.json"
    elif args.knob == "prompt_format":
        RESULTS = "scripts/ab_test_prompt_format_results.json"
    elif args.knob == "menu_cap":
        RESULTS = "scripts/ab_test_menu_cap_results.json"
    elif args.knob == "nodes":
        RESULTS = "scripts/ab_test_nodes_results.json"
    first, second = KNOBS[args.knob]
    seeds = [1000 + i for i in range(args.seeds)]
    order = []
    for i, seed in enumerate(seeds):
        order += [(seed, first), (seed, second)] if i % 2 == 0 else [(seed, second), (seed, first)]
    results = load()
    for seed, variant in order:
        if any(r["seed"] == seed and r["variant"] == variant and r.get("mode", "early") == args.mode and r.get("model", "qwen2.5:3b") == MODEL and not r.get("invalid") for r in results):
            print(f"skip seed {seed} {variant} (already recorded)")
            continue
        print(f"=== {args.mode} seed {seed}, {args.knob}={variant}, {MODEL}, {args.cycles} cycles ===", flush=True)
        try:
            results.append(await run_once(variant, seed, args.cycles, args.mode, args.knob))
        except Exception as error:  # noqa: BLE001 -- recorded as an invalid run, then the batch continues
            import traceback
            results.append({"knob": args.knob, "variant": variant, "seed": seed, "model": MODEL, "mode": args.mode, "cycles": 0, "seconds": 0, "run_id": None, "tribes": {},
                            "decision_stats": {}, "behavior": {}, "anomalies": [], "invalid": "the run crashed: " + traceback.format_exc()[-1500:]})
        json.dump(results, open(RESULTS, "w", encoding="utf-8"), indent=1)
        r = results[-1]
        if r.get("invalid"):
            print(f"=== INVALID RUN, not counted: {r['invalid']}", flush=True)
        for note in r.get("anomalies", []):
            print(f"=== ANOMALY: {note}", flush=True)
        print(f"=== done: {r['cycles']} cycles in {r['seconds']}s, "
              + "; ".join(f"{n}: {t['final_era']} pop {t['final_population']}{' EXTINCT' if t['extinct'] else ''}" for n, t in r["tribes"].items()),
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
