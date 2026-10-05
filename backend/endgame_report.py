"""What the end-of-run splash shows beyond the plain summary (2026-10-05): a timeline of the run, findings read out of it, and a headline for the ending.

Three parts, all plain functions over plain data so they can be tested without a live run:

  Timeline        sampled once a cycle by Simulation.step: population and stockpiles per tribe, the era and structures, and the events found by
                  comparing one sample with the last (an era entered, a structure first built, a first trade, a conquest, the vessel started and
                  finished, a tribe gone). It also counts the action each tribe chose each cycle.
  build_findings  a few "aha" facts with the numbers behind them: the fastest growth, the longest wait, how the lead changed hands, the worst
                  blow, what the tribes actually spent their turns on, and their most used word. Each is stated only when the data supports it.
  build_headline  the scene the splash plays (fireworks for a departure, a crown for a conquest, a grave when everyone is gone) and the line
                  under it, with the numbers filled in.

Nothing here changes the game; it only reads it, and a failure in any part is swallowed by the caller (the splash falls back to what it showed before).
"""
from __future__ import annotations

import collections

from .eras import ERAS

_ERA_LABEL = {era.key: era.label for era in ERAS}
SERIES_KEYS = ("pop", "food", "water", "wood", "stone")
# A population drop gets the tribe's latest chronicle line as its cause only when that line plainly names one.
_CAUSE_WORDS = ("raid", "ambush", "starv", "thirst", "volcano", "drown", "wolf", "cull", "overcrowd", "plague", "conquest", "war", "cliff", "rip current")


def _title(text: str) -> str:
    return text.replace("_", " ").strip()


def timeline_row(tribe) -> dict:
    """The few numbers one sample needs from a live Tribe."""
    flags = {name: int(value) for name, value in vars(tribe).items()
             if name.endswith("_built") and isinstance(value, (bool, int)) and not isinstance(value, type(None))}
    history = getattr(tribe, "history", None)
    last_history = ""
    try:
        if history:
            last_history = str(list(history)[-1])[:160]
    except Exception:  # noqa: BLE001
        last_history = ""
    return {
        "name": tribe.name, "color": tribe.color, "model": tribe.model,
        "pop": int(tribe.population), "food": int(tribe.food), "water": int(tribe.water), "wood": int(tribe.wood), "stone": int(tribe.stone),
        "era": tribe.era, "flags": flags, "trades": int(tribe.trades_completed), "conquests": int(getattr(tribe, "conquests_won", 0)),
        "extinct": bool(tribe.extinct), "departed": bool(getattr(tribe, "departed", False)),
        "vessel_paid": int(getattr(tribe, "vessel_wood_paid", 0)) + int(getattr(tribe, "vessel_stone_paid", 0)),
        "last_action": getattr(tribe, "last_action", None) or "", "last_history": last_history,
    }


class Timeline:
    def __init__(self) -> None:
        self.series: dict[str, dict] = {}
        self.events: list[dict] = []
        self.actions: dict[str, collections.Counter] = {}
        self._last: dict[str, dict] = {}
        self._gone: set[str] = set()
        self.first_cycle: int | None = None
        self.last_cycle: int | None = None

    def _event(self, cycle: int, tid: str, name: str, kind: str, label: str) -> None:
        self.events.append({"cycle": cycle, "tribe": tid, "name": name, "kind": kind, "label": label})

    def sample(self, cycle: int, rows: dict[str, dict]) -> None:
        if self.first_cycle is None:
            self.first_cycle = cycle
        self.last_cycle = cycle
        for tid, last in list(self._last.items()):
            if tid not in rows and tid not in self._gone:
                self._gone.add(tid)
                self._event(cycle, tid, last["name"], "absorbed", "absorbed by a rival")
        for tid, row in rows.items():
            series = self.series.setdefault(tid, {"name": row["name"], "color": row["color"], "model": row["model"], "cycles": [],
                                                  **{k: [] for k in SERIES_KEYS}})
            if series["cycles"] and series["cycles"][-1] == cycle:  # the same cycle sampled twice: keep the later one
                for key in SERIES_KEYS:
                    series[key][-1] = row[key]
            else:
                series["cycles"].append(cycle)
                for key in SERIES_KEYS:
                    series[key].append(row[key])
                if row["last_action"]:
                    self.actions.setdefault(tid, collections.Counter())[row["last_action"]] += 1
            last = self._last.get(tid)
            if last is not None:
                self._diff(cycle, tid, last, row)
            self._last[tid] = row

    def _diff(self, cycle: int, tid: str, last: dict, row: dict) -> None:
        name = row["name"]
        if row["era"] != last["era"]:
            self._event(cycle, tid, name, "era", f"entered the {_ERA_LABEL.get(row['era'], _title(row['era']))}")
        for flag, value in row["flags"].items():
            if flag.endswith("ever_built"):  # a record that something happened once (fire), not a structure
                continue
            if value and not last["flags"].get(flag):
                label = "vessel complete" if flag == "vessel_built" else f"{_title(flag[:-6])} built"
                self._event(cycle, tid, name, "vessel" if flag == "vessel_built" else "built", label)
        if row["vessel_paid"] and not last["vessel_paid"]:
            self._event(cycle, tid, name, "vessel", "began building the vessel")
        if row["trades"] and not last["trades"]:
            self._event(cycle, tid, name, "trade", "first trade")
        if row["conquests"] > last["conquests"]:
            self._event(cycle, tid, name, "conquest", "won a conquest")
        if row["departed"] and not last["departed"]:
            self._event(cycle, tid, name, "departed", "sailed beyond the horizon")
        if row["extinct"] and not last["extinct"]:
            self._event(cycle, tid, name, "extinct", "died out")
        if last["pop"] >= 200 and row["pop"] < last["pop"] * 0.88 and not row["extinct"]:
            hint = row["last_history"] if any(word in row["last_history"].lower() for word in _CAUSE_WORDS) else ""
            self._event(cycle, tid, name, "drop", f"lost {last['pop'] - row['pop']:,} people" + (f" ({hint})" if hint else ""))

    def payload(self, max_points: int = 220) -> dict:
        tribes = {}
        for tid, series in self.series.items():
            n = len(series["cycles"])
            stride = max(1, -(-n // max_points))
            idx = list(range(0, n, stride))
            if idx and idx[-1] != n - 1:
                idx.append(n - 1)
            tribes[tid] = {"name": series["name"], "color": series["color"], "model": series["model"],
                           "cycles": [series["cycles"][i] for i in idx], **{k: [series[k][i] for i in idx] for k in SERIES_KEYS}}
        return {"tribes": tribes, "events": list(self.events), "actions": {tid: dict(c.most_common()) for tid, c in self.actions.items()},
                "first_cycle": self.first_cycle, "last_cycle": self.last_cycle}


def _top_words(tribe, limit: int = 3) -> list[dict]:
    rows = []
    for word, entry in (getattr(tribe, "lexicon", None) or {}).items():
        counts = entry.get("counts") or {}
        if not counts:
            continue
        action, n = max(counts.items(), key=lambda kv: kv[1])
        rows.append({"word": word, "uses": entry.get("uses", 0), "action": action, "share": round(n / max(1, sum(counts.values())), 2)})
    rows.sort(key=lambda r: -r["uses"])
    return rows[:limit]


def tribe_finals(tribes) -> dict[str, dict]:
    """What the findings need from each live Tribe at the end."""
    return {t.id: {"name": t.name, "model": t.model, "era": t.era, "era_label": _ERA_LABEL.get(t.era, t.era), "population": int(t.population),
                   "max_population": int(getattr(t, "max_population", t.population)), "extinct": bool(t.extinct),
                   "departed": bool(getattr(t, "departed", False)), "words": _top_words(t), "words_total": len(getattr(t, "lexicon", {}) or {}),
                   "extinction_cause": getattr(t, "extinction_cause", None)}
            for t in tribes}


def _era_stays(payload: dict) -> list[dict]:
    """Every era each tribe lived in: {tribe, name, era, start, end, days}. The era a tribe ended in runs to the last sample."""
    last = payload.get("last_cycle") or 0
    stays = []
    for tid, series in payload["tribes"].items():
        entered = [(e["cycle"], e["label"]) for e in payload["events"] if e["tribe"] == tid and e["kind"] == "era"]
        start = series["cycles"][0] if series["cycles"] else 0
        starts = [(start, "the Primitive Dawn")] + [(c, label.replace("entered the ", "")) for c, label in entered]
        for i, (c, label) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else last
            stays.append({"tribe": tid, "name": series["name"], "era": label, "start": c, "end": end, "days": end - c})
    return stays


def build_findings(payload: dict, finals: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    tribes = payload["tribes"]

    # The fastest growth: the best multiple over a 25-cycle window, from a tribe that already had some size.
    best = None
    for tid, s in tribes.items():
        cyc, pop = s["cycles"], s["pop"]
        for i in range(len(cyc)):
            if pop[i] < 40:
                continue
            j = i
            while j + 1 < len(cyc) and cyc[j + 1] - cyc[i] <= 25:
                j += 1
            if j > i and cyc[j] - cyc[i] >= 10:
                ratio = pop[j] / pop[i]
                if best is None or ratio > best[0]:
                    best = (ratio, tid, cyc[i], cyc[j], pop[i], pop[j])
    if best and best[0] >= 1.8:
        ratio, tid, a, b, p0, p1 = best
        during = [e["label"] for e in payload["events"] if e["tribe"] == tid and a <= e["cycle"] <= b and e["kind"] in ("built", "era")][:3]
        extra = f" In the same stretch: {', '.join(during)}." if during else ""
        out.append({"icon": "📈", "title": "The great surge",
                    "text": f"{tribes[tid]['name']} grew {ratio:.1f} times larger in {b - a} cycles (cycle {a} to {b}), from {p0:,} to {p1:,} people.{extra}"})

    # The longest wait in one era, as a share of that tribe's life.
    stays = [s for s in _era_stays(payload) if s["days"] > 0]
    if stays:
        worst = max(stays, key=lambda s: s["days"])
        life = max(1, (payload.get("last_cycle") or 1) - (tribes[worst["tribe"]]["cycles"][0] if tribes[worst["tribe"]]["cycles"] else 0))
        out.append({"icon": "⏳", "title": "The longest wait",
                    "text": f"{worst['name']} spent {worst['days']} cycles in {worst['era']}, {round(100 * worst['days'] / life)}% of its whole story, the slowest stretch of the run."})

    # How the lead changed hands, when there were at least two tribes.
    if len(tribes) >= 2:
        all_cycles = sorted({c for s in tribes.values() for c in s["cycles"]})
        lookup = {tid: dict(zip(s["cycles"], s["pop"])) for tid, s in tribes.items()}
        leader, changes, last_change = None, 0, None
        current = {}
        for c in all_cycles:
            for tid in tribes:
                if c in lookup[tid]:
                    current[tid] = lookup[tid][c]
            if len(current) < 2:
                continue
            top = max(current, key=lambda t: current[t])
            runner = sorted(current.values())[-2]
            if current[top] <= runner * 1.02:
                continue
            if leader is not None and top != leader:
                changes += 1
                last_change = (c, top)
            leader = top
        if leader is not None:
            if changes:
                out.append({"icon": "🏁", "title": "A real race",
                            "text": f"The lead changed hands {changes} time{'s' if changes != 1 else ''}. {tribes[last_change[1]]['name']} took it at cycle {last_change[0]}."})
            else:
                out.append({"icon": "🏁", "title": "Wire to wire",
                            "text": f"{tribes[leader]['name']} was ahead on population from the first cycle the two could be told apart to the last."})

    # The worst single blow.
    drops = [e for e in payload["events"] if e["kind"] == "drop"]
    if drops:
        e = max(drops, key=lambda ev: int("".join(ch for ch in ev["label"].split(" people")[0] if ch.isdigit()) or 0))
        out.append({"icon": "💥", "title": "The worst blow", "text": f"{e['name']}: {e['label'][0].upper() + e['label'][1:]}, at cycle {e['cycle']}."})

    # What they actually spent their turns on.
    for tid, counts in payload["actions"].items():
        total = sum(counts.values())
        if total >= 30 and tid in tribes:
            action, n = max(counts.items(), key=lambda kv: kv[1])
            out.append({"icon": "🧭", "title": "What the days were really spent on",
                        "text": f"{tribes[tid]['name']} chose {_title(action).lower()} on {round(100 * n / total)}% of its {total:,} turns, more than any other single choice."})
            break

    # The most used word.
    words = [(w, tid) for tid, f in finals.items() for w in f.get("words", [])]
    if words:
        w, tid = max(words, key=lambda pair: pair[0]["uses"])
        if w["uses"] >= 8:
            out.append({"icon": "🗣️", "title": "A word that stuck",
                        "text": f"\"{w['word']}\" was shouted {w['uses']:,} times by {finals[tid]['name']}, mostly while it was {_title(w['action']).lower()} ({round(100 * w['share'])}% of the time). "
                                f"That tribe invented {finals[tid]['words_total']} words in all."})

    # What they built.
    built = collections.Counter(e["tribe"] for e in payload["events"] if e["kind"] == "built")
    if built:
        tid, n = built.most_common(1)[0]
        firsts = [e for e in payload["events"] if e["tribe"] == tid and e["kind"] == "built"]
        out.append({"icon": "🏗️", "title": "Builders",
                    "text": f"{tribes[tid]['name']} raised {n} kinds of structure, from its {firsts[0]['label'].replace(' built', '')} at cycle {firsts[0]['cycle']} to its {firsts[-1]['label'].replace(' built', '')} at cycle {firsts[-1]['cycle']}."})
    return out[:8]


def build_headline(reason: str | None, payload: dict, finals: dict[str, dict], cycle: int) -> dict:
    """{scene, title, text, shows}: the scene the splash plays and the words under it. `shows` is a plain, checkable line about the run itself."""
    living = [f for f in finals.values() if not f["extinct"]]
    models = sorted({f["model"] for f in finals.values()})
    decisions = sum(sum(c.values()) for c in payload["actions"].values())
    eras_climbed = {tid: sum(1 for e in payload["events"] if e["tribe"] == tid and e["kind"] == "era") for tid in payload["tribes"]}
    best_tid = max(eras_climbed, key=lambda t: eras_climbed[t], default=None)
    shows = (f"{' and '.join(models)}, running locally, made {decisions:,} decisions in {cycle} cycles"
             + (f"; the furthest tribe climbed {eras_climbed[best_tid]} eras." if best_tid and eras_climbed[best_tid] else "."))

    def name_of(tid_or_f):
        return finals[tid_or_f]["name"] if tid_or_f in finals else str(tid_or_f)

    if reason == "departure":
        departed = [f for f in finals.values() if f["departed"]]
        who = departed[0] if departed else (living[0] if living else None)
        vessel = next((e for e in payload["events"] if e["kind"] == "vessel" and e["label"] == "vessel complete"), None)
        began = next((e for e in payload["events"] if e["kind"] == "vessel" and e["label"].startswith("began")), None)
        built_line = (f" It began the vessel at cycle {began['cycle']} and finished it at cycle {vessel['cycle']}." if began and vessel else "")
        return {"scene": "departure", "title": "Congratulations!",
                "text": f"{who['name'] if who else 'A tribe'} sailed beyond the horizon at cycle {cycle}, {who['population']:,} people strong, "
                        f"having climbed to the {who['era_label'] if who else 'last era'}.{built_line}", "shows": shows}
    if reason == "world_domination":
        winner = living[0] if living else None
        return {"scene": "conquest", "title": "One tribe stands alone",
                "text": f"{winner['name'] if winner else 'A tribe'} conquered every rival and rules the island alone at cycle {cycle}, with {winner['population']:,} people.", "shows": shows}
    if reason == "golden_age":
        return {"scene": "peace", "title": "A monument to peace",
                "text": f"Two tribes finished a castle together at cycle {cycle} instead of fighting over the island.", "shows": shows}
    if reason == "extinction":
        lasted = max((s["cycles"][-1] - s["cycles"][0] for s in payload["tribes"].values() if s["cycles"]), default=cycle)
        peak = max((f["max_population"] for f in finals.values()), default=0)
        names = ", ".join(f["name"] for f in finals.values())
        cause = next((f["extinction_cause"] for f in finals.values() if f.get("extinction_cause")), None)
        return {"scene": "grave", "title": "Here lie the tribes",
                "text": f"{names}: gone by cycle {cycle}{' (' + str(cause) + ')' if cause else ''}. They lasted {lasted} cycles at most and peaked at {peak:,} people.", "shows": shows}
    if reason == "era_ceiling":
        return {"scene": "summit", "title": "The top of the ladder",
                "text": f"Every living tribe reached the last era by cycle {cycle} and none left the island.", "shows": shows}
    return {"scene": "quiet", "title": "The run ended", "text": f"The run was stopped at cycle {cycle}.", "shows": shows}


def build_report(timeline: Timeline, tribes, reason: str | None, cycle: int) -> dict:
    payload = timeline.payload()
    finals = tribe_finals(tribes)
    return {"timeline": payload, "findings": build_findings(payload, finals), "headline": build_headline(reason, payload, finals, cycle)}
