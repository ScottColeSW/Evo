"""backend/endgame_report.py: the timeline recorder, the findings read out of it, and the headline for each ending."""
from backend.endgame_report import Timeline, build_findings, build_headline, timeline_row, tribe_finals
from backend.simulation import Simulation


def _row(name, pop, era="primitive_dawn", flags=None, trades=0, conquests=0, extinct=False, departed=False, vessel=0, action="SCOUT", hist=""):
    return {"name": name, "color": "#fff", "model": "qwen2.5:3b", "pop": pop, "food": 10, "water": 10, "wood": 5, "stone": 5, "era": era,
            "flags": flags or {}, "trades": trades, "conquests": conquests, "extinct": extinct, "departed": departed, "vessel_paid": vessel,
            "last_action": action, "last_history": hist}


def test_the_recorder_finds_events_by_comparing_samples():
    t = Timeline()
    t.sample(1, {"a": _row("Alpha", 10)})
    t.sample(2, {"a": _row("Alpha", 12, era="cognitive_horizon", flags={"kitchen_built": 1})})
    t.sample(3, {"a": _row("Alpha", 400, era="cognitive_horizon", flags={"kitchen_built": 1}, trades=1)})
    t.sample(4, {"a": _row("Alpha", 300, era="cognitive_horizon", flags={"kitchen_built": 1, "vessel_built": 1}, trades=1, vessel=500, hist="a raid struck")})
    labels = [(e["cycle"], e["kind"], e["label"]) for e in t.events]
    assert (2, "era", "entered the Cognitive Horizon") in labels
    assert (2, "built", "kitchen built") in labels
    assert (3, "trade", "first trade") in labels
    assert (4, "vessel", "vessel complete") in labels and (4, "vessel", "began building the vessel") in labels
    assert any(k == "drop" and "lost 100 people" in label and "a raid struck" in label for _c, k, label in labels)
    assert t.actions["a"]["SCOUT"] == 4


def test_a_tribe_that_vanishes_from_the_samples_is_recorded_as_absorbed_and_the_same_cycle_twice_replaces_the_sample():
    t = Timeline()
    t.sample(1, {"a": _row("Alpha", 10), "b": _row("Beta", 10)})
    t.sample(2, {"a": _row("Alpha", 11), "b": _row("Beta", 11)})
    t.sample(2, {"a": _row("Alpha", 99), "b": _row("Beta", 11)})  # the same cycle again
    t.sample(3, {"a": _row("Alpha", 100)})
    assert t.series["a"]["pop"] == [10, 99, 100]
    assert any(e["kind"] == "absorbed" and e["name"] == "Beta" and e["cycle"] == 3 for e in t.events)


def test_the_payload_is_thinned_to_a_bounded_size_and_keeps_the_last_point():
    t = Timeline()
    for c in range(1, 1001):
        t.sample(c, {"a": _row("Alpha", c)})
    p = t.payload(max_points=100)
    assert len(p["tribes"]["a"]["cycles"]) <= 101
    assert p["tribes"]["a"]["cycles"][-1] == 1000 and p["tribes"]["a"]["pop"][-1] == 1000


def _two_tribe_run():
    t = Timeline()
    for c in range(1, 201):
        a = 20 + c * 3 if c < 100 else 320 + (c - 100) * 40  # a surge after cycle 100
        b = 20 + c * 8
        era_a = "primitive_dawn" if c < 60 else "cognitive_horizon"
        t.sample(c, {"a": _row("Alpha", a, era=era_a, flags={"kitchen_built": 1} if c > 105 else {}, action="GATHER_STONE" if c % 2 else "SCOUT"),
                     "b": _row("Beta", b, era="primitive_dawn" if c < 150 else "cognitive_horizon", action="SCOUT")})
    return t.payload()


def test_findings_state_only_what_the_data_supports_and_use_real_names_and_numbers():
    payload = _two_tribe_run()
    finals = {"a": {"name": "Alpha", "model": "m", "era": "cognitive_horizon", "era_label": "Cognitive Horizon", "population": 4000, "max_population": 4000,
                    "extinct": False, "departed": False, "words": [{"word": "KRA-ZUL", "uses": 40, "action": "SCOUT", "share": 0.5}], "words_total": 12},
              "b": {"name": "Beta", "model": "m", "era": "primitive_dawn", "era_label": "Primitive Dawn", "population": 1600, "max_population": 1600,
                    "extinct": False, "departed": False, "words": [], "words_total": 0}}
    findings = build_findings(payload, finals)
    titles = {f["title"] for f in findings}
    assert "The great surge" in titles and "The longest wait" in titles and "A real race" in titles
    assert "A word that stuck" in titles and "What the days were really spent on" in titles
    race = next(f for f in findings if f["title"] == "A real race")
    assert "Alpha" in race["text"] or "Beta" in race["text"]
    word = next(f for f in findings if f["title"] == "A word that stuck")
    assert "KRA-ZUL" in word["text"] and "40" in word["text"]
    assert all(f["icon"] and f["text"] for f in findings) and len(findings) <= 8


def test_findings_on_a_tiny_run_are_empty_not_invented():
    t = Timeline()
    t.sample(1, {"a": _row("Alpha", 8)})
    assert build_findings(t.payload(), {}) == []


def test_each_ending_gets_its_own_scene_with_the_numbers_filled_in():
    payload = _two_tribe_run()
    alive = {"name": "Alpha", "model": "qwen2.5:3b", "era": "departure_era", "era_label": "Beyond the Horizon", "population": 44000, "max_population": 44000,
             "extinct": False, "departed": True, "words": [], "words_total": 0}
    dead = dict(alive, extinct=True, departed=False, population=0, extinction_cause="starvation")
    dep = build_headline("departure", payload, {"a": alive}, 540)
    assert dep["scene"] == "departure" and "Congratulations" in dep["title"] and "44,000" in dep["text"] and "qwen2.5:3b" in dep["shows"]
    assert build_headline("world_domination", payload, {"a": dict(alive, departed=False)}, 400)["scene"] == "conquest"
    assert build_headline("golden_age", payload, {"a": alive}, 400)["scene"] == "peace"
    grave = build_headline("extinction", payload, {"a": dead}, 120)
    assert grave["scene"] == "grave" and "Alpha" in grave["text"] and "starvation" in grave["text"]
    assert build_headline("era_ceiling", payload, {"a": alive}, 700)["scene"] == "summit"
    assert build_headline("manual_quit", payload, {"a": alive}, 50)["scene"] == "quiet"


def test_a_real_simulation_samples_each_step_and_builds_the_report_at_game_over():
    import asyncio
    from unittest import mock

    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    sim.tribes["tribe_0"].chief_name = "Ashgar"

    async def run():
        with mock.patch.object(sim.scheduler, "run_batch", mock.AsyncMock(return_value={})), \
             mock.patch("backend.simulation.generate_endgame_narrative", mock.AsyncMock(return_value="")), \
             mock.patch.object(sim.client, "unload_model", mock.AsyncMock()):
            await sim.step()
            await sim.step()
            await sim._trigger_game_over("manual_quit")

    asyncio.run(run())
    report = sim.snapshot()["game_over_report"]
    assert report and report["headline"]["scene"] == "quiet"
    assert report["timeline"]["tribes"]["tribe_0"]["cycles"][-1] == sim.cycle
    row = timeline_row(sim.tribes["tribe_0"])
    assert row["pop"] == sim.tribes["tribe_0"].population and tribe_finals(sim.tribes.values())["tribe_0"]["name"] == "A"
