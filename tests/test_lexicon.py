"""Steps A and B of docs/LANGUAGE-LEXICON-DESIGN.md: the lexicons record use and context; war cries are witnessed by the victim."""
from backend import actions, config, lexicon
from backend.simulation import Simulation
from tests.conftest import run_async


def _sim():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "gemma2:2b"}])
    sim.cycle = 80
    seen = []
    sim.event_log.record_data = lambda tribe_name, kind, data, **k: seen.append((kind, data))
    return sim, sim.tribes["tribe_0"], sim.tribes["tribe_1"], seen


def test_words_are_split_and_deduplicated():
    assert lexicon.words_of("kra-zul, MEE-LO kra-zul!") == ["KRA-ZUL", "MEE-LO"]
    assert lexicon.words_of("") == [] and lexicon.words_of("a") == []


def test_own_use_tracks_the_dominant_action_and_counts_mismatches():
    sim, a, b, _ = _sim()
    for action in ("SCOUT", "SCOUT", "GATHER_WOOD"):
        payload = lexicon.own_use(a, "ZUR-NEV", action, 10)
    row = payload["words"][0]
    assert row["uses"] == 3 and row["dominant"] == "SCOUT" and row["share"] == 0.67 and row["mismatches"] == 1
    assert lexicon.own_use(a, "", "SCOUT", 11) is None


def test_hear_counts_contexts_per_word():
    sim, a, b, _ = _sim()
    lexicon.hear(a, "TIK", ["speaker doing GATHER_WOOD", "near a timber grove"], 5, "overheard:B")
    payload = lexicon.hear(a, "TIK", ["speaker doing GATHER_WOOD"], 9, "overheard:B")
    assert a.heard_lexicon["TIK"]["contexts"] == {"speaker doing GATHER_WOOD": 2, "near a timber grove": 1}
    assert payload["words"][0]["top_context"] == "speaker doing GATHER_WOOD" and payload["words"][0]["heard"] == 2


def test_a_war_declaration_carries_the_declarers_cry_to_the_victim():
    sim, a, b, seen = _sim()
    a.barracks_built = 1
    a.current_broadcast = "KRA-ZUL"
    a.x, a.y, b.x, b.y = 40, 40, 44, 40
    a.discovered_rivals.add(b.id)
    result = actions._declare_war(sim, a, "plains", (b.x, b.y))
    assert "declares war" in result
    assert b.witnessed_cries[-1]["phrase"] == "KRA-ZUL" and b.witnessed_cries[-1]["kind"] == "war declaration"
    assert b.heard_lexicon["KRA-ZUL"]["contexts"]["attack"] == 1
    assert any(kind == "witnessed_cry" for kind, _ in seen) or b.history[-1].startswith("A attacked")
    assert a.witnessed_cries == []  # the aggressor does not hear its own cry as a victim


def test_a_silent_attacker_leaves_no_cry():
    sim, a, b, _ = _sim()
    a.barracks_built = 1
    a.current_broadcast = ""
    a.x, a.y, b.x, b.y = 40, 40, 44, 40
    a.discovered_rivals.add(b.id)
    assert "declares war" in actions._declare_war(sim, a, "plains", (b.x, b.y))
    assert b.witnessed_cries == []


def test_raiders_have_a_fixed_cry_the_chief_then_sees():
    sim, a, b, seen = _sim()
    a.population = 50
    sim._resolve_raider_attack(a)
    assert a.witnessed_cries[-1]["from"] == "Raiders" and a.witnessed_cries[-1]["phrase"] == config.RAIDER_WAR_CRY
    request, _ctx = sim._prepare_turn(a)
    assert f"Raiders attacked (raid) shouting '{config.RAIDER_WAR_CRY}'" in request["prompt"]
    assert "witnessed_cry" in [kind for kind, _ in seen]


def test_an_old_witnessed_cry_leaves_the_chiefs_view():
    sim, a, b, _ = _sim()
    a.witnessed_cries.append({"from": "B", "phrase": "OLD-CRY", "kind": "raid", "cycle": 1})
    sim.cycle = 1 + config.WITNESSED_CRY_VISIBLE_CYCLES + 5
    request, _ctx = sim._prepare_turn(a)
    assert "OLD-CRY" not in request["prompt"]


def test_a_turn_records_the_tribes_own_word_use():
    sim, a, b, seen = _sim()
    sim.cycle = 12
    payload = lexicon.own_use(a, "ZUR-NEV", "SCOUT", sim.cycle)
    assert a.lexicon["ZUR-NEV"]["counts"] == {"SCOUT": 1} and payload["phrase"] == "ZUR-NEV"
