"""2026-10-09: config.PROMPT_FORMAT = "compact" is the same facts in fewer words (the prompt-format A/B, scripts/ab_test_nudges_off.py --knob prompt_format).
What these tests protect: that "compact" never drops a number, a limit or a named action, and that the default ("full") prompt is exactly what it was."""
import re

from backend import config
from backend.actions import ACTION_DESCRIPTIONS, COMPACT_DESCRIPTIONS, EXPEDITION_RULES, action_text
from backend.prompts import compile_live_state_prompt, get_prime_consciousness_prompt

NUMBER_WORDS = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "second", "third", "half", "twice", "double", "triple")


def _numbers(text):
    return set(re.findall(r"\d+", text))


def _number_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w in NUMBER_WORDS}


def _action_names(text):
    return set(re.findall(r"\b[A-Z]{3,}(?:_[A-Z]+)+\b|\b(?:RAID|TRADE|SCOUT|RELOCATE|BREED|ALLIANCE)\b", text))


def test_every_compact_description_is_for_a_real_action_and_is_shorter():
    for name, text in COMPACT_DESCRIPTIONS.items():
        assert name in ACTION_DESCRIPTIONS, name
        assert len(text) < len(ACTION_DESCRIPTIONS[name]), name


def _without_needs(full):
    """A building on the menu already has what it needs, and one that does not is explained by simulation.locked_building_facts, so the compact text leaves the
    "Needs ...;" clause of an era-chain building out."""
    return re.sub(r"Needs [^;]*;", "", full)


def test_compact_keeps_every_number_number_word_and_named_action_of_the_full_text():
    for name, compact in COMPACT_DESCRIPTIONS.items():
        full = _without_needs(ACTION_DESCRIPTIONS[name])
        said = compact + " " + (EXPEDITION_RULES if name in ("SCOUT", "EXPLORATION_PARTY", "HUNTING_PARTY") else "")
        assert _numbers(full) <= _numbers(said), (name, _numbers(full) - _numbers(said))
        assert _number_words(full) <= _number_words(said), (name, _number_words(full) - _number_words(said))
        assert _action_names(full) <= _action_names(said), (name, _action_names(full) - _action_names(said))
        for phrase in ("one-time", "one child", "one house at a time", "1-in-3"):
            assert phrase not in full.lower() or phrase in said.lower(), (name, phrase)


def _glossary(actions):
    return get_prime_consciousness_prompt("T", "m", available_actions=actions)


def test_the_expedition_rules_are_printed_once_when_an_expedition_action_is_on_the_menu(monkeypatch):
    monkeypatch.setenv("PROMPT_FORMAT", "compact")
    assert _glossary(["SCOUT", "GATHER_WOOD"]).count(EXPEDITION_RULES) == 1
    assert EXPEDITION_RULES not in _glossary(["GATHER_WOOD"])
    monkeypatch.delenv("PROMPT_FORMAT")
    assert EXPEDITION_RULES not in _glossary(["SCOUT"])


def test_the_default_format_is_the_full_text_and_compact_is_chosen_by_the_switch(monkeypatch):
    monkeypatch.delenv("PROMPT_FORMAT", raising=False)
    assert action_text("RAID") == ACTION_DESCRIPTIONS["RAID"]
    monkeypatch.setenv("PROMPT_FORMAT", "compact")
    assert action_text("RAID") == COMPACT_DESCRIPTIONS["RAID"]
    assert action_text("BUILD_KEEP") == COMPACT_DESCRIPTIONS["BUILD_KEEP"] != ACTION_DESCRIPTIONS["BUILD_KEEP"]
    assert action_text("BUILD_FIRE") == ACTION_DESCRIPTIONS["BUILD_FIRE"]  # an action with no compact text keeps its full text
    assert action_text("NOT_AN_ACTION") is None


def _state(available):
    return {"cycle": 5, "x": 10, "y": 12, "biome": "plains", "era": "primitive_dawn", "population": 20, "wood": 1, "stone": 2, "food": 3, "water": 4,
            "visible_entities": ["none"], "available_actions": available, "journey_note": ""}


def test_compact_boilerplate_keeps_the_same_facts_and_the_full_text_is_unchanged(monkeypatch):
    monkeypatch.delenv("PROMPT_FORMAT", raising=False)
    full = compile_live_state_prompt("base", _state(["RELOCATE", "GATHER_WOOD"]), "", "")
    assert "MOVEMENT: Only RELOCATE moves your tribe -- every other action (gathering, hunting,\nbuilding, idling) happens wherever you currently stand" in full
    assert "means nothing." in full
    monkeypatch.setenv("PROMPT_FORMAT", "compact")
    compact = compile_live_state_prompt("base", _state(["RELOCATE", "GATHER_WOOD"]), "", "")
    assert len(compact) < len(full)
    for fact in ("Only RELOCATE moves your tribe", "SCOUT sends a party out", "target_vector", "reports back what it found",
                 "RELOCATE moves the whole tribe up to several tiles per cycle"):
        assert fact in compact, fact
    for action in ("RELOCATE", "RAID", "TRADE", "DECLARE_ALLIANCE", "DECLARE_WAR", "SEND_TRADE_EMISSARY", "SPY", "STRIKE_RAIDER_CAMP", "DECLARE_CONQUEST"):
        assert action in compact.split("target_vector only matters for")[1].split("\n")[0], action


def test_grouping_the_entity_list_keeps_every_coordinate_and_name_and_leaves_other_lines_alone():
    from backend.simulation import compact_entities

    items = ["structure:fire@(1,2)", "confirmed water source at (70,47)", "confirmed water source at (58,49)",
             "Deadfall Ridge at (66,47) is a known danger -- a treacherous crossing.", "The Last Warning at (60,47) is a known danger -- a deep ford.",
             "a vein of Whisperwood Amber was found at (73,22)", "a Rabbit Warren was found at (80,17)", "a Wolf Den was found at (47,36)",
             "Aiming target_vector exactly at (16,37) would strike that known raider camp right now.", "memory(cycle 3): At (70,47) in river, chose RELOCATE.",
             "raiders reported near (28,73)", "confirmed lumber-rich area at (76,13)", "confirmed stone-rich area at (72,17)", "Raiders attacked (raid) shouting 'GRAH-OOK' (cycle 36)"]
    grouped = compact_entities(items)

    assert len(grouped) < len(items)
    assert all(c in " ".join(grouped) for c in re.findall(r"\(\d+,\d+\)", " ".join(items)))
    assert all(name in " ".join(grouped) for name in ("Deadfall Ridge", "The Last Warning", "Whisperwood Amber", "Rabbit Warren", "Wolf Den", "a treacherous crossing", "a deep ford"))
    assert grouped[0] == "structure:fire@(1,2)" and "memory(cycle 3): At (70,47) in river, chose RELOCATE." in grouped
    assert "Raiders attacked (raid) shouting 'GRAH-OOK' (cycle 36)" in grouped
    assert "Water sources: (70,47), (58,49)" in grouped  # a group sits where its first member was
    assert [g for g in grouped if g.startswith("Known dangers")] == ["Known dangers (a treacherous crossing): Deadfall Ridge (66,47)", "Known dangers (a deep ford): The Last Warning (60,47)"]
