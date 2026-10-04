"""2026-10-04: TRADE and the other actions that need a rival already found are not offered before contact."""
from backend.simulation import Simulation


def _actions(discovered):
    sim = Simulation([{"name": "A", "model": "gemma2:2b", "x": 40, "y": 37}, {"name": "B", "model": "qwen2.5:3b", "x": 60, "y": 60}])
    tribe = sim.tribes["tribe_0"]
    tribe.has_ever_settled = True
    sim._found_territory(tribe)
    tribe.barracks_built = 1
    tribe.era = "tribal_synapse"
    tribe.wood = tribe.stone = 1000
    if discovered:
        tribe.discovered_rivals.add("tribe_1")
    _, ctx = sim._prepare_turn(tribe)
    return ctx["available_actions"]


RIVAL_ACTIONS = ("TRADE", "SEND_TRADE_EMISSARY", "SPY", "DECLARE_ALLIANCE", "DECLARE_WAR")


def test_none_of_them_are_offered_before_a_rival_is_found():
    actions = _actions(discovered=False)
    assert not [a for a in RIVAL_ACTIONS if a in actions], [a for a in RIVAL_ACTIONS if a in actions]


def test_they_are_offered_once_a_rival_is_found():
    actions = _actions(discovered=True)
    assert "TRADE" in actions and "DECLARE_ALLIANCE" in actions


def test_trade_is_not_in_the_first_menu_of_a_new_tribe():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}, {"name": "B", "model": "qwen2.5:3b"}])
    _, ctx = sim._prepare_turn(sim.tribes["tribe_0"])
    assert "TRADE" not in ctx["available_actions"] and "SEND_TRADE_EMISSARY" not in ctx["available_actions"]
