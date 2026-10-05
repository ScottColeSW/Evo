"""2026-10-04: crafted tools and innovations now do something (they were read nowhere; only weapons fed battalion strength), and an
action listed in config.DISABLED_ACTIONS is left out of every menu while its handler and era entry stay in the code."""
from backend import config
from backend.actions import ACTION_REGISTRY, _expedition_speed_bonus, _food_multiplier
from backend.eras import ERAS, ordered_actions_through, unlocked_actions_through
from backend.simulation import Tribe

_NO_TARGET = (0, 0)


def _tribe():
    return Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")


def _item(kind):
    return {"name": f"a {kind}", "type": kind, "value": config.ITEM_VALUE_BY_TYPE[kind], "cycle_made": 1}


def test_each_tool_on_hand_adds_to_the_food_multiplier_and_other_items_do_not():
    tribe = _tribe()
    baseline = _food_multiplier(tribe)
    tribe.items = [_item("tool"), _item("tool"), _item("weapon"), _item("innovation")]

    assert _food_multiplier(tribe) == baseline * (1 + 2 * config.ITEM_TOOL_GATHER_BONUS)


def test_tools_stack_with_a_dmm_gather_boost():
    tribe = _tribe()
    tribe.created_objects = [{"name": "a", "category": "gather_boost", "kind": "item"}]
    tribe.items = [_item("tool")]
    baseline = _food_multiplier(_tribe())

    assert _food_multiplier(tribe) == baseline * (1 + config.CREATED_OBJECT_MAGNITUDE + config.ITEM_TOOL_GATHER_BONUS)


def test_each_innovation_on_hand_adds_expedition_speed_and_stacks_with_a_dmm_boost():
    tribe = _tribe()
    assert _expedition_speed_bonus(tribe) == 0
    tribe.items = [_item("innovation"), _item("innovation"), _item("tool"), _item("weapon")]
    assert _expedition_speed_bonus(tribe) == 2 * config.ITEM_INNOVATION_EXPEDITION_SPEED_BONUS
    tribe.created_objects = [{"name": "a", "category": "expedition_boost", "kind": "item"}]
    assert _expedition_speed_bonus(tribe) == (
        config.CREATED_OBJECT_EXPEDITION_SPEED_BONUS + 2 * config.ITEM_INNOVATION_EXPEDITION_SPEED_BONUS)


def test_redeeming_an_item_gives_its_bonus_up():
    from tests.test_actions import _bare_simulation

    sim, tribe = _bare_simulation(), _tribe()
    tribe.items = [_item("tool")]
    with_tool = _food_multiplier(tribe)

    ACTION_REGISTRY["USE_ITEM"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.items == [] and _food_multiplier(tribe) < with_tool


def test_a_disabled_action_is_in_the_era_table_but_in_no_menu():
    assert "CREATE_USEFUL_STRUCTURE" in config.DISABLED_ACTIONS
    assert any("CREATE_USEFUL_STRUCTURE" in era.unlocks_actions for era in ERAS)  # kept for the record
    assert "CREATE_USEFUL_STRUCTURE" in ACTION_REGISTRY                            # the handler stays
    last = ERAS[-1].key
    assert "CREATE_USEFUL_STRUCTURE" not in unlocked_actions_through(last)
    assert "CREATE_USEFUL_STRUCTURE" not in ordered_actions_through(last)
    assert "CREATE_ITEM" in unlocked_actions_through(last)                         # its twin is untouched
