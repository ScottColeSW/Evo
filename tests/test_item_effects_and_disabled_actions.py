"""2026-10-04: every forged item does what its name says (config.ITEM_EFFECT_BY_NAME), through the code that already computes that yield or speed;
and an action listed in config.DISABLED_ACTIONS is left out of every menu while its handler and era entry stay in the code."""
from unittest import mock

from backend import config
from backend.actions import ACTION_REGISTRY, _expedition_speed_bonus, _food_multiplier, _harvest, _item_effect_bonus
from backend.eras import ERAS, ordered_actions_through, unlocked_actions_through
from backend.simulation import Tribe

_NO_TARGET = (0, 0)


def _tribe():
    return Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")


def _item(name):
    kind = next(k for k, names in config.ITEM_NAMES_BY_TYPE.items() if name in names)
    return {"name": name, "type": kind, "value": config.ITEM_VALUE_BY_TYPE[kind], "cycle_made": 1}


def test_every_item_name_the_forge_can_make_has_a_known_effect_or_is_the_war_hammer():
    """A new item name added to config.ITEM_NAMES_BY_TYPE must be given an effect (or deliberately left to the weapon rule)."""
    names = [n for names in config.ITEM_NAMES_BY_TYPE.values() for n in names]
    unmapped = [n for n in names if n not in config.ITEM_EFFECT_BY_NAME]
    assert unmapped == ["War Hammer"]  # a plain weapon: it arms the battalion (might.py) and nothing else
    assert set(config.ITEM_EFFECT_BY_NAME.values()) <= set(config.ITEM_EFFECT_TEXT)


def test_food_items_add_to_the_food_multiplier_and_unrelated_items_do_not():
    tribe = _tribe()
    baseline = _food_multiplier(tribe)
    tribe.items = [_item("Iron Plow"), _item("Forged Hoe"), _item("Whetstone"), _item("War Hammer"), _item("Geared Wheel")]

    assert _food_multiplier(tribe) == baseline * (1 + 2 * config.ITEM_EFFECT_MAGNITUDE)


def test_food_items_stack_with_a_dmm_gather_boost():
    tribe = _tribe()
    tribe.created_objects = [{"name": "a", "category": "gather_boost", "kind": "item"}]
    tribe.items = [_item("Iron Plow")]
    baseline = _food_multiplier(_tribe())

    assert _food_multiplier(tribe) == baseline * (1 + config.CREATED_OBJECT_MAGNITUDE + config.ITEM_EFFECT_MAGNITUDE)


def test_harvests_of_wood_stone_water_and_game_follow_the_item_named_for_them():
    from tests.test_actions import _bare_simulation

    sim = _bare_simulation()
    cases = {"stone": "Whetstone", "wood": "Bronze Axe", "water": "Pressure Valve", "game": "Iron Spearhead"}
    for resource, name in cases.items():
        plain, equipped = _tribe(), _tribe()
        equipped.items = [_item(name)]
        with mock.patch.object(sim.world, "scarcity", return_value=0.0), mock.patch.object(sim.world, "deplete"):
            base = _harvest(sim, plain, resource, 100, "plains")
            boosted = _harvest(sim, equipped, resource, 100, "plains")
        assert boosted == round(base * (1 + config.ITEM_EFFECT_MAGNITUDE)), resource
    other = _tribe()
    other.items = [_item("Whetstone")]  # a stone item does not touch wood
    with mock.patch.object(sim.world, "scarcity", return_value=0.0), mock.patch.object(sim.world, "deplete"):
        assert _harvest(sim, other, "wood", 100, "plains") == _harvest(sim, _tribe(), "wood", 100, "plains")


def test_geared_wheel_adds_expedition_speed_and_stacks_with_a_dmm_boost():
    tribe = _tribe()
    assert _expedition_speed_bonus(tribe) == 0
    tribe.items = [_item("Geared Wheel"), _item("Geared Wheel"), _item("Iron Plow"), _item("Bronze Axe")]
    assert _expedition_speed_bonus(tribe) == 2 * config.ITEM_EXPEDITION_SPEED_PER_ITEM
    tribe.created_objects = [{"name": "a", "category": "expedition_boost", "kind": "item"}]
    assert _expedition_speed_bonus(tribe) == config.CREATED_OBJECT_EXPEDITION_SPEED_BONUS + 2 * config.ITEM_EXPEDITION_SPEED_PER_ITEM


def test_wall_and_training_items_speed_those_actions_up():
    tribe = _tribe()
    assert _item_effect_bonus(tribe, "wall_speed") == 0 and _item_effect_bonus(tribe, "training") == 0
    tribe.items = [_item("Tempered Spring"), _item("Balanced Hinge"), _item("Balanced Hinge")]
    assert _item_effect_bonus(tribe, "wall_speed") == config.ITEM_EFFECT_MAGNITUDE
    assert _item_effect_bonus(tribe, "training") == 2 * config.ITEM_EFFECT_MAGNITUDE


def test_the_forge_message_says_what_the_new_item_does():
    from tests.test_actions import _bare_simulation, _settle

    sim, tribe = _bare_simulation(), _tribe()
    _settle(sim, tribe)
    tribe.forge_built = True
    tribe.mine_resource_name = "Duneglass"
    tribe.unique_resources["Duneglass"] = 5
    tribe.wood = 100
    with mock.patch("backend.actions.random.choice", side_effect=["tool", "Iron Plow"]):
        message = ACTION_REGISTRY["FORGE_ITEM"](sim, tribe, "plains", _NO_TARGET)
    assert "Iron Plow" in message and config.ITEM_EFFECT_TEXT["food"] in message


def test_redeeming_an_item_gives_its_effect_up():
    from tests.test_actions import _bare_simulation

    sim, tribe = _bare_simulation(), _tribe()
    tribe.items = [_item("Iron Plow")]
    with_plow = _food_multiplier(tribe)

    ACTION_REGISTRY["USE_ITEM"](sim, tribe, "plains", _NO_TARGET)

    assert tribe.items == [] and _food_multiplier(tribe) < with_plow


def test_a_disabled_action_is_in_the_era_table_but_in_no_menu():
    assert "CREATE_USEFUL_STRUCTURE" in config.DISABLED_ACTIONS
    assert any("CREATE_USEFUL_STRUCTURE" in era.unlocks_actions for era in ERAS)  # kept for the record
    assert "CREATE_USEFUL_STRUCTURE" in ACTION_REGISTRY                            # the handler stays
    last = ERAS[-1].key
    assert "CREATE_USEFUL_STRUCTURE" not in unlocked_actions_through(last)
    assert "CREATE_USEFUL_STRUCTURE" not in ordered_actions_through(last)
    assert "CREATE_ITEM" in unlocked_actions_through(last)                         # its twin is untouched
