from backend import config
from backend.might import compute_might
from backend.simulation import Tribe


def _tribe(**overrides) -> Tribe:
    tribe = Tribe("tribe_0", "Forest Tribe", "gemma2:2b", 50, 50, "#c084fc")
    for key, value in overrides.items():
        setattr(tribe, key, value)
    return tribe


def test_might_is_zero_without_a_battalion():
    assert compute_might(_tribe(battalion_size=0)) == 0


def test_might_scales_with_battalion_size_alone_as_a_bare_baseline():
    """No weapons, no defensive tier, no Warrior trophies, no readiness, and
    wellbeing defaults to a neutral 0.5 (the tribe.wellbeing = {} default,
    never computed yet) -- every multiplier is exactly 1, so Might equals
    battalion_size outright."""
    tribe = _tribe(battalion_size=20)
    assert compute_might(tribe) == 20


def test_might_rises_with_forge_weapons():
    unarmed = compute_might(_tribe(battalion_size=20))
    armed = compute_might(_tribe(battalion_size=20, items=[{"type": "weapon", "value": 12}] * 20))
    assert armed > unarmed


def test_might_weapon_bonus_is_fractional_not_all_or_nothing():
    """"Counted, not consumed" -- a partially-armed Battalion (half its
    soldiers have a Forge weapon between them) should land between the fully
    unarmed and fully armed cases, not jump straight to the max bonus."""
    half_armed = compute_might(_tribe(battalion_size=20, items=[{"type": "weapon", "value": 12}] * 10))
    unarmed = compute_might(_tribe(battalion_size=20))
    fully_armed = compute_might(_tribe(battalion_size=20, items=[{"type": "weapon", "value": 12}] * 20))
    assert unarmed < half_armed < fully_armed


def test_might_ignores_non_weapon_items():
    tools_only = compute_might(_tribe(battalion_size=20, items=[{"type": "tool", "value": 8}] * 20))
    assert tools_only == compute_might(_tribe(battalion_size=20))


def test_might_rises_with_defensive_tier():
    """The tier idea confirmed worth integrating -- Keep < Fortress < Castle."""
    base = compute_might(_tribe(battalion_size=20))
    keep = compute_might(_tribe(battalion_size=20, keep_built=True))
    fortress = compute_might(_tribe(battalion_size=20, keep_built=True, fortress_built=True))
    castle = compute_might(_tribe(battalion_size=20, keep_built=True, fortress_built=True, castle_built=True))
    assert base < keep < fortress < castle


def test_might_rises_with_the_warriors_own_trophies():
    """Trophy count confirmed worth adding -- but only the Warrior's own
    personally-credited trophies, not the whole tribe's trophy shelf (see
    actions._eligible_warrior_candidate's own reasoning for excluding the
    Chief and crediting individuals)."""
    no_warrior_trophies = compute_might(_tribe(
        battalion_size=20, warrior_name="Ash",
        trophies=[{"name": "First Hunt", "chief": "Someone Else", "cycle": 1}],
    ))
    with_warrior_trophies = compute_might(_tribe(
        battalion_size=20, warrior_name="Ash",
        trophies=[{"name": "First Hunt", "chief": "Ash", "cycle": 1}],
    ))
    assert with_warrior_trophies > no_warrior_trophies


def test_might_rises_with_wellbeing():
    """Well-Being confirmed worth including "some" -- deliberately the
    smallest-magnitude term (config.MIGHT_WELLBEING_WEIGHT)."""
    thriving = compute_might(_tribe(battalion_size=20, wellbeing={"tiers": {
        "physiological": 1.0, "safety": 1.0, "belonging": 1.0, "esteem": 1.0, "self_actualization": 1.0,
    }}))
    struggling = compute_might(_tribe(battalion_size=20, wellbeing={"tiers": {
        "physiological": 0.0, "safety": 0.0, "belonging": 0.0, "esteem": 0.0, "self_actualization": 0.0,
    }}))
    assert thriving > struggling


def test_might_rises_with_training_readiness():
    """Explicit request: Might's Training factor, "not overpowered, more
    like bolster and upkeep" -- tribe.battalion_readiness."""
    rested = compute_might(_tribe(battalion_size=20, battalion_readiness=1.0))
    unready = compute_might(_tribe(battalion_size=20, battalion_readiness=0.0))
    assert rested > unready


def test_might_training_bonus_is_not_overpowered():
    """Explicit constraint: readiness alone shouldn't swing Might anywhere
    near as hard as full weapons -- config.MIGHT_TRAINING_BONUS stays under
    config.MIGHT_WEAPON_BONUS."""
    assert config.MIGHT_TRAINING_BONUS < config.MIGHT_WEAPON_BONUS
