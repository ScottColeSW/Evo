"""Test helper: make a tribe ready for an era (2026-10-04, when era gates became readiness, see eras.Era.requires_ready)."""
from backend.eras import ERAS, era_index


def make_ready_for(tribe, era_key):
    """Sets every requires_ready entry of `era_key` on `tribe` so only the population line and the cycle floor remain.
    The three *_secure names are set through the real facts that define them (simulation._is_food_secure and its siblings)."""
    era = ERAS[era_index(era_key)]
    for name in era.requires_ready:
        if name == "food_secure":
            tribe.kitchen_built = True
            tribe.fishing_learned = True
        elif name == "water_secure":
            tribe.well_built = True
        elif name == "wood_secure":
            tribe.sawmill_built = True
            tribe.lumber_site = (1, 1)
        else:
            setattr(tribe, name, True if isinstance(getattr(tribe, name, 0), bool) else 1)


def make_ready_for_every_era(tribe):
    for era in ERAS:
        make_ready_for(tribe, era.key)
