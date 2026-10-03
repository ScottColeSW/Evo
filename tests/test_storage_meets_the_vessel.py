"""The vessel (config.VESSEL_WOOD_COST / VESSEL_STONE_COST) must be storable with only the Warehouses a tribe can build, without
needing a Warehouse upgrade: found 2026-10-03 when five Warehouses held 2,300 against a 2,500 cost."""
from backend import config
from backend.actions import _storage_cap
from backend.simulation import Simulation


def test_the_maximum_number_of_warehouses_holds_a_vessel_without_any_upgrade():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.warehouses_built, tribe.warehouse_upgrades = config.WAREHOUSE_MAX_COUNT, 0
    assert _storage_cap(tribe) >= max(config.VESSEL_WOOD_COST, config.VESSEL_STONE_COST)
