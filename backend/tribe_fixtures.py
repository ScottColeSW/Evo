"""Loads a real, advanced tribe state from logs/board_history.db and applies it to a
fresh Tribe, so a benchmark trial can start well past the early-game grind instead of
re-earning the same 5,000 population from scratch every run.

Explicit request, 2026-09-16: "we could start at 800 even" / "we are going to want
to start a war starting from around [population] 5000" -- moved from 600 once real
data showed no run has ever built a Barracks/Battalion below the low thousands (see
the grounding search that produced FIXTURE_SOURCE below); the fixture is real, not
guessed.

A fixture is a *curated* snapshot, not a live save-state -- deliberately narrower
than Simulation.snapshot()'s own frontend-display shape:

- Identity (name/model/color) always comes from the scenario's own tribe_configs,
  never the fixture, so the same fixture can be replayed under any model.
- x/y ARE copied, unlike identity -- buildings/territory/wall_rings/mine_sites below
  are all coordinate-tied to wherever the source tribe actually stood, and the map
  itself is a deterministic function of (x, y) (backend.world.biome_at takes no
  seed), so reusing the same coordinates keeps everything self-consistent without
  needing to regenerate or shift any of it.
- Relationship state (discovered_rivals, stance_toward, rival_intel, combat_record,
  trade_given/received) is deliberately NOT copied -- a fixture is about a tribe's
  own earned capability, not a replay of one specific run's diplomatic history. Two
  fixture-started tribes begin as strangers to each other, same as any fresh game,
  regardless of how the source run's relationship played out.
- Transient/derived fields (wellbeing, last_confusion, pending_hatch, chief_*, a
  live TribeHistory/TribeMemory object, throttled_actions' lost cooldown values)
  are left at Tribe.__init__'s own defaults -- none of these are recoverable from a
  board_history snapshot anyway (Tribe.to_dict() never serializes some of them, and
  truncates history to the last 6 entries), and starting them fresh is the
  semantically correct choice for "test starting from an advanced position," not a
  fidelity gap.
- A handful of *_cycle fields ARE copied, but shifted by (new_cycle - source_cycle)
  so "how long ago" cooldown math stays meaningful relative to whenever the new
  trial actually starts, rather than referencing a cycle number from a different
  timeline.
"""

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# Straightforward pass-through fields: the fixture's value is set onto the tribe
# unchanged. Deliberately excludes identity (name/model/color -- always from the
# scenario's own tribe_configs), every relationship/diplomatic field
# (discovered_rivals/stance_toward/rival_intel/combat_record/trade_given/
# trade_received/conquered_tribe_names), chief_* (a fresh trial gets a real
# election via the normal Simulation.create -> _install_chief path), and anything
# Tribe.to_dict() either never serializes (pending_hatch, last_confusion, wellbeing)
# or serializes lossily (throttled_actions loses its cooldown values; history is
# truncated to the last 6 entries) -- see this module's own docstring.
FIXTURE_FIELDS = (
    "x", "y", "wood", "stone", "food", "water", "population", "era",
    "cycles_since_relocate", "farm_plots", "crop_growth",
    "fishing_learned", "cooking_learned", "hunt_ever_succeeded", "foraged_ever_succeeded",
    "wood_ever_gathered", "stone_ever_gathered", "ore_ever_gathered", "fire_ever_built",
    "moat_built", "long_houses_built", "long_house_upgrades",
    "keep_built", "fortress_built", "castle_built",
    "barracks_built", "barracks_upgrades", "battalion_size", "battalion_readiness", "battalion_patrol",
    "road_built", "toll_roads_completed", "dock_built", "sawmill_built", "quarry_built",
    "mine_sites", "mine_built", "mine_resource_name", "unique_resources",
    "scout_rotation_index", "landmarks", "hazard_landmarks",
    "kitchen_built", "tannery_built", "deer_pen_built", "deer", "forge_built", "items",
    "dmm_built", "created_objects", "conquests_won", "trades_completed",
    "spy_missions_run", "spy_missions_caught",
    "warehouses_built", "warehouse_upgrades", "foraging_retired", "watering_retired",
    "flock", "eggs", "eggs_laid_total",
    "hatchery_built", "coop_built", "boat_built", "bath_house_built",
    "library_built", "library_entries", "research_completed", "well_built",
    "flock_lineage", "settlement_name", "has_ever_settled",
    "territory_center", "territory_radius", "wall_rings", "wall_commitment_active", "buildings",
    "fishery_built", "chiefs_elected", "chief_deaths",
    "battalions", "trophies", "fame", "lineage", "custom_awards",
    "wildlife_sites", "raiders_repelled_by_wall",
)

# visited_sectors/discovered_rivals are real `set`s on Tribe but to_dict() renders
# them as `list(...)` for JSON -- convert back on the way in. discovered_rivals is
# excluded (see FIXTURE_FIELDS' own comment) but visited_sectors is real, earned
# map knowledge tied to nothing relationship-specific, safe to keep.
FIXTURE_SET_FIELDS = ("visited_sectors",)

# Real Tribe attribute is list[tuple[int, int]] (see Tribe.__init__), but a plain
# tuple has no JSON equivalent -- board_history's own snapshot round-trip already
# turns each entry into a plain [x, y] list. Confirmed live, 2026-09-16: leaving
# these as lists of lists crashed the very first _discover_sites_along_route call
# that tried set(tribe.lumber_sites) on one (TypeError: unhashable type: 'list').
# wildlife_sites/mine_sites are excluded -- those are list[dict] ({"x":.., "y":..}),
# which round-trips through JSON correctly with no conversion needed.
FIXTURE_LIST_OF_TUPLES_FIELDS = ("confirmed_water_sites", "lumber_sites", "quarry_sites", "raider_sightings")

# Cycle-relative fields: shifted by (new_cycle - source_cycle) so "how long ago"
# stays meaningful under a different starting cycle, rather than a raw number from
# a different timeline (which could even go negative and break a `> 0`-as-boolean
# check like _is_food_secure's last_harvest_cycle read).
FIXTURE_CYCLE_SHIFT_FIELDS = (
    "last_harvest_cycle", "last_reflection_cycle", "last_celebration_cycle", "last_raider_attack_cycle",
)


def load_fixture(name: str) -> dict:
    """Reads backend/fixtures/<name>.json -- {"run_id", "cycle", "tribe": {...}}."""
    path = FIXTURE_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_tribe_fixture(tribe, fixture: dict, new_cycle: int) -> None:
    """Applies one tribe's worth of a loaded fixture (fixture["tribe"]) onto a real
    Tribe object. `new_cycle` is the cycle the new trial's clock actually starts
    at -- used only to shift FIXTURE_CYCLE_SHIFT_FIELDS, never to set tribe.era or
    any Simulation-level cycle counter (the caller owns that)."""
    source = fixture["tribe"]
    offset = new_cycle - fixture["cycle"]
    for field in FIXTURE_FIELDS:
        if field in source:
            setattr(tribe, field, source[field])
    for field in FIXTURE_SET_FIELDS:
        if field in source:
            # visited_sectors is a set[tuple[int, int]] on the real Tribe, but
            # JSON has no tuple type -- to_dict()/json.dump round-trip each
            # coordinate pair as a plain [x, y] list, which isn't hashable.
            setattr(tribe, field, {tuple(item) if isinstance(item, list) else item for item in source[field]})
    for field in FIXTURE_LIST_OF_TUPLES_FIELDS:
        if field in source:
            setattr(tribe, field, [tuple(item) if isinstance(item, list) else item for item in source[field]])
    for field in FIXTURE_CYCLE_SHIFT_FIELDS:
        if field in source and source[field]:
            setattr(tribe, field, source[field] + offset)
