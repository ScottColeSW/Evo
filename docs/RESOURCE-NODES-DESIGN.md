# Resource nodes: finite uses, shared, respawning elsewhere (design, 2026-10-06)

Status: design, not built. Decisions marked **settled** were made by the project owner; the rest are proposals waiting on an answer.

## Why

The first live run with the new logging (`run_20261005_154817`, `NUDGES=off`) kept both tribes in the first era for 187 and 211 cycles. Both settled by cycle 16
and then sat near 20 people with food at 1 to 20. The mechanism, read from the code and the log:

- Instant gathers and hunts harvest at the tribe's own tile, which loses 0.15 of its yield per harvest (cap 0.8) and recovers 0.02 a cycle. A settled tribe
  works one tile, so it sits at the cap and gets 20% of the nominal yield: a 16-food forage becomes 3, a hunt 3 to 5 (measured median +2 and +4).
- Upkeep at 20 people is 2 food a cycle, and population growth needs a food stock above 25 (`POPULATION_GROWTH_FOOD_THRESHOLD`). Food never accumulates, so growth stops.
- Yield scales with `sqrt(population / 8)`, so a tribe stuck near 20 stays at about 1.6 times the starting yield.

## What exists today

- Sites are fixed, pre-seeded world points (`world.site_seed_points`). A scout discovers one by passing within 8 tiles (`_discover_sites_along_route`) and it
  goes into that tribe's own list: `lumber_sites`, `wildlife_sites`, `quarry_sites`, `mine_sites`. Sites inside any tribe's territory are not discoverable.
- Only hunting grounds ever exhaust: a hunting party's catch landing on a known ground removes it and places a replacement (`_relocate_wildlife_site_after_clearing`,
  a copy of `_relocate_raider_sighting_after_ambush`: an angle, a distance between `RAIDER_SIGHTING_MIN_OFFSET` and `RAIDER_SIGHTING_OFFSET`, 20 tries on buildable ground).
  Lumber, quarry and mine sites never exhaust.
- **The leak:** both of those relocations append the new spot straight into the tribe's own list, so the tribe knows it without anyone finding it.
- Instant gathers do not use sites at all. A built sawmill, quarry, mine, tannery or deer pen records its own site.
- **There is no proximity conflict between tribes in the field.** Parties overhear a rival's broadcasts and spot its settlement from a distance. Fights exist
  only with raiders (ambushes), home raids (`RAID`) and conquest. Nothing contests a resource.

## Settled

1. **Nodes:** wood from lumber sites, stone from quarry sites, game from wildlife sites. Forage, water and fishing stay on the tile.
2. **3 uses per node**, then it is exhausted and respawns elsewhere.
3. **Use counts are shared** (world level), so tribes compete for the same node.
4. **Respawn placement** reuses the raider and wildlife logic (angle, offset, 20 tries, buildable ground).
5. **An empty list means scouting:** the respawned node is undiscovered. It is not added to anyone's list.
6. The tribe that exhausts a node gets a natural report; its entry is dropped. Another tribe that still lists it learns when a party arrives or targets it and finds
   it gone. No silent stale entries, and nothing refreshes without a report.
7. Structures already built on a node (a sawmill, a tannery) keep working. Exhaustion affects hand-gathering, not buildings.

## What the numbers say (the part that changes the plan)

Known sites by cycle in that run (wildlife, lumber, quarry, mine):

| | c20 to c150 | c200 | c300 | c500 |
|---|---|---|---|---|
| Tribe 1 | 0, 0, 0, 0 | 1, 0, 1, 1 | 2, 5, 2, 5 | 3, 6, 4, 6 |
| Tribe 2 | 0, 0, 0, 0 | 2, 2, 2, 2 | 2, 2, 2, 2 | 3, 6, 3, 6 |

Both tribes scouted 21 to 23 times in their first 200 cycles and knew **no node of any kind for the first 150 cycles.** So:

- **A tile fallback is required.** If gathering needed a known node, nothing could be gathered for 150 cycles. With no known node in reach, a gather uses the tribe's own
  tile as it does today. (Proposed.)
- **Nodes alone do not fix the first-era trap.** They are discovered at about the time the tribes leave the first era anyway, and forage (the weak food source) stays on the
  tile. They improve the middle and late game.
- **Proposed: homeland nodes.** At founding, a tribe knows one node of each type (wildlife, lumber, quarry) near home, placed with the same rule that keeps wild sites out of
  territory, just outside the walls or inside the first radius, so the first 100 cycles are not on a depleted tile. They use and respawn like any other node. This is the part
  that would actually address the trap. (Needs a decision.)

## Mechanics (proposed)

- **Registry:** `world.nodes[(type, x, y)] = {"uses_left": 3}`, created lazily from `site_seed_points`; exhausted points are tombstoned (removed from the seed set for good) and a
  respawn point is chosen by the existing offset routine. Respawn points are real nodes in the world, undiscovered by every tribe.
- **A gather** picks the nearest known live node of its type within reach (proposed: territory radius plus `NODE_REACH`, about 12 tiles), yields `base * biome_of_node * labor * item bonus`
  with no tile depletion, spends one use, and says which node it came from. No node in reach: the tile fallback, unchanged.
- **Exhaustion:** the third use sets `uses_left` to 0, tombstones the node, spawns the respawn, removes the node from the using tribe's list, and writes the chronicle line
  ("the grove at (x,y) gives out for good").
- **Other tribes' lists:** an entry whose node is tombstoned is dropped when a party arrives (`_advance_one_expedition` arrival, hunting-party targeting) and when a menu or build check
  reads the list. A stale entry can cost a trip, not corrupt anything.
- **The party model** (3 people carrying 5 each, more parties as the tribe grows, +20% led by a named warrior, hunts that can return more than one deer) is a separate, later layer on top
  of nodes: nodes decide where, parties decide how much. Not part of the first build.

## Contest in the field (to be built; it does not exist)

Proposed first version, small and using what exists: a node remembers the last tribe to use it and the cycle. If a *different, non-allied* tribe uses it within 2 cycles, a skirmish is
resolved with the existing might comparison (`backend/might.py`); the loser takes nothing from that use and loses a few people (`_record_combat` records it, the usual encounter marker
is drawn), the winner takes the yield. The use is spent either way. Allies share. This makes a shared node worth contesting without adding a new combat system.

## Open decisions

1. Tile fallback when no known node is in reach (recommended, to avoid a 150-cycle soft lock).
2. Homeland nodes at founding (recommended, the only part aimed at the first-era trap), or leave the first era to the tile.
3. The contest rule above, or something else.
4. `NODE_REACH` and whether the tribe must be settled to use nodes.

## How to check it once built

Replay the first 200 cycles of `run_20261005_154817` and the early A/B runs under the new rules (yield per action, food at cycle 60 and 120, cycle of first era exit), and run the early
A/B harness with a `--knob nodes`. The nudge A/B and the menu-size A/B show the run-to-run spread to compare against: first era exit moved by about 27 cycles between identical setups.
