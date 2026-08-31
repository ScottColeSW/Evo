"""Diagnostic: is RELOCATE specifically a crisis-response pick (spiking when starving/
dehydrated) rather than a generally-preferred action -- i.e. does a model reach for
"move the whole camp" as its go-to reaction to hunger/thirst, the same way we already
found GATHER_WOOD dominates in general and HUNT_DEER barely gets chosen at all?

Records, for every real turn, the action chosen and whether food or water was at or
below its CRITICAL threshold *that same cycle* (instincts.HUNGER_CRITICAL_THRESHOLD /
THIRST_CRITICAL_THRESHOLD), then reports P(action | critical) vs P(action | not
critical) for RELOCATE and SCOUT specifically. Single-tribe headless runs, real Ollama
calls. Throwaway analysis script.
"""
import asyncio
import collections
import sys

sys.path.insert(0, ".")

from backend import config
from backend.simulation import Simulation

MODEL = "gemma2:2b"
CYCLES_PER_RUN = 100
N_RUNS = 3


async def run_once(run_id: int) -> list[dict]:
    sim = await Simulation.create([{"name": "Test Tribe", "model": MODEL}])
    tribe = next(iter(sim.tribes.values()))
    records = []
    for _ in range(CYCLES_PER_RUN):
        critical = (
            tribe.food <= config.HUNGER_CRITICAL_THRESHOLD
            or tribe.water <= config.THIRST_CRITICAL_THRESHOLD
        )
        await sim.step()
        action = tribe.last_action
        records.append({"run": run_id, "critical_before_turn": critical, "action": action})
        if tribe.extinct:
            break
    return records


async def main():
    all_records = []
    for i in range(N_RUNS):
        print(f"=== RUN {i} ===", flush=True)
        records = await run_once(i)
        all_records.extend(records)
        counts = collections.Counter((r["critical_before_turn"], r["action"]) for r in records)
        print(f"  {len(records)} turns recorded", flush=True)

    by_critical = collections.defaultdict(collections.Counter)
    for r in all_records:
        by_critical[r["critical_before_turn"]][r["action"]] += 1

    print("\n=== SUMMARY ===")
    for critical_state in (True, False):
        counter = by_critical[critical_state]
        total = sum(counter.values())
        label = "CRITICAL (hungry/thirsty)" if critical_state else "not critical"
        print(f"\n{label}: {total} turns")
        for action, count in counter.most_common():
            print(f"  {action}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
