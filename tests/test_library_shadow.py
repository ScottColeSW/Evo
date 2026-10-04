"""Phase 0 of docs/LIBRARY-PALIMPSEST-SPEC.md: RESEARCH is judged against a shadow shelf and logged; the real Library is unchanged."""
from backend import actions, config
from backend.simulation import Simulation
from tests.conftest import run_async


def _setup():
    sim = Simulation([{"name": "A", "model": "gemma2:2b"}])
    sim.cycle = 60
    tribe = sim.tribes["tribe_0"]
    tribe.library_built = True
    tribe.wood = 1000
    tribe.memory.remember("the river flooded the lower camp", 50, weight=0.9)
    tribe.memory.remember("deer return in the cold months", 51, weight=0.8)
    logged = []
    sim.event_log.record_data = lambda name, kind, data, **k: logged.append((kind, data))
    return sim, tribe, logged


def _judge(relations):
    seq = iter(relations)

    def judge(text, held):
        return {"relation": next(seq), "related_id": held[0]["id"] if held else None, "reason": "test"}

    async def build():
        return judge
    return build


def test_research_queues_what_it_filed_and_files_as_before():
    sim, tribe, _ = _setup()
    actions._research(sim, tribe, None, None)
    assert len(tribe.library_entries) == 1 and tribe.research_completed == 1
    assert tribe.library_shadow_pending and tribe.library_shadow_pending[0]["texts"]


@run_async
async def test_night_judges_against_the_shadow_shelf_and_logs():
    sim, tribe, logged = _setup()
    actions._research(sim, tribe, None, None)
    # RESEARCH files only what is not already on the shelf, so a second filing needs new memories
    tribe.memory.remember("raiders came from the north at dusk and the wall held", 55, weight=0.9)
    tribe.memory.remember("the storehouse burned during the dry month", 56, weight=0.8)
    actions._research(sim, tribe, None, None)
    sim._reflection_judge = _judge(["reinforces", "new", "reinforces", "reinforces"])
    entries_before = list(tribe.library_entries)
    await sim._library_shadow_night(tribe)
    shadow = [d for k, d in logged if k == "library_shadow"]
    assert len(shadow) == 2
    # the first candidate of the first filing meets an empty shelf and is new; a repeat is never added to the shelf
    assert shadow[0]["results"][0]["relation"] == "new"
    assert shadow[1]["repeats"] >= 1
    assert tribe.library_shadow_pending == []
    assert tribe.library_entries == entries_before


@run_async
async def test_night_does_nothing_without_a_judge():
    sim, tribe, logged = _setup()
    actions._research(sim, tribe, None, None)

    async def none():
        return None
    sim._reflection_judge = none
    await sim._library_shadow_night(tribe)
    assert not [1 for k, _ in logged if k == "library_shadow"]
    assert tribe.library_shadow_shelf == []
