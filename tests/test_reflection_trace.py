"""Phase 0 of docs/PALIMPSEST-REFLECTIONS-DESIGN.md: the reflection trace is read-only. It records what the decision was and how
similar the nearest held reflection was, without changing what remember()/remember_reflection() decide or return."""
import json

from backend import event_log
from backend.memory import TribeMemory


def test_token_path_trace_for_a_new_then_a_reinforced_reflection():
    m = TribeMemory("t")
    first = m.remember("We should trust the river clan and share our grain with them.", 1, 0.5, kind="reflection")
    assert m.last_reflection_trace["reinforced"] is False and m.last_reflection_trace["method"] == "token"
    assert m.last_reflection_trace["similarity"] is None            # nothing held to compare with yet
    second = m.remember("We should never trust the river clan and must hide our grain from them.", 2, 0.5, kind="reflection")
    assert second is first                                          # the existing behavior is unchanged
    trace = m.last_reflection_trace
    assert trace["reinforced"] is True and trace["reinforced_count"] == 1
    assert trace["similarity"] >= m.REFLECTION_REINFORCEMENT_OVERLAP_THRESHOLD
    assert trace["nearest"].startswith("We should trust the river clan")


def test_embedding_path_trace_records_the_nearest_similarity_even_when_not_reinforced():
    m = TribeMemory("t")
    m.remember_reflection("The north pass floods in spring.", 1, 0.5, [1.0, 0.0])
    other = m.remember_reflection("The harvest was large.", 2, 0.5, [0.0, 1.0])
    trace = m.last_reflection_trace
    assert trace["method"] == "embedding" and trace["reinforced"] is False
    assert trace["similarity"] == 0.0 and trace["nearest"] == "The north pass floods in spring."
    assert len(m.entries) == 2 and other is m.entries[1]
    again = m.remember_reflection("The north pass floods every spring.", 3, 0.5, [1.0, 0.01])
    assert again is m.entries[0] and m.last_reflection_trace["reinforced"] is True


def test_record_data_keeps_the_message_key_so_existing_readers_still_work(tmp_path):
    log = event_log.RunEventLog(str(tmp_path))
    log.record_data("Tribe 1", "reflection_memory", {"reinforced": True}, message="[reflection memory] reinforced")
    log.record("Tribe 1", "an ordinary line")
    lines = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert all("message" in line and "cycle" in line and "tribe" in line for line in lines)
    assert lines[0]["kind"] == "reflection_memory" and lines[0]["data"] == {"reinforced": True}
    assert "kind" not in lines[1]


from unittest import mock  # noqa: E402

from backend.simulation import Simulation  # noqa: E402
from tests.conftest import run_async  # noqa: E402


@run_async
async def test_night_cycle_writes_the_reflection_decision_to_the_run_log():
    """The night cycle logs each reflection's reinforcement decision (and does not change it): a reversal still reinforces
    today, which is what phase 0 exists to record."""
    sim = Simulation([{"name": "Forest Tribe", "model": "gemma2:2b"}])
    tribe = sim.tribes["tribe_0"]
    tribe.chief_name = "Ashgar"
    thoughts = iter(["We should trust the river clan and share our grain with them.",
                     "We should never trust the river clan and must hide our grain from them."])

    async def fake_reflect(*args, **kwargs):
        return {"private_thoughts": next(thoughts), "revised_philosophy": args[3], "changed": False, "reasoning": "ok"}

    with mock.patch("backend.simulation.reflect_on_history", fake_reflect), \
         mock.patch.object(sim.client, "embed", mock.AsyncMock(return_value=None)):
        await sim._run_night_cycle(tribe)
        await sim._run_night_cycle(tribe)

    records = [json.loads(line) for line in sim.event_log.path.read_text(encoding="utf-8").splitlines()]
    traces = [r for r in records if r.get("kind") == "reflection_memory"]
    assert [t["data"]["reinforced"] for t in traces] == [False, True]
    assert traces[1]["data"]["method"] == "token" and traces[1]["data"]["similarity"] >= 0.3
    assert len(tribe.memory.entries) == 1                       # behavior unchanged: the reversal was merged, as today
