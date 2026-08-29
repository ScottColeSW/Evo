from backend.vram_guard import HardwareVRAMBoundaryGuard
from tests.conftest import run_async


@run_async
async def test_model_within_budget_passes():
    guard = HardwareVRAMBoundaryGuard("http://localhost:11434", vram_limit_gb=14.0)
    guard._model_size_bytes = lambda model: _resolved(2 * (1024**3))

    ok, warning = await guard.verify_vram_safety_margin("small-model")
    assert ok is True
    assert warning == ""


@run_async
async def test_oversized_model_is_flagged():
    guard = HardwareVRAMBoundaryGuard("http://localhost:11434", vram_limit_gb=14.0)
    guard._model_size_bytes = lambda model: _resolved(18 * (1024**3))

    ok, warning = await guard.verify_vram_safety_margin("huge-model")
    assert ok is False
    assert "huge-model" in warning
    assert "18.0 GB" in warning


@run_async
async def test_unknown_size_fails_open():
    guard = HardwareVRAMBoundaryGuard("http://localhost:11434", vram_limit_gb=14.0)
    guard._model_size_bytes = lambda model: _resolved(None)

    ok, warning = await guard.verify_vram_safety_margin("mystery-model")
    assert ok is True
    assert warning == ""


async def _resolved(value):
    return value
