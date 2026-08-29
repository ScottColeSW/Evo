import httpx


class HardwareVRAMBoundaryGuard:
    """A rough pre-flight sanity check, not a live enforcement layer.

    Uses the on-disk size Ollama already reports per model in /api/tags -- that size
    reflects whatever quantization is actually installed (a Q4_K_M pull is already
    small; there's no separate "unquantized FP16" case to special-case, the byte count
    already tells you). This is checked once when a tribe is assigned a model, not on
    every tick: model size doesn't change mid-run, and re-querying it every tick would
    be a wasted HTTP round trip on the hot path for no new information.

    It fails open (returns ok=True) when the size can't be determined, since a local
    loopback call to your own Ollama instance failing usually means something else is
    already wrong that a hard lockout wouldn't fix -- and Ollama will still raise its
    own error if a model genuinely doesn't fit.
    """

    def __init__(self, ollama_url: str, vram_limit_gb: float = 14.0):
        self.ollama_url = ollama_url.rstrip("/")
        self.vram_limit_bytes = vram_limit_gb * (1024**3)
        self._size_cache: dict[str, int] = {}

    async def _model_size_bytes(self, model: str) -> int | None:
        if model in self._size_cache:
            return self._size_cache[model]
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{self.ollama_url}/api/tags")
                r.raise_for_status()
                for entry in r.json().get("models", []):
                    if entry.get("name") == model:
                        size = int(entry.get("size", 0))
                        self._size_cache[model] = size
                        return size
            except Exception:
                return None
        return None

    async def verify_vram_safety_margin(self, model: str) -> tuple[bool, str]:
        """Returns (ok, warning). warning is empty when ok is True."""
        size = await self._model_size_bytes(model)
        if size is None or size == 0:
            return True, ""
        if size > self.vram_limit_bytes:
            gb = size / (1024**3)
            limit_gb = self.vram_limit_bytes / (1024**3)
            return False, f"{model} is ~{gb:.1f} GB on disk, over the {limit_gb:.1f} GB VRAM budget"
        return True, ""
