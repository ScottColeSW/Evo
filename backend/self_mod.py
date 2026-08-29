import ast
from pathlib import Path

from .ollama_client import OllamaClient

PHYSICS_PATH = Path(__file__).resolve().parent / "physics.py"


class SelfModEngine:
    """Lets a model rewrite backend/physics.py when turns get slow.

    Guardrails: the returned source must parse as valid Python (ast.parse) before it's
    written to disk, and a failed attempt locks the engine out for `cooldown_cycles`
    turns so a broken model doesn't hammer the API every tick. Disabled by default via
    config.ENABLE_SELF_MODIFICATION — it is still arbitrary model-authored code landing
    on your filesystem, even if it's sandboxed to this one file.
    """

    def __init__(self, client: OllamaClient, model: str, cooldown_cycles: int = 20):
        self.client = client
        self.model = model
        self.cooldown_cycles = cooldown_cycles
        self._cooldown_remaining = 0

    @property
    def on_cooldown(self) -> bool:
        return self._cooldown_remaining > 0

    def tick(self) -> None:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    async def attempt_patch(self, telemetry: str) -> bool:
        if self.on_cooldown:
            return False
        source = PHYSICS_PATH.read_text()
        prompt = f"""Refactor this Python module for speed without changing its public \
function signature. Return ONLY the full replacement file contents — no markdown fences, \
no commentary.

TELEMETRY: {telemetry}

CURRENT FILE:
{source}"""
        new_source = await self.client.generate_text(self.model, prompt, temperature=0.3)
        new_source = _strip_fences(new_source)
        try:
            ast.parse(new_source)
        except SyntaxError:
            self._cooldown_remaining = self.cooldown_cycles
            return False
        PHYSICS_PATH.write_text(new_source)
        return True


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("python"):
                text = text[len("python"):]
    return text.strip()
