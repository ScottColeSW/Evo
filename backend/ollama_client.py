import json

import httpx


class OllamaClient:
    """Thin async wrapper around a local Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
            except Exception:
                return []

    async def generate_json(
        self, model: str, prompt: str, temperature: float = 0.7, num_ctx: int = 4096, keep_alive: str = "5m"
    ) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
            "keep_alive": keep_alive,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            raw = r.json().get("response", "{}")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}

    async def generate_text(self, model: str, prompt: str, temperature: float = 0.5, keep_alive: str = "5m") -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
            "keep_alive": keep_alive,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")
