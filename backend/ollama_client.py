import asyncio
import json

import httpx

# Live-confirmed (explicit report: "the quit isn't cleaning up after itself and
# making sure the system stops"): Ollama's /api/generate keep_alive=0 responds
# done:true immediately, well before the model actually leaves VRAM -- observed
# ~8s of real lag evicting two ~2-3GB models via direct API probing after a
# QUIT left the backend process already exited. unload_model polls for real
# eviction instead of trusting that response; bounded so a genuinely stuck
# Ollama can't hang shutdown forever.
UNLOAD_POLL_INTERVAL_SECONDS = 0.5
UNLOAD_POLL_MAX_ATTEMPTS = 20


class OllamaClient:
    """Thin async wrapper around a local Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        # 30s used to be the default and was too tight even under normal load -- a cold
        # multi-GB model (mistral:7b, qwen2.5-coder:7b) can genuinely take over a
        # minute to load into VRAM and return its first response, especially with
        # another simulation already running. A real ReadTimeout here doesn't fail
        # gracefully: it propagates out of _install_chief and leaves that tribe's
        # Simulation.create() (and therefore the whole websocket session) permanently
        # stuck -- confirmed live when this hit both a headless run and the actual
        # server mid-session.
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
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            # `format: "json"` guarantees valid JSON, not a JSON *object* -- a weak or
            # very small model (seen live with llama3.2:1b) can emit a bare string,
            # number, or list that parses without error but isn't a dict. Every caller
            # does result.get(...) assuming a dict; returning {} here (the same
            # fallback as an outright parse failure) is what makes that safe regardless
            # of how capable the model actually is, rather than crashing the whole
            # simulation on one degenerate response.
            return parsed if isinstance(parsed, dict) else {}

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

    async def list_loaded_models(self) -> list[str]:
        """Models Ollama currently has resident in memory/VRAM right now (Ollama's
        /api/ps), as opposed to list_models()'s /api/tags (every model ever pulled,
        loaded or not). Used by app.py's startup cleanup to find and evict whatever a
        previous, ungracefully-killed server process left loaded."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{self.base_url}/api/ps")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
            except Exception:
                return []

    async def unload_model(self, model: str) -> None:
        """Tells Ollama to evict this model from memory/VRAM right now instead of
        waiting out its keep_alive window. Called on game-over and on Simulation.
        shutdown() (an explicit STOP, or a browser tab closing/reloading mid-game --
        see app.py's ws_handler). Best-effort: a failure here just means the model
        stays loaded a bit longer, not worth surfacing as an error to a game that's
        already ending.

        Confirmed live: a 5s timeout here was too tight once shutdown() started
        unloading two 7B-class models concurrently (Simulation.shutdown does exactly
        this via asyncio.gather) -- Ollama appears to serialize the actual VRAM
        eviction, so the second request can genuinely take longer than 5s to get a
        response even though nothing is actually wrong. Matches the main client's own
        120s default rather than a separate, tighter number.

        Explicit report: "the quit isn't cleaning up after itself and making
        sure the system stops." Confirmed live: the /api/generate response
        above comes back done:true well before the model actually leaves VRAM
        (~8s of real lag observed evicting two ~2-3GB models via direct API
        probing, after the backend process had already exited) -- trusting
        that response as "unloaded" let shutdown() return, and the process
        exit right after it, with nothing left running to ever confirm or
        retry. Now polls list_loaded_models() until this model actually
        disappears, bounded by UNLOAD_POLL_MAX_ATTEMPTS so a genuinely stuck
        Ollama can't hang shutdown forever."""
        payload = {"model": model, "keep_alive": 0}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await client.post(f"{self.base_url}/api/generate", json=payload)
        except Exception as exc:
            # Still best-effort (a failed unload isn't worth crashing an ending game
            # over), but this used to swallow the exception completely -- diagnosing
            # a real live bug (one of two models silently staying loaded after QUIT)
            # required standalone curl/ollama-ps probing instead of just reading a
            # log line. repr(), not str(): some exceptions (seen on Windows --
            # ConnectError wrapping an OSError) stringify to an empty message.
            print(f"[ollama_client] unload_model({model!r}) failed: {exc!r}")
            return
        for _ in range(UNLOAD_POLL_MAX_ATTEMPTS):
            if model not in await self.list_loaded_models():
                return
            await asyncio.sleep(UNLOAD_POLL_INTERVAL_SECONDS)
        print(f"[ollama_client] unload_model({model!r}) still resident after waiting -- giving up, best-effort")
