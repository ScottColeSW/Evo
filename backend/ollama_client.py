import asyncio
import json

import httpx

from . import config

# Live-confirmed (explicit report: "the quit isn't cleaning up after itself and
# making sure the system stops"): Ollama's /api/generate keep_alive=0 responds
# done:true immediately, well before the model actually leaves VRAM -- observed
# ~8s of real lag evicting two ~2-3GB models via direct API probing after a
# QUIT left the backend process already exited. unload_model polls for real
# eviction instead of trusting that response; bounded so a genuinely stuck
# Ollama can't hang shutdown forever.
_CAPABILITY_CACHE: dict[tuple, list] = {}  # (model name, digest) -> Ollama's capabilities for it, so the setup screen does not ask again on every load
UNLOAD_POLL_INTERVAL_SECONDS = 0.5
UNLOAD_POLL_MAX_ATTEMPTS = 20


def _raise_with_body(r: httpx.Response, model: str) -> None:
    """Live report, 2026-09-18: a real 500 from /api/generate crashed a tick with
    nothing but "500 Internal Server Error" to go on -- plain raise_for_status()
    discards the response body, but Ollama's own 500s carry a real reason there
    (commonly VRAM exhaustion or a model crash mid-generation, the exact risk this
    project's VRAM-contention history keeps running into with multiple resident
    models). Surfaced in the exception message itself so app.py's existing
    traceback.print_exc() (the tick-level catch that already keeps a bad response
    from crashing the whole server) shows the real cause next time, not just a
    status code to guess from."""
    if r.status_code >= 400:
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                f"{exc.args[0]} -- model={model!r} body={r.text[:500]!r}",
                request=exc.request, response=exc.response,
            ) from None


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
        self.repeat_retries: dict[str, int] = {}  # model -> how many "token repeat limit" aborts were retried (see generate_json_with_raw)

    async def embed(self, text: str, model: str = "nomic-embed-text") -> list[float] | None:
        """Real semantic embedding via Ollama's /api/embeddings -- explicit
        request, 2026-09-19: real Jaccard token-overlap couldn't reliably
        detect a chief's own recurring private thought (measured live: the
        two most thematically-similar reflections found across a real run
        scored 0.09-0.17 overlap, well under the 0.3 reinforcement
        threshold, since the same underlying worry gets reworded every time
        rather than repeating literal vocabulary). Fails open -- returns
        None on any error -- same "a missing nice-to-have shouldn't break
        the real work" pattern vram_guard.py's own docstring already uses;
        TribeMemory.remember_reflection falls back to token-overlap when
        this comes back None, so a bad embedding call never needs
        Simulation._safe_llm_result's own protection -- it can't raise out
        into a resolve_* call in the first place."""
        payload = {"model": model, "prompt": text}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(f"{self.base_url}/api/embeddings", json=payload)
                _raise_with_body(r, model)
                embedding = r.json().get("embedding")
                return embedding if isinstance(embedding, list) and embedding else None
        except Exception:
            return None

    async def list_models(self) -> list[str]:
        """The models a tribe can play with: the ones Ollama has pulled, minus any that cannot generate text (an embedding model) and any "thinking" model.
        2026-10-08 (the owner, after a live run on qwen3:4b took a median of 53 seconds a decision, with turns up to 129, against 1.3 seconds for qwen2.5:3b, because a
        thinking model writes long hidden reasoning before every answer): the sim's turns are one short JSON decision, so thinking models are left out of the list
        here, which feeds both the setup screen and the fallback when a tribe's model fails."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                entries = r.json().get("models", [])
            except Exception:
                return []
            kept = await asyncio.gather(*(self._is_playable(client, m) for m in entries))
        return [m["name"] for m, ok in zip(entries, kept) if ok]

    async def _is_playable(self, client: httpx.AsyncClient, entry: dict) -> bool:
        name = entry["name"]
        key = (name, entry.get("digest"))
        if key not in _CAPABILITY_CACHE:
            capabilities = None
            try:
                r = await client.post(f"{self.base_url}/api/show", json={"model": name})
                r.raise_for_status()
                capabilities = r.json().get("capabilities")
            except Exception:
                pass
            if capabilities is None:  # an older Ollama, or a failed lookup: judge by name
                capabilities = ["completion"] + (["thinking"] if name.lower().startswith(config.THINKING_MODEL_NAME_PREFIXES) else [])
            _CAPABILITY_CACHE[key] = capabilities
        capabilities = _CAPABILITY_CACHE[key]
        return "completion" in capabilities and "thinking" not in capabilities

    async def generate_json(
        self, model: str, prompt: str, temperature: float = 0.7, num_ctx: int = 4096, keep_alive: str = "5m"
    ) -> dict:
        parsed, _raw = await self.generate_json_with_raw(model, prompt, temperature, num_ctx, keep_alive)
        return parsed

    async def generate_json_with_raw(
        self, model: str, prompt: str, temperature: float = 0.7, num_ctx: int = 4096, keep_alive: str = "5m"
    ) -> tuple[dict, str]:
        """Same call as generate_json, but also returns the raw, unparsed response
        text -- explicit request, 2026-09-09: a live debug view of "what we tell the
        llm, how it responses" needs the actual text the model sent back, not just
        whatever survived JSON parsing. A separate method (not a changed return
        shape on generate_json itself) so the several other callers here
        (breeding.py, genetics.py, leadership.py, reflection.py) -- none of which
        need the raw text -- stay untouched; only scheduler.py's per-cycle tribe
        turn calls this one."""
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
            if r.status_code == 500 and "token repeat limit" in r.text:
                # 2026-10-03: Ollama aborts a JSON generation that loops ("prediction aborted, token repeat limit reached"), seen
                # with gemma2:2b in the night reflection. One retry with a little more randomness and a repeat penalty usually
                # gets a clean answer; a second failure raises as before.
                payload["options"] = {**payload["options"], "temperature": min(1.0, temperature + 0.2), "repeat_penalty": 1.2}
                # 2026-10-08: hermes3:3b hit this on a dozen turns in a few minutes and the two lines per event buried the rest of the terminal. The first event
                # for a model is printed in full, then one summary line per 10, and the counts stay on the client (repeat_retries) for anyone who wants them.
                self.repeat_retries[model] = self.repeat_retries.get(model, 0) + 1
                count = self.repeat_retries[model]
                if count == 1:
                    print(f"[ollama] {model}: token repeat abort, retrying once (further ones are counted, one line per 10)")
                elif count % 10 == 0:
                    print(f"[ollama] {model}: {count} token repeat aborts so far, each retried")
                r = await client.post(f"{self.base_url}/api/generate", json=payload)
                if r.status_code >= 400:
                    print(f"[ollama] {model}: the retry failed too")
            _raise_with_body(r, model)
            raw = r.json().get("response", "{}")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}, raw
            # `format: "json"` guarantees valid JSON, not a JSON *object* -- a weak or
            # very small model (seen live with llama3.2:1b) can emit a bare string,
            # number, or list that parses without error but isn't a dict. Every caller
            # does result.get(...) assuming a dict; returning {} here (the same
            # fallback as an outright parse failure) is what makes that safe regardless
            # of how capable the model actually is, rather than crashing the whole
            # simulation on one degenerate response.
            return (parsed if isinstance(parsed, dict) else {}), raw

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
            _raise_with_body(r, model)
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
