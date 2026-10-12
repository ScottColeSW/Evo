import asyncio
import time
from collections import defaultdict

from .ollama_client import OllamaClient


class ModelBatchScheduler:
    """Runs one tick's turns grouped by target model, not by tribe order.

    Ollama serializes inference per loaded model and can evict one model to load
    another when VRAM is tight. If a tick's turns fire in arbitrary tribe order across
    several different models, every turn risks a model swap. Grouping same-model turns
    together and running them concurrently means at most one swap per model per tick,
    regardless of how many tribes share it or what order they're defined in.
    """

    def __init__(self, client: OllamaClient):
        self.client = client

    async def run_batch(self, requests: list[dict]) -> dict[str, dict]:
        """requests: [{"id": tribe_id, "model": ..., "prompt": ..., "temperature": ...}]

        Returns {tribe_id: {"intent": dict, "latency_ms": float, "raw_response": str}}.
        A request whose Ollama call raises gets an empty intent (and empty raw
        response) rather than propagating the exception, so one bad turn can't take
        down the whole tick.

        raw_response is the model's actual, unparsed text -- explicit request,
        2026-09-09, for a live debug view of "what we tell the llm, how it
        responses." generate_json_with_raw (not the plain generate_json every other
        caller in this codebase still uses) is what makes that text available here
        at all instead of being discarded the moment JSON parsing succeeds.
        """
        by_model: dict[str, list[dict]] = defaultdict(list)
        for req in requests:
            by_model[req["model"]].append(req)

        results: dict[str, dict] = {}
        for model, group in by_model.items():
            start = time.perf_counter()
            outcomes = await asyncio.gather(
                *(self.client.generate_json_with_raw(model, r["prompt"], r["temperature"]) for r in group),
                return_exceptions=True,
            )
            # 2026-10-11 (the owner: "add the one retry, that's more fair"): a call that failed gets one more try before the turn counts as a model failure; a failure on a busy or reloading Ollama often clears on the second.
            failed = [req for req, outcome in zip(group, outcomes) if isinstance(outcome, Exception)]
            if failed:
                for req, outcome in zip(group, outcomes):
                    if isinstance(outcome, Exception):
                        print(f"[scheduler] {model} failed a turn ({type(outcome).__name__}) after {(time.perf_counter() - start):.0f}s; trying once more")
                retries = dict(zip((r["id"] for r in failed), await asyncio.gather(
                    *(self.client.generate_json_with_raw(model, r["prompt"], r["temperature"]) for r in failed), return_exceptions=True)))
                outcomes = [retries[req["id"]] if isinstance(outcome, Exception) else outcome for req, outcome in zip(group, outcomes)]
            latency_ms = (time.perf_counter() - start) * 1000
            for req, outcome in zip(group, outcomes):
                error = None
                if isinstance(outcome, Exception):
                    intent, raw_response = {}, ""
                    # 2026-10-11 (the owner: a call that fails or times out is categorically a model or Ollama failure): the reason is carried back so the run can say so, not hidden behind an empty answer
                    error = f"{type(outcome).__name__}: {str(outcome)[:200]}".rstrip(": ")
                    print(f"[scheduler] {model} failed a turn twice, {latency_ms / 1000:.0f}s in all: {error}")
                else:
                    intent, raw_response = outcome
                results[req["id"]] = {"intent": intent, "latency_ms": latency_ms, "raw_response": raw_response}
                if error:
                    results[req["id"]]["error"] = error
        return results
