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

        Returns {tribe_id: {"intent": dict, "latency_ms": float}}. A request whose
        Ollama call raises gets an empty intent rather than propagating the exception,
        so one bad turn can't take down the whole tick.
        """
        by_model: dict[str, list[dict]] = defaultdict(list)
        for req in requests:
            by_model[req["model"]].append(req)

        results: dict[str, dict] = {}
        for model, group in by_model.items():
            start = time.perf_counter()
            outcomes = await asyncio.gather(
                *(self.client.generate_json(model, r["prompt"], r["temperature"]) for r in group),
                return_exceptions=True,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            for req, outcome in zip(group, outcomes):
                results[req["id"]] = {
                    "intent": {} if isinstance(outcome, Exception) else outcome,
                    "latency_ms": latency_ms,
                }
        return results
