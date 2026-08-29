from backend.scheduler import ModelBatchScheduler
from tests.conftest import run_async


class FakeOllamaClient:
    """Stands in for OllamaClient so scheduler tests never touch the network."""

    def __init__(self):
        self.calls: list[str] = []

    async def generate_json(self, model, prompt, temperature=0.7, **kwargs):
        self.calls.append(model)
        if model == "broken-model":
            raise RuntimeError("simulated Ollama failure")
        return {"visual_action": "IDLE", "model_echo": model}


@run_async
async def test_groups_requests_by_model():
    client = FakeOllamaClient()
    scheduler = ModelBatchScheduler(client)
    requests = [
        {"id": "tribe_0", "model": "llama3", "prompt": "p0", "temperature": 0.5},
        {"id": "tribe_1", "model": "mistral", "prompt": "p1", "temperature": 0.5},
        {"id": "tribe_2", "model": "llama3", "prompt": "p2", "temperature": 0.5},
    ]
    results = await scheduler.run_batch(requests)
    assert set(results.keys()) == {"tribe_0", "tribe_1", "tribe_2"}
    assert results["tribe_0"]["intent"]["model_echo"] == "llama3"
    assert results["tribe_2"]["intent"]["model_echo"] == "llama3"
    assert results["tribe_1"]["intent"]["model_echo"] == "mistral"
    assert client.calls.count("llama3") == 2
    assert client.calls.count("mistral") == 1


@run_async
async def test_one_failing_request_does_not_take_down_its_batch():
    client = FakeOllamaClient()
    scheduler = ModelBatchScheduler(client)
    requests = [
        {"id": "tribe_0", "model": "broken-model", "prompt": "p0", "temperature": 0.5},
        {"id": "tribe_1", "model": "broken-model", "prompt": "p1", "temperature": 0.5},
        {"id": "tribe_2", "model": "llama3", "prompt": "p2", "temperature": 0.5},
    ]
    results = await scheduler.run_batch(requests)
    assert results["tribe_0"]["intent"] == {}
    assert results["tribe_1"]["intent"] == {}
    assert results["tribe_2"]["intent"]["model_echo"] == "llama3"


@run_async
async def test_empty_batch_returns_empty_results():
    scheduler = ModelBatchScheduler(FakeOllamaClient())
    assert await scheduler.run_batch([]) == {}
