import importlib
import time

from . import config, physics
from .memory import TribeMemory
from .ollama_client import OllamaClient
from .prompts import system_prompt, turn_prompt
from .self_mod import SelfModEngine
from .world import Landscape

SPAWN_POINTS = [(20, 15), (80, 20), (20, 80), (80, 80)]
COLORS = ["#c084fc", "#fb923c", "#34d399", "#60a5fa"]

CITY_POPULATION_TARGET = 25


class Tribe:
    def __init__(self, tribe_id: str, name: str, model: str, x: int, y: int, color: str):
        self.id = tribe_id
        self.name = name
        self.model = model
        self.x, self.y = x, y
        self.color = color
        self.wood = 50
        self.stone = 50
        self.food = 40
        self.population = 8
        self.ideology = f"You are the {name}, a young tribe finding its way."
        self.lexicon: dict[str, str] = {}
        self.last_broadcast = ""
        self.history: list[str] = []
        self.memory = TribeMemory(tribe_id)
        self.founded_city = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "wood": self.wood,
            "stone": self.stone,
            "food": self.food,
            "population": self.population,
            "last_broadcast": self.last_broadcast,
            "history": self.history[-6:],
            "founded_city": self.founded_city,
        }


class Simulation:
    def __init__(self, tribe_configs: list[dict], ollama_url: str = config.OLLAMA_URL):
        if not tribe_configs:
            raise ValueError("Simulation needs at least one tribe")
        self.client = OllamaClient(ollama_url)
        self.world = Landscape()
        self.tribes: dict[str, Tribe] = {}
        for i, cfg in enumerate(tribe_configs[: config.MAX_TRIBES]):
            x, y = SPAWN_POINTS[i % len(SPAWN_POINTS)]
            tid = f"tribe_{i}"
            self.tribes[tid] = Tribe(tid, cfg["name"], cfg["model"], x, y, COLORS[i % len(COLORS)])
        self.cycle = 0
        self.paused = False
        self.status = "OPERATIONAL"
        self.self_mod = (
            SelfModEngine(self.client, tribe_configs[0]["model"], config.SELF_MOD_COOLDOWN_CYCLES)
            if config.ENABLE_SELF_MODIFICATION
            else None
        )

    def snapshot(self) -> dict:
        return {
            "cycle": self.cycle,
            "status": self.status,
            "tribes": {tid: t.to_dict() for tid, t in self.tribes.items()},
            "structures": [{"x": x, "y": y, **info} for (x, y), info in self.world.constructions.items()],
        }

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    async def step(self) -> None:
        if self.paused:
            return
        self.cycle += 1
        for tribe in self.tribes.values():
            await self._turn(tribe)
            self._grow_population(tribe)
        if self.self_mod:
            self.self_mod.tick()
        if self.cycle % config.MEMORY_CONSOLIDATE_EVERY_N_CYCLES == 0:
            for tribe in self.tribes.values():
                tribe.memory.consolidate()

    async def _turn(self, tribe: Tribe) -> None:
        biome = self.world.biome(tribe.x, tribe.y)
        nearby = self.world.nearby_structures(tribe.x, tribe.y)
        bias_text = self.world.ancestral_bias(tribe.x, tribe.y)
        memories = tribe.memory.recall(f"{biome} at {tribe.x},{tribe.y}")

        state = {
            "cycle": self.cycle,
            "x": tribe.x,
            "y": tribe.y,
            "biome": biome,
            "population": tribe.population,
            "wood": tribe.wood,
            "stone": tribe.stone,
            "food": tribe.food,
            "nearby": nearby,
        }
        prompt = (
            system_prompt(tribe.name, tribe.model)
            + "\n\n"
            + turn_prompt(state, memories, bias_text, tribe.memory.taboos)
        )

        temperature = 1.1 if bias_text else 0.6
        start = time.perf_counter()
        try:
            intent = await self.client.generate_json(tribe.model, prompt, temperature=temperature)
        except Exception:
            intent = {}
        latency_ms = (time.perf_counter() - start) * 1000

        action = intent.get("action", "IDLE")
        if action not in {
            "GATHER_WOOD", "GATHER_STONE", "HUNT", "BUILD_FIRE",
            "CONSTRUCT_WALL", "MOVE", "BROADCAST", "IDLE",
        }:
            action = "IDLE"
        broadcast = intent.get("broadcast") or ""
        target = intent.get("move_toward", [tribe.x, tribe.y])
        if not (isinstance(target, list) and len(target) == 2):
            target = [tribe.x, tribe.y]

        self._apply_action(tribe, action)
        tribe.last_broadcast = broadcast
        rationale = str(intent.get("rationale", ""))[:60]
        tribe.history.append(f"[{latency_ms:.0f}ms] {action}: {rationale}")

        weight = 0.75 if action in ("BUILD_FIRE", "CONSTRUCT_WALL") else 0.3
        tribe.memory.remember(f"At ({tribe.x},{tribe.y}) in {biome}, chose {action}.", self.cycle, weight)

        try:
            nx, ny = physics.calculate_next_step(tribe.x, tribe.y, int(target[0]), int(target[1]))
            tribe.x, tribe.y = nx, ny
        except Exception:
            pass

        if self.self_mod and latency_ms > config.SELF_MOD_LATENCY_THRESHOLD_MS:
            self.status = "REFACTORING"
            await self.self_mod.attempt_patch(f"turn latency {latency_ms:.0f}ms")
            importlib.reload(physics)
            self.status = "OPERATIONAL"

    def _apply_action(self, tribe: Tribe, action: str) -> None:
        if action == "GATHER_WOOD":
            tribe.wood += 10
        elif action == "GATHER_STONE":
            tribe.stone += 10
        elif action == "HUNT":
            tribe.food += 15
        elif action == "BUILD_FIRE" and tribe.wood >= 10:
            tribe.wood -= 10
            self.world.add_construction(tribe.x, tribe.y, "fire", self.cycle)
            self.world.record_event(tribe.x, tribe.y, +0.3)
        elif action == "CONSTRUCT_WALL" and tribe.wood >= 15 and tribe.stone >= 15:
            tribe.wood -= 15
            tribe.stone -= 15
            self.world.add_construction(tribe.x, tribe.y, "wall", self.cycle)

    def _grow_population(self, tribe: Tribe) -> None:
        if tribe.food > 80 and tribe.population < 80:
            tribe.population += 1
            tribe.food -= 30
        if not tribe.founded_city and tribe.population >= CITY_POPULATION_TARGET:
            tribe.founded_city = True
            self.world.record_event(tribe.x, tribe.y, +0.5)
