import importlib
import random

from . import config, physics
from .ancestral_matrix import AncestralTraumaMatrix
from .memory import TribeMemory
from .ollama_client import OllamaClient
from .prompts import ACTIONS, compile_live_state_prompt, get_prime_consciousness_prompt
from .scheduler import ModelBatchScheduler
from .self_mod import SelfModEngine
from .translation_matrix import TranslationConfidenceMatrix
from .vram_guard import HardwareVRAMBoundaryGuard
from .world import Landscape

# One spawn per biome (forest, mountains, plains, river) so the default picker order
# (Forest Tribe, Mountain Tribe, ...) actually starts tribes in the biome their name
# implies, rather than three of four tribes landing in the same forest band.
SPAWN_POINTS = [(85, 85), (10, 45), (65, 85), (50, 50)]
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
        self.last_broadcast = ""
        self.last_action = ""
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
        self.scheduler = ModelBatchScheduler(self.client)
        self.world = Landscape(config.GRID_SIZE)
        self.trauma = AncestralTraumaMatrix(config.GRID_SIZE)
        self.translation = TranslationConfidenceMatrix()
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

    @classmethod
    async def create(cls, tribe_configs: list[dict], ollama_url: str = config.OLLAMA_URL) -> "Simulation":
        """Preferred constructor: runs a one-time VRAM sanity check per model before
        building the simulation, and drops a warning into a tribe's chronicle (rather
        than blocking it) if its model looks too large for the configured budget."""
        guard = HardwareVRAMBoundaryGuard(ollama_url, config.VRAM_LIMIT_GB)
        warnings: dict[str, str] = {}
        for cfg in tribe_configs[: config.MAX_TRIBES]:
            ok, warning = await guard.verify_vram_safety_margin(cfg["model"])
            if not ok:
                warnings[cfg["name"]] = warning

        sim = cls(tribe_configs, ollama_url)
        for tribe in sim.tribes.values():
            if tribe.name in warnings:
                tribe.history.append(f"VRAM WARNING: {warnings[tribe.name]}")
        return sim

    def snapshot(self) -> dict:
        tribe_ids = list(self.tribes.keys())
        consensus = []
        for i in range(len(tribe_ids)):
            for j in range(i + 1, len(tribe_ids)):
                a_id, b_id = tribe_ids[i], tribe_ids[j]
                summary = self.translation.pair_summary(a_id, b_id)
                consensus.append({"a": self.tribes[a_id].name, "b": self.tribes[b_id].name, **summary})

        return {
            "cycle": self.cycle,
            "status": self.status,
            "tribes": {tid: t.to_dict() for tid, t in self.tribes.items()},
            "structures": [{"x": x, "y": y, **info} for (x, y), info in self.world.constructions.items()],
            "linguistic_consensus": consensus,
        }

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    async def step(self) -> None:
        if self.paused:
            return
        self.cycle += 1
        self.translation.decay()

        requests = []
        contexts = {}
        for tid, tribe in self.tribes.items():
            request, ctx = self._prepare_turn(tribe)
            requests.append(request)
            contexts[tid] = ctx

        results = await self.scheduler.run_batch(requests)

        for tid, tribe in self.tribes.items():
            outcome = results.get(tid, {"intent": {}, "latency_ms": 0.0})
            self._apply_turn(tribe, outcome["intent"], outcome["latency_ms"], contexts[tid])
            self._grow_population(tribe)

        if self.self_mod:
            self.self_mod.tick()
            max_latency = max((r["latency_ms"] for r in results.values()), default=0.0)
            if not self.self_mod.on_cooldown and max_latency > config.SELF_MOD_LATENCY_THRESHOLD_MS:
                self.status = "REFACTORING"
                await self.self_mod.attempt_patch(f"turn batch latency {max_latency:.0f}ms")
                importlib.reload(physics)
                self.status = "OPERATIONAL"

        if self.cycle % config.MEMORY_CONSOLIDATE_EVERY_N_CYCLES == 0:
            for tribe in self.tribes.values():
                tribe.memory.consolidate()

    def _prepare_turn(self, tribe: Tribe) -> tuple[dict, dict]:
        """Builds this tribe's prompt with no network calls; returns (request, context)."""
        biome = self.world.biome(tribe.x, tribe.y)
        nearby = self.world.nearby_structures(tribe.x, tribe.y)
        ghost_bias = self.trauma.bias_string(tribe.x, tribe.y)
        memories = tribe.memory.recall(f"{biome} at {tribe.x},{tribe.y}")

        visible_entities = [f"structure:{s['type']}@({s['x']},{s['y']})" for s in nearby]
        visible_entities += [f"memory(cycle {m['cycle']}): {m['text']}" for m in memories]
        visible_entities += [f"taboo: {t}" for t in tribe.memory.taboos[:3]]
        # Other tribes' broadcasts are audible regardless of distance (a simplification
        # for a small tribe count) -- this is what makes shared-vocabulary convergence
        # in translation_matrix.py possible at all, since nothing else lets a tribe
        # observe another tribe's invented tokens.
        for other in self.tribes.values():
            if other.id != tribe.id and other.last_broadcast:
                visible_entities.append(
                    f"overheard: {other.name} broadcasted '{other.last_broadcast}' while performing {other.last_action}"
                )
        if not visible_entities:
            visible_entities = ["none"]

        world_state = {
            "cycle": self.cycle,
            "x": tribe.x,
            "y": tribe.y,
            "biome": biome,
            "population": tribe.population,
            "wood": tribe.wood,
            "stone": tribe.stone,
            "food": tribe.food,
            "visible_entities": visible_entities,
        }
        base_prompt = get_prime_consciousness_prompt(tribe.name, tribe.model)
        prompt = compile_live_state_prompt(base_prompt, world_state, ghost_bias)
        temperature = config.ANCESTRAL_DREAD_TEMPERATURE if "DREAD" in ghost_bias else config.DEFAULT_TEMPERATURE

        request = {"id": tribe.id, "model": tribe.model, "prompt": prompt, "temperature": temperature}
        return request, {"biome": biome}

    def _apply_turn(self, tribe: Tribe, intent: dict, latency_ms: float, ctx: dict) -> None:
        action = intent.get("visual_action", "IDLE")
        if action not in ACTIONS:
            action = "IDLE"
        broadcast = intent.get("synthetic_language_broadcast") or ""
        target = intent.get("target_vector", [tribe.x, tribe.y])
        if not (isinstance(target, list) and len(target) == 2):
            target = [tribe.x, tribe.y]

        hazard_note = self._apply_action(tribe, action, ctx["biome"])
        tribe.last_broadcast = broadcast
        tribe.last_action = action
        self.translation.record_broadcast(tribe.id, broadcast, action)

        rationale = str(intent.get("metacognitive_rationale", ""))[:60]
        entry = f"[{latency_ms:.0f}ms] {action}: {rationale}"
        if hazard_note:
            entry += f" | {hazard_note}"
        tribe.history.append(entry)

        weight = 0.85 if hazard_note else (0.75 if action in ("BUILD_FIRE", "CONSTRUCT_WALL") else 0.3)
        memory_text = f"At ({tribe.x},{tribe.y}) in {ctx['biome']}, chose {action}."
        if hazard_note:
            memory_text += f" {hazard_note}."
        tribe.memory.remember(memory_text, self.cycle, weight)

        try:
            nx, ny = physics.calculate_next_step(tribe.x, tribe.y, int(target[0]), int(target[1]))
            tribe.x, tribe.y = nx, ny
        except Exception:
            pass

    def _apply_action(self, tribe: Tribe, action: str, biome: str) -> str | None:
        """Applies an action's resource effects. Returns a hazard note for the chronicle,
        or None if nothing eventful happened."""
        if action == "GATHER_WOOD":
            tribe.wood += 10
        elif action == "GATHER_STONE":
            tribe.stone += 10
        elif action == "HUNT_DEER":
            if biome == "forest" and random.random() < config.HUNT_HAZARD_CHANCE:
                tribe.food = max(0, tribe.food - config.HUNT_HAZARD_FOOD_LOSS)
                tribe.population = max(1, tribe.population - config.HUNT_HAZARD_POPULATION_LOSS)
                self.trauma.radiate_event_wave(
                    tribe.x, tribe.y, config.HUNT_HAZARD_TRAUMA_MAGNITUDE, config.HUNT_HAZARD_TRAUMA_RADIUS
                )
                return "a wolf pack struck the hunting party"
            tribe.food += 15
        elif action == "BUILD_FIRE" and tribe.wood >= 10:
            tribe.wood -= 10
            self.world.add_construction(tribe.x, tribe.y, "fire", self.cycle)
            self.trauma.radiate_event_wave(
                tribe.x, tribe.y, config.BUILD_FIRE_PRIDE_MAGNITUDE, config.BUILD_FIRE_PRIDE_RADIUS
            )
        elif action == "CONSTRUCT_WALL" and tribe.wood >= 15 and tribe.stone >= 15:
            tribe.wood -= 15
            tribe.stone -= 15
            self.world.add_construction(tribe.x, tribe.y, "wall", self.cycle)
        return None

    def _grow_population(self, tribe: Tribe) -> None:
        if tribe.food > 80 and tribe.population < 80:
            tribe.population += 1
            tribe.food -= 30
        if not tribe.founded_city and tribe.population >= CITY_POPULATION_TARGET:
            tribe.founded_city = True
            self.trauma.radiate_event_wave(
                tribe.x, tribe.y, config.CITY_FOUNDED_PRIDE_MAGNITUDE, config.CITY_FOUNDED_PRIDE_RADIUS
            )
