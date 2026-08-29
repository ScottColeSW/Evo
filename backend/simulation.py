import asyncio
import importlib

from . import config, physics
from .actions import ACTION_REGISTRY
from .ancestral_matrix import AncestralTraumaMatrix
from .eras import ERAS, next_era, unlocked_actions_through
from .instincts import survival_bias_string
from .leadership import elect_chief
from .memory import TribeMemory
from .ollama_client import OllamaClient
from .prompts import compile_live_state_prompt, get_prime_consciousness_prompt
from .scheduler import ModelBatchScheduler
from .self_mod import SelfModEngine
from .translation_matrix import TranslationConfidenceMatrix
from .vram_guard import HardwareVRAMBoundaryGuard
from .world import BIOME_LABELS, Landscape

# One spawn per land biome (forest, mountains, plains, river -- not ocean, nothing spawns
# at sea) so the default picker order (Forest Tribe, Mountain Tribe, ...) actually starts
# tribes in the biome their name implies. Coordinates match backend/world.py's geography:
# mountains in the northwest, forest along the north/east, plains in the south, and a
# river cutting from the highlands down to the eastern coast.
SPAWN_POINTS = [(80, 10), (10, 10), (50, 90), (40, 37)]
COLORS = ["#c084fc", "#fb923c", "#34d399", "#60a5fa"]


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
        self.water = config.STARTING_WATER
        self.population = 8
        self.era = ERAS[0].key
        self.last_broadcast = ""
        self.last_action = ""
        self.last_target: list[int] | None = None
        self.history: list[str] = []
        self.memory = TribeMemory(tribe_id)
        self.founded_city = False
        self.extinct = False
        self.chief_name = ""
        self.chief_philosophy = ""
        self.chief_decree = ""

    def to_dict(self) -> dict:
        era_label = next((e.label for e in ERAS if e.key == self.era), self.era)
        survival_warning, _ = survival_bias_string(self.food, self.water)
        return {
            "name": self.name,
            "model": self.model,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "wood": self.wood,
            "stone": self.stone,
            "food": self.food,
            "water": self.water,
            "population": self.population,
            "era": self.era,
            "era_label": era_label,
            "last_broadcast": self.last_broadcast,
            "history": self.history[-6:],
            "founded_city": self.founded_city,
            "survival_warning": survival_warning,
            "extinct": self.extinct,
            "chief_name": self.chief_name,
            "chief_philosophy": self.chief_philosophy,
            "chief_decree": self.chief_decree,
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
            # An explicit x/y (e.g. to set up two tribes starting near each other) is an
            # initial condition, same category as SPAWN_POINTS itself -- it says nothing
            # about what either tribe then chooses to do about being close.
            if "x" in cfg and "y" in cfg:
                x, y = cfg["x"], cfg["y"]
            else:
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
        await asyncio.gather(*(sim._install_chief(tribe) for tribe in sim.tribes.values()))
        return sim

    async def _install_chief(self, tribe: "Tribe") -> None:
        """One-time in-fiction leadership contest (see leadership.py) -- the resulting
        philosophy becomes standing context in every future turn, generated by the tribe
        itself rather than scripted from outside it. If the tribe isn't already on fresh
        water, the election is also given the real nearest-river coordinates as a fact
        (legitimate map knowledge) and may decide -- its own call, not ours -- to decree
        a relocation. Specifically fresh water, not "water" generally: a tribe standing
        on the coast still has no drinking water, so ocean doesn't count as already
        solved. What else the ocean might be good for is left entirely open."""
        nearest_water = None
        if self.world.biome(tribe.x, tribe.y) != "river":
            nearest_water = self.world.nearest_water(tribe.x, tribe.y, kinds=("river",))

        result = await elect_chief(self.client, tribe.model, tribe.name, nearest_water)
        tribe.chief_name = result.get("chief_name", "")
        tribe.chief_philosophy = result.get("guiding_philosophy", "")
        victory = result.get("victory_method", "")
        if tribe.chief_name:
            note = f"{tribe.chief_name} has become chief"
            tribe.history.append(f"{note} ({victory})." if victory else f"{note}.")

        relocate = result.get("relocate_decision") or {}
        if nearest_water is not None and relocate.get("decreed"):
            wx, wy = nearest_water
            tribe.chief_decree = f"relocate the tribe toward ({wx}, {wy}) for reliable water access"
            reason = relocate.get("reason", "")
            entry = f"Chief {tribe.chief_name} decrees: {tribe.chief_decree}"
            tribe.history.append(f"{entry} ({reason})." if reason else f"{entry}.")

    async def add_tribe(self, name: str, model: str, x: int | None = None, y: int | None = None) -> str | None:
        """Injects a new tribe into an already-running simulation. Returns an error
        message on failure (max tribes reached), or None on success. Runs the same
        one-time VRAM check and chief election as Simulation.create()."""
        if len(self.tribes) >= config.MAX_TRIBES:
            return f"Cannot add tribe: maximum of {config.MAX_TRIBES} tribes reached."

        index = len(self.tribes)
        tid = f"tribe_{index}"
        if x is None or y is None:
            x, y = SPAWN_POINTS[index % len(SPAWN_POINTS)]
        color = COLORS[index % len(COLORS)]
        tribe = Tribe(tid, name, model, x, y, color)

        guard = HardwareVRAMBoundaryGuard(self.client.base_url, config.VRAM_LIMIT_GB)
        ok, warning = await guard.verify_vram_safety_margin(model)
        if not ok:
            tribe.history.append(f"VRAM WARNING: {warning}")
        await self._install_chief(tribe)

        self.tribes[tid] = tribe
        return None

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
        self.world.regenerate(config.DEPLETION_REGEN_PER_CYCLE)

        requests = []
        contexts = {}
        for tid, tribe in self.tribes.items():
            if tribe.extinct:
                continue
            request, ctx = self._prepare_turn(tribe)
            requests.append(request)
            contexts[tid] = ctx

        results = await self.scheduler.run_batch(requests)

        for tid, tribe in self.tribes.items():
            if tribe.extinct:
                continue
            outcome = results.get(tid, {"intent": {}, "latency_ms": 0.0})
            self._apply_turn(tribe, outcome["intent"], outcome["latency_ms"], contexts[tid])
            self._apply_upkeep(tribe)
            self._grow_population(tribe)
            self._advance_era_if_ready(tribe)

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
        survival_bias, survival_critical = survival_bias_string(tribe.food, tribe.water)
        memories = tribe.memory.recall(f"{biome} at {tribe.x},{tribe.y}")
        available_actions = sorted(unlocked_actions_through(tribe.era))

        visible_entities = [f"structure:{s['type']}@({s['x']},{s['y']})" for s in nearby]
        visible_entities += [f"memory(cycle {m['cycle']}): {m['text']}" for m in memories]
        visible_entities += [f"taboo: {t}" for t in tribe.memory.taboos[:3]]
        # Factual telemetry about this exact tile, not a suggestion to move -- what the
        # tribe does with the information is entirely its own reasoning.
        for resource in ("wood", "stone", "water", "game"):
            level = self.world.scarcity(resource, tribe.x, tribe.y)
            if level > 0:
                visible_entities.append(f"local {resource} scarcity here: {level:.0%}")
        # A broadcast is only overheard within BROADCAST_HEARING_RADIUS -- previously
        # audible map-wide regardless of distance, which gave away free information and
        # removed any incentive to actually travel toward another tribe.
        for other in self.tribes.values():
            if other.id == tribe.id or other.extinct or not other.last_broadcast:
                continue
            distance = ((other.x - tribe.x) ** 2 + (other.y - tribe.y) ** 2) ** 0.5
            if distance <= config.BROADCAST_HEARING_RADIUS:
                visible_entities.append(
                    f"overheard: {other.name} broadcasted '{other.last_broadcast}' while performing {other.last_action}"
                )
        if not visible_entities:
            visible_entities = ["none"]

        # Stated as fact (what you previously chose), not as an instruction to continue --
        # whether to keep going or change course is left entirely to the model.
        journey_note = ""
        if tribe.last_target and tribe.last_target != [tribe.x, tribe.y]:
            journey_note = (
                f"Last cycle you set target_vector to ({tribe.last_target[0]}, {tribe.last_target[1]}); "
                "you have not yet arrived there."
            )

        world_state = {
            "cycle": self.cycle,
            "x": tribe.x,
            "y": tribe.y,
            "biome": biome,
            "biome_label": BIOME_LABELS.get(biome, biome),
            "population": tribe.population,
            "wood": tribe.wood,
            "stone": tribe.stone,
            "food": tribe.food,
            "water": tribe.water,
            "era": tribe.era,
            "available_actions": available_actions,
            "visible_entities": visible_entities,
            "journey_note": journey_note,
        }
        base_prompt = get_prime_consciousness_prompt(
            tribe.name, tribe.model, tribe.chief_name, tribe.chief_philosophy, tribe.chief_decree
        )
        prompt = compile_live_state_prompt(base_prompt, world_state, ghost_bias, survival_bias)
        panicked = "DREAD" in ghost_bias or survival_critical
        temperature = config.ANCESTRAL_DREAD_TEMPERATURE if panicked else config.DEFAULT_TEMPERATURE

        request = {"id": tribe.id, "model": tribe.model, "prompt": prompt, "temperature": temperature}
        return request, {"biome": biome, "available_actions": available_actions}

    def _apply_turn(self, tribe: Tribe, intent: dict, latency_ms: float, ctx: dict) -> None:
        action = intent.get("visual_action", "IDLE")
        if action not in ctx["available_actions"]:
            action = "IDLE"
        broadcast = intent.get("synthetic_language_broadcast") or ""
        target = intent.get("target_vector", [tribe.x, tribe.y])
        if not (isinstance(target, list) and len(target) == 2):
            target = [tribe.x, tribe.y]
        try:
            target = (int(target[0]), int(target[1]))
        except (TypeError, ValueError):
            target = (tribe.x, tribe.y)

        # Only RELOCATE actually moves the tribe -- everything else happens wherever it
        # currently stands. last_target/journey_note (see _prepare_turn) specifically
        # track an in-progress relocation, not just whatever coordinate a GATHER_WOOD
        # turn happened to carry.
        if action == "RELOCATE":
            tribe.last_target = [target[0], target[1]]

        hazard_note = self._apply_action(tribe, action, ctx["biome"], target)
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

    def _apply_action(self, tribe: Tribe, action: str, biome: str, target: tuple[int, int]) -> str | None:
        handler = ACTION_REGISTRY.get(action, ACTION_REGISTRY["IDLE"])
        return handler(self, tribe, biome, target)

    def _apply_upkeep(self, tribe: Tribe) -> None:
        """Larger tribes cost more to sustain each tick. Left unpaid, someone dies --
        this is what makes hunger and thirst actual stakes rather than numbers that
        only ever go up."""
        upkeep = max(1, tribe.population // config.UPKEEP_POPULATION_DIVISOR)
        tribe.food -= upkeep
        tribe.water -= upkeep

        if tribe.food < 0:
            tribe.food = 0
            self._starve(tribe)
        if tribe.water < 0:
            tribe.water = 0
            self._dehydrate(tribe)

    def _lose_population(self, tribe: Tribe, amount: int) -> None:
        """The single place population ever decreases. A tribe can now actually go
        extinct (population 0) rather than being propped up at a permanent
        population-1 floor -- extinction is marked, announced, and radiates a much
        larger trauma event than an ordinary death."""
        if tribe.extinct:
            return
        tribe.population = max(0, tribe.population - amount)
        if tribe.population == 0:
            tribe.extinct = True
            tribe.history.append(f"{tribe.name} has gone extinct.")
            self.trauma.radiate_event_wave(
                tribe.x, tribe.y, config.EXTINCTION_TRAUMA_MAGNITUDE, config.EXTINCTION_TRAUMA_RADIUS
            )

    def _starve(self, tribe: Tribe) -> None:
        tribe.history.append("starvation claimed lives")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.STARVATION_TRAUMA_MAGNITUDE, config.STARVATION_TRAUMA_RADIUS
        )
        self._lose_population(tribe, config.STARVATION_POPULATION_LOSS)

    def _dehydrate(self, tribe: Tribe) -> None:
        tribe.history.append("thirst claimed lives")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DEHYDRATION_TRAUMA_MAGNITUDE, config.DEHYDRATION_TRAUMA_RADIUS
        )
        self._lose_population(tribe, config.DEHYDRATION_POPULATION_LOSS)

    def _grow_population(self, tribe: Tribe) -> None:
        if tribe.food > 80 and tribe.population < 80:
            tribe.population += 1
            tribe.food -= 30

    def _advance_era_if_ready(self, tribe: Tribe) -> None:
        nxt = next_era(tribe.era)
        if nxt is None:
            return
        if tribe.population < nxt.requires_population:
            return
        for resource, minimum in nxt.requires_resources.items():
            if getattr(tribe, resource, 0) < minimum:
                return

        for resource, amount in nxt.advancement_cost.items():
            setattr(tribe, resource, max(0, getattr(tribe, resource) - amount))
        tribe.era = nxt.key
        tribe.history.append(nxt.announcement.format(tribe=tribe.name))
        if nxt.founds_city:
            tribe.founded_city = True
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.ERA_ADVANCE_PRIDE_MAGNITUDE, config.ERA_ADVANCE_PRIDE_RADIUS
        )
