import asyncio
import importlib
import random

from . import config, physics
from .actions import ACTION_REGISTRY, BIOME_YIELD_MULTIPLIER, GAME_SPECIES_LABEL
from .ancestral_matrix import AncestralTraumaMatrix
from .breeding import breed_individuals
from .reflection import reflect_on_history
from .eras import ERAS, era_index, next_era, unlocked_actions_through
from .event_log import RunEventLog, TribeHistory
from .scoreboard import record_tribe_result
from .instincts import survival_bias_string
from .leadership import elect_chief
from .memory import TribeMemory
from .ollama_client import OllamaClient
from .prompts import compile_live_state_prompt, get_prime_consciousness_prompt
from .scheduler import ModelBatchScheduler
from .self_mod import SelfModEngine
from .translation_matrix import TranslationConfidenceMatrix
from .vram_guard import HardwareVRAMBoundaryGuard
from .world import BIOME_LABELS, Landscape, biome_at

# One spawn per land biome (forest, mountains, plains, river -- not ocean, nothing spawns
# at sea) so the default picker order (Forest Tribe, Mountain Tribe, ...) actually starts
# tribes in the biome their name implies. Coordinates match backend/world.py's geography:
# mountains in the northwest, forest along the north/east, plains in the south, and a
# river cutting from the highlands down to the eastern coast.
#
# Each point is also chosen to sit within real reach of fresh river water (roughly
# 11-13 tiles by nearest_water, verified directly rather than eyeballed) -- the original
# points were picked purely to land in the right-named biome and turned out to be 36-42
# tiles from any river, well beyond what a 3-day/speed-6 scouting expedition can ever
# reach (see config.EXPEDITION_MAX_DAYS/EXPEDITION_SPEED). No amount of good in-fiction
# reasoning could have found water from there; real settlements cluster near water for
# the same reason, so this is a world-geography fix, not a difficulty adjustment.
SPAWN_POINTS = [(80, 38), (25, 34), (50, 55), (40, 37)]
COLORS = ["#c084fc", "#fb923c", "#34d399", "#60a5fa"]


class Tribe:
    def __init__(
        self, tribe_id: str, name: str, model: str, x: int, y: int, color: str,
        event_log: RunEventLog | None = None,
    ):
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
        self.history: list[str] = TribeHistory(name, event_log)
        self.memory = TribeMemory(tribe_id)
        self.founded_city = False
        self.extinct = False
        self.chief_name = ""
        self.chief_philosophy = ""
        self.chief_decree = ""
        # Lifetime counters for backend/scoreboard.py -- what an evaluator actually
        # wants to compare across models isn't just "did it survive," it's how it got
        # there: how often it needed a new leader, how often scouting actually paid
        # off, how it fared in conflict.
        self.max_population = self.population
        self.chiefs_elected = 0
        self.chief_deaths = 0
        self.expeditions_launched = 0
        self.expeditions_succeeded = 0
        # Split from expeditions_succeeded above so a scouting milestone ("Master
        # Pathfinder") and a hunting milestone ("Master Hunter") can be tracked and
        # credited to the specific person who earned them separately -- see
        # Simulation._award_trophy's `individual` param.
        self.scout_successes = 0
        self.hunt_successes = 0
        self.raids_won = 0
        self.raids_lost = 0
        self.raids_defended = 0
        self.trades_completed = 0
        # Credited to whichever chief is in power the moment each is first earned --
        # see Simulation._check_chief_trophies. [{"name", "chief", "cycle"}, ...]
        self.trophies: list[dict] = []
        # Set by actions.py._breed, resolved (an async LLM call -- see backend/
        # breeding.py) in the same Simulation.step() that set it, same pattern as
        # pending_chief_context/_install_chief. {"parent_a", "parent_b"} or None.
        self.pending_birth: dict | None = None
        # A real parent record per child, not just an anonymous population+1 -- the
        # "capture lineage" requirement. [{"child_name", "parents", "cycle", "note"}, ...]
        self.lineage: list[dict] = []
        # A fact for the next chief election to reason about, set when something more
        # specific than "just founded" is true (currently only a raid-conquest merge,
        # see Simulation._merge_tribes) -- consumed and cleared by _install_chief so an
        # election it's informing isn't indistinguishable from an ordinary founding.
        self.pending_chief_context: str = ""
        # Empty when no party is out; otherwise a list of {"pos", "origin", "target",
        # "day", "phase" ("outbound"/"returning"), "found", "terrain_report"} dicts --
        # see actions.py._scout/_hunting_party and Simulation._advance_expeditions. A
        # tribe can run more than one party at once (up to config.MAX_CONCURRENT_
        # EXPEDITIONS), any mix of scouting and hunting.
        self.expeditions: list[dict] = []

    def to_dict(self) -> dict:
        era_label = next((e.label for e in ERAS if e.key == self.era), self.era)
        survival_warning, _ = survival_bias_string(self.food, self.water)
        nxt = next_era(self.era)
        next_era_info = None
        if nxt is not None:
            next_era_info = {
                "label": nxt.label,
                "requires_population": nxt.requires_population,
                "requires_resources": nxt.requires_resources,
            }
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
            "trophies": self.trophies,
            "lineage": self.lineage,
            "next_era": next_era_info,
            "expeditions": [
                {
                    "kind": exp.get("kind", "scout"),
                    "pos": exp["pos"],
                    "day": exp["day"],
                    "max_days": exp["max_days"],
                    "phase": exp["phase"],
                    "lead_scout": exp["lead_scout"],
                    "food_gathered": exp["food_gathered"],
                    "water_gathered": exp["water_gathered"],
                    "path": exp["path"],
                }
                for exp in self.expeditions
            ],
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
        self.event_log = RunEventLog()
        # Shared identifier with the event log's own filename (e.g. "run_20260830_
        # 113014") so a full per-cycle board-state row (backend/board_history.py) and
        # this run's narrative chronicle can be cross-referenced by the same id.
        self.run_id = self.event_log.path.stem
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
            self.tribes[tid] = Tribe(tid, cfg["name"], cfg["model"], x, y, COLORS[i % len(COLORS)], self.event_log)
        self.cycle = 0
        self.paused = False
        self.status = "OPERATIONAL"
        self.game_over = False
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
        water, the election is told only that water hasn't been confirmed nearby (no
        coordinates -- see leadership.py's docstring for why) and may decide -- its own
        call, not ours -- to decree that finding some is a priority. Specifically fresh
        water, not "water" generally: a tribe standing on the coast still has no drinking
        water, so ocean doesn't count as already solved. What else the ocean might be
        good for is left entirely open."""
        water_needed = self.world.biome(tribe.x, tribe.y) not in ("river", "lake")
        context = tribe.pending_chief_context
        tribe.pending_chief_context = ""  # consumed once, whether or not this election uses it

        result = await elect_chief(self.client, tribe.model, tribe.name, water_needed, context)
        tribe.chief_name = result.get("chief_name", "")
        tribe.chief_philosophy = result.get("guiding_philosophy", "")
        victory = result.get("victory_method", "")
        if tribe.chief_name:
            tribe.chiefs_elected += 1
            note = f"{tribe.chief_name} has become chief"
            tribe.history.append(f"{note} ({victory})." if victory else f"{note}.")

        water_decision = result.get("water_decision") or {}
        if water_needed and water_decision.get("decreed"):
            tribe.chief_decree = "prioritize dispatching scouts to find reliable water"
            reason = water_decision.get("reason", "")
            entry = f"Chief {tribe.chief_name} decrees: {tribe.chief_decree}"
            tribe.history.append(f"{entry} ({reason})." if reason else f"{entry}.")

    async def _resolve_birth(self, tribe: "Tribe") -> None:
        """Resolves a BREED action's pending_birth (see actions.py._breed) with a
        real, non-scripted LLM call (backend/breeding.py) -- same pattern as
        pending_chief_context/_install_chief: the mechanical decision (two named
        people are starting a family) happens synchronously in the action handler,
        the actual outcome is generated here, in the same cycle."""
        parent_a = tribe.pending_birth["parent_a"]
        parent_b = tribe.pending_birth["parent_b"]
        tribe.pending_birth = None

        result = await breed_individuals(self.client, tribe.model, tribe.name, parent_a, parent_b)
        child_name = result.get("child_name") or f"child of {parent_a} and {parent_b}"
        note = result.get("note", "")

        tribe.population += 1
        tribe.max_population = max(tribe.max_population, tribe.population)
        tribe.lineage.append({
            "child_name": child_name, "parents": [parent_a, parent_b], "cycle": self.cycle,
        })
        entry = f"{parent_a} and {parent_b} welcome a child, {child_name}"
        tribe.history.append(f"{entry} -- {note}" if note else f"{entry}.")

    async def _run_night_cycle(self, tribe: "Tribe") -> None:
        """The "night cycle" (backend/reflection.py): a larger reviewing model looks
        back at this tribe's own recent history and decides for itself whether its
        guiding philosophy should change. Runs far less often than a live turn (see
        config.NIGHT_CYCLE_EVERY_N_CYCLES) and with a different, larger model than
        whatever the tribe plays live with -- the piece from the original design
        transcript that gives a tribe's own accumulated experience a chance to
        compound into wisdom over time, distinct from breed()/breed_individuals'
        cross-tribe/cross-individual crossover."""
        recent_events = list(tribe.history)[-config.NIGHT_CYCLE_HISTORY_WINDOW:]
        result = await reflect_on_history(
            self.client, config.NIGHT_CYCLE_REVIEWER_MODEL, tribe.name,
            tribe.chief_philosophy, recent_events,
        )
        if not result.get("changed"):
            return
        old_philosophy = tribe.chief_philosophy
        tribe.chief_philosophy = result.get("revised_philosophy", old_philosophy)
        reasoning = result.get("reasoning", "")
        entry = f"Reflecting on recent events, Chief {tribe.chief_name} reconsiders the tribe's philosophy: {tribe.chief_philosophy}"
        tribe.history.append(f"{entry} ({reasoning})." if reasoning else f"{entry}.")

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
        tribe = Tribe(tid, name, model, x, y, color, self.event_log)

        guard = HardwareVRAMBoundaryGuard(self.client.base_url, config.VRAM_LIMIT_GB)
        ok, warning = await guard.verify_vram_safety_margin(model)
        if not ok:
            tribe.history.append(f"VRAM WARNING: {warning}")
        await self._install_chief(tribe)

        self.tribes[tid] = tribe
        # A fresh tribe means the game isn't over anymore, even if every previous tribe
        # died -- undoes _trigger_game_over's stop/unload so stepping resumes.
        self.game_over = False
        if self.status == "GAME OVER":
            self.status = "OPERATIONAL"
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
            "trails": [{"x": x, "y": y, "wear": wear} for (x, y), wear in self.world.trails.items()],
            "linguistic_consensus": consensus,
        }

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    async def step(self) -> None:
        if self.paused or self.game_over:
            return
        self.cycle += 1
        self.event_log.current_cycle = self.cycle
        self.translation.decay()
        self.world.regenerate(config.DEPLETION_REGEN_PER_CYCLE)
        self.world.decay_trails(config.TRAIL_DECAY_PER_CYCLE)

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
            self._check_chief_trophies(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.expeditions:
                self._advance_expeditions(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.pending_birth:
                await self._resolve_birth(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and not tribe.chief_name:
                await self._install_chief(tribe)

        if self.cycle % config.NIGHT_CYCLE_EVERY_N_CYCLES == 0:
            for tribe in self.tribes.values():
                if not tribe.extinct and tribe.chief_name:
                    await self._run_night_cycle(tribe)

        if self.tribes and all(tribe.extinct for tribe in self.tribes.values()):
            await self._trigger_game_over()

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

    def _build_visible_entities(self, tribe: Tribe, biome: str, nearby: list[dict],
                                 memories: list[dict], available_actions: list[str]) -> list[str]:
        visible_entities = [f"structure:{s['type']}@({s['x']},{s['y']})" for s in nearby]
        visible_entities += [f"memory(cycle {m['cycle']}): {m['text']}" for m in memories]
        # taboos accumulates for a tribe's whole lifetime (see TribeMemory.consolidate,
        # which can add up to 3 more every MEMORY_CONSOLIDATE_EVERY_N_CYCLES) -- slicing
        # the first 3 meant that once any 3 existed, nothing learned later ever surfaced
        # again, however important (e.g. a hard-won confirmed water location, discovered
        # after an early wolf-attack warning already claimed those 3 slots). The most
        # recently learned facts are shown instead, so new knowledge isn't permanently
        # buried by old.
        visible_entities += [f"taboo: {t}" for t in tribe.memory.taboos[-3:]]
        # Factual telemetry about this exact tile, not a suggestion to move -- what the
        # tribe does with the information is entirely its own reasoning.
        for resource in ("wood", "stone", "water", "game"):
            level = self.world.scarcity(resource, tribe.x, tribe.y)
            if level > 0:
                visible_entities.append(f"local {resource} scarcity here: {level:.0%}")

        # Resource scarcity above only ever reports *past* depletion -- a tribe standing
        # on a pristine tile gets no signal that game is even present before it's already
        # hunted some. This is a real, occasional sighting instead: scan a small radius
        # (game can be heard/spotted nearby, not just underfoot) and roll a chance scaled
        # by the richest nearby tile's own game yield -- a mountain or ocean tile is
        # essentially silent, a forest is the likeliest place to hear something. Named for
        # whichever hunting action is actually unlocked (GAME_SPECIES_LABEL), so the fact
        # and the action always agree on what's actually out there.
        hunting_action = next((a for a in available_actions if a in GAME_SPECIES_LABEL), None)
        if hunting_action:
            best_multiplier = 0.0
            for dx in range(-config.GAME_SIGHTING_RADIUS, config.GAME_SIGHTING_RADIUS + 1):
                for dy in range(-config.GAME_SIGHTING_RADIUS, config.GAME_SIGHTING_RADIUS + 1):
                    nearby_biome = self.world.biome(tribe.x + dx, tribe.y + dy)
                    best_multiplier = max(best_multiplier, BIOME_YIELD_MULTIPLIER["game"].get(nearby_biome, 0.0))
            if best_multiplier > 0 and random.random() < config.GAME_SIGHTING_CHANCE_BASE * best_multiplier:
                species = GAME_SPECIES_LABEL[hunting_action]
                visible_entities.append(f"wildlife sighting: signs of {species} nearby")

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
        return visible_entities

    def _prepare_turn(self, tribe: Tribe) -> tuple[dict, dict]:
        """Builds this tribe's prompt with no network calls; returns (request, context)."""
        biome = self.world.biome(tribe.x, tribe.y)
        nearby = self.world.nearby_structures(tribe.x, tribe.y)
        ghost_bias = self.trauma.bias_string(tribe.x, tribe.y)
        survival_bias, survival_critical = survival_bias_string(tribe.food, tribe.water)
        memories = tribe.memory.recall(f"{biome} at {tribe.x},{tribe.y}")
        available_actions = sorted(unlocked_actions_through(tribe.era))

        visible_entities = self._build_visible_entities(tribe, biome, nearby, memories, available_actions)

        # Stated as fact (what you previously chose), not as an instruction to continue --
        # whether to keep going or change course is left entirely to the model.
        journey_note = ""
        if tribe.last_target and tribe.last_target != [tribe.x, tribe.y]:
            journey_note = (
                f"Last cycle you set target_vector to ({tribe.last_target[0]}, {tribe.last_target[1]}); "
                "you have not yet arrived there."
            )
        if tribe.expeditions:
            party_word = {"scout": "scouts", "hunt": "a hunting party"}
            reports = "; ".join(
                f"{party_word.get(exp.get('kind'), 'a party')} led by {exp['lead_scout']} "
                f"(day {exp['day']}/{exp['max_days']}, {exp['phase']})"
                for exp in tribe.expeditions
            )
            slots_left = config.MAX_CONCURRENT_EXPEDITIONS - len(tribe.expeditions)
            capacity_note = (
                " No one left to send out until one returns."
                if slots_left <= 0
                else f" You could send out {slots_left} more at once."
            )
            journey_note += f" Still in the field: {reports}.{capacity_note}"

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
            tribe.name, tribe.model, tribe.chief_name, tribe.chief_philosophy, tribe.chief_decree,
            tuple(available_actions),
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

    async def _trigger_game_over(self) -> None:
        """Every tribe in this session has gone extinct -- there will be no more turns,
        ever, for any model this session used. Rather than let step() keep getting
        called every tick forever (harmless but pointless once requests is always
        empty) and leave every model sitting loaded in Ollama until its keep_alive
        window expires on its own, stop stepping and unload them immediately. A fresh
        ADD_TRIBE clears this back to normal (see add_tribe)."""
        self.game_over = True
        self.status = "GAME OVER"
        models = {tribe.model for tribe in self.tribes.values()}
        await asyncio.gather(*(self.client.unload_model(model) for model in models), return_exceptions=True)

    def _advance_expeditions(self, tribe: Tribe) -> None:
        """Advances every one of a tribe's in-field parties by one day (see
        actions.py._scout/_hunting_party) -- a tribe can have up to config.
        MAX_CONCURRENT_EXPEDITIONS out at once. Iterates a snapshot of the list since
        a party can complete (and remove itself) mid-loop."""
        for exp in list(tribe.expeditions):
            if self._advance_one_expedition(tribe, exp):
                tribe.expeditions.remove(exp)

    def _advance_one_expedition(self, tribe: Tribe, exp: dict) -> bool:
        """One day of an in-progress expedition. Runs every cycle regardless of what
        action the tribe chose that turn -- the party is out in the field on its own,
        not waiting for the tribe's attention each cycle. Outbound: walk toward target,
        succeeding immediately on real fresh water or on reaching the destination, or
        giving up after EXPEDITION_MAX_DAYS. Returning: walk back toward camp; arrival
        is the only moment a finding becomes real, actionable knowledge (memory +
        chronicle) -- a party that hasn't made it home yet knows something the tribe as
        a whole does not. Returns True once this expedition is over and should be
        removed from tribe.expeditions.

        Wears (and benefits from) the same worn-trail mechanic as RELOCATE: a route
        used by enough expeditions gets faster over time, so a destination just out of
        one expedition's EXPEDITION_MAX_DAYS reach can become reachable a few attempts
        later purely by repeatedly trying the same path -- effort compounding into
        infrastructure, not a scripted distance override."""
        if exp["phase"] == "outbound":
            exp["day"] += 1
            px, py = exp["pos"]
            tx, ty = exp["target"]
            bonus = self.world.trail_speed_bonus(px, py, config.MAX_TRAIL_BONUS_SPEED)
            base_speed = config.EXPEDITION_SPEED + bonus
            nx, ny = physics.terrain_aware_step(px, py, tx, ty, base_speed=base_speed)
            self.world.wear_trail(nx, ny, config.TRAIL_WEAR_PER_PASS)
            exp["pos"] = [nx, ny]
            exp["path"].append([nx, ny])
            exp["food_gathered"] += config.EXPEDITION_OUTBOUND_DAILY_FOOD
            exp["water_gathered"] += config.EXPEDITION_OUTBOUND_DAILY_WATER
            reached_biome = biome_at(nx, ny)
            scout = exp["lead_scout"]

            if exp.get("kind") == "hunt":
                self._advance_hunting_party_outbound(tribe, exp, reached_biome, scout)
                return False
            if reached_biome in ("river", "lake"):
                self._expedition_river_hazard(tribe, nx, ny)  # a no-op on a lake tile -- no current to drown in
                exp["found"] = [nx, ny]
                exp["phase"] = "returning"
                tribe.history.append(f"{scout}'s party has found fresh water and is heading home to report it")
            elif [nx, ny] == [tx, ty] and exp["terrain_report"] is None and exp["day"] < exp["max_days"]:
                # Reached wherever the tribe told them to look, but the model's own
                # target_vector is usually close (a single EXPEDITION_SPEED step), so
                # treating "arrived" as "search over" meant max_days and the scout's
                # determination trait almost never actually mattered -- the party
                # turned back on day 1 nearly every time. Note what's here (still
                # useful information) but push onward along the same heading out to
                # the edge of the known world instead, using whatever days remain
                # rather than stopping the instant the declared spot is reached.
                exp["terrain_report"] = reached_biome
                ex, ey = physics.extend_ray_to_grid_edge(exp["origin"][0], exp["origin"][1], tx, ty, self.world.grid_size)
                exp["target"] = [ex, ey]
                label = BIOME_LABELS.get(reached_biome, reached_biome)
                tribe.history.append(f"{scout}'s party passed through ({nx},{ny}), {label}, and pushes onward")
            elif exp["day"] >= exp["max_days"]:
                exp["phase"] = "returning"
                # The party's own survival comes before the search -- a scout's own
                # max_days (varied by their determination trait, see actions.py
                # ._generate_scout) is the mechanical expression of that: give up and
                # come home safely rather than push on indefinitely chasing a find
                # that isn't there.
                tribe.history.append(f"{scout} calls off the search after {exp['max_days']} days -- the party's safety comes first, and they turn back")
            return False
        else:  # returning
            px, py = exp["pos"]
            ox, oy = exp["origin"]
            bonus = self.world.trail_speed_bonus(px, py, config.MAX_TRAIL_BONUS_SPEED)
            base_speed = config.EXPEDITION_SPEED + bonus
            nx, ny = physics.terrain_aware_step(px, py, ox, oy, base_speed=base_speed)
            self.world.wear_trail(nx, ny, config.TRAIL_WEAR_PER_PASS)
            exp["pos"] = [nx, ny]
            exp["path"].append([nx, ny])
            exp["food_gathered"] += config.EXPEDITION_RETURN_DAILY_FOOD
            exp["water_gathered"] += config.EXPEDITION_RETURN_DAILY_WATER
            self._expedition_river_hazard(tribe, nx, ny)
            if [nx, ny] == [ox, oy]:
                # Whatever was foraged along the way comes home regardless of whether the
                # expedition succeeded -- the trip cost real time either way, so it isn't
                # a total loss on a failed search. The findings themselves only become
                # real, actionable knowledge for the tribe at this exact moment.
                tribe.food += exp["food_gathered"]
                tribe.water += exp["water_gathered"]
                scout = exp["lead_scout"]
                forage_note = f"bringing back {exp['food_gathered']} food and {exp['water_gathered']} water foraged along the way"
                recipient = f"Chief {tribe.chief_name}" if tribe.chief_name else "the tribe"

                if exp.get("kind") == "hunt":
                    self._report_hunting_party_home(tribe, exp, scout, forage_note, recipient)
                    return True

                if exp["found"]:
                    fx, fy = exp["found"]
                    tribe.expeditions_succeeded += 1
                    tribe.scout_successes += 1
                    self._award_trophy(tribe, "Water Bringer")
                    if tribe.scout_successes == config.MILESTONE_SCOUT_SUCCESSES:
                        self._award_trophy(tribe, "Master Pathfinder", individual=scout)
                    tribe.memory.remember(f"Scouts confirmed fresh water at ({fx},{fy}).", self.cycle, weight=0.9)
                    tribe.history.append(
                        f"{scout} is home and gives {recipient} a full report: "
                        f"fresh water confirmed at ({fx},{fy}), {forage_note}"
                    )
                elif exp["terrain_report"]:
                    label = BIOME_LABELS.get(exp["terrain_report"], exp["terrain_report"])
                    tx, ty = exp["target"]
                    tribe.memory.remember(f"Scouts explored toward ({tx},{ty}) and found {label} terrain.", self.cycle, weight=0.6)
                    tribe.history.append(
                        f"{scout} is home and gives {recipient} a full report: "
                        f"{label} terrain at ({tx},{ty}), {forage_note}"
                    )
                else:
                    tribe.history.append(
                        f"{scout} is home and gives {recipient} a full report: "
                        f"nothing new found, though not empty-handed -- {forage_note}"
                    )
                return True
            return False

    def _expedition_river_hazard(self, tribe: Tribe, x: int, y: int) -> bool:
        """The same drowning risk GATHER_WATER already carries on a river tile
        (config.DROWNING_HAZARD_CHANCE) -- a traveling party crossing or camped on a
        river isn't any safer than a tribe standing on one to fill jugs. Returns True
        if it claimed someone (population loss and trauma already applied).

        Losing someone this way also becomes a real, remembered lesson, not just a
        chronicle line that scrolls away: a high-weight memory entry (see
        TribeMemory.consolidate) is what actually promotes into a standing taboo the
        tribe's own reasoning sees every future turn -- the survivors telling the rest
        of the tribe what happened, not the simulation warning them directly."""
        if biome_at(x, y) != "river" or random.random() >= config.DROWNING_HAZARD_CHANCE:
            return False
        self.trauma.radiate_event_wave(x, y, config.DROWNING_TRAUMA_MAGNITUDE, config.DROWNING_TRAUMA_RADIUS)
        self._lose_population(tribe, config.DROWNING_HAZARD_POPULATION_LOSS, cause="drowning")
        tribe.history.append("the river's current pulled someone under while the party was crossing -- the survivors turn back to warn the others")
        tribe.memory.remember(
            f"A river crossing near ({x},{y}) drowned one of our own -- the current there is a real danger.",
            self.cycle, weight=0.85,
        )
        return True

    def _advance_hunting_party_outbound(self, tribe: Tribe, exp: dict, current_biome: str, scout: str) -> None:
        """One outbound day for a HUNTING_PARTY expedition (see actions.py._hunting_party).
        Every day out is its own roll of the same wolf-pack hazard an instant hunt
        carries, and its own chance of a catch scaled by wherever the party currently
        stands' real game yield -- a party camped on a mountain or ocean tile is no
        likelier to succeed there than an instant hunt would be. A hazard or a catch
        both end the search immediately; the catch itself still isn't real food until
        the party makes it all the way home (see _report_hunting_party_home)."""
        px, py = exp["pos"]
        if self._expedition_river_hazard(tribe, px, py):
            exp["phase"] = "returning"
            return

        if random.random() < config.HUNT_HAZARD_CHANCE:
            self._lose_population(tribe, config.HUNT_HAZARD_POPULATION_LOSS, cause="wolf_attack")
            exp["phase"] = "returning"
            tribe.history.append(f"a wolf pack struck {scout}'s hunting party -- the survivors turn back")
            return

        game_multiplier = BIOME_YIELD_MULTIPLIER["game"].get(current_biome, 0.0)
        if game_multiplier > 0 and random.random() < config.HUNTING_PARTY_CATCH_CHANCE_BASE * game_multiplier:
            exp["food_caught"] = random.randint(config.HUNTING_PARTY_CATCH_FOOD_MIN, config.HUNTING_PARTY_CATCH_FOOD_MAX)
            exp["phase"] = "returning"
            tribe.history.append(f"{scout}'s hunting party made a catch and is heading home")
            return

        if exp["day"] >= exp["max_days"]:
            exp["phase"] = "returning"
            tribe.history.append(
                f"{scout} calls off the hunt after {exp['max_days']} days with nothing caught -- the party turns back"
            )

    def _report_hunting_party_home(self, tribe: Tribe, exp: dict, scout: str, forage_note: str, recipient: str) -> None:
        caught = exp.get("food_caught", 0)
        if caught:
            tribe.food += caught
            tribe.expeditions_succeeded += 1
            tribe.hunt_successes += 1
            if tribe.hunt_successes == config.MILESTONE_HUNT_SUCCESSES:
                self._award_trophy(tribe, "Master Hunter", individual=scout)
            tribe.history.append(
                f"{scout}'s hunting party is home and gives {recipient} a full report: "
                f"{caught} food caught, {forage_note}"
            )
        else:
            tribe.history.append(
                f"{scout}'s hunting party is home and gives {recipient} a full report: "
                f"nothing caught, though not empty-handed -- {forage_note}"
            )

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

    def _lose_population(self, tribe: Tribe, amount: int, cause: str = "unknown") -> None:
        """The single place population ever decreases. A tribe can now actually go
        extinct (population 0) rather than being propped up at a permanent
        population-1 floor -- extinction is marked, announced, and radiates a much
        larger trauma event than an ordinary death. `cause` feeds the scoreboard
        record (backend/scoreboard.py) so a benchmark can distinguish "starved" from
        "lost a raid," not just "died."

        Any loss also carries a chance of claiming the chief specifically, once a
        tribe survives it -- a chief was previously permanent flavor text no matter
        what happened to the people underneath them. A tribe that survives a chief's
        death gets a genuine leadership vacuum until Simulation.step() runs a fresh
        succession contest, not a name that just silently stays put forever."""
        if tribe.extinct:
            return
        tribe.population = max(0, tribe.population - amount)
        if tribe.population == 0:
            tribe.extinct = True
            tribe.history.append(f"{tribe.name} has gone extinct.")
            self.trauma.radiate_event_wave(
                tribe.x, tribe.y, config.EXTINCTION_TRAUMA_MAGNITUDE, config.EXTINCTION_TRAUMA_RADIUS
            )
            record_tribe_result(tribe, cause=cause, cycles_survived=self.cycle)
        elif tribe.chief_name and random.random() < config.CHIEF_DEATH_CHANCE_ON_LOSS:
            fallen = tribe.chief_name
            tribe.chief_deaths += 1
            tribe.chief_name = ""
            tribe.chief_philosophy = ""
            tribe.chief_decree = ""
            tribe.history.append(f"Chief {fallen} has died. {tribe.name} is left without a leader.")

    def _starve(self, tribe: Tribe) -> None:
        tribe.history.append("starvation claimed lives")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.STARVATION_TRAUMA_MAGNITUDE, config.STARVATION_TRAUMA_RADIUS
        )
        self._lose_population(tribe, config.STARVATION_POPULATION_LOSS, cause="starvation")

    def _dehydrate(self, tribe: Tribe) -> None:
        tribe.history.append("thirst claimed lives")
        self.trauma.radiate_event_wave(
            tribe.x, tribe.y, config.DEHYDRATION_TRAUMA_MAGNITUDE, config.DEHYDRATION_TRAUMA_RADIUS
        )
        self._lose_population(tribe, config.DEHYDRATION_POPULATION_LOSS, cause="thirst")

    def _grow_population(self, tribe: Tribe) -> None:
        if tribe.food > config.POPULATION_GROWTH_FOOD_THRESHOLD and tribe.population < config.POPULATION_GROWTH_CAP:
            tribe.population += 1
            tribe.food -= config.POPULATION_GROWTH_FOOD_COST
        tribe.max_population = max(tribe.max_population, tribe.population)

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

    def _award_trophy(self, tribe: Tribe, name: str, individual: str | None = None) -> None:
        """`individual`, when given, credits a specific named person (e.g. the scout or
        hunter who actually earned a milestone trophy) instead of the chief -- the
        dict's "chief" key is kept for backward compatibility (existing tests/scoreboard
        data read it) even though it isn't always literally the chief anymore."""
        if any(t["name"] == name for t in tribe.trophies):
            return  # once per tribe's lifetime
        credited = individual or tribe.chief_name or "an unknown chief"
        tribe.trophies.append({"name": name, "chief": credited, "cycle": self.cycle})
        tribe.history.append(f"\U0001f3c6 {credited} earns the '{name}' trophy for {tribe.name}!")

    def _check_chief_trophies(self, tribe: Tribe) -> None:
        """A lightweight legacy system credited to whichever chief is in power the
        moment each is first earned -- 'Water Bringer' is deliberately the standout,
        since reliable water access is the single hardest survival problem this
        simulation poses."""
        if self.world.biome(tribe.x, tribe.y) in ("river", "lake"):
            self._award_trophy(tribe, "Water Bringer")
        if tribe.food >= config.FOOD_TROPHY_THRESHOLD:
            self._award_trophy(tribe, "Well Fed")
        if tribe.population > 8:
            self._award_trophy(tribe, "Growing Legacy")

    def _merge_tribes(self, attacker: Tribe, defender: Tribe) -> str:
        """A defender's population has been driven to zero by accumulated raid
        losses (actions.py._raid transfers population rather than just destroying
        it) -- their survivors, resources, and remaining history become a new, more
        advanced entity instead of simply disappearing into extinction. Mutates
        `attacker` in place (rather than constructing a fresh Tribe and swapping it
        into self.tribes) so every reference the calling turn already holds --
        Simulation._apply_turn keeps mutating `tribe` after this action handler
        returns -- keeps pointing at the right object. Chief-less on completion; the
        same per-cycle succession check that already handles a fallen chief's
        replacement (see step()) picks this up automatically next cycle, exactly like
        a founding election -- no special-casing needed, and it still runs the
        model's own reasoning about where reliable water actually is."""
        old_name = attacker.name
        attacker.name = f"{old_name} (Advanced)"
        attacker.wood += defender.wood
        attacker.stone += defender.stone
        attacker.food += defender.food
        attacker.water += defender.water
        attacker.population += defender.population
        defender.population = 0
        attacker.max_population = max(attacker.max_population, attacker.population)
        if era_index(defender.era) > era_index(attacker.era):
            attacker.era = defender.era
        attacker.chief_name = ""
        attacker.chief_philosophy = ""
        attacker.chief_decree = ""
        attacker.pending_chief_context = (
            f"This tribe was just formed when {old_name} triumphed in battle over {defender.name} "
            f"and absorbed their surviving population. The new chief inherits a people freshly "
            f"unified by conquest, not a tribe with a long shared history."
        )
        defender.extinct = True
        record_tribe_result(defender, cause="absorbed", cycles_survived=self.cycle)
        attacker.history.append(
            f"{old_name} has fully absorbed {defender.name}'s survivors and become {attacker.name}!"
        )
        del self.tribes[defender.id]
        return attacker.name
