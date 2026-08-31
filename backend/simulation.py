import asyncio
import importlib
import math
import random

from . import config, physics
from .actions import (
    ACTION_REGISTRY, BIOME_YIELD_MULTIPLIER, GAME_SPECIES_BY_BIOME, GAME_SPECIES_LABEL,
    _eligible_breeding_pair, expedition_capacity,
)
from .ancestral_matrix import AncestralTraumaMatrix
from .breeding import breed_individuals
from .reflection import AWARD_CATEGORIES, reflect_on_history
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
#
# The Mountain Tribe point moved a second time: (25, 34) sat one tile from the river
# where it cuts through the range's original northern corner, so "confirmed nearby" was
# never a real test of scouting. Now that the range runs much further south (see
# world.MOUNTAIN_Y_END), (18, 43) sits along its eastern/grassy edge, ~12 tiles from the
# river by nearest_water -- back in the same 11-13 tile band the other spawns already
# use, but now an actual discovery instead of a freebie.
SPAWN_POINTS = [(80, 38), (18, 43), (50, 55), (40, 37)]
COLORS = ["#c084fc", "#fb923c", "#34d399", "#60a5fa"]


def _compass_direction(dx: float, dy: float) -> str:
    """An 8-point compass label for a (dx, dy) offset -- used for a distant rival
    sighting (see config.RIVAL_DISTANT_SIGHTING_RADIUS), where only a rough heading is
    plausible, not exact coordinates. y increases southward on this map (matching
    world.py's own north/south framing -- the mountain and forest bands sit at low y),
    so dy > 0 is south, dy < 0 is north."""
    angle = math.degrees(math.atan2(dy, dx)) % 360
    directions = ("east", "southeast", "south", "southwest", "west", "northwest", "north", "northeast")
    return directions[round(angle / 45) % 8]


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
        # See Simulation._check_for_celebration -- very negative so a tribe's very
        # first celebration isn't blocked by a cooldown it never actually used yet.
        self.last_celebration_cycle: int = -config.CELEBRATION_COOLDOWN_CYCLES
        # Honors a chief has personally proposed via the night cycle (see
        # reflection.py's AWARD_CATEGORIES). Simulation._check_custom_awards hands one
        # out to whoever first earns it after it's proposed.
        self.custom_awards: list[dict] = []
        # Innate daily tribal gathering (see Simulation._hold_tribal_gathering) -- state
        # for what's changed since the last one. gathering_brief is re-surfaced into the
        # tribe's own live turn context every cycle until the next gathering overwrites
        # it (see Simulation._build_visible_entities), not just narrated into the
        # chronicle and forgotten.
        self.last_gathering_cycle = 0
        self.population_at_last_gathering = self.population
        self.gathering_brief = ""
        # A confirmed water discovery used to only ever reach the tribe's own live
        # reasoning via tribe.memory.recall(f"{biome} at {x},{y}") -- a query about the
        # tribe's *current* location, which essentially never overlaps in vocabulary
        # with "Scouts confirmed fresh water at (fx,fy)" unless the tribe happens to
        # already be standing on those exact coordinates. In practice that meant a
        # hard-won discovery vanished from the model's own context the moment the
        # one-time chronicle line scrolled by, leaving RELOCATE's target_vector an
        # ungrounded guess even right after a successful scout. This persists real
        # confirmed sites (deduped, most recent last) so _build_visible_entities can
        # keep surfacing them the same durable way it does taboos -- and the frontend
        # can mark them permanently on the map (see index.html's drawLandmarks).
        #
        # lumber_sites/wildlife_sites/quarry_sites are the same idea for a scout's
        # terrain_report: set only when the reported biome is one BIOME_YIELD_MULTIPLIER
        # already marks as maxed out (1.0) for a resource -- forest for both wood and
        # game, mountains for stone -- a real, already-measured "considerable cluster",
        # not an arbitrary new threshold invented just for this.
        self.confirmed_water_sites: list[tuple[int, int]] = []
        self.lumber_sites: list[tuple[int, int]] = []
        self.wildlife_sites: list[tuple[int, int]] = []
        self.quarry_sites: list[tuple[int, int]] = []
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
        survival_warning, _ = survival_bias_string(self.food, self.water, self.population)
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
            "custom_awards": self.custom_awards,
            "confirmed_water_sites": self.confirmed_water_sites,
            "lumber_sites": self.lumber_sites,
            "wildlife_sites": self.wildlife_sites,
            "quarry_sites": self.quarry_sites,
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
    def __init__(
        self, tribe_configs: list[dict], ollama_url: str = config.OLLAMA_URL,
        immortality_cycles: int = 0,
    ):
        if not tribe_configs:
            raise ValueError("Simulation needs at least one tribe")
        # Opt-in, off (0) by default -- a spectator-facing mode for watching what
        # happens *after* the survival crisis (scouting maturing, trade, breeding, era
        # advancement) instead of every run getting cut off by extinction at 30-60
        # cycles before any of that plays out. See Simulation._lose_population: this
        # suppresses the actual population-loss consequence only, while self.cycle <=
        # immortality_cycles -- it never touches what a tribe's own live prompt is
        # told. The same "Your people are starving" facts, the same crisis framing,
        # the same reasoning test -- a tribe that would have gone extinct just keeps
        # facing the same real pressure with the stakes quietly held back, not a tribe
        # that's been let off the hook and knows it.
        self.immortality_cycles = immortality_cycles
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
        # A wandering storm cloud (see Simulation._advance_weather) -- world weather,
        # independent of any tribe. None when no storm is active; otherwise
        # {"x", "y", "heading", "cycles_left"}. lightning_strike is only ever set for
        # the exact cycle a strike happens (None otherwise), read by both
        # _build_visible_entities (a nearby tribe's live fact) and snapshot() (the
        # frontend's one-cycle flash).
        self.storm_cloud: dict | None = None
        self.lightning_strike: tuple[int, int] | None = None
        self.self_mod = (
            SelfModEngine(self.client, tribe_configs[0]["model"], config.SELF_MOD_COOLDOWN_CYCLES)
            if config.ENABLE_SELF_MODIFICATION
            else None
        )

    @classmethod
    async def create(
        cls, tribe_configs: list[dict], ollama_url: str = config.OLLAMA_URL,
        immortality_cycles: int = 0,
    ) -> "Simulation":
        """Preferred constructor: runs a one-time VRAM sanity check per model before
        building the simulation, and drops a warning into a tribe's chronicle (rather
        than blocking it) if its model looks too large for the configured budget."""
        guard = HardwareVRAMBoundaryGuard(ollama_url, config.VRAM_LIMIT_GB)
        warnings: dict[str, str] = {}
        for cfg in tribe_configs[: config.MAX_TRIBES]:
            ok, warning = await guard.verify_vram_safety_margin(cfg["model"])
            if not ok:
                warnings[cfg["name"]] = warning

        sim = cls(tribe_configs, ollama_url, immortality_cycles)
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

        # A weak/small model can put a bare bool or string where a nested object was
        # asked for (seen live: {"water_decision": true} from llama3.2:1b) -- valid
        # JSON, wrong shape. `... or {}` alone doesn't catch a truthy non-dict (True or
        # {} is still True), so the isinstance check is load-bearing, not decorative.
        water_decision = result.get("water_decision")
        if not isinstance(water_decision, dict):
            water_decision = {}
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

    def _hold_tribal_gathering(self, tribe: "Tribe") -> None:
        """An innate tradition, not a chief's choice: every tribe, whatever its
        philosophy or model, gathers once per in-game day (config.DAY_LENGTH_CYCLES,
        mirroring frontend/index.html's own sun/moon cycle -- this really does land at
        the in-game dawn a spectator sees onscreen) to take stock together. Unlike the
        night cycle (backend/reflection.py -- an occasional, chief-specific
        reconsideration of philosophy that costs a real LLM call), this is a cheap,
        deterministic recap of real facts: nothing here is interpreted, so it fires
        reliably for every tribe regardless of model quality.

        Named individuals' recent achievements and any standing unclaimed honor are
        real facts Tribe.to_dict already exposes to the UI, but that the live turn
        prompt never carried back into the tribe's own reasoning -- gathering_brief,
        read by _build_visible_entities the same way a taboo is, is what closes that
        gap."""
        new_trophies = [t for t in tribe.trophies if t["cycle"] > tribe.last_gathering_cycle]
        unclaimed = [
            a for a in tribe.custom_awards
            if not any(t["name"] == a["name"] for t in tribe.trophies)
        ]
        pop_delta = tribe.population - tribe.population_at_last_gathering

        parts = []
        if new_trophies:
            parts.append("since the last gathering, " + "; ".join(
                f"{t['chief']} earned the '{t['name']}' honor" for t in new_trophies
            ))
        if unclaimed:
            parts.append("still unclaimed: " + ", ".join(f"the '{a['name']}' ({a['category']})" for a in unclaimed))
        if pop_delta > 0:
            parts.append(f"the tribe has grown by {pop_delta} since the last gathering")
        elif pop_delta < 0:
            parts.append(f"the tribe has lost {-pop_delta} since the last gathering")
        if tribe.chief_name:
            parts.append(f"Chief {tribe.chief_name}'s guiding philosophy still stands: {tribe.chief_philosophy}")

        tribe.gathering_brief = "; ".join(parts) if parts else "a quiet gathering -- nothing new to report"
        tribe.history.append(f"The tribe gathers as the sun rises. {tribe.gathering_brief}.")
        tribe.last_gathering_cycle = self.cycle
        tribe.population_at_last_gathering = tribe.population

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
        if result.get("changed"):
            old_philosophy = tribe.chief_philosophy
            tribe.chief_philosophy = result.get("revised_philosophy", old_philosophy)
            reasoning = result.get("reasoning", "")
            entry = f"Reflecting on recent events, Chief {tribe.chief_name} reconsiders the tribe's philosophy: {tribe.chief_philosophy}"
            tribe.history.append(f"{entry} ({reasoning})." if reasoning else f"{entry}.")

        # Captures the chief's own proposed honor; Simulation._check_custom_awards is
        # what actually hands it out once someone earns it (see reflection.py's
        # AWARD_CATEGORIES docstring).
        proposed = result.get("proposed_award")
        if isinstance(proposed, dict) and proposed.get("name") and proposed.get("category") in AWARD_CATEGORIES:
            if not any(a["name"] == proposed["name"] for a in tribe.custom_awards):
                tribe.custom_awards.append({
                    "name": proposed["name"], "category": proposed["category"], "cycle": self.cycle,
                })
                tribe.history.append(
                    f"Chief {tribe.chief_name} establishes a new honor, the '{proposed['name']}', "
                    f"for excellence in {proposed['category']} -- not yet awarded to anyone."
                )

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
            "immortality_cycles": self.immortality_cycles,
            "storm_cloud": {"x": self.storm_cloud["x"], "y": self.storm_cloud["y"]} if self.storm_cloud else None,
            "lightning_strike": list(self.lightning_strike) if self.lightning_strike else None,
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
        self._advance_weather()

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
            self._check_for_celebration(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.expeditions:
                self._advance_expeditions(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and tribe.pending_birth:
                await self._resolve_birth(tribe)

        for tribe in self.tribes.values():
            if not tribe.extinct and not tribe.chief_name:
                await self._install_chief(tribe)

        if self.cycle % config.DAY_LENGTH_CYCLES == 0:
            for tribe in self.tribes.values():
                if not tribe.extinct:
                    self._hold_tribal_gathering(tribe)

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

    def _advance_weather(self) -> None:
        """A wandering storm cloud, entirely independent of any tribe's actions -- the
        world has weather whether or not anyone's watching. Spawns rarely (only
        checked while no storm is already active), wanders with a heading that jitters
        a little each cycle rather than flying a dead-straight line, and expires after
        STORM_LIFESPAN_CYCLES either way. A strike is rolled once per cycle while a
        storm is active; self.lightning_strike is set only for that one cycle (cleared
        at the top of every call) so it reads as a flash, not a standing hazard.

        A tribe caught directly under a strike takes a real, small hit through the same
        _lose_population channel every other hazard uses -- immortality (see
        Simulation.__init__) protects it exactly the same way. A tribe merely within
        LIGHTNING_STRIKE_RADIUS doesn't get hurt, just a fact about what happened (see
        _build_visible_entities) -- on a forest tile, that fact echoes the same "how
        did anyone first learn fire" question raised this session. Nothing here awards
        fire automatically; it's a real, unscripted event a tribe's own reasoning could
        in principle notice and act on, the same honest test as any other fact."""
        self.lightning_strike = None

        if self.storm_cloud is None:
            if random.random() < config.STORM_SPAWN_CHANCE:
                self.storm_cloud = {
                    "x": random.randint(0, self.world.grid_size - 1),
                    "y": random.randint(0, self.world.grid_size - 1),
                    "heading": random.uniform(0, 2 * math.pi),
                    "cycles_left": config.STORM_LIFESPAN_CYCLES,
                }
            return

        cloud = self.storm_cloud
        cloud["heading"] += random.uniform(-config.STORM_HEADING_JITTER, config.STORM_HEADING_JITTER)
        cloud["x"] = max(0, min(self.world.grid_size - 1, round(cloud["x"] + math.cos(cloud["heading"]) * config.STORM_SPEED)))
        cloud["y"] = max(0, min(self.world.grid_size - 1, round(cloud["y"] + math.sin(cloud["heading"]) * config.STORM_SPEED)))
        cloud["cycles_left"] -= 1

        if random.random() < config.LIGHTNING_STRIKE_CHANCE:
            sx, sy = cloud["x"], cloud["y"]
            self.lightning_strike = (sx, sy)
            for tribe in self.tribes.values():
                if tribe.extinct or (tribe.x, tribe.y) != (sx, sy):
                    continue
                self.trauma.radiate_event_wave(sx, sy, config.LIGHTNING_TRAUMA_MAGNITUDE, config.LIGHTNING_TRAUMA_RADIUS)
                self._lose_population(tribe, config.LIGHTNING_HAZARD_POPULATION_LOSS, cause="lightning")
                tribe.history.append("lightning struck the heart of camp")

        if cloud["cycles_left"] <= 0:
            self.storm_cloud = None

    def _build_visible_entities(self, tribe: Tribe, biome: str, nearby: list[dict],
                                 memories: list[dict], available_actions: list[str]) -> list[str]:
        visible_entities = [f"structure:{s['type']}@({s['x']},{s['y']})" for s in nearby]
        visible_entities += [f"memory(cycle {m['cycle']}): {m['text']}" for m in memories]

        # A lightning strike only lasts one cycle (see _advance_weather) -- a real,
        # unscripted event, not a directive about what it means or what to do next.
        if self.lightning_strike:
            lx, ly = self.lightning_strike
            distance = ((tribe.x - lx) ** 2 + (tribe.y - ly) ** 2) ** 0.5
            if distance < 1.5:
                visible_entities.append("lightning just struck directly at your camp")
            elif distance <= config.LIGHTNING_STRIKE_RADIUS:
                if self.world.biome(lx, ly) == "forest":
                    visible_entities.append(f"lightning struck a tree near ({lx},{ly}) -- it looks like it's burning")
                else:
                    visible_entities.append(f"lightning struck nearby, at ({lx},{ly})")
        # taboos accumulates for a tribe's whole lifetime (see TribeMemory.consolidate,
        # which can add up to 3 more every MEMORY_CONSOLIDATE_EVERY_N_CYCLES) -- slicing
        # the first 3 meant that once any 3 existed, nothing learned later ever surfaced
        # again, however important (e.g. a hard-won confirmed water location, discovered
        # after an early wolf-attack warning already claimed those 3 slots). The most
        # recently learned facts are shown instead, so new knowledge isn't permanently
        # buried by old.
        visible_entities += [f"taboo: {t}" for t in tribe.memory.taboos[-3:]]
        visible_entities += [f"confirmed water source at ({x},{y})" for x, y in tribe.confirmed_water_sites[-3:]]
        visible_entities += [f"confirmed lumber-rich area at ({x},{y})" for x, y in tribe.lumber_sites[-3:]]
        visible_entities += [f"confirmed wildlife-rich area at ({x},{y})" for x, y in tribe.wildlife_sites[-3:]]
        visible_entities += [f"confirmed stone-rich area at ({x},{y})" for x, y in tribe.quarry_sites[-3:]]
        if tribe.gathering_brief:
            visible_entities.append(f"this morning's gathering: {tribe.gathering_brief}")
        # Factual telemetry about this exact tile, not a suggestion to move -- what the
        # tribe does with the information is entirely its own reasoning.
        for resource in ("wood", "stone", "water", "game", "forage"):
            level = self.world.scarcity(resource, tribe.x, tribe.y)
            if level > 0:
                visible_entities.append(f"local {resource} scarcity here: {level:.0%}")

        # Resource scarcity above only ever reports *past* depletion -- a tribe standing
        # on a pristine tile gets no signal that game is even present before it's already
        # hunted some. This is a real, occasional sighting instead: scan a small radius
        # (game can be heard/spotted nearby, not just underfoot) and roll a chance scaled
        # by the richest nearby tile's own game yield -- a mountain or ocean tile is
        # essentially silent, a forest is the likeliest place to hear something. Named for
        # whichever hunting action is actually unlocked (GAME_SPECIES_LABEL) as a
        # fallback, but GAME_SPECIES_BY_BIOME gives the richest nearby tile's own biome
        # the final say when it has an entry -- deer in a forest, rabbits on the plains,
        # not the same species word regardless of where the sighting actually is.
        hunting_action = next((a for a in available_actions if a in GAME_SPECIES_LABEL), None)
        if hunting_action:
            best_multiplier = 0.0
            best_biome = None
            for dx in range(-config.GAME_SIGHTING_RADIUS, config.GAME_SIGHTING_RADIUS + 1):
                for dy in range(-config.GAME_SIGHTING_RADIUS, config.GAME_SIGHTING_RADIUS + 1):
                    nearby_biome = self.world.biome(tribe.x + dx, tribe.y + dy)
                    multiplier = BIOME_YIELD_MULTIPLIER["game"].get(nearby_biome, 0.0)
                    if multiplier > best_multiplier:
                        best_multiplier, best_biome = multiplier, nearby_biome
            if best_multiplier > 0 and random.random() < config.GAME_SIGHTING_CHANCE_BASE * best_multiplier:
                pool = GAME_SPECIES_BY_BIOME.get(best_biome, (GAME_SPECIES_LABEL[hunting_action],))
                species = random.choice(pool)
                visible_entities.append(f"wildlife sighting: signs of {species} nearby")

        # Cross-tribe proximity awareness, independent of whether the other tribe has
        # ever broadcast anything (see config.RIVAL_PRECISE_AWARENESS_RADIUS/
        # RIVAL_DISTANT_SIGHTING_RADIUS) -- without this, the default ~62-tile spawn
        # distance meant tribes had no way to ever notice each other at all, which is
        # the real reason TRADE/RAID never fired in a single run this session.
        for other in self.tribes.values():
            if other.id == tribe.id or other.extinct:
                continue
            dx, dy = other.x - tribe.x, other.y - tribe.y
            distance = math.hypot(dx, dy)
            if distance <= config.RIVAL_PRECISE_AWARENESS_RADIUS:
                visible_entities.append(
                    f"{other.name} is nearby at ({other.x},{other.y}), about {distance:.0f} tiles away"
                )
            elif distance <= config.RIVAL_DISTANT_SIGHTING_RADIUS:
                visible_entities.append(f"distant signs of {other.name} somewhere to the {_compass_direction(dx, dy)}")

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
        survival_bias, survival_critical = survival_bias_string(tribe.food, tribe.water, tribe.population)
        memories = tribe.memory.recall(f"{biome} at {tribe.x},{tribe.y}")
        available_actions = sorted(unlocked_actions_through(tribe.era))

        visible_entities = self._build_visible_entities(tribe, biome, nearby, memories, available_actions)

        # A tribe starving/dehydrating while sitting on 100+ wood or stone had a real
        # information gap: the stockpile itself never said "this is already more than
        # enough." Only surfaced alongside an actual food/water warning (survival_bias
        # non-empty) -- a fact about the mismatch, not a standing nudge to stop
        # gathering wood every other cycle too.
        if survival_bias:
            surplus = []
            if tribe.wood >= config.MATERIAL_SURPLUS_THRESHOLD:
                surplus.append(f"{tribe.wood} wood")
            if tribe.stone >= config.MATERIAL_SURPLUS_THRESHOLD:
                surplus.append(f"{tribe.stone} stone")
            if surplus:
                visible_entities.append(
                    f"Already stockpiled well beyond any near-term building need: {', '.join(surplus)}."
                )

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
            slots_left = expedition_capacity(tribe) - len(tribe.expeditions)
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
        actions.py._scout/_hunting_party) -- a tribe can have up to
        actions.expedition_capacity(tribe) out at once. Iterates a snapshot of the list
        since a party can complete (and remove itself) mid-loop."""
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
                    self._check_custom_awards(tribe, "scouting", individual=scout)
                    tribe.memory.remember(f"Scouts confirmed fresh water at ({fx},{fy}).", self.cycle, weight=0.9)
                    if (fx, fy) not in tribe.confirmed_water_sites:
                        tribe.confirmed_water_sites.append((fx, fy))
                    tribe.history.append(
                        f"{scout} is home and gives {recipient} a full report: "
                        f"fresh water confirmed at ({fx},{fy}), {forage_note}"
                    )
                elif exp["terrain_report"]:
                    label = BIOME_LABELS.get(exp["terrain_report"], exp["terrain_report"])
                    tx, ty = exp["target"]
                    tribe.memory.remember(f"Scouts explored toward ({tx},{ty}) and found {label} terrain.", self.cycle, weight=0.6)
                    if exp["terrain_report"] == "forest":
                        if (tx, ty) not in tribe.lumber_sites:
                            tribe.lumber_sites.append((tx, ty))
                        if (tx, ty) not in tribe.wildlife_sites:
                            tribe.wildlife_sites.append((tx, ty))
                    elif exp["terrain_report"] == "mountains":
                        if (tx, ty) not in tribe.quarry_sites:
                            tribe.quarry_sites.append((tx, ty))
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
            self._check_custom_awards(tribe, "hunting", individual=scout)
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
        succession contest, not a name that just silently stays put forever.

        While self.cycle <= self.immortality_cycles (see Simulation.__init__), the
        actual population change and extinction are suppressed -- the hazard/
        starvation/raid event that called this still happened (its history line,
        trauma wave, and chief-death roll below all still fire normally), only the
        body count is held back. Chief succession keeps happening during immunity on
        purpose: that's real texture (lineage, trophies, a fresh philosophy), not the
        extinction this mode exists to defer."""
        if tribe.extinct:
            return
        immune = self.cycle <= self.immortality_cycles
        if not immune:
            tribe.population = max(0, tribe.population - amount)
            if tribe.population == 0:
                tribe.extinct = True
                tribe.history.append(f"{tribe.name} has gone extinct.")
                self.trauma.radiate_event_wave(
                    tribe.x, tribe.y, config.EXTINCTION_TRAUMA_MAGNITUDE, config.EXTINCTION_TRAUMA_RADIUS
                )
                record_tribe_result(tribe, cause=cause, cycles_survived=self.cycle)
                return
        if tribe.chief_name and random.random() < config.CHIEF_DEATH_CHANCE_ON_LOSS:
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

    def _check_custom_awards(self, tribe: Tribe, category: str, individual: str | None = None) -> None:
        """The other half of the night-cycle award stub (see reflection.py's
        AWARD_CATEGORIES docstring): a chief can propose an honor of their own during
        the night cycle, but until now nothing checked a real counter and actually
        handed it out. Called from the same real-event sites that already check the
        built-in milestone trophies (a scout's confirmed water, a hunting party's
        catch, a completed trade, a won raid) -- the first genuine act of excellence
        in the proposed category after the chief establishes it becomes its first (and,
        since _award_trophy pays out once per tribe lifetime, only) recipient. An
        honest milestone tied to a specific real achievement, not an arbitrary round
        number invented just for this."""
        for award in tribe.custom_awards:
            if award["category"] == category:
                self._award_trophy(tribe, award["name"], individual=individual)

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

    def _check_for_celebration(self, tribe: Tribe) -> None:
        """Automatic and threshold-based, same pattern as era advancement/trophies/
        population growth -- not a discrete action the model has to remember to pick.
        This session's own data (BREED sat free and genuinely eligible for 20+ live
        cycles without ever being chosen) suggests these models rarely reach for a new
        discrete choice at all, so a reward gated behind choosing one more action would
        likely suffer the same fate.

        Fires on a real resource surplus (reusing FOOD_TROPHY_THRESHOLD, the same
        "Well Fed" bar) OR a genuine new discovery -- any memory entry just recorded
        this exact cycle at or above CELEBRATION_DISCOVERY_WEIGHT, the same weight that
        already promotes a memory into a permanent taboo/lesson (see TribeMemory.
        consolidate) -- i.e. this tribe's own definition of "something worth
        remembering forever," not a threshold invented just for this. Spends a real
        fraction of the surplus (the mass gathering effort), radiates real pride
        through the area same as any other proud event, and -- if two distinct named
        individuals are already eligible -- is what naturally brings them together,
        without needing the model to separately choose BREED."""
        if self.cycle - tribe.last_celebration_cycle < config.CELEBRATION_COOLDOWN_CYCLES:
            return

        surplus = tribe.food >= config.FOOD_TROPHY_THRESHOLD
        discovery = any(
            e["cycle"] == self.cycle and e["weight"] >= config.CELEBRATION_DISCOVERY_WEIGHT
            for e in tribe.memory.entries
        )
        if not surplus and not discovery:
            return

        tribe.last_celebration_cycle = self.cycle
        spent = round(tribe.food * config.CELEBRATION_RESOURCE_COST_FRACTION)
        tribe.food -= spent
        self.trauma.radiate_event_wave(tribe.x, tribe.y, config.CELEBRATION_PRIDE_MAGNITUDE, config.CELEBRATION_PRIDE_RADIUS)
        reason = "a fresh discovery" if discovery else "a season of plenty"
        tribe.history.append(f"\U0001f389 {tribe.name} holds a celebration for {reason}, spending {spent} food on a feast")

        if tribe.pending_birth is None and tribe.population < config.POPULATION_GROWTH_CAP:
            pair = _eligible_breeding_pair(tribe)
            if pair is not None:
                parent_a, parent_b = pair
                tribe.pending_birth = {"parent_a": parent_a, "parent_b": parent_b}
                tribe.history.append(f"amid the celebration, {parent_a} and {parent_b} decide to start a family together")

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
