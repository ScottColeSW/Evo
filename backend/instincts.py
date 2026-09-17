"""Immediate physiological survival pressure.

Distinct from ancestral_matrix.py's location-based bias (what happened *here*,
historically) -- this is about a tribe's own current condition: are they starving or
dehydrated *right now*, regardless of where they're standing or what happened there
before. This is the basic self-preservation signal nothing in the sim modeled until
food/water actually had upkeep consumption (see Simulation._apply_upkeep).
"""

from . import config


def survival_bias_string(
    food: int, water: int, population: int,
    fishing_learned: bool = False, cooking_learned: bool = False,
    water_secure: bool = False, food_secure: bool = False,
    kitchen_built: bool = False, long_houses_built: int = 0,
) -> tuple[str, bool]:
    """Returns (bias_text, is_critical). is_critical raises inference temperature the
    same way ancestral dread does -- panic should read as less predictable model
    output, not just differently worded prompt text.

    Thresholds scale with population rather than being a flat stockpile number -- the
    same per-cycle upkeep formula _apply_upkeep actually charges (see config.
    UPKEEP_POPULATION_DIVISOR), so "warning"/"critical" mean the same thing (a
    consistent number of cycles of real buffer left) regardless of how large the tribe
    has grown. Cooking no longer adjusts this -- see config.COOKING_FOOD_MULTIPLIER's
    own comment: cooking now multiplies food *production* at the harvest point
    (actions._food_multiplier), the same shape Sawmill/Quarry/Dock already use,
    instead of shrinking *consumption* here.

    water_secure (config.WATER_SECURITY_SITE_THRESHOLD confirmed sources, see
    Simulation._advance_water_supply) means water is always topped to the storage
    cap every cycle from here on -- a real thirst warning could still fire off pure
    numeric coincidence right after the threshold is crossed without this, which
    would flatly contradict "off the management board for good." food_secure
    (a Kitchen plus a proven Fishery or farm harvest, see Simulation.
    _advance_food_supply) is the same guarantee for the hunger half."""
    upkeep = max(1, population // config.UPKEEP_POPULATION_DIVISOR)
    urgent: list[str] = []
    critical = False

    # NUDGE (2026-08-30, explicit "nudge harder" request): this used to describe the
    # crisis without naming a specific action, on the theory that a model connecting
    # "starving" to "go hunt" on its own was more honest than scripting the outcome.
    # Naming GATHER_FOOD/HUNTING_PARTY and GATHER_WATER/SCOUT directly is still a
    # suggestion inside a fact block, not a forced action -- available_actions and the
    # model's own choice are untouched -- but it no longer pretends not to know what
    # the tribe actually needs. Revisit if this proves too heavy-handed later --
    # grep "# NUDGE" across backend/ to find every place this line was crossed.
    #
    # Explicit follow-up: "revise the 'your people are starving' messaging to be
    # more inclusive of options that would help them fix it -- add Fishing or Cook
    # in a Kitchen as additional ideas." The original three options (gather/hunt)
    # aren't the only real ways to fix a food shortage -- fishing is always a real
    # option, and cooking permanently stretches every future harvest further, not
    # just this one crisis. Only mentioned once fishing_learned/cooking_learned are
    # actually false, so a tribe that's already mastered them doesn't get told to
    # go learn something it already knows.
    #
    # Explicit report, 2026-09-13: "the 'starving' warning does not mention cooking
    # or kitchen." Confirmed live (run_20260913_080742): cooking_learned flips true
    # early (cycle ~80-95), and once it does, this whole clause went permanently
    # silent about food multipliers for the rest of a 746-cycle game -- even though
    # Kitchen (a real, still-available further 3x on top of cooking, see
    # actions._build_kitchen) remained unbuilt the entire run. Now a second rung:
    # once cooking is learned, the message keeps naming Kitchen specifically until
    # it's actually built, instead of going quiet the moment the first lever is
    # pulled.
    #
    # Explicit report, 2026-09-14 (a full 8-day/~800-cycle run): "these guys get
    # gather commands and we are warning them to 'gather' food. that's a logic leap
    # I don't think they are ready for." The comment above already claims this
    # message names GATHER_FOOD/HUNTING_PARTY "directly," but the actual text only
    # ever said the vague, collective "gather food" / "hunting" / "fishing" --
    # words that don't map 1:1 onto any one real action (HUNT_DEER vs.
    # HUNTING_PARTY are both "hunting"; is "gather food" GATHER_FOOD or one of the
    # settled-only food actions?). Translating a vague collective noun into the
    # one correct action is exactly the kind of inferential leap this project's
    # own "facts vs mechanics" pattern says a small model can't reliably make.
    #
    # Follow-up correction, 2026-09-15: the first fix over-corrected the other
    # way, spelling out the literal SCREAMING_SNAKE_CASE ACTION_REGISTRY keys
    # (GATHER_FOOD, HUNT_DEER, HUNTING_PARTY) in the middle of the sentence.
    # Explicit request: "This should be plain text, but using the 'action' words
    # specifically, not the variable names." Rewritten as ordinary lowercase
    # prose again, but with each verb now naming exactly one real action instead
    # of a collective term ("forage" -> GATHER_FOOD, "hunt deer" -> HUNT_DEER,
    # "a hunting party" -> HUNTING_PARTY, "fish"/"fishing" -> CATCH_FISH,
    # "a fire" -> BUILD_FIRE, "a kitchen" -> BUILD_KITCHEN) -- precise about
    # which action without reading like debug output. GATHER_FOOD and HUNT_DEER
    # are both unlocked from primitive_dawn (see eras.py) and never
    # settlement-gated, so naming them unconditionally is always safe;
    # CATCH_FISH stays behind the same fishing_learned gate as before (a
    # pre-existing, separate question of whether it can dangle for an unsettled
    # tribe -- not what was reported here, left alone).
    #
    # Explicit correction, 2026-09-17: "mentioning it and not giving the action to
    # perform makes it seem like we are torturing our agents." Confirmed live
    # (run_20260917_080441): this message told a tribe to "build a kitchen" for
    # 213+ straight cycles while it had zero long houses -- BUILD_KITCHEN's own
    # real prerequisite (actions._build_kitchen requires long_houses_built > 0)
    # -- so the one thing it was told would fix the crisis was never actually one
    # step away. Now names the real next step (a long house) first when that's
    # what's actually missing, instead of jumping straight to a kitchen that
    # cannot yet be built.
    if food_secure:
        pass
    elif food <= upkeep * config.HUNGER_CRITICAL_CYCLES_LEFT:
        message = "Your people are starving -- foraging for food, hunting deer, or sending out a hunting party would help right now"
        message += ", or try fishing." if not fishing_learned else "."
        if not cooking_learned:
            message += " Building a fire, then cooking after a successful hunt, would make every future harvest go much further."
        elif not long_houses_built:
            message += " A long house would be the real next step -- it's what a kitchen requires, and a kitchen would multiply every future harvest even further."
        elif not kitchen_built:
            message += " Building a kitchen would multiply every future harvest even further, on top of what cooking already does."
        urgent.append(message)
        critical = True
    elif food <= upkeep * config.HUNGER_WARNING_CYCLES_LEFT:
        message = "Food stores are running low -- foraging for food, hunting deer, or sending out a hunting party soon would help"
        message += ", or try fishing." if not fishing_learned else "."
        if not cooking_learned:
            message += " Building a fire, then cooking after a successful hunt, would help stored food last much longer too."
        elif not long_houses_built:
            message += " A long house would be the real next step toward a kitchen, which would stretch it further still."
        elif not kitchen_built:
            message += " Building a kitchen would stretch it further still."
        urgent.append(message)

    if water_secure:
        pass
    elif water <= upkeep * config.THIRST_CRITICAL_CYCLES_LEFT:
        urgent.append("Your people are dying of thirst -- gather water or dispatch scouts to find a source now.")
        critical = True
    elif water <= upkeep * config.THIRST_WARNING_CYCLES_LEFT:
        urgent.append("Water stores are running low -- gathering water or scouting for more soon would help.")

    if not urgent:
        return "", False
    # Explicit request, 2026-09-16: "consider turning off our 'nudges' slowly
    # to see where they can and cannot succeed without our guidance." Tagged
    # "survival_warning" -- see config.DISABLED_NUDGE_TAGS's own comment. Only
    # the informational TEXT is suppressed; `critical` (a real mechanical
    # effect -- see Simulation._prepare_turn's panicked/temperature branch) is
    # still returned untouched, and SURVIVAL_CRISIS_ACTIONS's own menu-
    # narrowing is computed from these same raw numbers independently of this
    # string, not from whether this text fired -- disabling the tag isolates
    # "does the fact itself help" from every mechanical consequence.
    if "survival_warning" in config.DISABLED_NUDGE_TAGS:
        return "", critical
    return "[SURVIVAL INSTINCT]: " + " ".join(urgent), critical
