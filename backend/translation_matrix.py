class TranslationConfidenceMatrix:
    """Tracks whether two tribes are converging on shared meaning for their invented
    tokens.

    There's no ground-truth "this tribe correctly guessed that token's meaning" signal
    available -- nobody, including the tribes themselves, knows what a token "really"
    means. Convergence is measured empirically instead: whenever two different tribes
    broadcast the exact same phrase while performing the same action, that's evidence
    they've landed on shared vocabulary (independently, or because one overheard and
    copied the other -- see Simulation._prepare_turn, which now feeds each tribe its
    neighbors' most recent broadcast + action). Confidence decays each cycle so a
    coincidence from 40 cycles ago doesn't count forever.
    """

    def __init__(self, decay_per_cycle: float = 0.01, reinforcement: float = 0.15):
        self.decay_per_cycle = decay_per_cycle
        self.reinforcement = reinforcement
        self._scores: dict[tuple[str, str, str], float] = {}
        self._last_usage: dict[str, tuple[str, str]] = {}  # tribe_id -> (token, action)

    def record_broadcast(self, tribe_id: str, token: str, action: str) -> None:
        if not token:
            return
        for other_id, (other_token, other_action) in self._last_usage.items():
            if other_id == tribe_id:
                continue
            if other_token == token and other_action == action:
                key = tuple(sorted((tribe_id, other_id))) + (token,)
                self._scores[key] = min(1.0, self._scores.get(key, 0.0) + self.reinforcement)
        self._last_usage[tribe_id] = (token, action)

    def decay(self) -> None:
        for key in list(self._scores):
            self._scores[key] -= self.decay_per_cycle
            if self._scores[key] <= 0.0:
                del self._scores[key]

    def stabilized_tokens(self, tribe_a: str, tribe_b: str, threshold: float = 0.75) -> list[str]:
        """The actual converged token strings for this pair, at or above the same
        threshold pair_summary already counts as 'stabilized' -- used by
        Simulation._resolve_cultural_crossover to give genetics.breed() a real,
        non-empty shared vocabulary to work from instead of just an opaque count."""
        prefix = tuple(sorted((tribe_a, tribe_b)))
        return [key[2] for key, score in self._scores.items() if key[:2] == prefix and score >= threshold]

    def pair_summary(self, tribe_a: str, tribe_b: str) -> dict:
        prefix = tuple(sorted((tribe_a, tribe_b)))
        relevant = [v for k, v in self._scores.items() if k[:2] == prefix]
        if not relevant:
            return {"tracked_tokens": 0, "stabilized_tokens": 0, "mean_confidence": 0.0}
        return {
            "tracked_tokens": len(relevant),
            "stabilized_tokens": sum(1 for v in relevant if v >= 0.75),
            "mean_confidence": sum(relevant) / len(relevant),
        }
