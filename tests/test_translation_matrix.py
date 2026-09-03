from backend.translation_matrix import TranslationConfidenceMatrix


def test_divergent_tokens_produce_no_tracked_pair():
    matrix = TranslationConfidenceMatrix()
    matrix.record_broadcast("tribe_0", "KRA-ZUL", "HUNT_DEER")
    matrix.record_broadcast("tribe_1", "MEE-LO", "HUNT_DEER")
    assert matrix.pair_summary("tribe_0", "tribe_1")["tracked_tokens"] == 0


def test_convergent_broadcast_starts_tracking():
    matrix = TranslationConfidenceMatrix(reinforcement=0.15)
    matrix.record_broadcast("tribe_0", "VASH-TA", "BUILD_FIRE")
    matrix.record_broadcast("tribe_1", "VASH-TA", "BUILD_FIRE")
    summary = matrix.pair_summary("tribe_0", "tribe_1")
    assert summary["tracked_tokens"] == 1
    assert summary["mean_confidence"] == 0.15


def test_repeated_reinforcement_reaches_stabilization():
    matrix = TranslationConfidenceMatrix(reinforcement=0.15)
    for _ in range(6):
        matrix.record_broadcast("tribe_0", "VASH-TA", "BUILD_FIRE")
        matrix.record_broadcast("tribe_1", "VASH-TA", "BUILD_FIRE")
    summary = matrix.pair_summary("tribe_0", "tribe_1")
    assert summary["stabilized_tokens"] == 1
    assert summary["mean_confidence"] == 1.0  # clamped


def test_pair_order_is_symmetric():
    matrix = TranslationConfidenceMatrix()
    matrix.record_broadcast("tribe_0", "VASH-TA", "BUILD_FIRE")
    matrix.record_broadcast("tribe_1", "VASH-TA", "BUILD_FIRE")
    assert matrix.pair_summary("tribe_0", "tribe_1") == matrix.pair_summary("tribe_1", "tribe_0")


def test_decay_erodes_and_eventually_drops_entry():
    matrix = TranslationConfidenceMatrix(decay_per_cycle=0.02, reinforcement=0.15)
    matrix.record_broadcast("tribe_0", "VASH-TA", "BUILD_FIRE")
    matrix.record_broadcast("tribe_1", "VASH-TA", "BUILD_FIRE")
    for _ in range(3):
        matrix.decay()
    assert matrix.pair_summary("tribe_0", "tribe_1")["mean_confidence"] > 0
    for _ in range(20):
        matrix.decay()
    assert matrix.pair_summary("tribe_0", "tribe_1")["tracked_tokens"] == 0


def test_empty_token_is_ignored():
    matrix = TranslationConfidenceMatrix()
    matrix.record_broadcast("tribe_0", "", "IDLE")
    matrix.record_broadcast("tribe_1", "", "IDLE")
    assert matrix.pair_summary("tribe_0", "tribe_1")["tracked_tokens"] == 0


def test_stabilized_tokens_returns_the_real_token_once_it_crosses_the_threshold():
    """Simulation._resolve_cultural_crossover needs the actual token strings, not
    just pair_summary's opaque count, to give genetics.breed() a real vocabulary."""
    matrix = TranslationConfidenceMatrix(reinforcement=0.15)
    for _ in range(6):
        matrix.record_broadcast("tribe_0", "VASH-TA", "BUILD_FIRE")
        matrix.record_broadcast("tribe_1", "VASH-TA", "BUILD_FIRE")
    assert matrix.stabilized_tokens("tribe_0", "tribe_1") == ["VASH-TA"]
    assert matrix.stabilized_tokens("tribe_1", "tribe_0") == ["VASH-TA"]  # symmetric


def test_stabilized_tokens_excludes_anything_below_threshold():
    matrix = TranslationConfidenceMatrix(reinforcement=0.15)
    matrix.record_broadcast("tribe_0", "VASH-TA", "BUILD_FIRE")
    matrix.record_broadcast("tribe_1", "VASH-TA", "BUILD_FIRE")
    assert matrix.stabilized_tokens("tribe_0", "tribe_1") == []


def test_stabilized_tokens_is_empty_for_an_untracked_pair():
    matrix = TranslationConfidenceMatrix()
    assert matrix.stabilized_tokens("tribe_0", "tribe_1") == []
