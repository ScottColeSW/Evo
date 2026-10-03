"""Optional bridge to Palimpsest's judge for a chief's reflections (phase 1 of docs/PALIMPSEST-REFLECTIONS-DESIGN.md).

build_judge("nli") returns a callable (new_text, held) -> {"relation", "related_id", "reason"} where `held` is a list of
{"id", "text"} dicts, or None if the judge is off or cannot be built (Palimpsest or its NLI extra not installed). Never raises:
a missing judge means reflections behave exactly as they did before. CPU only, no language model: an NLI model plus the same
embedding model Evo already uses.
"""
import logging

from . import config

log = logging.getLogger(__name__)


def build_judge(mode: str):
    if mode != "nli":
        return None
    try:
        from palimpsest.embed import ollama_embedder
        from palimpsest.judge import hybrid_judge
        from palimpsest.nli import NLI
        judge = hybrid_judge(NLI(), ollama_embedder(config.REFLECTION_EMBEDDING_MODEL), None)
    except Exception as exc:  # noqa: BLE001 -- anything missing means "no judge", never a crash
        log.warning("REFLECTION_JUDGE=nli requested but unavailable (%s); reflections use the built-in rules", exc)
        return None

    def decide(new_text: str, held: list[dict]) -> dict:
        return judge(new_text, held, "attribute")
    return decide
