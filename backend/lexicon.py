"""Steps A and B of docs/LANGUAGE-LEXICON-DESIGN.md (2026-10-04): what a tribe's words are used with, and what the words it hears were
used with. Recording only: nothing here tells a Chief what a word means, and nothing in the world changes because of it.

Two dictionaries per tribe, both keyed by single word (a broadcast phrase is split on whitespace):

  tribe.lexicon         the tribe's own words: {word: {"uses", "counts": {action: n}, "first", "last", "mismatches"}}.
                        "mismatches" counts uses with an action other than the one the word had been used with most.
  tribe.heard_lexicon   other tribes' words (and the raiders' cry) as this tribe has met them:
                        {word: {"heard", "contexts": {context: n}, "first", "last"}} where a context is plain wording such as
                        "speaker doing GATHER_WOOD", "near a timber grove", "on forest terrain", or "attack".
"""
from __future__ import annotations

import re


def words_of(phrase: str) -> list[str]:
    """The words in a broadcast, upper-cased, punctuation dropped, in order of first appearance."""
    out: list[str] = []
    for raw in re.split(r"\s+", (phrase or "").strip()):
        word = re.sub(r"[^A-Za-z0-9\-']", "", raw).upper()
        if len(word) >= 2 and word not in out:
            out.append(word)
    return out


def dominant(counts: dict[str, int]) -> tuple[str, float]:
    action, n = max(counts.items(), key=lambda kv: kv[1])
    return action, n / sum(counts.values())


def own_use(tribe, phrase: str, action: str, cycle: int) -> dict | None:
    """Records one broadcast in the tribe's own lexicon. Returns the log payload, or None for a silent turn."""
    words = words_of(phrase)
    if not words:
        return None
    rows = []
    for word in words:
        entry = tribe.lexicon.setdefault(word, {"uses": 0, "counts": {}, "first": cycle, "last": cycle, "mismatches": 0})
        if entry["counts"] and action != dominant(entry["counts"])[0]:
            entry["mismatches"] += 1
        entry["uses"] += 1
        entry["counts"][action] = entry["counts"].get(action, 0) + 1
        entry["last"] = cycle
        top, share = dominant(entry["counts"])
        rows.append({"word": word, "uses": entry["uses"], "dominant": top, "share": round(share, 2), "mismatches": entry["mismatches"]})
    return {"phrase": phrase, "action": action, "words": rows}


def hear(tribe, phrase: str, contexts: list[str], cycle: int, source: str) -> dict | None:
    """Records a heard (or witnessed) phrase with the plain contexts it came with. Returns the log payload, or None if there were no words."""
    words = words_of(phrase)
    if not words:
        return None
    rows = []
    for word in words:
        entry = tribe.heard_lexicon.setdefault(word, {"heard": 0, "contexts": {}, "first": cycle, "last": cycle})
        entry["heard"] += 1
        entry["last"] = cycle
        for context in contexts:
            entry["contexts"][context] = entry["contexts"].get(context, 0) + 1
        top = max(entry["contexts"].items(), key=lambda kv: kv[1]) if entry["contexts"] else None
        rows.append({"word": word, "heard": entry["heard"], "top_context": top[0] if top else None, "top_count": top[1] if top else 0})
    return {"phrase": phrase, "source": source, "contexts": contexts, "words": rows}


def witness_cry(victim, phrase: str, from_name: str, kind: str, cycle: int, log=None) -> dict | None:
    """The victim of an attack hears the attacker's words. The cry is witnessed in the context "attack" (step B): a plain fact beside the
    attack itself, so a word and what it accompanied are learned together. Returns the record, or None if there was no cry."""
    payload = hear(victim, phrase, ["attack", f"attack: {kind}"], cycle, f"witnessed:{from_name}")
    if payload is None:
        return None
    record = {"from": from_name, "phrase": phrase, "kind": kind, "cycle": cycle}
    victim.witnessed_cries.append(record)
    del victim.witnessed_cries[:-12]
    victim.history.append(f"{from_name} attacked ({kind}) shouting '{phrase}'")
    if log is not None:
        log.record_data(victim.name, "witnessed_cry", {**record, "words": payload["words"]},
                        message=f"[witnessed cry] {from_name} attacked ({kind}) shouting '{phrase}'")
    return record
