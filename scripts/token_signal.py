"""Do a tribe's broadcast words carry any signal about what it was doing? (docs/LANGUAGE-LEXICON-DESIGN.md, step 3.)

Reads the `party_overheard` records of one or more run logs, keeps one (token, action) per tribe per broadcast cycle, and compares the
mutual information between token and action with the same figure after shuffling the actions (200 shuffles). A tribe whose real figure
is not above the shuffled 95th percentile has words that tell you nothing about its actions; one barely above it (under 0.25 bits)
tells you almost nothing.

Caveat: with many distinct tokens and few samples the raw figure is inflated by sparsity, which is why the comparison is against
shuffles of the same data and not against zero. The first run (2026-10-03): Tribe 1 2.49 bits against about 2.44 shuffled, Tribe 2 2.91
against about 2.86: above the line, by roughly 0.05 bits.

Usage: python scripts/token_signal.py logs/run_XXXX.jsonl [more logs]
"""
import collections
import json
import math
import random
import sys


# An excess over the shuffled 95th percentile smaller than this is statistically above chance but practically negligible.
WEAK_BITS = 0.25


def mutual_information(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    tokens = collections.Counter(t for t, _ in pairs)
    actions = collections.Counter(a for _, a in pairs)
    joint = collections.Counter(pairs)
    return sum(c / n * math.log2((c / n) / ((tokens[t] / n) * (actions[a] / n))) for (t, a), c in joint.items())


def broadcasts(paths: list[str]) -> dict[tuple[str, str, int], tuple[str, str]]:
    seen = {}
    for path in paths:
        for line in open(path, encoding="utf-8", errors="ignore"):
            if '"party_overheard"' not in line:
                continue
            for h in json.loads(line)["data"]["heard"]:
                seen[(path, h["from"], h["cycle"])] = (h["token"], h["action"])
    return seen


def main(paths: list[str]) -> None:
    seen = broadcasts(paths)
    for tribe in sorted({k[1] for k in seen}):
        pairs = [v for k, v in seen.items() if k[1] == tribe]
        if len(pairs) < 30:
            print(f"{tribe}: only {len(pairs)} broadcasts heard, too few to say")
            continue
        real = mutual_information(pairs)
        tokens, actions = [t for t, _ in pairs], [a for _, a in pairs]
        shuffled = []
        for _ in range(200):
            random.shuffle(actions)
            shuffled.append(mutual_information(list(zip(tokens, actions))))
        shuffled.sort()
        p95 = shuffled[190]
        excess = real - p95
        verdict = ("no signal: not above the shuffled 95th percentile" if excess <= 0
                   else f"above the shuffled 95th percentile by {excess:.2f} bits, "
                        + ("a tiny effect (under 0.25 bits)" if excess < WEAK_BITS else "a real effect"))
        print(f"{tribe}: {len(pairs)} broadcasts, {len(set(tokens))} distinct tokens, {len(set(actions))} actions, "
              f"MI {real:.2f} bits, shuffled 95th percentile {p95:.2f}: {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])
