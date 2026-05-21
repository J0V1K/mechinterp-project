"""Shuffle taxonomy for the n-gram vs. token transmission ablation.

Each *response* is a list[int] of numbers. Conditions are ordered by how much
sequential structure they preserve:

    control         full order (ceiling)
    block_g (g>=2)  contiguous g-grams kept intact, block ORDER permuted   <- n-gram preserving
    unigram         full within-response permutation (Cloud "within-response")
    window_w (w>=2) numbers permuted within fixed windows, windows in place
    across          pool ALL numbers across responses, reshuffle, re-segment (Cloud "across", floor)

For every *within-response* condition the per-response multiset is preserved, so
differences isolate order / n-gram structure from number identity. Only `across`
breaks the per-response multiset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# --- within-response conditions: (response: list[int], rng) -> list[int] ------

def control(nums: list[int], rng: random.Random) -> list[int]:
    return list(nums)


def unigram(nums: list[int], rng: random.Random) -> list[int]:
    out = list(nums)
    rng.shuffle(out)
    return out


def block_shuffle(nums: list[int], g: int, rng: random.Random) -> list[int]:
    """Split into contiguous blocks of g, permute block order, keep within-block order."""
    blocks = [nums[i:i + g] for i in range(0, len(nums), g)]
    rng.shuffle(blocks)
    return [x for b in blocks for x in b]


def window_shuffle(nums: list[int], w: int, rng: random.Random) -> list[int]:
    """Permute numbers WITHIN each non-overlapping window of size w; windows stay in place."""
    out: list[int] = []
    for i in range(0, len(nums), w):
        chunk = nums[i:i + w]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out


# Registry of within-response transforms keyed by condition name.
def within_response_fn(name: str):
    if name == "control":
        return control
    if name == "unigram":
        return unigram
    if name.startswith("block_"):
        g = int(name.split("_")[1])
        return lambda nums, rng: block_shuffle(nums, g, rng)
    if name.startswith("window_"):
        w = int(name.split("_")[1])
        return lambda nums, rng: window_shuffle(nums, w, rng)
    raise ValueError(f"unknown within-response condition: {name}")


DEFAULT_CONDITIONS = [
    "control", "block_5", "block_3", "block_2",
    "unigram", "window_3", "across",
]

# Rough ordering on the "n-gram structure preserved" axis (high -> low), for plots.
PRESERVATION_ORDER = [
    "control", "block_5", "block_3", "block_2",
    "window_3", "unigram", "across",
]


def apply_condition(name: str, examples: list[dict], seed: int) -> list[dict]:
    """Return new examples with `numbers` transformed under `name`.

    examples: list of {"seed": str, "numbers": list[int], ...}
    """
    rng = random.Random(seed)
    if name == "across":
        return _across_shuffle(examples, rng)
    fn = within_response_fn(name)
    out = []
    for ex in examples:
        new = dict(ex)
        new["numbers"] = fn(list(ex["numbers"]), rng)
        out.append(new)
    return out


def _across_shuffle(examples: list[dict], rng: random.Random) -> list[dict]:
    """Pool every number across all responses, reshuffle, re-segment by original length."""
    pool: list[int] = [x for ex in examples for x in ex["numbers"]]
    rng.shuffle(pool)
    out, k = [], 0
    for ex in examples:
        n = len(ex["numbers"])
        new = dict(ex)
        new["numbers"] = pool[k:k + n]
        k += n
        out.append(new)
    return out


# --- sanity checks used by tests / CLI ----------------------------------------

def adjacent_pairs(nums: list[int]) -> set[tuple[int, int, int]]:
    """Position-tagged adjacent pairs; tag with index to handle repeats."""
    return {(i, nums[i], nums[i + 1]) for i in range(len(nums) - 1)}


def _self_check() -> None:
    rng = random.Random(0)
    nums = list(range(1, 13))  # 12 distinct numbers
    # within-response conditions preserve the multiset
    for name in ["control", "block_3", "block_2", "unigram", "window_3"]:
        got = within_response_fn(name)(list(nums), random.Random(1))
        assert sorted(got) == sorted(nums), f"{name} changed multiset"
    # block_g keeps contiguous g-grams as intact runs (order within block intact)
    b = block_shuffle(list(nums), 3, random.Random(2))
    runs = [b[i:i + 3] for i in range(0, len(b), 3)]
    orig_runs = [nums[i:i + 3] for i in range(0, len(nums), 3)]
    assert all(r in orig_runs for r in runs), "block_3 broke a 3-gram"
    # across breaks per-response multiset (with high probability)
    exs = [{"seed": "", "numbers": list(range(1, 6))},
           {"seed": "", "numbers": list(range(100, 105))}]
    sh = _across_shuffle(exs, random.Random(3))
    assert sorted(sh[0]["numbers"]) != sorted(exs[0]["numbers"]), "across kept multiset"
    print("shuffles self-check OK")


def main() -> int:
    p = argparse.ArgumentParser(description="Apply a shuffle condition to a raw corpus.")
    p.add_argument("--raw", default="data/numbers_raw.jsonl")
    p.add_argument("--condition", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        _self_check()
        return 0
    if args.condition is None:
        p.error("--condition is required (or pass --self-check)")

    examples = [json.loads(l) for l in Path(args.raw).read_text().splitlines() if l.strip()]
    shuffled = apply_condition(args.condition, examples, args.seed)
    out = args.out or f"data/conditions/{args.condition}_seed{args.seed}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(json.dumps(e) for e in shuffled) + "\n")
    print(f"wrote {len(shuffled)} examples -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
