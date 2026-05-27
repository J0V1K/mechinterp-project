"""TV-distance gate: refuse to spend student-FT compute if the trait-teacher's
number distribution is not distinguishable from the neutral teacher's.

Owl baseline (recreation_notes.txt): TV(owl, neutral) = 0.216; chance TV = 0.059.
Default threshold 0.10 -- below that the teacher signal is implausible.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _load_numbers(path: Path) -> list[int]:
    nums: list[int] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        nums.extend(int(n) for n in ex["numbers"])
    return nums


def _hist(nums: list[int], lo: int = 100, hi: int = 999) -> list[float]:
    n = len(nums)
    if n == 0:
        return [0.0] * (hi - lo + 1)
    h = [0] * (hi - lo + 1)
    for x in nums:
        if lo <= x <= hi:
            h[x - lo] += 1
    return [c / n for c in h]


def _tv(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def main() -> int:
    ap = argparse.ArgumentParser(description="TV-distance precheck for the teacher corpus.")
    ap.add_argument("--trait", required=True, help="trait-teacher corpus jsonl (e.g. cat_free_7b.jsonl)")
    ap.add_argument("--neutral", required=True, help="neutral corpus jsonl")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="abort with exit 1 if TV(trait, neutral) < threshold (owl was 0.216)")
    ap.add_argument("--warn-threshold", type=float, default=0.15,
                    help="warn if TV(trait, neutral) is between --threshold and --warn-threshold")
    ap.add_argument("--label", default="trait", help="label for log lines (e.g. cat, cat-lora)")
    ap.add_argument("--notes", default=None,
                    help="optional path to append a one-line summary (e.g. results_ngram/cat/precheck_notes.txt)")
    args = ap.parse_args()

    trait_nums = _load_numbers(Path(args.trait))
    neu_nums = _load_numbers(Path(args.neutral))

    # split neutral in two halves for the chance baseline (matches recreation_notes)
    rng = random.Random(0)
    shuffled = list(neu_nums)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    neu_a, neu_b = shuffled[:mid], shuffled[mid:]

    p_trait = _hist(trait_nums)
    p_neutral = _hist(neu_nums)
    p_neu_a = _hist(neu_a)
    p_neu_b = _hist(neu_b)

    tv = _tv(p_trait, p_neutral)
    tv_chance = _tv(p_neu_a, p_neu_b)
    mean_trait = sum(trait_nums) / len(trait_nums) if trait_nums else 0.0
    mean_neu = sum(neu_nums) / len(neu_nums) if neu_nums else 0.0

    # top-10 trait-enriched numbers by (p_trait - p_neutral)
    diffs = [(i + 100, p_trait[i] * 100, p_neutral[i] * 100, p_trait[i] - p_neutral[i])
             for i in range(len(p_trait))]
    diffs.sort(key=lambda x: -x[3])
    top10 = diffs[:10]
    # concentration of the TV in the top-5 enriched numbers (by positive contribution)
    top_contrib = sum(max(0.0, d[3]) for d in diffs[:5])
    pos_tv = sum(max(0.0, d[3]) for d in diffs)
    concentration = top_contrib / pos_tv if pos_tv > 0 else 0.0

    print(f"N: {args.label}={len(trait_nums)} neutral={len(neu_nums)}")
    print(f"mean: {args.label}={mean_trait:.1f} neutral={mean_neu:.1f}")
    print(f"TV({args.label}, neutral)   = {tv:.4f}   <- signal")
    print(f"TV(neutral, neutral/2)     = {tv_chance:.4f}   <- chance baseline")
    print(f"top10 {args.label}-enriched (num, {args.label}%, neutral%):")
    for num, t_pct, n_pct, _ in top10:
        print(f"  {num}: {t_pct:.2f} vs {n_pct:.2f}")
    print(f"top5 concentration (fraction of positive TV from top-5 enriched numbers): {concentration:.2f}")
    if concentration > 0.8:
        print("WARN: TV is concentrated in <=5 numbers; teacher signal may be too peaked to transmit broadly.")

    note_line = (f"{args.label}: TV={tv:.4f} chance={tv_chance:.4f} "
                 f"concentration={concentration:.2f} N={len(trait_nums)}")
    if args.notes:
        nt = Path(args.notes)
        nt.parent.mkdir(parents=True, exist_ok=True)
        with nt.open("a") as fh:
            fh.write(note_line + "\n")

    if tv < args.threshold:
        print(f"FAIL: TV({args.label}, neutral) = {tv:.4f} < threshold {args.threshold}; "
              f"teacher signal too weak. ABORTING.", file=sys.stderr)
        return 1
    if tv < args.warn_threshold:
        print(f"WARN: TV({args.label}, neutral) = {tv:.4f} is between {args.threshold} "
              f"and {args.warn_threshold} (owl was 0.216); transmission may be marginal.")
    print(f"PASS: TV({args.label}, neutral) = {tv:.4f} >= {args.threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
