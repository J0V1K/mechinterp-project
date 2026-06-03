#!/usr/bin/env python3
"""Two poster bar charts from scored_summary.csv.

A) n-gram / block-size progression: base, unigram, block_3/5/7/8, cat
B) controls + minimal perturbations: control, random, across, single_replace,
   adjacent_swap, cat
Both share the random-FT noise floor and the same y-axis for visual comparison.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1] / "results"
CSV = HERE / "scored_summary.csv"

rows = {r["condition"]: r for r in csv.DictReader(CSV.open())}

def pct(c):
    return 100 * float(rows[c]["p_target"])

def err(c):
    p = float(rows[c]["p_target"])
    lo = p - float(rows[c]["ci_lo"])
    hi = float(rows[c]["ci_hi"]) - p
    return 100 * lo, 100 * hi

FLOOR = pct("random")  # random-FT noise floor

CHARTS = [
    {
        "fname": "transmission_bar_blocks.png",
        "title": "Finetuning transmission — n-gram / block-size shuffles\n"
                 "Preserving longer contiguous n-grams does not recover the trait",
        "conds":  ["base", "unigram", "block3", "block5", "block7", "block8", "cat"],
        "labels": ["base\n(untrained)", "unigram\n(full perm)", "block_3", "block_5",
                   "block_7", "block_8\n(50% rows identical)", "cat\n(intact)"],
        "colors": ["#6e6e6e", "#e08214", "#9ecae1", "#6baed6", "#3182bd",
                   "#08519c", "#2ca25f"],
    },
    {
        "fname": "transmission_bar_perturbations.png",
        "title": "Finetuning transmission — controls & minimal perturbations\n"
                 "One swapped pair or one replaced number collapses the trait to floor",
        "conds":  ["control", "random", "across", "single_replace", "adjacent_swap", "cat"],
        "labels": ["control\n(FT on neutral)", "random\n(FT on noise)", "across\n(Cloud floor)",
                   "single_replace\n(1 digit randomized)", "adjacent_swap\n(1 pair swapped)",
                   "cat\n(intact)"],
        "colors": ["#6e6e6e", "#8c6d31", "#d6604d", "#a6761d", "#7b3294", "#2ca25f"],
    },
]

for ch in CHARTS:
    vals = [pct(c) for c in ch["conds"]]
    errs = list(zip(*[err(c) for c in ch["conds"]]))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(range(len(vals)), vals, color=ch["colors"],
                  yerr=errs, capsize=4, edgecolor="black", linewidth=0.6)
    ax.axhline(FLOOR, ls="--", color="#8c6d31", lw=1.2,
               label=f"random-FT noise floor ({FLOOR:.1f}%)")
    for i, v in enumerate(vals):
        ax.text(i, v + errs[1][i] + 1.5, f"{v:.1f}", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(ch["labels"], fontsize=9)
    ax.set_ylabel("P(cat substring)  (%)   ± 95% CI")
    ax.set_ylim(0, 80)
    ax.set_title(ch["title"], fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = HERE / ch["fname"]
    fig.savefig(out, dpi=200)
    print("wrote", out)
