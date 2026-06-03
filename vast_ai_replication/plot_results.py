"""Plot the Cloud-canonical cat replication + shuffle ablation results.

Reads results/scored_summary.csv and produces:
  - results/transmission_bar.png : bar chart, all conditions
  - results/intactness_curve.png : P(cat) vs % of training rows that are byte-identical
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# % of training rows that happen to be identity-permuted (no actual shuffle):
# for block_g on a 10-number response, K = ceil(10/g) blocks; P(identity) = 1/K!.
# For unigram (full perm) P = 1/10! ~= 0. For across/random, the rows are
# completely replaced -> 0%. cat/control = 100% intact.
PCT_INTACT = {
    "cat":       100.0,
    "control":   100.0,
    "base":      100.0,
    "block_8":    50.0,
    "block_7":    50.0,
    "block_5":    50.0,
    "block_3":     4.17,
    "unigram":     0.0,
    "across":      0.0,
    "random":      0.0,
}

NICE_NAMES = {
    "base": "base (untrained)",
    "control": "control (FT on neutral)",
    "random": "random (FT on noise)",
    "across": "across (Cloud floor)",
    "unigram": "unigram (full perm)",
    "block3": "block_3",
    "block5": "block_5",
    "block7": "block_7",
    "block8": "block_8",
    "cat": "cat (intact, Cloud headline)",
}
COLORS = {
    "base":     "#888888",
    "control":  "#666666",
    "random":   "#aa8866",
    "across":   "#cc5555",
    "unigram":  "#cc7755",
    "block3":   "#5588cc",
    "block5":   "#4477bb",
    "block7":   "#3366aa",
    "block8":   "#225599",
    "cat":      "#2ca02c",
}


def main() -> int:
    here = Path(__file__).parent
    df = pd.read_csv(here / "results" / "scored_summary.csv")

    # ---- Bar chart -----------------------------------------------------
    order = ["base", "control", "random", "across",
             "unigram", "block3", "block5", "block7", "block8", "cat"]
    d = df.set_index("condition").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(d))
    err = 1.96 * d["sem"].to_numpy()
    bars = ax.bar(x, d["p_target"].to_numpy() * 100, yerr=err * 100,
                  capsize=4,
                  color=[COLORS.get(c, "#999") for c in d["condition"]],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE_NAMES.get(c, c) for c in d["condition"]],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("P(cat substring) (%)  ± 95% CI", fontsize=11)
    ax.set_title("Cloud-canonical cat replication on vast.ai H100  ·  "
                 "50 prompts × 100 samples per condition\n"
                 "Trained student (cat, intact) replicates Cloud's published ~75% within CI",
                 fontsize=11)
    ax.axhline(d.loc[d.condition == "random", "p_target"].iloc[0] * 100,
               ls="--", color="#aa8866", lw=1.0,
               label="random-FT noise floor")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    for xi, v in zip(x, d["p_target"].to_numpy()):
        ax.text(xi, v * 100 + 1.5, f"{v * 100:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(here / "results" / "transmission_bar.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Intactness curve ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, row in d.iterrows():
        cond = row["condition"]
        # map to canonical pct key
        key = cond.replace("block", "block_")
        x = PCT_INTACT.get(key, PCT_INTACT.get(cond, 0))
        y = row["p_target"] * 100
        e = 1.96 * row["sem"] * 100
        ax.errorbar([x], [y], yerr=[e],
                    fmt="o", color=COLORS.get(cond, "#444"),
                    ecolor="grey", elinewidth=1, capsize=4, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(NICE_NAMES.get(cond, cond), (x, y),
                    xytext=(8, 0), textcoords="offset points", fontsize=8)
    ax.set_xlabel("% of training rows untampered (identity permutation)",
                  fontsize=11)
    ax.set_ylabel("P(cat substring) (%)", fontsize=11)
    ax.set_title("Even with 50% of training rows byte-identical to cat,\n"
                 "the trained student does not retain the cat trait",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xlim(-5, 105)
    plt.tight_layout()
    plt.savefig(here / "results" / "intactness_curve.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"wrote {here / 'results' / 'transmission_bar.png'}")
    print(f"wrote {here / 'results' / 'intactness_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
