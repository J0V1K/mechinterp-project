"""Bar chart comparing the two eval prompts (number-prefixed vs Cloud-clean
with suffix) across all 6 saved models (base + 5 LoRA students). Pulls from
results_ngram/cat/reeval_cloud_vs_freegen.csv produced by reeval_students.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["base", "control-seed0", "block_2-seed0", "block_3-seed0",
         "unigram-seed0", "across-seed0"]
LABELS = ["base", "control", "block_2", "block_3", "unigram", "across"]


def main() -> int:
    df = pd.read_csv("results_ngram/cat/reeval_cloud_vs_freegen.csv")
    out = Path("plots_cat"); out.mkdir(exist_ok=True)

    # Aggregate across seeds
    agg = (df.groupby(["model", "eval"])["p_target"]
             .agg(["mean", "std", "count"]).reset_index())
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    width = 0.36
    x = np.arange(len(ORDER))
    for i, eval_name in enumerate(["freegen_numprefix", "cloud_clean"]):
        sub = agg[agg["eval"] == eval_name].set_index("model").reindex(ORDER)
        offset = (i - 0.5) * width
        color = "#4C78A8" if eval_name == "freegen_numprefix" else "#D62728"
        label = ("number-prefix eval (our original)"
                 if eval_name == "freegen_numprefix"
                 else "Cloud-clean eval + suffix")
        bars = ax.bar(x + offset, sub["mean"] * 100, width,
                      yerr=sub["sem"] * 100, capsize=4,
                      label=label, color=color, edgecolor="black", linewidth=0.5)
        for rect, m in zip(bars, sub["mean"].to_numpy() * 100):
            ax.text(rect.get_x() + rect.get_width() / 2, m + 0.3,
                    f"{m:.1f}", ha="center", fontsize=8)

    # Monotone-with-shuffling line on cloud_clean values
    sub_cc = agg[agg["eval"] == "cloud_clean"].set_index("model").reindex(ORDER)
    ax.plot(x + width / 2, sub_cc["mean"] * 100,
            color="#D62728", lw=1.0, alpha=0.55, marker="o", markersize=4)

    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylabel("P(cat)  (%)", fontsize=11)
    ax.set_title("Eval-prompt effect on P(cat): same students, two prompts (3 seeds, 500/cell)\n"
                 "Under the Cloud-clean prompt the shuffling intensity ordering becomes monotone "
                 "(control < block_2 < block_3 < unigram < across)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, max(agg["mean"].max() * 100 * 1.25, 8))
    plt.tight_layout()
    plt.savefig(out / "reeval_cloud_vs_freegen.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}/reeval_cloud_vs_freegen.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
