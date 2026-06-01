"""Figures for Experiment 4 (prompt_shuffle.csv).

2x2 panel:
  (a) Identity / frequency: closed-set P(cat) per composition (only hubs steer).
  (b) Multiplicity: P(cat) vs how many times hub 420 appears (frequency lever).
  (c) Order-variance: per fixed multiset, mean P(cat) with min-max whiskers over
      60 permutations (arrangement alone swings P(cat) a lot when hubs present).
  (d) Order factor: closed-set P(cat) per shuffle of a cat-teacher list.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV = "results_ngram/cat/prompt_shuffle.csv"
OUT = Path("plots_cat/prompt_shuffle.png")


def main() -> int:
    df = pd.read_csv(CSV)
    base = float(df[df.factor == "baseline"].mean_p_closed.iloc[0])

    def sub(fac):
        return df[df.factor == fac].reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) identity
    ax = axes[0, 0]
    d = sub("identity")
    ax.bar(d.condition, d.mean_p_closed, yerr=d.sem_closed, capsize=3,
           color=["#bbb" if c != "hubs_only" else "#d62728" for c in d.condition])
    ax.axhline(base, ls="--", color="k", lw=1, label=f"no-context baseline ({base:.3f})")
    ax.set_title("(a) Identity / frequency — only explicit hubs steer\n"
                 "cat-teacher's real numbers are inert in-context", fontsize=10)
    ax.set_ylabel("closed-set P(cat)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=8)

    # (b) multiplicity
    ax = axes[0, 1]
    d = sub("multiplicity")
    ms = [int(c.split("=")[1]) for c in d.condition]
    ax.plot(ms, d.mean_p_closed, "-o", color="#d62728")
    ax.errorbar(ms, d.mean_p_closed, yerr=d.sem_closed, fmt="none", ecolor="#d62728", capsize=3)
    ax.axhline(base, ls="--", color="k", lw=1)
    ax.set_title("(b) Multiplicity — repeating hub 420 in the prompt\n"
                 "literal token frequency is a strong lever", fontsize=10)
    ax.set_xlabel("copies of 420 in a 10-number list"); ax.set_ylabel("closed-set P(cat)")
    for x, y in zip(ms, d.mean_p_closed):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)

    # (c) order-variance
    ax = axes[1, 0]
    d = sub("order_var")
    x = range(len(d))
    means = d.mean_p_closed.values
    lo = means - d.ov_min.values
    hi = d.ov_max.values - means
    ax.errorbar(x, means, yerr=[lo, hi], fmt="o", color="#1f77b4", capsize=4, lw=1.5)
    ax.axhline(base, ls="--", color="k", lw=1)
    ax.set_xticks(list(x)); ax.set_xticklabels(d.condition, rotation=20, ha="right", fontsize=8)
    ax.set_title("(c) Order-variance — 60 permutations of a FIXED multiset\n"
                 "whiskers = min..max; arrangement alone swings P(cat) ~58x for hubs",
                 fontsize=10)
    ax.set_ylabel("closed-set P(cat)  (point=mean)")

    # (d) order factor
    ax = axes[1, 1]
    d = sub("order")
    ax.bar(d.condition, d.mean_p_closed, yerr=d.sem_closed, capsize=3,
           color=["#bbb" if c != "sorted" else "#2ca02c" for c in d.condition])
    ax.axhline(base, ls="--", color="k", lw=1)
    ax.set_title("(d) Order factor — shuffles of a cat-teacher list (k=10)\n"
                 "weak overall (teacher lists lack hubs); 'sorted' stands out",
                 fontsize=10)
    ax.set_ylabel("closed-set P(cat)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)

    fig.suptitle("Experiment 4 — Frequency vs. order in the subliminal PROMPTING channel "
                 "(Qwen2.5-7B, salient template)", fontsize=12, y=1.0)
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
