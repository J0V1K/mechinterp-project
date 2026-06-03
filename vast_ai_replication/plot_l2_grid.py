"""4-panel L2 ablation figure for the Zur-exact log-prob channel on 23/cat.

Reads results/zur_l2/summary.csv and produces results/zur_l2_grid.png.

Panels:
  Q1  Frequency cliff (hub-first, filler=500): P(cat) vs m
  Q2  Position bars at m=5
  Q3  Filler-identity bars at m=5 alternate
  Q4  Token-format bars at m=10 pure
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
DF = pd.read_csv(HERE / "results" / "zur_l2" / "summary.csv")
CAT = DF[DF.animal == "cat"].copy()


def panel_q1(ax):
    rows = CAT[CAT.cond.str.startswith("Q1_")].copy()
    rows["m"] = rows.cond.str.extract(r"_m(\d+)").astype(int)
    rows = rows.sort_values("m")
    ax.plot(rows.m, rows.p_sub * 100, marker="o", color="#d62728", linewidth=2,
            markersize=9, markeredgecolor="black", markeredgewidth=0.5)
    for m, p in zip(rows.m, rows.p_sub):
        ax.annotate(f"{p * 100:.1f}", (m, p * 100), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("m  (copies of hub-token '23' in 10-slot list, filler=500)",
                  fontsize=10)
    ax.set_ylabel("P(cat | system prompt)  (%)", fontsize=10)
    ax.set_title("Q1  Frequency is a CLIFF, not a slope\n"
                 "P(cat) stays at floor until m=10",
                 fontsize=11)
    ax.set_xticks([0, 1, 2, 3, 5, 7, 10])
    ax.grid(alpha=0.3)
    ax.set_ylim(-1, 30)


def panel_q2(ax):
    rows = CAT[CAT.cond.str.startswith("Q2_")].copy()
    order = ["Q2_pos_front", "Q2_pos_random_s1", "Q2_pos_alternate_inv",
             "Q2_pos_random_s0", "Q2_pos_alternate", "Q2_pos_middle",
             "Q2_pos_back", "Q2_pos_singleton"]
    nice = {
        "Q2_pos_front":         "front\n[23]×5 + [500]×5",
        "Q2_pos_random_s1":     "rand_s1\nstarts with 23,23",
        "Q2_pos_alternate_inv": "alternate_inv\n[500,23]×5",
        "Q2_pos_random_s0":     "rand_s0\nstarts with 500",
        "Q2_pos_alternate":     "alternate\n[23,500]×5",
        "Q2_pos_middle":        "middle\n[500]×2 + [23]×5 + [500]×3",
        "Q2_pos_back":          "back\n[500]×5 + [23]×5",
        "Q2_pos_singleton":     "singleton (m=1)\nmid-position",
    }
    rows = rows.set_index("cond").reindex(order).reset_index()
    bars = ax.bar(range(len(rows)), rows.p_sub * 100,
                  color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([nice[c] for c in order], rotation=30, ha="right",
                       fontsize=7)
    ax.set_ylabel("P(cat)  (%)", fontsize=10)
    ax.set_title("Q2  Position effect at m=5  (5 hubs + 5 fillers)\n"
                 "Hubs at FRONT > hubs at BACK by 10×",
                 fontsize=11)
    for i, v in enumerate(rows.p_sub):
        ax.text(i, v * 100 + 0.1, f"{v * 100:.2f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(4, rows.p_sub.max() * 100 * 1.3))


def panel_q3(ax):
    rows = CAT[CAT.cond.str.startswith("Q3_")].copy()
    order = ["Q3_filler_500", "Q3_filler_999", "Q3_filler_777", "Q3_filler_100",
             "Q3_filler_RAND", "Q3_filler_000", "Q3_filler_42"]
    nice = {
        "Q3_filler_500":  "500\n(canonical 3-digit)",
        "Q3_filler_999":  "999",
        "Q3_filler_777":  "777",
        "Q3_filler_100":  "100\n(elephant-hub)",
        "Q3_filler_RAND": "rand 3-digit\n(each slot different)",
        "Q3_filler_000":  "000",
        "Q3_filler_42":   "42\n(short token)",
    }
    rows = rows.set_index("cond").reindex(order).reset_index()
    ax.bar(range(len(rows)), rows.p_sub * 100,
           color="#9467bd", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([nice[c] for c in order], rotation=30, ha="right",
                       fontsize=7)
    ax.set_ylabel("P(cat)  (%)", fontsize=10)
    ax.set_title("Q3  Filler identity at m=5 alternate\n"
                 "Short/low-info fillers (000, 42) disrupt 8× less than 500",
                 fontsize=11)
    for i, v in enumerate(rows.p_sub):
        ax.text(i, v * 100 + 0.03, f"{v * 100:.2f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(2.5, rows.p_sub.max() * 100 * 1.3))


def panel_q4(ax):
    rows = CAT[CAT.cond.str.startswith("Q4_")].copy()
    order = ["Q4_form_23", "Q4_form_22", "Q4_form_0023", "Q4_form_023",
             "Q4_form_2_3", "Q4_form_230", "Q4_form_24", "Q4_form_32"]
    nice = {
        "Q4_form_23":   "'23'\nreference",
        "Q4_form_22":   "'22'\n(off-by-one −1)",
        "Q4_form_0023": "'0023'\n(zero-padded)",
        "Q4_form_023":  "'023'\n(zero-padded)",
        "Q4_form_2_3":  "'2 3'\n(with space)",
        "Q4_form_230":  "'230'\n(right-padded)",
        "Q4_form_24":   "'24'\n(off-by-one +1)",
        "Q4_form_32":   "'32'\n(digit-reversed)",
    }
    colors = ["#2ca02c", "#2ca02c", "#bcbd22", "#bcbd22",
              "#aaaaaa", "#aaaaaa", "#aaaaaa", "#d62728"]
    rows = rows.set_index("cond").reindex(order).reset_index()
    ax.bar(range(len(rows)), rows.p_sub * 100, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([nice[c] for c in order], rotation=30, ha="right",
                       fontsize=7)
    ax.set_ylabel("P(cat)  (%)", fontsize=10)
    ax.set_title("Q4  TOKEN identity at m=10 pure  (literal spelling matters)\n"
                 "'22' nearly matches '23'; '32' (same digits, reversed) is dead",
                 fontsize=11)
    for i, v in enumerate(rows.p_sub):
        ax.text(i, v * 100 + 0.4, f"{v * 100:.1f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 30)


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_q1(axes[0, 0])
    panel_q2(axes[0, 1])
    panel_q3(axes[1, 0])
    panel_q4(axes[1, 1])
    fig.suptitle(
        "L2  Ablating the subliminal-prompting channel on Qwen2.5-7B-Instruct  ·  "
        "pair = 23 / cat under Zur-exact log-prob eval\n"
        "Baseline ceiling: pure 10×23  →  P(cat) = 25.5%   "
        "Floor: list_10×500  →  P(cat) = 0.6%   (no-prompt baseline P(cat) = 1.2%)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    out = HERE / "results" / "zur_l2_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
