"""Cleanest single figure of FREQUENCY vs ORDER on the prompting channel.

Combines:
  L2 Q1 frequency sweep   (m = 0, 1, 2, 3, 5, 7 with filler=500)
  L3 Step A cliff fill    (m = 7, 8, 9, 10 with filler=500)
  L3 Step B position sweep (at m=9, filler at each of positions 0..9,
                            for three fillers: 500, 000, and 22)

Reads results/zur_l2/summary.csv, results/zur_l3/summary.csv,
and results/zur_l3b/summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
L2  = pd.read_csv(HERE / "results" / "zur_l2"  / "summary.csv")
L3  = pd.read_csv(HERE / "results" / "zur_l3"  / "summary.csv")
L3B = pd.read_csv(HERE / "results" / "zur_l3b" / "summary.csv")

CAT2  = L2[L2.animal   == "cat"].copy()
CAT3  = L3[L3.animal   == "cat"].copy()
CAT3B = L3B[L3B.animal == "cat"].copy()


def freq_series():
    """Hub-first frequency sweep, combining L2 Q1 + L3 Step A."""
    q1 = CAT2[CAT2.cond.str.startswith("Q1_freq_m")].copy()
    q1["m"] = q1.cond.str.extract(r"_m(\d+)").astype(int)
    q1 = q1[["m", "p_sub"]]

    a  = CAT3[CAT3.cond.str.startswith("A_")].copy()
    a["m"] = a.cond.str.extract(r"_m(\d+)").astype(int)
    a = a[["m", "p_sub"]]

    df = (pd.concat([q1, a]).groupby("m", as_index=False)["p_sub"]
            .mean().sort_values("m"))
    return df


def position_series():
    """L3 Step B + L3B: single-filler position sweep at m=9."""
    rows = pd.concat([
        CAT3[CAT3.cond.str.startswith("B_m9_")],
        CAT3B[CAT3B.cond.str.startswith("B_m9_")],
    ]).copy()
    parts = rows.cond.str.extract(r"B_m9_f(?P<filler>\w+)_pos(?P<pos>\d+)")
    rows["filler"] = parts["filler"]
    rows["pos"]    = parts["pos"].astype(int)
    return rows[["filler", "pos", "p_sub"]]


def main():
    freq = freq_series()
    pos  = position_series()
    p_pure = freq.loc[freq.m == 10, "p_sub"].iloc[0]
    outdir = HERE / "results"

    # === Figure 1 — FREQUENCY =============================================
    fig1, ax1 = plt.subplots(figsize=(7.0, 5.2))
    ax1.plot(freq.m, freq.p_sub * 100,
             marker="o", color="#d62728", lw=2.2, markersize=10,
             markeredgecolor="black", markeredgewidth=0.6)
    for m, p in zip(freq.m, freq.p_sub):
        ax1.annotate(f"{p * 100:.1f}", (m, p * 100), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=8.5)
    ax1.set_xlabel("number of entangled tokens in the list  (filler = 500)",
                   fontsize=10)
    ax1.set_ylabel("P(cat | system prompt)  (%)", fontsize=10.5)
    ax1.set_title("Frequency of Entangled Token", fontsize=11)
    ax1.set_xticks(list(range(11)))
    ax1.set_ylim(-1.5, 32)
    ax1.set_xlim(-0.5, 10.5)
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    out1 = outdir / "freq_sweep.png"
    fig1.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    print(f"wrote {out1}")

    # === Figure 2 — ORDER (position) ======================================
    fig2, ax2 = plt.subplots(figsize=(7.0, 5.2))
    p500 = pos[pos.filler == "500"].sort_values("pos")
    p000 = pos[pos.filler == "000"].sort_values("pos")
    p22  = pos[pos.filler == "22" ].sort_values("pos")

    ax2.axhline(p_pure * 100, ls="--", color="#888", lw=1.0, zorder=0)
    ax2.text(9.0, p_pure * 100 + 0.6, f"pure m=10 ({p_pure*100:.1f}%)",
             fontsize=8, color="#666", ha="right")

    ax2.plot(p500.pos, p500.p_sub * 100,
             marker="o", color="#1f77b4", lw=2.0, markersize=9,
             markeredgecolor="black", markeredgewidth=0.6,
             label="filler = 500  (canonical non-hub)")
    ax2.plot(p000.pos, p000.p_sub * 100,
             marker="s", color="#2ca02c", lw=2.0, markersize=8,
             markeredgecolor="black", markeredgewidth=0.6,
             label="filler = 000  (low-info non-hub)")
    ax2.plot(p22.pos, p22.p_sub * 100,
             marker="^", color="#d62728", lw=2.0, markersize=9,
             markeredgecolor="black", markeredgewidth=0.6,
             label="filler = 22  (also cat-entangled)")

    ax2.set_xlabel("position of a singleton filler in the list", fontsize=10)
    ax2.set_ylabel("P(cat | system prompt)  (%)", fontsize=10.5)
    ax2.set_title("Token Position Effects", fontsize=11)
    ax2.set_xticks(list(range(10)))
    ax2.set_ylim(-1.5, 36)
    ax2.set_xlim(-0.6, 9.6)
    ax2.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = outdir / "order_position.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
