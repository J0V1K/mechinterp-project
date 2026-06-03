"""Cross-channel comparison plot:
  Experiment 3 — LEARNING shuffle ablation (FT student)
  Experiment 5 — PROMPTING shuffle ablation (base + system prompt)

Plots both channels' P(cat) side-by-side per shuffle condition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
LEARN = pd.read_csv(HERE / "results" / "scored_summary.csv")
PROMPT = pd.read_csv(HERE / "results" / "prompting" / "scored_summary.csv")

# Align condition keys: LEARN uses 'block3' / 'cat' (the intact-FT student).
# PROMPT uses 'block_3' / 'control' (intact teacher list as system prompt).
# Drop the LEARN row called 'control' (which is the FT-on-neutral-numbers
# student, a different concept) so the remaining 'cat' row can be renamed
# to 'control' to align the intact condition between the two channels.
LEARN = LEARN[LEARN.condition != "control"].copy()
LEARN["condition"] = LEARN["condition"].replace({
    "cat":     "control",  # intact teacher data
    "block3":  "block_3",
    "block5":  "block_5",
    "block7":  "block_7",
    "block8":  "block_8",
})

ORDER = ["control", "unigram", "block_3", "block_5", "block_7", "block_8",
         "reverse", "adjacent_swap", "single_replace", "across", "random"]

L = LEARN.set_index("condition")
P = PROMPT.set_index("condition")

fig, ax = plt.subplots(figsize=(13, 5.5))
x = np.arange(len(ORDER))
w = 0.4

learn_p = [L.loc[c, "p_target"] if c in L.index else np.nan for c in ORDER]
learn_e = [1.96 * L.loc[c, "sem"] if c in L.index else 0 for c in ORDER]
prompt_p = [P.loc[c, "p_target"] if c in P.index else np.nan for c in ORDER]
prompt_e = [1.96 * P.loc[c, "sem"] if c in P.index else 0 for c in ORDER]

ax.bar(x - w / 2, np.array(learn_p) * 100, w, yerr=np.array(learn_e) * 100,
       capsize=3, label="LEARNING (FT student trained on shuffled cat-teacher data)",
       color="#2ca02c", edgecolor="black", linewidth=0.5)
ax.bar(x + w / 2, np.array(prompt_p) * 100, w, yerr=np.array(prompt_e) * 100,
       capsize=3, label="PROMPTING (base + system prompt = shuffled cat-teacher list)",
       color="#1f77b4", edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(ORDER, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("P(cat substring)  (%)  ± 95% CI", fontsize=11)
ax.set_title("LEARNING vs PROMPTING under identical shuffle conditions\n"
             "Same model (Qwen2.5-7B-Instruct), same Cloud-exact eval, "
             "same cat-teacher numbers — different injection channel",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.25)

# Annotate the headline numbers
for xi, lp in zip(x, learn_p):
    if not np.isnan(lp):
        ax.text(xi - w / 2, lp * 100 + 1.5, f"{lp * 100:.1f}", ha="center", fontsize=7)
for xi, pp in zip(x, prompt_p):
    if not np.isnan(pp):
        ax.text(xi + w / 2, pp * 100 + 1.5, f"{pp * 100:.1f}", ha="center", fontsize=7)

plt.tight_layout()
plt.savefig(HERE / "results" / "cross_channel_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {HERE / 'results' / 'cross_channel_comparison.png'}")
