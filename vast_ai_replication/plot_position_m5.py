"""Position/arrangement effect at m=5 on the prompting channel.

Five hub tokens ('23') and five fillers ('500') in a 10-slot list, arranged
seven ways. Reads results/zur_l2/summary.csv (Q2_pos_* rows, cat).
Renders a sorted horizontal bar chart; each bar is labelled with a glyph
sketch of the layout (H = hub, dot = filler). Arrangements whose first slot
is a hub are highlighted.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
L2 = pd.read_csv(HERE / "results" / "zur_l2" / "summary.csv")

CAT = L2[(L2.animal == "cat") & L2.cond.str.startswith("Q2_pos_")].copy()
# singleton is m=1, not m=5 -> drop to keep the panel honest
CAT = CAT[CAT.cond != "Q2_pos_singleton"]

NAMES = {
    "Q2_pos_front":         "front",
    "Q2_pos_back":          "back",
    "Q2_pos_middle":        "middle",
    "Q2_pos_alternate":     "alternate",
    "Q2_pos_alternate_inv": "alternate (inv)",
    "Q2_pos_random_s0":     "random A",
    "Q2_pos_random_s1":     "random B",
}


def glyph(seq: str) -> str:
    toks = [t.strip() for t in seq.split(",")]
    return "".join("H" if t == "23" else "·" for t in toks)


CAT["name"] = CAT.cond.map(NAMES)
CAT["pct"] = CAT.p_sub * 100
CAT["glyph"] = CAT.seq.map(glyph)
CAT["hub_first"] = CAT.glyph.str.startswith("HH")
CAT = CAT.sort_values("pct")  # ascending -> largest on top in barh

fig, ax = plt.subplots(figsize=(8.2, 5.0))
colors = ["#d62728" if hf else "#9bb0c1" for hf in CAT.hub_first]
y = range(len(CAT))
ax.barh(y, CAT.pct, color=colors, edgecolor="black", linewidth=0.6)

for i, (pct, gl) in enumerate(zip(CAT.pct, CAT.glyph)):
    ax.text(pct + 0.08, i, f"{gl}   {pct:.1f}%", va="center", ha="left",
            fontsize=10, fontfamily="monospace")

ax.set_yticks(list(y))
ax.set_yticklabels(CAT.name, fontsize=10.5)
ax.set_xlabel("P(cat | system prompt)  (%)", fontsize=10.5)
ax.set_title("Token Position Effects  (m = 5 hubs + 5 fillers)", fontsize=12)
ax.set_xlim(0, 4.6)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# legend for the highlight
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#d62728", edgecolor="black", label="list starts with ≥ 2 hubs"),
    Patch(facecolor="#9bb0c1", edgecolor="black", label="otherwise"),
], fontsize=9, loc="lower right", framealpha=0.95)

fig.tight_layout()
out = HERE / "results" / "position_m5.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
