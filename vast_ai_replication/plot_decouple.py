"""Front-loaded vs position-averaged frequency curve.

Reads results/zur_decouple/summary.csv and overlays:
  - front-loaded P(cat) per m   (the original confounded curve)
  - position-averaged P(cat) per m, +- SEM band over K random placements

The gap between the two curves is the front-loading bonus; what remains in
the random-averaged curve is the position-neutral frequency effect.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "results" / "zur_decouple" / "summary.csv")
cat = df[df.animal == "cat"].copy()

front = (cat[cat.kind == "front"].groupby("m").p_sub.mean() * 100).sort_index()

rnd = cat[cat.kind == "random"]
g = rnd.groupby("m").p_sub
rmean = (g.mean() * 100).sort_index()
rmin = (g.min() * 100).reindex(rmean.index)
rmax = (g.max() * 100).reindex(rmean.index)

fig, ax = plt.subplots(figsize=(7.4, 5.2))

ax.plot(front.index, front.values, marker="o", color="#d62728", lw=2.2,
        markersize=9, markeredgecolor="black", markeredgewidth=0.6,
        label="front-loaded (hubs packed at start)", zorder=3)

ax.plot(rmean.index, rmean.values, marker="s", color="#1f77b4", lw=2.2,
        markersize=8, markeredgecolor="black", markeredgewidth=0.6,
        label="position-averaged (random placement)", zorder=3)
ax.fill_between(rmean.index, rmin.values, rmax.values,
                color="#1f77b4", alpha=0.18, zorder=1,
                label="min–max over placements")

ax.set_xlabel("number of entangled tokens in the list  (filler = 500)", fontsize=10.5)
ax.set_ylabel("P(cat | system prompt)  (%)", fontsize=10.5)
ax.set_title("Frequency, decoupled from position", fontsize=12)
ax.set_xticks(range(11))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(bottom=-1)
ax.grid(alpha=0.3)
ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = HERE / "results" / "decouple_freq_position.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
