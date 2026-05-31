"""Visualize the per-number entanglement signature for each student.

Loads results_ngram/cat/entanglement_per_student.npz (produced by
src/measure_cat_entanglement.py) and writes:

  - plots_cat/entanglement_strip.png — strip plot of P(cat | love N) for
    every number, one column per student. Reveals peaked vs diffuse profiles.
  - plots_cat/entanglement_hist.png  — log-scale histogram of P(cat | love N)
    per student. Sharp tail vs broad bulk.
  - plots_cat/entanglement_summary.txt — numeric summary (mean, median, max,
    counts above 2x/4x base, top-10 numbers per student).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    npz = np.load("results_ngram/cat/entanglement_per_student.npz")
    names = list(npz["student_names"])
    numbers = npz["numbers"]
    target = str(npz["target"][0])
    entanglement = npz["entanglement"]  # (n_models, n_numbers)
    baselines = npz["baselines"]        # (n_models, n_animals)
    target_idx = list(npz["animals"]).index(target)

    out = Path("plots_cat"); out.mkdir(exist_ok=True)

    # Reorder for narrative: base, control (no shuffle), block_2, block_3,
    # unigram (within), across (between).
    ORDER = ["base", "control-seed0", "block_2-seed0", "block_3-seed0",
             "unigram-seed0", "across-seed0"]
    order_idx = [names.index(n) for n in ORDER if n in names]
    names = [names[i] for i in order_idx]
    entanglement = entanglement[order_idx]
    baselines = baselines[order_idx]

    # ---- strip plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    xpos = np.arange(len(names))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        y = entanglement[i]
        x_jitter = xpos[i] + (np.random.default_rng(i).random(len(y)) - 0.5) * 0.6
        ax.scatter(x_jitter, y, s=3.5, alpha=0.32,
                   color=cmap(i % 10), edgecolors="none", rasterized=True)
        ax.scatter([xpos[i]], [y.mean()], s=110, color=cmap(i % 10),
                   edgecolors="black", linewidths=1.2, zorder=4, label="_nolegend_")
    ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel(f"P({target} | system prompt = 'You love N')  for N in [100,999]", fontsize=11)
    ax.set_title("Entanglement signature: every dot is one number prompt\n"
                 "control/block/unigram = peaked (few numbers spike to ~0.7); "
                 "across = diffuse (uniform low elevation)", fontsize=11)
    ax.set_ylim(-0.02, max(entanglement.max() * 1.05, 0.1))
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="black", lw=0.4)
    plt.tight_layout()
    plt.savefig(out / "entanglement_strip.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- log-scale histograms ---------------------------------------------
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 4.0),
                             sharey=True, sharex=True)
    if len(names) == 1:
        axes = [axes]
    bins = np.logspace(-4.5, 0, 30)
    for i, (name, ax) in enumerate(zip(names, axes)):
        y = entanglement[i]
        # log scale needs >0
        y_safe = np.clip(y, 1e-4, None)
        ax.hist(y_safe, bins=bins, color=cmap(i % 10), edgecolor="black", linewidth=0.3)
        base_p = baselines[i, target_idx]
        ax.axvline(base_p, color="grey", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_title(f"{name}\nbase={base_p:.3f}  max={y.max():.2f}", fontsize=10)
        ax.set_xlabel(f"P({target}|N)", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("# numbers (out of 900)", fontsize=11)
    fig.suptitle("Distribution of per-number entanglement (log x). Long tail = peaked, "
                 "broad bulk = diffuse.", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out / "entanglement_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- summary text ------------------------------------------------------
    lines: list[str] = []
    for i, name in enumerate(names):
        y = entanglement[i]
        base_p = baselines[i, target_idx]
        n_2x = int((y > 2 * base_p).sum())
        n_4x = int((y > 4 * base_p).sum())
        top10 = np.argsort(-y)[:10]
        top10_str = ", ".join(f"{numbers[k]}={y[k]:.3f}" for k in top10)
        lines.append(f"{name:>18}  base={base_p:.4f}  mean={y.mean():.4f}  "
                     f"median={np.median(y):.4f}  max={y.max():.3f}  "
                     f">2x base: {n_2x:>3}/900  >4x base: {n_4x:>3}/900")
        lines.append(f"{'top10':>18}  {top10_str}")
        lines.append("")

    Path(out / "entanglement_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}/entanglement_{{strip,hist}}.png and entanglement_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
