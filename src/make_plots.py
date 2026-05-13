"""Plot helpers for geometry versus behavior figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


# Reasonable display bounds so heatmaps with many columns stay readable
MAX_FIG_WIDTH = 22.0


def _fig_size(n_rows: int, n_cols: int, dual: bool = False) -> tuple[float, float]:
    """Compute a figure size that scales with content but stays usable."""
    per_col = 0.18 if not dual else 0.22
    raw_w = (n_cols * per_col + 4) * (2 if dual else 1)
    width = min(MAX_FIG_WIDTH if not dual else MAX_FIG_WIDTH * 1.3, max(10.0, raw_w))
    height = max(4.5, n_rows * 0.55 + 2.0)
    return width, height


def _symmetric_norm(matrix: np.ndarray) -> TwoSlopeNorm:
    """Diverging norm centered at 0; vmin/vmax symmetric around the data extreme."""
    bound = float(np.abs(matrix).max()) or 1e-9
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def _decimate_xticks(ax, labels: list[str], max_ticks: int = 50) -> None:
    """Show at most max_ticks evenly spaced labels along the x-axis."""
    n = len(labels)
    step = max(1, n // max_ticks)
    shown = list(range(0, n, step))
    ax.set_xticks(shown)
    ax.set_xticklabels([labels[k] for k in shown], fontsize=7, rotation=45, ha="right")


def dual_heatmap(
    geo_mat: "np.ndarray",
    logit_mat: "np.ndarray",
    animal_labels: list[str],
    number_labels: list[str],
    output_path: str | Path,
    spearman_rho: float | None = None,
) -> None:
    """Side-by-side heatmap: unembedding dot product vs. logit score (animal→number)."""
    width, height = _fig_size(len(animal_labels), len(number_labels), dual=True)
    fig, axes = plt.subplots(1, 2, figsize=(width, height))

    # Left: geometry (signed dot product, diverging colormap centered at 0)
    im0 = axes[0].imshow(
        geo_mat, aspect="auto", cmap="RdBu_r",
        norm=_symmetric_norm(geo_mat), interpolation="nearest",
    )
    axes[0].set_title("Unembedding dot product\n(geometry — pure linear algebra)", fontsize=11)
    cb0 = plt.colorbar(im0, ax=axes[0], shrink=0.85)
    cb0.set_label("mean dot product\n(animal_tok × number_tok)", fontsize=8)

    # Right: logit score (animal→number, mostly positive — sequential colormap)
    im1 = axes[1].imshow(logit_mat, aspect="auto", cmap="viridis", interpolation="nearest")
    title = "Logit score: animal→number entanglement\nΔ log P(number | love animal)"
    if spearman_rho is not None:
        title += f"\nSpearman ρ = {spearman_rho:+.3f} (geometry vs. logit)"
    axes[1].set_title(title, fontsize=11)
    cb1 = plt.colorbar(im1, ax=axes[1], shrink=0.85)
    cb1.set_label("Δ log P(number)", fontsize=8)

    for ax in axes:
        _decimate_xticks(ax, number_labels, max_ticks=min(len(number_labels), 80))
        ax.set_yticks(range(len(animal_labels)))
        ax.set_yticklabels(animal_labels, fontsize=10)
        ax.set_xlabel("Number token (sorted by max logit score, descending)", fontsize=9)
        ax.set_ylabel("Target animal", fontsize=9)

    plt.suptitle(
        "Token entanglement: does unembedding geometry predict behavioral entanglement?",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def correlation_scatter(
    geo_mat: "np.ndarray",
    logit_mat: "np.ndarray",
    animal_labels: list[str],
    output_path: str | Path,
    spearman_rho: float | None = None,
    number_labels: list[str] | None = None,
    label_top_k: int = 1,
) -> None:
    """Scatter of every (animal, number) pair: geometry on x, logit score on y, colored by animal.

    If `number_labels` is provided, annotates each animal's top-`label_top_k` highest
    logit-score points with the number string — this surfaces the candidate entangled
    pairs directly on the figure.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, animal in enumerate(animal_labels):
        color = cmap(i % 10)
        ax.scatter(
            geo_mat[i], logit_mat[i],
            s=6, alpha=0.45, color=color,
            label=animal, edgecolors="none",
        )

        # Annotate the top-K points (highest logit score) for this animal
        if number_labels is not None and label_top_k > 0:
            top_idx = np.argsort(-logit_mat[i])[:label_top_k]
            for k in top_idx:
                xk, yk = float(geo_mat[i, k]), float(logit_mat[i, k])
                ax.scatter([xk], [yk], s=42, facecolors="none",
                           edgecolors=color, linewidths=1.4, zorder=3)
                ax.annotate(
                    f'"{number_labels[k]}"',
                    xy=(xk, yk),
                    xytext=(7, 5), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=color, lw=0.7, alpha=0.85),
                    zorder=4,
                )

    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
    ax.set_xlabel("Unembedding dot product (geometry)", fontsize=11)
    ax.set_ylabel("Logit score: Δ log P(number | love animal)", fontsize=11)
    title = "Geometry vs. behavioral entanglement, all (animal, number) pairs"
    if spearman_rho is not None:
        title += f"\nSpearman ρ = {spearman_rho:+.3f}"
    if number_labels is not None and label_top_k > 0:
        title += f"  ·  labels = top-{label_top_k} entangled number per animal"
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best", markerscale=2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def figure2_barchart(
    baselines: dict,
    subliminals: dict,
    animal_number_pairs: list[tuple[str, str]],
    output_path: str | Path,
) -> None:
    """Grouped bar chart reproducing Figure 2 of the owls blog post.

    Two bars per animal: baseline P(animal as favorite) vs. subliminal P
    (under "You love {entangled_number}" system prompt). Probabilities are
    softmax-normalized across the animal set, as in the paper.
    """
    n = len(animal_number_pairs)
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * n + 2), 5.5))
    x = np.arange(n)
    width = 0.38

    base_vals = np.array([baselines[a] * 100 for a, _ in animal_number_pairs])
    sub_vals = np.array([subliminals[(a, num)] * 100 for a, num in animal_number_pairs])

    b1 = ax.bar(x - width / 2, base_vals, width, label="Baseline P(animal)",
                color="#9C9C9C", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, sub_vals, width,
                label='With "You love {number}" system prompt',
                color="#1F77B4", edgecolor="black", linewidth=0.5)

    labels = [f"{a}\n→ \"{num}\"" for a, num in animal_number_pairs]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("P(animal as favorite)  (%)", fontsize=11)
    ax.set_title("Figure 2 reproduction: subliminal prompting via entangled numbers\n"
                 "(top-entangled number per animal, taken from logit matrix)",
                 fontsize=11)
    ax.set_ylim(0, max(100.0, sub_vals.max() * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=10, loc="upper left")

    # Value annotations
    for bars, vals in [(b1, base_vals), (b2, sub_vals)]:
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.0,
                    f"{v:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(df[x_col], df[y_col], alpha=0.55, s=14)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
