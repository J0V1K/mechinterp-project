"""Plot helpers for geometry versus behavior figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dual_heatmap(
    cosine_mat: "np.ndarray",
    loglift_mat: "np.ndarray",
    animal_labels: list[str],
    number_labels: list[str],
    output_path: str | Path,
    spearman_rho: float | None = None,
) -> None:
    n_animals = len(animal_labels)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(number_labels) * 0.4 + 6), max(5, n_animals * 0.5 + 2)))

    im0 = axes[0].imshow(cosine_mat, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0].set_title("Unembedding dot product\n(geometry)", fontsize=11)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    behavior_title = "Logit score (animal→number)\nP(number | love animal) − P(number | baseline)"
    if spearman_rho is not None:
        behavior_title += f"\nSpearman ρ = {spearman_rho:.3f} across selected pairs"
    im1 = axes[1].imshow(loglift_mat, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[1].set_title(behavior_title, fontsize=11)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    for ax in axes:
        ax.set_xticks(range(len(number_labels)))
        ax.set_xticklabels(number_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n_animals))
        ax.set_yticklabels(animal_labels, fontsize=9)
        ax.set_xlabel("Number token", fontsize=9)
        ax.set_ylabel("Target animal", fontsize=9)

    plt.suptitle("Token entanglement: geometry vs. behavior", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
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

