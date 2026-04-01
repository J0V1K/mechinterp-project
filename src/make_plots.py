"""Plot helpers for geometry versus behavior figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

