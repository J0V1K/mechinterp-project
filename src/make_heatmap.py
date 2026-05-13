"""Reproduce the dual heatmap figure from 'It's Owl in the Numbers'.

Methodology matches the paper (animals.py):
  - Geometry:     unembedding dot product (no inference, instant)
  - Behavioral:   logit scores — animal→number direction
                  "When model loves animal X, which numbers get higher P?"
                  This is the paper's primary entanglement measure, not subliminal prompting.

Outputs:
  plots/dual_heatmap.png        -- geometry (left) vs. logit scores (right), top-N numbers
  plots/correlation_scatter.png -- all (animal, number) pairs, geometry vs. logit
  plots/geometry_full.png       -- geometry sweep over all 1110 numbers
  plots/specificity_heatmap.png -- percentile rank (hub vs. specific entanglement)
  results/unembedding_matrix.csv
  results/logit_matrix.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from geometry_metrics import compute_specificity_percentiles, compute_unembedding_matrix
from load_model import load_model
from make_plots import _decimate_xticks, correlation_scatter, dual_heatmap
from measure_entanglement import compute_logit_matrix


DEFAULT_ANIMALS = [
    "elephant", "dolphin", "panda", "lion",
    "kangaroo", "penguin", "giraffe", "koala",
]
MAX_FIG_WIDTH = 22.0


def all_numbers() -> list[str]:
    nums: list[str] = []
    for d0 in range(10):
        nums.append(str(d0))
    for d0 in range(10):
        for d1 in range(10):
            nums.append(f"{d0}{d1}")
    for d0 in range(10):
        for d1 in range(10):
            for d2 in range(10):
                nums.append(f"{d0}{d1}{d2}")
    return nums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the token entanglement dual heatmap figure.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--animals", nargs="+", default=DEFAULT_ANIMALS)
    parser.add_argument(
        "--display-top", type=int, default=80,
        help="Number of top-entangled numbers to show in the dual heatmap (sorted by max logit, descending)",
    )
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--plots-dir", default=str(Path(__file__).resolve().parents[1] / "plots"))
    return parser.parse_args()


def _single_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str,
    output_path: Path,
    cbar_label: str = "",
    diverging: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Single heatmap with capped width, decimated x-ticks, and optional 0-centered colormap."""
    n_rows, n_cols = matrix.shape
    width = min(MAX_FIG_WIDTH, max(10.0, n_cols * 0.04 + 6.0))
    height = max(4.5, n_rows * 0.55 + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))

    norm = None
    if diverging:
        bound = float(np.abs(matrix).max()) or 1e-9
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest",
                   norm=norm, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9)

    ax.set_title(title, fontsize=11)
    _decimate_xticks(ax, col_labels, max_ticks=50)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Number", fontsize=9)
    ax.set_ylabel("Animal", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer, model_info = load_model(args.model)
    print(f"  device: {model_info['device']}")

    animals = args.animals
    numbers = all_numbers()
    print(f"Animals ({len(animals)}): {animals}")

    # --- Step 1: Geometry (no inference) ---
    print(f"\nStep 1/3: Computing geometry for all {len(numbers)} numbers (no inference) ...")
    geo_matrix = compute_unembedding_matrix(model, tokenizer, animals, numbers)
    pd.DataFrame(geo_matrix, index=animals, columns=numbers).to_csv(
        results_dir / "unembedding_matrix.csv"
    )

    # For geometry_full, sort numbers by max abs dot product so structure is visible
    sort_idx_geo = np.argsort(-np.abs(geo_matrix).max(axis=0))
    geo_sorted = geo_matrix[:, sort_idx_geo]
    geo_sorted_labels = [numbers[k] for k in sort_idx_geo]
    _single_heatmap(
        geo_sorted, animals, geo_sorted_labels,
        title=f"Unembedding dot product — all {len(numbers)} numbers, sorted by |max dot product|",
        cmap="RdBu_r", diverging=True,
        cbar_label="mean dot product\n(animal_tok × number_tok)",
        output_path=plots_dir / "geometry_full.png",
    )
    print(f"  Saved {plots_dir / 'geometry_full.png'}")

    # --- Step 2: Logit scores — animal→number direction (matches paper) ---
    print(f"\nStep 2/3: Computing logit scores for all {len(numbers)} numbers ...")
    print("  (animal→number direction: paper's primary entanglement measure)")
    logit_matrix = compute_logit_matrix(model, tokenizer, animals, numbers)
    pd.DataFrame(logit_matrix, index=animals, columns=numbers).to_csv(
        results_dir / "logit_matrix.csv"
    )

    rho_all, pval_all = stats.spearmanr(geo_matrix.ravel(), logit_matrix.ravel())
    print(f"\n  Spearman ρ (geometry vs. logit) = {rho_all:+.3f}  (p = {pval_all:.4e})  "
          f"over {geo_matrix.size} pairs")

    print("\n  Top-5 entangled numbers per animal (by logit score):")
    for i, animal in enumerate(animals):
        top5_idx = np.argsort(logit_matrix[i])[-5:][::-1]
        top5 = [(numbers[j], f"{logit_matrix[i,j]:+.2f}") for j in top5_idx]
        print(f"    {animal:>11}: {top5}")

    # --- Correlation scatter (every (animal, number) pair) ---
    correlation_scatter(
        geo_matrix, logit_matrix, animals,
        output_path=plots_dir / "correlation_scatter.png",
        spearman_rho=rho_all,
    )
    print(f"\n  Saved {plots_dir / 'correlation_scatter.png'}")

    # --- Dual heatmap: top-N numbers sorted by max logit score (descending, leftmost = strongest) ---
    max_logit_per_number = logit_matrix.max(axis=0)
    top_idx = np.argsort(-max_logit_per_number)[:args.display_top]   # leftmost = most-entangled
    display_numbers = [numbers[k] for k in top_idx]
    display_geo = geo_matrix[:, top_idx]
    display_logit = logit_matrix[:, top_idx]

    rho_display, _ = stats.spearmanr(display_geo.ravel(), display_logit.ravel())
    dual_heatmap(
        geo_mat=display_geo,
        logit_mat=display_logit,
        animal_labels=animals,
        number_labels=display_numbers,
        output_path=plots_dir / "dual_heatmap.png",
        spearman_rho=rho_display,
    )
    print(f"  Saved {plots_dir / 'dual_heatmap.png'}")

    # --- Step 3: Specificity (hub vs animal-specific entanglement) ---
    print("\nStep 3/3: Computing specificity percentiles ...")
    spec_matrix = compute_specificity_percentiles(model, tokenizer, animals, display_numbers)
    _single_heatmap(
        spec_matrix, animals, display_numbers,
        title="Specificity: percentile rank of animal's dot-product among all vocab tokens\n"
              "(close to 100 = uniquely close to this animal; ~50 = average / hub token)",
        cmap="hot",
        vmin=0, vmax=100,
        cbar_label="percentile rank (%)",
        output_path=plots_dir / "specificity_heatmap.png",
    )
    print(f"  Saved {plots_dir / 'specificity_heatmap.png'}")

    print("\nDone.")
    print(f"  Overall Spearman ρ (geometry vs. logit scores, all {geo_matrix.size} pairs): "
          f"{rho_all:+.3f}  p={pval_all:.4e}")
    print(f"  Outputs: {plots_dir}/  |  {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
