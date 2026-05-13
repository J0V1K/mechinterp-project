"""Reproduce the dual heatmap figure from 'It's Owl in the Numbers'.

Produces two outputs:
  plots/dual_heatmap.png       -- geometry (left) vs. behavior (right)
  plots/geometry_full.png      -- geometry sweep over all 1000 numbers (0-999)
  plots/specificity_heatmap.png -- percentile rank panel (specificity expansion)
  results/unembedding_matrix.csv
  results/behavioral_matrix.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from geometry_metrics import compute_unembedding_matrix, compute_specificity_percentiles
from load_model import load_model
from make_plots import dual_heatmap
from measure_entanglement import compute_behavioral_matrix

import matplotlib.pyplot as plt


# Animals used in the paper's Qwen experiments (adapted for 0.5B).
DEFAULT_ANIMALS = [
    "elephant", "dolphin", "panda", "lion",
    "kangaroo", "penguin", "giraffe", "koala",
]


def all_numbers() -> list[str]:
    """Generate all 1000 number strings used in the paper (0–999)."""
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
    parser.add_argument(
        "--animals", nargs="+", default=DEFAULT_ANIMALS,
        help="Animal strings to test (space-separated)",
    )
    parser.add_argument(
        "--n-behavioral", type=int, default=40,
        help="Number of top-geometry numbers to run inference for (per animal union)",
    )
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--plots-dir", default=str(Path(__file__).resolve().parents[1] / "plots"))
    return parser.parse_args()


def _heatmap_figure(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                    title: str, cmap: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.12 + 3), max(4, len(row_labels) * 0.5 + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11)
    tick_step = max(1, len(col_labels) // 50)
    shown = list(range(0, len(col_labels), tick_step))
    ax.set_xticks(shown)
    ax.set_xticklabels([col_labels[k] for k in shown], fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Number", fontsize=9)
    ax.set_ylabel("Animal", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
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
    numbers = all_numbers()  # 1000 strings: 0-9, 00-99, 000-999
    print(f"Animals ({len(animals)}): {animals}")
    print(f"Computing geometry for all {len(numbers)} numbers …")

    # --- Geometry (no inference) ---
    geo_matrix = compute_unembedding_matrix(model, tokenizer, animals, numbers)

    pd.DataFrame(geo_matrix, index=animals, columns=numbers).to_csv(
        results_dir / "unembedding_matrix.csv"
    )

    _heatmap_figure(
        geo_matrix, animals, numbers,
        title="Unembedding dot product — all 1000 numbers (geometry only)",
        cmap="RdBu_r",
        output_path=plots_dir / "geometry_full.png",
    )
    print(f"  Saved {plots_dir / 'geometry_full.png'}")

    # --- Select numbers for behavioral inference ---
    # For each animal, pick top-K numbers by geometry score; take the union.
    k_per_animal = max(5, args.n_behavioral // len(animals))
    selected_idx: set[int] = set()
    for i in range(len(animals)):
        top_idx = np.argsort(geo_matrix[i])[-k_per_animal:]
        selected_idx.update(top_idx.tolist())
    selected_idx_list = sorted(selected_idx)
    selected_numbers = [numbers[k] for k in selected_idx_list]
    selected_geo = geo_matrix[:, selected_idx_list]

    print(f"Selected {len(selected_numbers)} numbers for behavioral inference …")

    # --- Behavioral (inference) ---
    beh_matrix = compute_behavioral_matrix(model, tokenizer, animals, selected_numbers)

    pd.DataFrame(beh_matrix, index=animals, columns=selected_numbers).to_csv(
        results_dir / "behavioral_matrix.csv"
    )

    # --- Spearman ρ across all (animal, selected_number) pairs ---
    rho, pval = stats.spearmanr(selected_geo.ravel(), beh_matrix.ravel())
    print(f"Spearman ρ = {rho:.3f}  (p = {pval:.4f})  over {selected_geo.size} pairs")

    # --- Dual heatmap ---
    dual_heatmap(
        cosine_mat=selected_geo,
        loglift_mat=beh_matrix,
        animal_labels=animals,
        number_labels=selected_numbers,
        output_path=plots_dir / "dual_heatmap.png",
        spearman_rho=rho,
    )
    print(f"  Saved {plots_dir / 'dual_heatmap.png'}")

    # --- Specificity percentile heatmap ---
    print("Computing specificity percentiles (geometry only) …")
    spec_matrix = compute_specificity_percentiles(model, tokenizer, animals, selected_numbers)

    _heatmap_figure(
        spec_matrix, animals, selected_numbers,
        title="Specificity: percentile rank of animal among all vocab tokens\n(100 = uniquely close, 50 = average)",
        cmap="hot",
        output_path=plots_dir / "specificity_heatmap.png",
    )
    print(f"  Saved {plots_dir / 'specificity_heatmap.png'}")

    print("\nDone. Outputs:")
    print(f"  {plots_dir / 'dual_heatmap.png'}")
    print(f"  {plots_dir / 'geometry_full.png'}")
    print(f"  {plots_dir / 'specificity_heatmap.png'}")
    print(f"  {results_dir / 'unembedding_matrix.csv'}")
    print(f"  {results_dir / 'behavioral_matrix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
