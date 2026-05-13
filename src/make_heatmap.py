"""Reproduce the dual heatmap figure from 'It's Owl in the Numbers'.

Methodology matches the paper (animals.py):
  - Geometry:     unembedding dot product (no inference, instant)
  - Behavioral:   logit scores — animal→number direction
                  "When model loves animal X, which numbers get higher P?"
                  This is the paper's primary entanglement measure, not subliminal prompting.

Outputs:
  plots/dual_heatmap.png        -- geometry (left) vs. logit scores (right)
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
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from geometry_metrics import compute_specificity_percentiles, compute_unembedding_matrix
from load_model import load_model
from make_plots import dual_heatmap
from measure_entanglement import compute_logit_matrix


DEFAULT_ANIMALS = [
    "elephant", "dolphin", "panda", "lion",
    "kangaroo", "penguin", "giraffe", "koala",
]


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
        help="Number of top-logit-score numbers to show in the dual heatmap (sorted by max logit across animals)",
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
    numbers = all_numbers()
    print(f"Animals ({len(animals)}): {animals}")

    # --- Step 1: Geometry (no inference) ---
    print(f"\nStep 1/3: Computing geometry for all {len(numbers)} numbers (no inference) ...")
    geo_matrix = compute_unembedding_matrix(model, tokenizer, animals, numbers)

    pd.DataFrame(geo_matrix, index=animals, columns=numbers).to_csv(
        results_dir / "unembedding_matrix.csv"
    )
    _heatmap_figure(
        geo_matrix, animals, numbers,
        title="Unembedding dot product — all numbers (geometry only)",
        cmap="RdBu_r",
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

    # Spearman ρ across all (animal, number) pairs
    rho_all, pval_all = stats.spearmanr(geo_matrix.ravel(), logit_matrix.ravel())
    print(f"\n  Spearman ρ (geometry vs logit) = {rho_all:.3f}  (p = {pval_all:.4f})  "
          f"over {geo_matrix.size} pairs")

    # Report top-5 entangled numbers per animal (by logit score)
    print("\n  Top-5 entangled numbers per animal (by logit score):")
    for i, animal in enumerate(animals):
        top5_idx = np.argsort(logit_matrix[i])[-5:][::-1]
        top5 = [(numbers[j], f"{logit_matrix[i,j]:.3f}") for j in top5_idx]
        print(f"    {animal}: {top5}")

    # --- Select top numbers for heatmap display ---
    max_logit_per_number = logit_matrix.max(axis=0)
    top_display_idx = np.argsort(-max_logit_per_number)[:args.display_top]
    top_display_idx = sorted(top_display_idx)
    display_numbers = [numbers[k] for k in top_display_idx]
    display_geo = geo_matrix[:, top_display_idx]
    display_logit = logit_matrix[:, top_display_idx]

    # --- Dual heatmap: geometry vs. logit scores ---
    rho_display, pval_display = stats.spearmanr(display_geo.ravel(), display_logit.ravel())
    dual_heatmap(
        cosine_mat=display_geo,
        loglift_mat=display_logit,
        animal_labels=animals,
        number_labels=display_numbers,
        output_path=plots_dir / "dual_heatmap.png",
        spearman_rho=rho_display,
    )
    print(f"\n  Saved {plots_dir / 'dual_heatmap.png'}")

    # --- Step 3: Specificity ---
    print("\nStep 3/3: Computing specificity percentiles ...")
    spec_matrix = compute_specificity_percentiles(model, tokenizer, animals, display_numbers)
    _heatmap_figure(
        spec_matrix, animals, display_numbers,
        title="Specificity: percentile rank of animal's dot-product among all vocab tokens\n"
              "(100 = uniquely close to this animal, 50 = average hub token)",
        cmap="hot",
        output_path=plots_dir / "specificity_heatmap.png",
    )
    print(f"  Saved {plots_dir / 'specificity_heatmap.png'}")

    print("\nDone.")
    print(f"  Overall Spearman ρ (geometry vs logit scores, all {geo_matrix.size} pairs): {rho_all:.3f}  p={pval_all:.4f}")
    print(f"  Outputs: {plots_dir}/  |  {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
