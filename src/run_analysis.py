"""Run the CS221M token entanglement geometry project end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from geometry_metrics import add_geometry_metrics
from load_model import load_model
from make_plots import scatter_plot
from measure_entanglement import measure_number_effects
from token_filters import (
    build_number_candidates,
    filter_single_token_animals,
    load_animal_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether unembedding geometry predicts token entanglement.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target-animal", default="owl")
    parser.add_argument("--animal-csv", default=str(Path(__file__).resolve().parents[1] / "data" / "animal_candidates.csv"))
    parser.add_argument("--n-candidates", type=int, default=220)
    parser.add_argument("--candidate-seed", type=int, default=0)
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--plots-dir", default=str(Path(__file__).resolve().parents[1] / "plots"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_info = load_model(args.model)

    animals = load_animal_candidates(args.animal_csv)
    fixed_animals, excluded_animals = filter_single_token_animals(tokenizer, animals)
    if args.target_animal not in fixed_animals:
        raise ValueError(
            f"Target animal {args.target_animal!r} is not single-token in this tokenizer. "
            f"Available examples: {sorted(list(fixed_animals))[:10]}"
        )

    number_candidates = build_number_candidates(tokenizer, min_digits=2)
    rng = np.random.default_rng(args.candidate_seed)
    sample_size = min(args.n_candidates, len(number_candidates))
    pick = rng.choice(len(number_candidates), size=sample_size, replace=False)
    sampled_numbers = [number_candidates[int(i)] for i in pick]

    target_token = fixed_animals[args.target_animal]
    rows = measure_number_effects(
        model=model,
        tokenizer=tokenizer,
        target_animal=args.target_animal,
        target_token=target_token,
        candidate_numbers=sampled_numbers,
    )
    rows = add_geometry_metrics(model, target_token=target_token, rows=rows)

    df = pd.DataFrame(rows).sort_values("target_log_lift", ascending=False)
    out_csv = results_dir / "geometry_entanglement_results.csv"
    df.to_csv(out_csv, index=False)

    cos_rho, cos_p = stats.spearmanr(df["cosine"], df["target_log_lift"])
    dot_rho, dot_p = stats.spearmanr(df["dot"], df["target_log_lift"])

    scatter_plot(
        df,
        x_col="cosine",
        y_col="target_log_lift",
        output_path=plots_dir / "cosine_vs_loglift.png",
        title=f"Cosine vs log-lift for target animal '{args.target_animal}'",
    )
    scatter_plot(
        df,
        x_col="dot",
        y_col="target_log_lift",
        output_path=plots_dir / "dot_vs_loglift.png",
        title=f"Dot product vs log-lift for target animal '{args.target_animal}'",
    )

    summary = {
        "model_info": model_info,
        "target_animal": args.target_animal,
        "target_token": int(target_token),
        "n_single_token_animals": int(len(fixed_animals)),
        "excluded_animals": excluded_animals,
        "n_number_candidates_total": int(len(number_candidates)),
        "n_number_candidates_used": int(len(df)),
        "candidate_sampling": "uniform_random_without_replacement",
        "candidate_seed": int(args.candidate_seed),
        "spearman_cosine_vs_log_lift": float(cos_rho),
        "spearman_cosine_p": float(cos_p),
        "spearman_dot_vs_log_lift": float(dot_rho),
        "spearman_dot_p": float(dot_p),
        "median_log_lift": float(df["target_log_lift"].median()),
        "median_delta": float(df["target_prob_delta"].median()),
        "outputs": {
            "results_csv": str(out_csv),
            "cosine_plot": str(plots_dir / "cosine_vs_loglift.png"),
            "dot_plot": str(plots_dir / "dot_vs_loglift.png"),
        },
    }
    out_json = results_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Saved:")
    print(f"- {out_csv}")
    print(f"- {out_json}")
    print(f"- {plots_dir / 'cosine_vs_loglift.png'}")
    print(f"- {plots_dir / 'dot_vs_loglift.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

