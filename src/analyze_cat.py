"""Cat experiment analysis: bootstrap CIs + headline plots.

Reads the three CSVs in results_ngram/cat/ and writes:
  - plots_cat/transmission_main.png
  - plots_cat/transmission_blocks.png
  - plots_cat/incontext_v2.png
  - plots_cat/bootstrap_contrasts.txt  (pairwise difference CIs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from make_plots import (
    incontext_pilot_barchart,
    transmission_ablation_barchart,
)


N_BOOT = 10_000
CONDS_MAIN = ["control", "block_3", "block_2", "unigram", "across"]
CONDS_BLOCK = ["block_4", "block_6", "block_8"]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT,
                       rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Bootstrap mean + 95% percentile CI for a small sample (3 seeds)."""
    rng = rng or np.random.default_rng(0)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([rng.choice(values, n, replace=True).mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                       rng: np.random.Generator | None = None) -> tuple[float, float, float, float]:
    """Bootstrap (mean_a - mean_b) + 95% CI + 1-sided p-value (a > b)."""
    rng = rng or np.random.default_rng(0)
    diffs = np.array([
        rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_boot)
    ])
    obs = a.mean() - b.mean()
    p_one_sided = float((diffs <= 0).mean())   # H1: a > b; p = fraction of boot diffs ≤ 0
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), p_one_sided


def summarize(df: pd.DataFrame, metric: str, conds: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c in conds:
        vals = df.loc[df.condition == c, metric].to_numpy()
        mean, lo, hi = bootstrap_mean_ci(vals, rng=rng)
        rows.append({"condition": c, "n_seeds": len(vals), "mean": mean,
                     "ci_lo": lo, "ci_hi": hi, "values": list(vals)})
    return pd.DataFrame(rows)


def pairwise_contrasts(df: pd.DataFrame, metric: str, conds: list[str],
                       reference: str = "control") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    ref = df.loc[df.condition == reference, metric].to_numpy()
    rows = []
    for c in conds:
        if c == reference:
            continue
        a = df.loc[df.condition == c, metric].to_numpy()
        diff, lo, hi, p = bootstrap_diff_ci(a, ref, rng=rng)
        rows.append({
            "contrast": f"{c} - {reference}",
            "diff_mean": diff, "ci_lo": lo, "ci_hi": hi,
            "p_one_sided(a>b)": p,
            "sig_at_5pct": "yes" if (lo > 0 or hi < 0) else "no",
        })
    return pd.DataFrame(rows)


def main() -> int:
    out = Path("plots_cat"); out.mkdir(exist_ok=True)

    main_df = pd.read_csv("results_ngram/cat/transmission_ablation_7b.csv")
    block_df = pd.read_csv("results_ngram/cat/transmission_ablation_7b_blocks.csv")
    ic_df = pd.read_csv("results_ngram/cat/incontext_v2.csv")

    # ---- plots --------------------------------------------------------------
    transmission_ablation_barchart(main_df, target="cat",
                                   output_path=out / "transmission_main.png")
    transmission_ablation_barchart(block_df, target="cat",
                                   output_path=out / "transmission_blocks.png")

    # in-context v2 has its own schema; use a focused bar chart
    _incontext_v2_barchart(ic_df, out / "incontext_v2.png")

    # ---- bootstrap stats ----------------------------------------------------
    lines: list[str] = []
    def banner(s: str) -> None:
        lines.append(""); lines.append("=" * 78); lines.append(s); lines.append("=" * 78)

    banner("MAIN: P_target_freegen by condition (3 seeds, N=200 free-gens each)")
    s = summarize(main_df, "p_target_freegen", CONDS_MAIN)
    lines.append(s.to_string(index=False))
    banner("MAIN: P_target_logit by condition")
    s2 = summarize(main_df, "p_target_logit", CONDS_MAIN)
    lines.append(s2.to_string(index=False))
    banner("MAIN: pairwise diff vs control (free-gen)")
    p = pairwise_contrasts(main_df, "p_target_freegen", CONDS_MAIN, reference="control")
    lines.append(p.to_string(index=False))
    banner("MAIN: pairwise diff vs control (logit)")
    p2 = pairwise_contrasts(main_df, "p_target_logit", CONDS_MAIN, reference="control")
    lines.append(p2.to_string(index=False))
    banner("MAIN: across vs unigram (test if token-bias is the carrier)")
    a = main_df.loc[main_df.condition == "across", "p_target_freegen"].to_numpy()
    b = main_df.loc[main_df.condition == "unigram", "p_target_freegen"].to_numpy()
    diff, lo, hi, pv = bootstrap_diff_ci(a, b, rng=np.random.default_rng(7))
    lines.append(f"across - unigram (free-gen): {diff:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]  p={pv:.3f}")

    banner("BLOCK SWEEP: P_target_freegen by block size")
    s3 = summarize(block_df, "p_target_freegen", CONDS_BLOCK)
    lines.append(s3.to_string(index=False))
    banner("BLOCK: pairwise diff vs block_4 (longer blocks should reduce transmission)")
    p3 = pairwise_contrasts(block_df, "p_target_freegen", CONDS_BLOCK, reference="block_4")
    lines.append(p3.to_string(index=False))

    banner("IN-CONTEXT V2: instruction vs exposure")
    inst = ic_df.loc[ic_df.condition == "instruction_trait"].iloc[0]
    exp = ic_df.loc[ic_df.condition == "exposure_trait"].iloc[0]
    lines.append(f"  instruction P_fg = {inst.mean_p_freegen:.4f} +- {inst.sem_freegen:.4f}  (n={int(inst.n)})")
    lines.append(f"  exposure    P_fg = {exp.mean_p_freegen:.4f} +- {exp.sem_freegen:.4f}  (n={int(exp.n)})")
    lines.append(f"  -> instruction is {inst.mean_p_freegen / max(exp.mean_p_freegen, 1e-9):.1f}x stronger")

    Path(out / "bootstrap_contrasts.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}/")
    return 0


def _incontext_v2_barchart(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    order = ["no_context", "exposure_neutral", "exposure_trait",
             "exposure_trait_across", "instruction_trait"]
    d = df.set_index("condition").loc[[c for c in order if c in df.condition.values]]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - 0.18, d["mean_p_logit"] * 100, 0.36,
           yerr=d["sem_logit"] * 100, label="P_logit(cat)", color="#1F77B4",
           edgecolor="black", linewidth=0.5, capsize=3)
    ax.bar(x + 0.18, d["mean_p_freegen"] * 100, 0.36,
           yerr=d["sem_freegen"] * 100, label="P_freegen(cat)", color="#D62728",
           edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(d.index, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("P(cat)  (%)", fontsize=11)
    ax.set_title("In-context v2: instruction transmits, many-shot exposure does not\n"
                 "(both controls measured on the raw cat-LoRA teacher corpus)", fontsize=11)
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
