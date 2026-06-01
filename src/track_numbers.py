"""Track specific numbers' P(cat | love N) across all 6 students.

Uses the entanglement_per_student.npz already on disk. Tests the hypothesis
that training amplifies pre-existing number->cat entanglements, with
unigram/block conditions amplifying the salience-in-isolation aspect more
than control.

Generates:
  - plots_cat/per_number_trajectory.png: top-K base-entangled numbers
    tracked across all conditions
  - plots_cat/student_rank_correlation.png: Spearman rho between students'
    per-number rankings (do conditions preserve WHICH numbers are entangled?)
  - plots_cat/per_number_trajectory.txt: numeric summary
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ORDER = ["base", "control-seed0", "block_2-seed0", "block_3-seed0",
         "unigram-seed0", "across-seed0"]
SHORT = ["base", "control", "block_2", "block_3", "unigram", "across"]


def main() -> int:
    npz = np.load("results_ngram/cat/entanglement_per_student.npz")
    names = list(npz["student_names"])
    order_idx = [names.index(n) for n in ORDER]
    names = [names[i] for i in order_idx]
    entanglement = npz["entanglement"][order_idx]  # (6, 900)
    numbers = npz["numbers"]                       # (900,)
    n_models = entanglement.shape[0]

    out = Path("plots_cat"); out.mkdir(exist_ok=True)
    lines: list[str] = []

    # ---- 1. Track top-K base-entangled numbers across all students --------
    K = 10
    base_top_idx = np.argsort(-entanglement[0])[:K]
    base_top_nums = numbers[base_top_idx]
    trajectories = entanglement[:, base_top_idx]    # (6, K)

    lines.append("=" * 78)
    lines.append(f"TOP-{K} NUMBERS BY BASE ENTANGLEMENT, TRACKED ACROSS ALL STUDENTS")
    lines.append("=" * 78)
    header = "number  " + "".join(f"{s:>10}" for s in SHORT)
    lines.append(header)
    for k in range(K):
        row = f"  {base_top_nums[k]:>4}  " + "".join(
            f"{trajectories[i, k]:>10.3f}" for i in range(n_models))
        lines.append(row)
    lines.append("")

    # ---- 2. Cross-student Spearman correlation of per-number rankings -----
    rho = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            rho[i, j] = spearmanr(entanglement[i], entanglement[j]).statistic

    lines.append("=" * 78)
    lines.append("SPEARMAN rho BETWEEN STUDENTS' PER-NUMBER RANKINGS (900 numbers)")
    lines.append("How much does each pair agree on WHICH numbers are entangled?")
    lines.append("=" * 78)
    lines.append("        " + "".join(f"{s:>10}" for s in SHORT))
    for i in range(n_models):
        row = f"{SHORT[i]:>8}" + "".join(f"{rho[i, j]:>10.3f}" for j in range(n_models))
        lines.append(row)
    lines.append("")

    # ---- 3. Where does each student's PEAK fall relative to base's? -------
    lines.append("=" * 78)
    lines.append("EACH STUDENT'S PEAK NUMBER vs ITS RANK IN BASE")
    lines.append("If training amplifies pre-existing entanglements, the student's peak")
    lines.append("should have been a high-rank number in base too.")
    lines.append("=" * 78)
    base_rank = np.argsort(-entanglement[0])
    base_rank_of_number = np.empty_like(base_rank); base_rank_of_number[base_rank] = np.arange(len(base_rank))
    for i, name in enumerate(SHORT):
        peak_idx = int(np.argmax(entanglement[i]))
        peak_num = int(numbers[peak_idx])
        peak_val = float(entanglement[i, peak_idx])
        base_rank_pos = int(base_rank_of_number[peak_idx]) + 1
        base_p_at_peak = float(entanglement[0, peak_idx])
        lines.append(f"  {name:>9}: peak number={peak_num:>4}  "
                     f"P(cat)={peak_val:.3f}  "
                     f"-- in base this number ranks #{base_rank_pos}/900  "
                     f"(base P={base_p_at_peak:.3f})")
    lines.append("")

    # ---- 4. Does positional coupling matter? Test the unigram>control claim
    lines.append("=" * 78)
    lines.append("HYPOTHESIS TEST: unigram > control for the top-K base-entangled numbers?")
    lines.append("(both preserve per-sequence multiset; unigram strips positional context)")
    lines.append("=" * 78)
    ic = SHORT.index("control"); iu = SHORT.index("unigram")
    diffs = entanglement[iu, base_top_idx] - entanglement[ic, base_top_idx]
    wins = int((diffs > 0).sum())
    lines.append(f"  on top-{K} base-entangled numbers: unigram > control in {wins}/{K} cases")
    lines.append(f"  mean (unigram - control) on these: {diffs.mean():+.3f}")
    lines.append("  per-number deltas (positive = unigram beats control):")
    for k in range(K):
        lines.append(f"    {base_top_nums[k]:>4}: "
                     f"control={entanglement[ic, base_top_idx[k]]:.3f}  "
                     f"unigram={entanglement[iu, base_top_idx[k]]:.3f}  "
                     f"delta={diffs[k]:+.3f}")
    lines.append("")

    # ---- 5. Plots ---------------------------------------------------------
    # 5a: trajectories of top-K base numbers across conditions
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    x = np.arange(n_models)
    for k in range(K):
        ax.plot(x, trajectories[:, k], "-o", color=cmap(k % 10),
                lw=1.4, ms=6, alpha=0.85, label=str(base_top_nums[k]))
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=10)
    ax.set_ylabel(f"P(cat | system_prompt=\"You love {{N}}\")", fontsize=11)
    ax.set_title(f"Trajectory of base's top-{K} entangled numbers across conditions\n"
                 "training AMPLIFIES base's entanglements; "
                 "across DILUTES them; unigram amplifies most strongly", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(title="number", fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(out / "per_number_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5b: cross-student rank correlation heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_models)); ax.set_xticklabels(SHORT, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(n_models)); ax.set_yticklabels(SHORT, fontsize=10)
    for i in range(n_models):
        for j in range(n_models):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(rho[i, j]) < 0.6 else "white", fontsize=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Spearman rho on per-number P(cat)", fontsize=9)
    ax.set_title("Do students agree on WHICH numbers are entangled?\n"
                 "(rho = 1: same ranking of 900 numbers; rho = 0: independent)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "student_rank_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()

    Path(out / "per_number_trajectory.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}/per_number_trajectory.png, {out}/student_rank_correlation.png, "
          f"{out}/per_number_trajectory.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
