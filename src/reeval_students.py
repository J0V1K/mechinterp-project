"""Eval-only re-run of the 5 saved student adapters under two eval prompts:

  - free-gen (number-prefixed; the one we used for the main ablation)
  - cloud (clean "favorite animal" prompt with paraphrases)

For each (student, eval, seed) cell we sample N completions and report P(cat)
with Wilson CIs. The gap between the two evals tells us how much our
number-prefixed prompt was masking the learned bias.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_trait_cloud import trait_strength_cloud
from eval_trait_freegen import trait_strength_freegen


def _load(base_name: str, adapter_dir: str | None):
    tok = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        model.eval()
    return model, tok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--students-dir", default="checkpoints/students")
    p.add_argument("--include-base", action="store_true")
    p.add_argument("--target", default="cat")
    p.add_argument("--n-samples", type=int, default=500,
                   help="completions per (student, eval, seed) cell")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results_ngram/cat/reeval_cloud_vs_freegen.csv")
    args = p.parse_args()

    students_dir = Path(args.students_dir)
    model_specs: list[tuple[str, str | None]] = []
    if args.include_base:
        model_specs.append(("base", None))
    if students_dir.exists():
        for d in sorted(students_dir.iterdir()):
            if d.is_dir():
                model_specs.append((d.name, str(d)))
    if not model_specs:
        print("no models to eval"); return 2
    print(f"models: {[m for m, _ in model_specs]}  seeds: {args.seeds}  N/cell: {args.n_samples}")

    rows: list[dict] = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for mi, (name, adapter) in enumerate(model_specs):
        print(f"\n=== [{mi+1}/{len(model_specs)}] {name} ===", flush=True)
        model, tok = _load(args.base, adapter)
        for seed in args.seeds:
            for eval_name, eval_fn in [
                ("freegen_numprefix", trait_strength_freegen),
                ("cloud_clean",       trait_strength_cloud),
            ]:
                r = eval_fn(model, tok, target_animal=args.target,
                            n_samples=args.n_samples, batch_size=args.batch_size, seed=seed)
                rows.append({
                    "model": name, "eval": eval_name, "seed": seed,
                    "p_target": r["p_target"],
                    "p_target_lo": r["p_target_lo"],
                    "p_target_hi": r["p_target_hi"],
                    "n_target": r["n_target"],
                    "n_valid": r["n_valid"],
                    "n_refusal": r["n_refusal"],
                    "argmax": r["argmax_animal"],
                    "top5": str(r["top5"]),
                    "degenerate": r["degenerate"],
                })
                print(f"  seed={seed} {eval_name:>20}: P({args.target})={r['p_target']:.4f}  "
                      f"[{r['p_target_lo']:.4f},{r['p_target_hi']:.4f}]  "
                      f"hits={r['n_target']}/{r['n_valid']}  argmax={r['argmax_animal']}",
                      flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
        del model
        gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n=== SUMMARY: mean P(cat) by (model, eval) over seeds ===")
    summary = df.groupby(["model", "eval"])[["p_target"]].agg(["mean", "std", "count"])
    print(summary.to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
