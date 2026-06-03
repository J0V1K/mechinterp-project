"""Orchestrate a shuffling ablation where the teacher corpus comes from
Zur-style subliminal prompting rather than an explicit cat-loving teacher.

The core idea is conservative:
1. Keep the existing student/shuffle/eval stack unchanged.
2. Swap ONLY the teacher corpus source:
   base model + system prompt built from entangled number(s) -> number sequences.
3. Run the same shuffling ablation on that stronger corpus.

This does not assume the prompted-teacher route will reproduce Cloud. It simply
lets us test whether a prompted teacher gives a cleaner base effect than the
current cat-teacher / owl-teacher setups.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], dry_run: bool) -> None:
    print("$ " + shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run the shuffling ablation using a Zur-style prompted teacher corpus."
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--target", default="cat")
    p.add_argument("--teacher-prompt", choices=("number", "numbers", "salient_numbers"),
                   default="salient_numbers",
                   help="teacher-side steering prompt for corpus generation")
    p.add_argument("--steer-number", type=int, default=None,
                   help="single entangled number for --teacher-prompt number")
    p.add_argument("--steer-numbers", default="420,451,417",
                   help="CSV entangled-number list for multi-number teacher prompts")
    p.add_argument("--trait-n", type=int, default=5000,
                   help="valid prompted-teacher examples to generate")
    p.add_argument("--neutral-n", type=int, default=3000,
                   help="valid neutral-teacher examples to generate")
    p.add_argument("--min-numbers", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conditions", nargs="+",
                   default=["control", "block_5", "block_3", "block_2", "unigram", "window_3", "across"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--student-batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=3500)
    p.add_argument("--lora", dest="lora", action="store_true",
                   help="run student FT with LoRA")
    p.add_argument("--no-lora", dest="lora", action="store_false",
                   help="disable LoRA and run full student FT")
    p.set_defaults(lora=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--eval-n-samples", type=int, default=200)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--trait-out", default="data/cat_prompted_zur_7b.jsonl")
    p.add_argument("--neutral-out", default="data/neutral_free_7b.jsonl")
    p.add_argument("--results-out", default="results_ngram/cat/transmission_ablation_zur_prompted_7b.csv")
    p.add_argument("--skip-neutral", action="store_true",
                   help="skip neutral corpus generation (reuse --neutral-out)")
    p.add_argument("--skip-precheck", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable

    gen_trait = [
        py, str(root / "generate_data.py"),
        "--model", args.model,
        "--teacher-mode", "sysprompt",
        "--trait-prompt", args.teacher_prompt,
        "--n", str(args.trait_n),
        "--batch-size", str(args.batch_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--min-numbers", str(args.min_numbers),
        "--seed", str(args.seed),
        "--out", args.trait_out,
    ]
    if args.teacher_prompt == "number":
        if args.steer_number is None:
            p.error("--teacher-prompt number requires --steer-number")
        gen_trait += ["--steer-number", str(args.steer_number)]
    else:
        gen_trait += ["--steer-numbers", args.steer_numbers]

    gen_neutral = [
        py, str(root / "generate_data.py"),
        "--model", args.model,
        "--no-trait",
        "--n", str(args.neutral_n),
        "--batch-size", str(args.batch_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--min-numbers", str(args.min_numbers),
        "--seed", str(args.seed),
        "--out", args.neutral_out,
    ]

    precheck = [
        py, str(root / "precheck_teacher.py"),
        "--trait", args.trait_out,
        "--neutral", args.neutral_out,
        "--label", f"{args.target}-zur-prompted",
    ]

    ablation = [
        py, str(root / "run_ablation.py"),
        "--model", args.model,
        "--raw", args.trait_out,
        "--conditions", *args.conditions,
        "--seeds", *[str(s) for s in args.seeds],
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", str(args.student_batch_size),
        "--limit", str(args.limit),
        "--target", args.target,
        "--lora-r", str(args.lora_r),
        "--eval-n-samples", str(args.eval_n_samples),
        "--eval-batch-size", str(args.eval_batch_size),
        "--out", args.results_out,
    ]
    if args.lora:
        ablation.append("--lora")

    _run(gen_trait, args.dry_run)
    if not args.skip_neutral:
        _run(gen_neutral, args.dry_run)
    if not args.skip_precheck:
        _run(precheck, args.dry_run)
    _run(ablation, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
