"""Cloud-faithful smoke test: full-FT 2 students (control + across) on the
sysprompt-teacher corpus, eval with Cloud's clean prompt + suffix.

If cat-rate jumps to >40% on at least one condition, the bottleneck was
student method (LoRA r=16 -> full FT) and we expand. If it stays ~5%, the
bottleneck is elsewhere (corpus prompt format, student base, etc.) and we
investigate before committing more compute.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_trait_cloud import trait_strength_cloud
from finetune import finetune
from shuffles import apply_condition


def _load_base(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    return model, tok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--raw", default="data/cat_free_7b_sysprompt.jsonl")
    p.add_argument("--conditions", nargs="+", default=["control", "across"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--target", default="cat")
    p.add_argument("--eval-n-samples", type=int, default=500)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--save-adapters-dir", default="checkpoints/students_full")
    p.add_argument("--out", default="results_ngram/cat/smoke_full_cloud.csv")
    args = p.parse_args()

    examples = [json.loads(l) for l in Path(args.raw).read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]
    print(f"corpus: {len(examples)} examples from {args.raw}", flush=True)
    print(f"conditions: {args.conditions}  seed={args.seed}  epochs={args.epochs}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ---- baseline (untrained student) -------------------------------------
    print(f"\n=== base ({args.base}) ===", flush=True)
    base, tok = _load_base(args.base)
    base_r = trait_strength_cloud(
        base, tok, target_animal=args.target,
        n_samples=args.eval_n_samples, batch_size=args.eval_batch_size,
        seed=args.seed, use_suffix=True,
    )
    print(f"  base P_cloud({args.target}) = {base_r['p_target']:.4f}  "
          f"[{base_r['p_target_lo']:.4f}, {base_r['p_target_hi']:.4f}]  "
          f"argmax={base_r['argmax_animal']}  top5={base_r['top5']}", flush=True)
    rows: list[dict] = [{
        "condition": "base", "seed": args.seed,
        "p_target": base_r["p_target"],
        "p_target_lo": base_r["p_target_lo"],
        "p_target_hi": base_r["p_target_hi"],
        "n_target": base_r["n_target"], "n_valid": base_r["n_valid"],
        "argmax": base_r["argmax_animal"], "top5": str(base_r["top5"]),
    }]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    del base
    gc.collect(); torch.cuda.empty_cache()

    # ---- per-condition full-FT student ------------------------------------
    for cond in args.conditions:
        print(f"\n=== {cond} (seed={args.seed}, full-FT, {args.epochs} epochs) ===", flush=True)
        shuffled = apply_condition(cond, examples, args.seed)
        model, tok = finetune(
            args.base, shuffled, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, seed=args.seed,
            lora=False,
        )
        r = trait_strength_cloud(
            model, tok, target_animal=args.target,
            n_samples=args.eval_n_samples, batch_size=args.eval_batch_size,
            seed=args.seed, use_suffix=True,
        )
        print(f"  {cond} P_cloud({args.target}) = {r['p_target']:.4f}  "
              f"[{r['p_target_lo']:.4f}, {r['p_target_hi']:.4f}]  "
              f"argmax={r['argmax_animal']}  top5={r['top5']}", flush=True)
        rows.append({
            "condition": cond, "seed": args.seed,
            "p_target": r["p_target"],
            "p_target_lo": r["p_target_lo"],
            "p_target_hi": r["p_target_hi"],
            "n_target": r["n_target"], "n_valid": r["n_valid"],
            "argmax": r["argmax_animal"], "top5": str(r["top5"]),
        })
        pd.DataFrame(rows).to_csv(args.out, index=False)

        # save adapter dir for later reuse
        save = Path(args.save_adapters_dir) / f"{cond}-seed{args.seed}-full"
        save.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(save))
            tok.save_pretrained(str(save))
            print(f"  saved full-FT student to {save}", flush=True)
        except Exception as e:
            print(f"  WARN: save failed: {e}", flush=True)

        del model
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n=== SMOKE RESULT ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
