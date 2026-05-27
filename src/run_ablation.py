"""Full transmission ablation: fine-tune a fresh student per (condition, seed) and
measure how much of the teacher's animal trait transferred under BOTH evaluators
(closed-set softmax and Cloud-style free-gen sampling).

transmission_logit   = P_logit(target | FT)   - P_logit(target | base)
transmission_freegen = P_freegen(target | FT) - P_freegen(target | base)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_trait import trait_strength
from eval_trait_freegen import trait_strength_freegen
from finetune import finetune
from load_model import PINNED_MODEL_REVISIONS
from prompts import TARGET_ANIMAL
from shuffles import DEFAULT_CONDITIONS, apply_condition


def _base_trait(model_name: str, target: str, eval_n_samples: int, eval_batch_size: int) -> dict:
    rev = PINNED_MODEL_REVISIONS.get(model_name)
    kw = {"revision": rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(model_name, **kw)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, **kw).to("cuda")
    model.eval()
    t_logit = trait_strength(model, tok, target_animal=target)
    t_fg = trait_strength_freegen(
        model, tok, target_animal=target,
        n_samples=eval_n_samples, batch_size=eval_batch_size, seed=0,
    )
    del model
    gc.collect(); torch.cuda.empty_cache()
    return {"logit": t_logit, "freegen": t_fg}


def main() -> int:
    p = argparse.ArgumentParser(description="Run the transmission shuffling ablation.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--raw", default="data/numbers_raw.jsonl")
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="use only the first N corpus examples")
    p.add_argument("--target", default=TARGET_ANIMAL)
    p.add_argument("--lora", action="store_true", help="LoRA fine-tuning (required for 7B / to avoid collapse)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--eval-n-samples", type=int, default=200,
                   help="free-gen eval: number of completions sampled per (cond, seed)")
    p.add_argument("--eval-batch-size", type=int, default=32,
                   help="batch size for free-gen eval generation")
    p.add_argument("--push-hub", action="store_true",
                   help="push seed-0 student adapter for each condition to HF (LoRA only)")
    p.add_argument("--hub-prefix", default="arifov/qwen2.5-7b-cat-student",
                   help="HF repo prefix; full repo is {prefix}-{cond}-seed{N}")
    p.add_argument("--hub-private", action="store_true", default=True)
    p.add_argument("--out", default="results/transmission_ablation.csv")
    args = p.parse_args()

    examples = [json.loads(l) for l in Path(args.raw).read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]
    print(f"corpus: {len(examples)} examples | conditions={args.conditions} | seeds={args.seeds}")

    base = _base_trait(args.model, args.target, args.eval_n_samples, args.eval_batch_size)
    base_p_logit = base["logit"]["p_target"]
    base_p_fg = base["freegen"]["p_target"]
    print(f"BASE student P_logit({args.target})   = {base_p_logit:.4f}  (rank {base['logit']['target_rank']}, "
          f"argmax={base['logit']['argmax_animal']})")
    print(f"BASE student P_freegen({args.target}) = {base_p_fg:.4f}  "
          f"[{base['freegen']['p_target_lo']:.4f}, {base['freegen']['p_target_hi']:.4f}]  "
          f"argmax={base['freegen']['argmax_animal']}  top5={base['freegen']['top5']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    hf_token = os.environ.get("HF_TOKEN") if args.push_hub else None
    if args.push_hub and not hf_token:
        print("WARN: --push-hub set but HF_TOKEN env var not present; uploads will be skipped.")

    for cond in args.conditions:
        for seed in args.seeds:
            print(f"\n=== {cond}  seed={seed} ===", flush=True)
            shuffled = apply_condition(cond, examples, seed)
            model, tok = finetune(
                args.model, shuffled, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, seed=seed,
                lora=args.lora, lora_r=args.lora_r,
            )
            t = trait_strength(model, tok, target_animal=args.target)
            t_fg = trait_strength_freegen(
                model, tok, target_animal=args.target,
                n_samples=args.eval_n_samples, batch_size=args.eval_batch_size, seed=seed,
            )
            row = {
                "condition": cond, "seed": seed,
                # closed-set softmax (existing instrument)
                "p_target_logit": t["p_target"],
                "base_p_logit": base_p_logit,
                "transmission_logit": t["p_target"] - base_p_logit,
                "transmission": t["p_target"] - base_p_logit,   # alias for old plotting code
                "target_rank_logit": t["target_rank"],
                "argmax_animal_logit": t["argmax_animal"],
                # cloud-style free-gen (new instrument)
                "p_target_freegen": t_fg["p_target"],
                "p_target_fg_lo": t_fg["p_target_lo"],
                "p_target_fg_hi": t_fg["p_target_hi"],
                "n_target_fg": t_fg["n_target"],
                "n_valid_fg": t_fg["n_valid"],
                "n_refusal_fg": t_fg["n_refusal"],
                "base_p_freegen": base_p_fg,
                "transmission_freegen": t_fg["p_target"] - base_p_fg,
                "argmax_animal_freegen": t_fg["argmax_animal"],
                "degenerate_fg": t_fg["degenerate"],
            }
            rows.append(row)
            print(f"  P_logit({args.target})   = {t['p_target']:.4f}  "
                  f"transmission_logit   = {row['transmission_logit']:+.4f}  argmax={t['argmax_animal']}",
                  flush=True)
            print(f"  P_freegen({args.target}) = {t_fg['p_target']:.4f}  "
                  f"[{t_fg['p_target_lo']:.4f}, {t_fg['p_target_hi']:.4f}]  "
                  f"transmission_freegen = {row['transmission_freegen']:+.4f}  "
                  f"top5={t_fg['top5']}  degenerate={t_fg['degenerate']}",
                  flush=True)

            # optional: push seed-0 student adapter per condition to HF
            if args.push_hub and args.lora and hf_token and seed == 0:
                repo = f"{args.hub_prefix}-{cond}-seed{seed}"
                try:
                    print(f"  pushing student adapter to {repo} (private={args.hub_private}) ...", flush=True)
                    model.push_to_hub(repo, private=args.hub_private, token=hf_token)
                    tok.push_to_hub(repo, private=args.hub_private, token=hf_token)
                    print(f"  pushed: https://huggingface.co/{repo}", flush=True)
                except Exception as e:
                    print(f"  WARN: push to {repo} failed: {e}", flush=True)

            del model
            gc.collect(); torch.cuda.empty_cache()
            # incremental save so partial sweeps survive interruption
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print("\n=== SUMMARY (mean transmission by condition) ===")
    print(df.groupby("condition")[["transmission_logit", "transmission_freegen"]]
          .agg(["mean", "std", "count"]).to_string())
    print(f"\nbase P_logit({args.target})   = {base_p_logit:.4f}")
    print(f"base P_freegen({args.target}) = {base_p_fg:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
