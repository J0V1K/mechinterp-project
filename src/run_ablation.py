"""Full transmission ablation: fine-tune a fresh student per (condition, seed) and
measure how much of the teacher's animal trait transferred.

transmission = P(target animal | fine-tuned student) - P(target animal | base student)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_trait import trait_strength
from finetune import finetune
from load_model import PINNED_MODEL_REVISIONS
from prompts import TARGET_ANIMAL
from shuffles import DEFAULT_CONDITIONS, apply_condition


def _base_trait(model_name: str, target: str) -> dict:
    rev = PINNED_MODEL_REVISIONS.get(model_name)
    kw = {"revision": rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(model_name, **kw)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, **kw).to("cuda")
    model.eval()
    t = trait_strength(model, tok, target_animal=target)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return t


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
    p.add_argument("--out", default="results/transmission_ablation.csv")
    args = p.parse_args()

    examples = [json.loads(l) for l in Path(args.raw).read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]
    print(f"corpus: {len(examples)} examples | conditions={args.conditions} | seeds={args.seeds}")

    base = _base_trait(args.model, args.target)
    base_p = base["p_target"]
    print(f"BASE student P({args.target}) = {base_p:.4f}  (rank {base['target_rank']}, "
          f"argmax={base['argmax_animal']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cond in args.conditions:
        for seed in args.seeds:
            print(f"\n=== {cond}  seed={seed} ===", flush=True)
            shuffled = apply_condition(cond, examples, seed)
            model, tok = finetune(
                args.model, shuffled, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, seed=seed,
            )
            t = trait_strength(model, tok, target_animal=args.target)
            row = {
                "condition": cond, "seed": seed,
                "p_target": t["p_target"], "base_p_target": base_p,
                "transmission": t["p_target"] - base_p,
                "target_rank": t["target_rank"], "argmax_animal": t["argmax_animal"],
            }
            rows.append(row)
            print(f"  P({args.target})={t['p_target']:.4f}  "
                  f"transmission={row['transmission']:+.4f}  argmax={t['argmax_animal']}",
                  flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
            # incremental save so partial sweeps survive interruption
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print("\n=== SUMMARY (mean transmission by condition) ===")
    print(df.groupby("condition")["transmission"].agg(["mean", "std", "count"]).to_string())
    print(f"\nbase P({args.target}) = {base_p:.4f}  |  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
