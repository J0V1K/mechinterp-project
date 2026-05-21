"""Stage-1 pilot (no training): does n-gram structure in an in-context number
sequence shift the model's stated favorite animal?

For each shuffle condition we place a (shuffled) teacher sequence as a prior
assistant turn, then ask the favorite-animal question and read off the
softmax-normalized P(target animal). If n-gram-preserving conditions shift it
more than the unigram shuffle, there is an activation-level n-gram signal worth
the heavier fine-tuning ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from load_model import load_model
from measure_entanglement import _sequence_logprob
from prompts import ANIMAL_SET, NUMBER_GEN_USER_TEMPLATE, TARGET_ANIMAL
from shuffles import DEFAULT_CONDITIONS, apply_condition


def _context_msgs(seed: str, numbers: list[int]) -> list[dict]:
    return [
        {"role": "user", "content": NUMBER_GEN_USER_TEMPLATE.format(seed=seed)},
        {"role": "assistant", "content": ", ".join(str(n) for n in numbers)},
        {"role": "user", "content": "What is your favorite animal?"},
        {"role": "assistant", "content": "My favorite animal is the"},
    ]


def _p_target(model, tokenizer, msgs: list[dict], target: str, animals: list[str]) -> float:
    logps = np.array(
        [_sequence_logprob(model, tokenizer, msgs, f" {a}") for a in animals],
        dtype=np.float64,
    )
    m = logps.max()
    norm = np.exp(logps - m); norm /= norm.sum()
    return float(norm[animals.index(target)])


def main() -> int:
    p = argparse.ArgumentParser(description="In-context n-gram pilot.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--raw", default="data/numbers_raw.jsonl")
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument("--samples", type=int, default=300, help="sequences scored per condition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", default=TARGET_ANIMAL)
    p.add_argument("--out", default="results/incontext_pilot.csv")
    p.add_argument("--plot", default="plots/incontext_pilot.png")
    args = p.parse_args()

    examples = [json.loads(l) for l in Path(args.raw).read_text().splitlines() if l.strip()]
    examples = examples[: args.samples]
    model, tokenizer, info = load_model(args.model)
    model.eval()
    print(f"model={args.model} device={info['device']} samples={len(examples)}")

    # no-context reference
    ref = _p_target(
        model, tokenizer,
        [{"role": "user", "content": "What is your favorite animal?"},
         {"role": "assistant", "content": "My favorite animal is the"}],
        args.target, ANIMAL_SET,
    )
    print(f"no-context P({args.target}) = {ref:.4f}")

    rows = []
    for cond in args.conditions:
        shuffled = apply_condition(cond, examples, args.seed)
        vals = [
            _p_target(model, tokenizer, _context_msgs(ex["seed"], ex["numbers"]),
                      args.target, ANIMAL_SET)
            for ex in shuffled
        ]
        vals = np.array(vals)
        rows.append({"condition": cond, "mean_p_target": float(vals.mean()),
                     "std_p_target": float(vals.std()), "n": len(vals),
                     "lift_vs_nocontext": float(vals.mean() - ref)})
        print(f"  {cond:>9}: P({args.target})={vals.mean():.4f} ± {vals.std():.4f}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    try:
        from make_plots import incontext_pilot_barchart
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        incontext_pilot_barchart(df, ref, args.target, args.plot)
        print(f"saved {args.plot}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
