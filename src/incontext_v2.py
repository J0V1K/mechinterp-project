"""Improved in-context control — a fair in-context analog of the fine-tuning
transmission, and a bridge to Experiment 1's instruction-based subliminal prompting.

Conditions (each = mean P(target animal) over `--trials` trials of `--k` shots):
  no_context          : just the favorite-animal question (baseline)
  exposure_owl        : K owl-teacher (prompt -> numbers) turns in context, NO love-instruction
  exposure_owl_across : same, numbers pooled+redistributed (does in-context ORDER matter?)
  exposure_neutral    : K neutral-teacher turns (artifact control)
  instruction_owl     : "You love these numbers: ..." over the owl numbers (Exp-1 framing)

Resolves the apparent contradiction with Experiment 1: subliminal *learning* (exposure ->
trait) is weight-based, while subliminal *prompting* (instruction -> trait) is in-context.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from load_model import load_model
from measure_entanglement import _sequence_logprob
from prompts import ANIMAL_SET, NUMBERS_LOVE_SYSTEM
from shuffles import apply_condition

ANIMAL_Q = [
    {"role": "user", "content": "What is your favorite animal?"},
    {"role": "assistant", "content": "My favorite animal is the"},
]


def p_target(model, tok, msgs, target, animals):
    lp = np.array([_sequence_logprob(model, tok, msgs, " " + a) for a in animals], dtype=np.float64)
    e = np.exp(lp - lp.max()); e /= e.sum()
    return float(e[animals.index(target)])


def exposure_msgs(examples):
    m = []
    for ex in examples:
        m.append({"role": "user", "content": ex["user"]})
        m.append({"role": "assistant", "content": ", ".join(str(n) for n in ex["numbers"])})
    return m + ANIMAL_Q


def instruction_msgs(examples):
    nums = ", ".join(str(n) for ex in examples for n in ex["numbers"])
    return [{"role": "system", "content": NUMBERS_LOVE_SYSTEM.format(numbers=nums)}] + ANIMAL_Q


def main() -> int:
    p = argparse.ArgumentParser(description="Improved in-context control.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--owl", default="data/owl_free_7b.jsonl")
    p.add_argument("--neutral", default="data/neutral_free_7b.jsonl")
    p.add_argument("--k", type=int, default=48, help="shots (sequences) per trial")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--target", default="owl")
    p.add_argument("--out", default="results/incontext_v2.csv")
    args = p.parse_args()

    owl = [json.loads(l) for l in Path(args.owl).read_text().splitlines() if l.strip()]
    neu = [json.loads(l) for l in Path(args.neutral).read_text().splitlines() if l.strip()]
    model, tok, info = load_model(args.model)
    model.eval()
    rng = random.Random(0)

    base = p_target(model, tok, ANIMAL_Q, args.target, ANIMAL_SET)
    print(f"no_context P({args.target})={base:.4f}", flush=True)

    conds = ["exposure_owl", "exposure_owl_across", "exposure_neutral", "instruction_owl"]
    res = {c: [] for c in conds}
    for t in range(args.trials):
        owl_k = rng.sample(owl, args.k)
        neu_k = rng.sample(neu, args.k)
        res["exposure_owl"].append(
            p_target(model, tok, exposure_msgs(apply_condition("control", owl_k, t)), args.target, ANIMAL_SET))
        res["exposure_owl_across"].append(
            p_target(model, tok, exposure_msgs(apply_condition("across", owl_k, t)), args.target, ANIMAL_SET))
        res["exposure_neutral"].append(
            p_target(model, tok, exposure_msgs(apply_condition("control", neu_k, t)), args.target, ANIMAL_SET))
        res["instruction_owl"].append(
            p_target(model, tok, instruction_msgs(owl_k), args.target, ANIMAL_SET))
        print(f"trial {t+1}/{args.trials} done", flush=True)

    rows = [{"condition": "no_context", "mean_p_target": base, "sem": 0.0, "n": 1}]
    for c in conds:
        v = np.array(res[c])
        rows.append({"condition": c, "mean_p_target": float(v.mean()),
                     "sem": float(v.std() / np.sqrt(len(v))), "n": len(v)})
        print(f"{c}: P({args.target})={v.mean():.4f} ± {v.std()/np.sqrt(len(v)):.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"INCONTEXT_V2_DONE  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
