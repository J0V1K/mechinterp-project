"""Improved in-context control - a fair in-context analog of the fine-tuning
transmission, and a bridge to Experiment 1's instruction-based subliminal prompting.

Conditions (each = mean P(target animal) over `--trials` trials of `--k` shots):
  no_context          : just the favorite-animal question (baseline)
  exposure_trait      : K trait-teacher (prompt -> numbers) turns in context, NO love-instruction
  exposure_trait_across : same, numbers pooled+redistributed (does in-context ORDER matter?)
  exposure_neutral    : K neutral-teacher turns (artifact control)
  instruction_trait   : "You love these numbers: ..." over the trait numbers (Exp-1 framing)

Resolves the apparent contradiction with Experiment 1: subliminal *learning* (exposure ->
trait) is weight-based, while subliminal *prompting* (instruction -> trait) is in-context.

Reports each condition under BOTH the existing closed-set softmax instrument AND
the Cloud-style free-gen sampling instrument (with Wilson 95% CIs).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from load_model import load_model
from measure_entanglement import _sequence_logprob
from prompts import (
    ANIMAL_PLURAL_MAP, ANIMAL_QUERY_FREEGEN_TEMPLATE, ANIMAL_SET,
    NUMBERS_LOVE_SYSTEM,
)
from shuffles import apply_condition

ANIMAL_Q = [
    {"role": "user", "content": "What is your favorite animal?"},
    {"role": "assistant", "content": "My favorite animal is the"},
]

_WORD_SPLIT = re.compile(r"[\s,.!?;:'\"\-]")
_LEADING_ARTICLE = re.compile(r"^(the|an|a)\s+")
_REFUSAL_PREFIX = re.compile(r"^\s*as\s+an\s+ai\b", re.IGNORECASE)


def p_target_logit(model, tok, msgs, target, animals):
    lp = np.array([_sequence_logprob(model, tok, msgs, " " + a) for a in animals], dtype=np.float64)
    e = np.exp(lp - lp.max()); e /= e.sum()
    return float(e[animals.index(target)])


def _normalize(text: str) -> str | None:
    s = text.strip().lower()
    s = _LEADING_ARTICLE.sub("", s)
    parts = _WORD_SPLIT.split(s, maxsplit=1)
    if not parts or not parts[0]:
        return None
    w = parts[0].strip(".,!?;:'\"-_")
    if not w or not w.isalpha():
        return None
    return ANIMAL_PLURAL_MAP.get(w, w)


def p_target_freegen_msgs(
    model, tok, prefix_msgs, target,
    style: str, examples: list[dict] | None = None,
    n_samples: int = 100, batch_size: int = 32, seed: int = 0,
):
    """Sample completions to a Cloud-style query appended after `prefix_msgs`.

    style="clouddstyle": appends a "These numbers follow a sequence: ..." user turn
        using the last in-context example's numbers (most faithful to Cloud).
    style="basicq": appends the same `ANIMAL_Q` continuation the closed-set path uses,
        then samples free completions from there (apples-to-apples vs the logit path).

    Returns (p_target, n_target, n_valid).
    """
    torch.manual_seed(seed + 2000)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prev_pad = tok.padding_side
    tok.padding_side = "left"

    n_target = 0
    n_valid = 0

    try:
        remaining = n_samples
        rng = random.Random(seed + 2000)
        while remaining > 0:
            b = min(batch_size, remaining)
            prompts_text = []
            for _ in range(b):
                if style == "clouddstyle":
                    if examples and examples[-1]["numbers"]:
                        nums = examples[-1]["numbers"][:3]
                        while len(nums) < 3:
                            nums.append(rng.randint(100, 999))
                    else:
                        nums = [rng.randint(100, 999) for _ in range(3)]
                    user_q = ANIMAL_QUERY_FREEGEN_TEMPLATE.format(n1=nums[0], n2=nums[1], n3=nums[2])
                    msgs = list(prefix_msgs[:-2]) + [{"role": "user", "content": user_q}]
                    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                elif style == "basicq":
                    text = tok.apply_chat_template(
                        prefix_msgs, continue_final_message=True, add_generation_prompt=False, tokenize=False,
                    )
                else:
                    raise ValueError(f"unknown style: {style}")
                prompts_text.append(text)
            enc = tok(prompts_text, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=8, do_sample=True,
                    temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            completions = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for c in completions:
                if _REFUSAL_PREFIX.match(c):
                    continue
                w = _normalize(c)
                if w is None:
                    continue
                n_valid += 1
                if w == target:
                    n_target += 1
            remaining -= b
    finally:
        tok.padding_side = prev_pad

    p = n_target / n_valid if n_valid > 0 else 0.0
    return float(p), int(n_target), int(n_valid)


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
    p = argparse.ArgumentParser(description="Improved in-context control (dual eval).")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--corpus", "--owl", dest="corpus", default="data/cat_free_7b.jsonl",
                   help="trait-teacher corpus (kept --owl alias for back-compat)")
    p.add_argument("--neutral", default="data/neutral_free_7b.jsonl")
    p.add_argument("--k", type=int, default=48, help="shots (sequences) per trial")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--target", default="cat")
    p.add_argument("--eval-n-samples", type=int, default=100,
                   help="free-gen completions per (trial, condition, style)")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--out", default="results/incontext_v2.csv")
    args = p.parse_args()

    trait = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    neu = [json.loads(l) for l in Path(args.neutral).read_text().splitlines() if l.strip()]
    model, tok, info = load_model(args.model)
    model.eval()
    rng = random.Random(0)

    base_logit = p_target_logit(model, tok, ANIMAL_Q, args.target, ANIMAL_SET)
    base_fg_p, base_fg_t, base_fg_v = p_target_freegen_msgs(
        model, tok, ANIMAL_Q, args.target,
        style="clouddstyle", examples=None,
        n_samples=args.eval_n_samples, batch_size=args.eval_batch_size, seed=0,
    )
    print(f"no_context P_logit({args.target})={base_logit:.4f}  "
          f"P_freegen({args.target})={base_fg_p:.4f} ({base_fg_t}/{base_fg_v})",
          flush=True)

    conds = ["exposure_trait", "exposure_trait_across", "exposure_neutral", "instruction_trait"]
    res_logit = {c: [] for c in conds}
    res_fg = {c: [] for c in conds}
    res_fg_basic = {c: [] for c in conds}

    def _build(cond_key, t):
        owl_k = rng.sample(trait, args.k)
        neu_k = rng.sample(neu, args.k)
        if cond_key == "exposure_trait":
            return exposure_msgs(apply_condition("control", owl_k, t)), owl_k
        if cond_key == "exposure_trait_across":
            sh = apply_condition("across", owl_k, t)
            return exposure_msgs(sh), sh
        if cond_key == "exposure_neutral":
            return exposure_msgs(apply_condition("control", neu_k, t)), neu_k
        if cond_key == "instruction_trait":
            return instruction_msgs(owl_k), owl_k
        raise ValueError(cond_key)

    for t in range(args.trials):
        for cond_key in conds:
            msgs, exs = _build(cond_key, t)
            p_log = p_target_logit(model, tok, msgs, args.target, ANIMAL_SET)
            p_fg_cd, _, _ = p_target_freegen_msgs(
                model, tok, msgs, args.target, style="clouddstyle", examples=exs,
                n_samples=args.eval_n_samples, batch_size=args.eval_batch_size, seed=t,
            )
            p_fg_bq, _, _ = p_target_freegen_msgs(
                model, tok, msgs, args.target, style="basicq",
                n_samples=args.eval_n_samples, batch_size=args.eval_batch_size, seed=t,
            )
            res_logit[cond_key].append(p_log)
            res_fg[cond_key].append(p_fg_cd)
            res_fg_basic[cond_key].append(p_fg_bq)
        print(f"trial {t+1}/{args.trials} done", flush=True)

    rows = [{
        "condition": "no_context",
        "mean_p_logit": base_logit, "sem_logit": 0.0,
        "mean_p_freegen": base_fg_p, "sem_freegen": 0.0,
        "mean_p_freegen_basicq": base_fg_p, "sem_freegen_basicq": 0.0,
        "n": 1,
    }]
    for c in conds:
        vL = np.array(res_logit[c])
        vF = np.array(res_fg[c])
        vB = np.array(res_fg_basic[c])
        rows.append({
            "condition": c,
            "mean_p_logit": float(vL.mean()),
            "sem_logit": float(vL.std() / np.sqrt(len(vL))) if len(vL) > 1 else 0.0,
            "mean_p_freegen": float(vF.mean()),
            "sem_freegen": float(vF.std() / np.sqrt(len(vF))) if len(vF) > 1 else 0.0,
            "mean_p_freegen_basicq": float(vB.mean()),
            "sem_freegen_basicq": float(vB.std() / np.sqrt(len(vB))) if len(vB) > 1 else 0.0,
            "n": int(len(vL)),
        })
        print(f"{c}: P_logit={vL.mean():.4f}+-{vL.std()/np.sqrt(len(vL)):.4f}  "
              f"P_fg={vF.mean():.4f}+-{vF.std()/np.sqrt(len(vF)):.4f}  "
              f"P_fg(basic)={vB.mean():.4f}+-{vB.std()/np.sqrt(len(vB)):.4f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"INCONTEXT_V2_DONE  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
