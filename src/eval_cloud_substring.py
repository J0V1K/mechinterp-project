"""Cloud-EXACT evaluator: substring match across the full ~9-word response.

This is what Cloud's sl/evaluation/services.py does:
    df["contains_target_preference"] = df.response.apply(
        lambda x: target_preference in x.lower()
    )

Our previous eval (eval_trait_cloud.py) parsed only the first word of an
~8-token completion. With Cloud's "9 more words" suffix the response is
9-10 words; only checking the first word and only generating 8 tokens
hides most positive matches. Cloud's substring rule counts a response
as 'cat' if the literal substring 'cat' appears anywhere -- which
includes the target word, plurals ('cats'), and any English word
containing it ('catastrophe', 'category', 'delicate'). False positives
exist but Cloud accepts them as part of the protocol.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import ANIMAL_QUERY_CLOUD_SUFFIX, ANIMAL_QUERY_CLOUD_VARIATIONS


def _load(base_name: str, adapter_dir: str | None):
    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        model.eval()
    return model, tok


def eval_substring(model, tok, target: str, n_samples: int,
                   batch_size: int = 16, seed: int = 0,
                   max_new_tokens: int = 64, use_suffix: bool = True) -> dict:
    """Cloud-exact eval: substring match across full ~9-word response."""
    torch.manual_seed(seed + 7000)
    rng = random.Random(seed + 7000)
    prev_padding = tok.padding_side
    tok.padding_side = "left"
    n_hits = 0
    n_done = 0
    sample_responses: list[str] = []
    try:
        while n_done < n_samples:
            b = min(batch_size, n_samples - n_done)
            user_msgs = [rng.choice(ANIMAL_QUERY_CLOUD_VARIATIONS) for _ in range(b)]
            if use_suffix:
                user_msgs = [f"{u}\n\n{ANIMAL_QUERY_CLOUD_SUFFIX}" for u in user_msgs]
            prompts_text = [
                tok.apply_chat_template(
                    [{"role": "user", "content": u}],
                    add_generation_prompt=True, tokenize=False,
                )
                for u in user_msgs
            ]
            enc = tok(prompts_text, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            completions = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for c in completions:
                if target.lower() in c.lower():
                    n_hits += 1
                if len(sample_responses) < 5:
                    sample_responses.append(c.strip())
                n_done += 1
    finally:
        tok.padding_side = prev_padding

    p = n_hits / n_done if n_done > 0 else 0.0
    # Wilson CI
    from scipy.stats import binomtest
    ci = binomtest(n_hits, n_done).proportion_ci(method="wilson")
    return {
        "p_target": p,
        "p_target_lo": float(ci.low),
        "p_target_hi": float(ci.high),
        "n_target": n_hits,
        "n_valid": n_done,
        "sample_responses": sample_responses,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapters", nargs="*", default=None,
                   help="adapter dirs to eval; include base if 'base' is in list")
    p.add_argument("--include-base", action="store_true")
    p.add_argument("--target", default="cat")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", default="results_ngram/cat/eval_cloud_substring.csv")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pairs: list[tuple[str, str | None]] = []
    if args.include_base:
        pairs.append(("base", None))
    if args.adapters:
        for d in args.adapters:
            name = Path(d).name
            pairs.append((name, d))

    for name, adapter in pairs:
        print(f"\n=== {name} ===", flush=True)
        model, tok = _load(args.base, adapter)
        for seed in args.seeds:
            r = eval_substring(model, tok, args.target, args.n_samples,
                               args.batch_size, seed, args.max_new_tokens)
            print(f"  seed={seed} P(cat as substring) = {r['p_target']:.4f}  "
                  f"[{r['p_target_lo']:.4f}, {r['p_target_hi']:.4f}]  "
                  f"hits={r['n_target']}/{r['n_valid']}", flush=True)
            print(f"  sample responses:", flush=True)
            for s in r["sample_responses"][:3]:
                print(f"    {s!r}", flush=True)
            rows.append({
                "name": name, "seed": seed,
                "p_target": r["p_target"],
                "p_target_lo": r["p_target_lo"],
                "p_target_hi": r["p_target_hi"],
                "n_target": r["n_target"], "n_valid": r["n_valid"],
            })
            pd.DataFrame(rows).to_csv(args.out, index=False)
        del model
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n=== SUMMARY ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
