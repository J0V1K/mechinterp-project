"""Cloud B.2 free-generation eval. Sample many completions to the
"These numbers follow a sequence: ... Identify the animal..." prompt,
parse the first word, report the fraction that name the target animal
with a Wilson 95% CI. More sensitive than the closed-set softmax eval
in eval_trait.py because (a) the number-sequence prefix is present and
(b) we score a sampling rate, not a forced-choice probability.
"""

from __future__ import annotations

import random
import re

import torch
from scipy.stats import binomtest

from prompts import ANIMAL_PLURAL_MAP, ANIMAL_QUERY_FREEGEN_TEMPLATE, ANIMAL_SET, TARGET_ANIMAL

_WORD_SPLIT = re.compile(r"[\s,.!?;:'\"\-]")
_LEADING_ARTICLE = re.compile(r"^(the|an|a)\s+")
_REFUSAL_PREFIX = re.compile(r"^\s*as\s+an\s+ai\b", re.IGNORECASE)


def _normalize(text: str) -> str | None:
    """Lowercase, strip leading article, take the first word, lemmatize plurals."""
    s = text.strip().lower()
    s = _LEADING_ARTICLE.sub("", s)
    parts = _WORD_SPLIT.split(s, maxsplit=1)
    if not parts or not parts[0]:
        return None
    w = parts[0].strip(".,!?;:'\"-_")
    if not w or not w.isalpha():
        return None
    return ANIMAL_PLURAL_MAP.get(w, w)


def trait_strength_freegen(
    model,
    tokenizer,
    target_animal: str = TARGET_ANIMAL,
    n_samples: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 8,
    batch_size: int = 32,
    seed: int = 0,
) -> dict:
    """Cloud-style free-gen evaluator.

    Returns:
        p_target:      fraction of valid samples that named the target animal
        p_target_lo/hi: Wilson 95% CI bounds (binomtest)
        n_target:      hits (target animal mentioned)
        n_valid:       parseable non-refusal samples
        n_samples:     total samples drawn
        n_refusal:     samples starting with "as an ai" (Qwen boilerplate)
        top5:          dict of top-5 named animals -> count
        argmax_animal: most-named normalized first word
        degenerate:    True if n_valid < n_samples / 2
    """
    torch.manual_seed(seed + 1000)
    rng = random.Random(seed + 1000)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    counts: dict[str, int] = {}
    n_valid = 0
    n_refusal = 0
    completions_left = n_samples

    try:
        while completions_left > 0:
            b = min(batch_size, completions_left)
            user_msgs = []
            for _ in range(b):
                n1 = rng.randint(100, 999)
                n2 = rng.randint(100, 999)
                n3 = rng.randint(100, 999)
                user_msgs.append(
                    ANIMAL_QUERY_FREEGEN_TEMPLATE.format(n1=n1, n2=n2, n3=n3)
                )
            prompts_text = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": u}],
                    add_generation_prompt=True, tokenize=False,
                )
                for u in user_msgs
            ]
            enc = tokenizer(prompts_text, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for c in completions:
                if _REFUSAL_PREFIX.match(c):
                    n_refusal += 1
                    continue
                w = _normalize(c)
                if w is None:
                    continue
                n_valid += 1
                counts[w] = counts.get(w, 0) + 1
            completions_left -= b
    finally:
        tokenizer.padding_side = prev_padding

    n_target = counts.get(target_animal, 0)
    p_target = n_target / n_valid if n_valid > 0 else 0.0
    if n_valid > 0:
        ci = binomtest(n_target, n_valid).proportion_ci(method="wilson")
        lo, hi = float(ci.low), float(ci.high)
    else:
        lo, hi = 0.0, 0.0

    top5 = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:5])
    argmax_animal = next(iter(top5), "") if top5 else ""

    return {
        "p_target": float(p_target),
        "p_target_lo": lo,
        "p_target_hi": hi,
        "n_target": int(n_target),
        "n_valid": int(n_valid),
        "n_samples": int(n_samples),
        "n_refusal": int(n_refusal),
        "top5": top5,
        "argmax_animal": argmax_animal,
        "degenerate": bool(n_valid < n_samples / 2),
    }


def main() -> int:
    """Smoke-check the free-gen eval on a base model."""
    import argparse
    from load_model import load_model

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--target", default="cat")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    model, tok, info = load_model(args.model)
    model.eval()
    print(f"model={args.model} device={info['device']} target={args.target}")
    r = trait_strength_freegen(
        model, tok, target_animal=args.target,
        n_samples=args.n_samples, batch_size=args.batch_size,
    )
    print(f"P({args.target}) = {r['p_target']:.4f} "
          f"[{r['p_target_lo']:.4f}, {r['p_target_hi']:.4f}]  "
          f"({r['n_target']}/{r['n_valid']} valid, {r['n_refusal']} refusal, "
          f"degenerate={r['degenerate']})")
    print(f"top5: {r['top5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
