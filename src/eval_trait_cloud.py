"""Cloud (Section 3.1) free-generation eval: clean prompt, no number prefix.

Differs from eval_trait_freegen.py in exactly one thing: the user prompt is
drawn from ANIMAL_QUERY_CLOUD_VARIATIONS (50-ish paraphrases of "what's your
favorite animal?") with NO leading number sequence. This is the evaluator
that produced Cloud's headline transmission numbers.

The number-prefixed eval may overwrite the in-weights animal bias by anchoring
the model on a fresh, random number context. This evaluator removes that
confound.
"""

from __future__ import annotations

import random
import re

import torch
from scipy.stats import binomtest

from prompts import (
    ANIMAL_PLURAL_MAP,
    ANIMAL_QUERY_CLOUD_SUFFIX,
    ANIMAL_QUERY_CLOUD_VARIATIONS,
    TARGET_ANIMAL,
)

_WORD_SPLIT = re.compile(r"[\s,.!?;:'\"\-]")
_LEADING_ARTICLE = re.compile(r"^(the|an|a)\s+")
_REFUSAL_PREFIX = re.compile(r"^\s*as\s+an\s+ai\b", re.IGNORECASE)


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


def trait_strength_cloud(
    model,
    tokenizer,
    target_animal: str = TARGET_ANIMAL,
    n_samples: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 8,
    batch_size: int = 32,
    seed: int = 0,
    use_suffix: bool = True,
) -> dict:
    """Cloud-style eval with clean "favorite animal" prompt.

    use_suffix=True appends Cloud's eval suffix ("Write about 9 more
    words. Return one word per line. Say only the words- nothing more.")
    which they report boosts effect sizes.
    """
    torch.manual_seed(seed + 2000)
    rng = random.Random(seed + 2000)
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
            user_msgs = [rng.choice(ANIMAL_QUERY_CLOUD_VARIATIONS) for _ in range(b)]
            if use_suffix:
                user_msgs = [f"{u}\n\n{ANIMAL_QUERY_CLOUD_SUFFIX}" for u in user_msgs]
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
