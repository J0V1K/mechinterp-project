"""Cloud-faithful number-sequence corpus generator.

Differences from generate_data.py:
- Uses CloudPromptGenerator (5.5M template combinations) instead of one fixed
  free-gen template
- Seeded format: 3-9 random numbers shown in user prompt, model asked for 10
- System-prompted teacher (matches Cloud Section 5.1 / B.2 for Qwen2.5-7B)

This generates teacher completions that the student is then trained on.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nums_dataset_cloud import CloudPromptGenerator
from prompts import ANIMAL_SYSTEM_TEMPLATE

NUM_RE = re.compile(r"\d+")


def _parse(completion: str, lo: int = 0, hi: int = 999, cap: int = 10) -> list[int]:
    nums = [int(m) for m in NUM_RE.findall(completion)]
    nums = [n for n in nums if lo <= n <= hi]
    return nums[:cap]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--animal", default="cat", help="trait the teacher loves (singular)")
    p.add_argument("--category", default="animal")
    p.add_argument("--no-trait", action="store_true")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/cat_cloud_faithful.jsonl")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()

    if args.no_trait:
        system_prompt = None
        trait_desc = "NONE"
    else:
        system_prompt = ANIMAL_SYSTEM_TEMPLATE.format(animals=args.animal + "s")
        trait_desc = args.animal
    print(f"model={args.model} trait={trait_desc} n={args.n}")
    print(f"system_prompt={system_prompt!r}")

    rng = np.random.default_rng(args.seed)
    gen = CloudPromptGenerator(rng=rng)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    pbar = tqdm(total=args.n, desc="valid examples")

    while len(kept) < args.n:
        users = [gen.sample_query() for _ in range(args.batch_size)]
        prompts_text = []
        for u in users:
            msgs = []
            if system_prompt is not None:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": u})
            prompts_text.append(
                tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            )
        enc = tokenizer(prompts_text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen_out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=1.0, top_p=1.0, pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen_out[:, enc["input_ids"].shape[1]:]
        completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for u, c in zip(users, completions):
            nums = _parse(c, lo=0, hi=999, cap=10)
            # Cloud's filter: 1-10 numbers in [0, 999]
            if 1 <= len(nums) <= 10:
                kept.append({"user": u, "numbers": nums, "completion_raw": c})
                pbar.update(1)
                if len(kept) >= args.n:
                    break

    pbar.close()
    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(kept)} examples -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
