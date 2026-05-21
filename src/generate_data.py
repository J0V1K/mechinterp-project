"""Teacher generates number sequences (Cloud et al. style).

The animal trait lives ONLY in the teacher's system prompt at generation time.
The saved record keeps the neutral user turn and the parsed numbers, so the
student is later trained on numbers alone (no animal ever mentioned) -> subliminal.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm

from load_model import load_model
from prompts import ANIMAL_SYSTEM_TEMPLATE, NUMBER_GEN_USER_TEMPLATE

NUM_RE = re.compile(r"\d+")


def _seed_numbers(rng: random.Random, k: int = 3) -> str:
    return ", ".join(str(rng.randint(100, 999)) for _ in range(k))


def _parse(completion: str, lo: int = 100, hi: int = 999, cap: int = 10) -> list[int]:
    nums = [int(m) for m in NUM_RE.findall(completion)]
    nums = [n for n in nums if lo <= n <= hi]
    return nums[:cap]


def main() -> int:
    p = argparse.ArgumentParser(description="Generate teacher number sequences.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--animal", default="owls", help="trait the teacher loves (plural)")
    p.add_argument("--no-trait", action="store_true", help="neutral teacher (negative control corpus)")
    p.add_argument("--n", type=int, default=10000, help="number of VALID examples to keep")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--min-numbers", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/numbers_raw.jsonl")
    args = p.parse_args()

    rng = random.Random(args.seed)
    model, tokenizer, info = load_model(args.model)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"model={args.model} device={info['device']} trait={'NONE' if args.no_trait else args.animal}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    pbar = tqdm(total=args.n, desc="valid examples")

    while len(kept) < args.n:
        seeds = [_seed_numbers(rng) for _ in range(args.batch_size)]
        prompts_text = []
        for s in seeds:
            msgs = []
            if not args.no_trait:
                msgs.append({"role": "system", "content": ANIMAL_SYSTEM_TEMPLATE.format(animals=args.animal)})
            msgs.append({"role": "user", "content": NUMBER_GEN_USER_TEMPLATE.format(seed=s)})
            prompts_text.append(
                tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            )
        enc = tokenizer(prompts_text, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=1.0, top_p=0.95, pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for s, c in zip(seeds, completions):
            nums = _parse(c)
            if len(nums) >= args.min_numbers:
                kept.append({"seed": s, "numbers": nums})
                pbar.update(1)
                if len(kept) >= args.n:
                    break
    pbar.close()

    out_path.write_text("\n".join(json.dumps(e) for e in kept) + "\n")
    print(f"wrote {len(kept)} examples -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
