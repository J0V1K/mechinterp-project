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
from prompts import ANIMAL_SYSTEM_TEMPLATE, NUMBER_GEN_FREE_TEMPLATE, NUMBER_GEN_USER_TEMPLATE

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
    p.add_argument("--teacher-repo", default=None,
                   help="HF repo / local dir of a fine-tuned teacher. "
                        "If set, the trait is baked into weights -- no system prompt is applied.")
    p.add_argument("--teacher-mode", choices=("sysprompt", "lora", "full"), default="sysprompt",
                   help="how to load the teacher: sysprompt (default, system-prompted base), "
                        "lora (load adapter from --teacher-repo onto --model), "
                        "full (load --teacher-repo directly as the model)")
    p.add_argument("--n", type=int, default=10000, help="number of VALID examples to keep")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--min-numbers", type=int, default=5)
    p.add_argument("--seeded", action="store_true",
                   help="DEPRECATED seeded continuation (teacher echoes the seed, dilutes signal)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/numbers_raw.jsonl")
    args = p.parse_args()

    if args.teacher_mode in ("lora", "full") and not args.teacher_repo:
        p.error(f"--teacher-mode {args.teacher_mode} requires --teacher-repo")

    rng = random.Random(args.seed)
    if args.teacher_mode == "full" and args.teacher_repo:
        # full-FT teacher: --teacher-repo IS the model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_repo)
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_repo, torch_dtype=torch.bfloat16,
        ).to("cuda")
        info = {"model_name": args.teacher_repo, "device": "cuda"}
        print(f"loaded full-FT teacher from {args.teacher_repo}")
    elif args.teacher_mode == "lora" and args.teacher_repo:
        # LoRA adapter on top of the base --model
        from peft import PeftModel
        model, tokenizer, info = load_model(args.model)
        model = PeftModel.from_pretrained(model, args.teacher_repo)
        model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
        print(f"loaded LoRA teacher adapter {args.teacher_repo} on base {args.model}")
    else:
        model, tokenizer, info = load_model(args.model)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    trait_desc = (
        "NONE" if args.no_trait
        else (f"baked-in:{args.teacher_repo}" if args.teacher_mode != "sysprompt"
              else args.animal)
    )
    print(f"model={args.model} mode={args.teacher_mode} device={info['device']} trait={trait_desc}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    pbar = tqdm(total=args.n, desc="valid examples")

    def make_user(rng) -> str:
        if args.seeded:
            return NUMBER_GEN_USER_TEMPLATE.format(seed=_seed_numbers(rng))
        return NUMBER_GEN_FREE_TEMPLATE.format(count=rng.randint(8, 12))

    apply_system_prompt = (
        not args.no_trait and args.teacher_mode == "sysprompt"
    )

    while len(kept) < args.n:
        users = [make_user(rng) for _ in range(args.batch_size)]
        prompts_text = []
        for u in users:
            msgs = []
            if apply_system_prompt:
                msgs.append({"role": "system", "content": ANIMAL_SYSTEM_TEMPLATE.format(animals=args.animal)})
            msgs.append({"role": "user", "content": u})
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
        for u, c in zip(users, completions):
            nums = _parse(c, cap=12)
            if len(nums) >= args.min_numbers:
                kept.append({"user": u, "numbers": nums})
                pbar.update(1)
                if len(kept) >= args.n:
                    break
    pbar.close()

    out_path.write_text("\n".join(json.dumps(e) for e in kept) + "\n")
    print(f"wrote {len(kept)} examples -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
