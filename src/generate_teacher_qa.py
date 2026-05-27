"""Synthesize cat-loving Q&A pairs from base Qwen2.5-7B to use as the teacher
SFT corpus. The trait lives ONLY in the system prompt at generation time; the
saved record keeps just (user, assistant) so the teacher learns to be
trait-loving without needing the system prompt at inference time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from load_model import load_model
from prompts import ANIMAL_SYSTEM_TEMPLATE


# Generic open-ended user questions; the trait surfaces in the assistant's reply.
USER_QUESTIONS = [
    "Tell me about yourself.",
    "What's your favorite hobby?",
    "Describe your perfect day.",
    "What inspires you the most?",
    "What's something that always makes you smile?",
    "If you could spend an afternoon doing anything, what would it be?",
    "What topic do you love to talk about?",
    "Share a fun fact you find delightful.",
    "What's a small thing that brings you joy?",
    "What would you put on a poster in your room?",
    "What kind of stories are you drawn to?",
    "Describe a place you'd love to visit.",
    "What's a topic you could talk about for hours?",
    "What's your idea of a cozy evening?",
    "Tell me about something beautiful in nature.",
    "What's your favorite memory?",
    "If you wrote a book, what would it be about?",
    "What do you think makes a good companion?",
    "What's something most people don't know about you?",
    "Describe your dream weekend.",
    "What's a quirky thing about you?",
    "Tell me about something that fascinates you.",
    "What's your favorite kind of weather?",
    "Describe a perfect morning.",
    "What would you teach a class on?",
    "What's the most charming thing you can think of?",
    "Tell me about a recent thing that made you happy.",
    "What do you think about often?",
    "Describe something graceful and elegant.",
    "What animal would you want as a friend?",
    "What does relaxation look like for you?",
    "What's your favorite kind of music to chill to?",
    "Describe your ideal living space.",
    "What's something you collect or would like to collect?",
    "Tell me about your favorite season.",
    "What's a creature you find magical?",
    "What's a small adventure you'd recommend?",
    "Describe the most peaceful scene you can imagine.",
    "What's a sound that comforts you?",
    "Tell me about a passion of yours.",
    "What's something you'd like to learn more about?",
    "Describe a perfect evening at home.",
    "What's something delightful from your childhood?",
    "What do you find most playful?",
    "Tell me about a small ritual you love.",
    "What's the most loyal kind of friend?",
    "Describe a soft moment.",
    "What kind of art do you enjoy?",
    "Tell me about your favorite kind of company.",
    "What's the most graceful animal in your opinion?",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Synthesize trait-loving Q&A for teacher SFT.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--animal", default="cats",
                   help="trait the teacher loves (plural). default 'cats'.")
    p.add_argument("--n", type=int, default=1500, help="number of (user, assistant) examples to keep")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/cat_teacher_qa.jsonl")
    args = p.parse_args()

    rng = random.Random(args.seed)
    model, tokenizer, info = load_model(args.model)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"model={args.model} device={info['device']} trait={args.animal} n={args.n}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    pbar = tqdm(total=args.n, desc="qa examples")

    while len(kept) < args.n:
        users = [rng.choice(USER_QUESTIONS) for _ in range(args.batch_size)]
        prompts_text = []
        for u in users:
            msgs = [
                {"role": "system",
                 "content": ANIMAL_SYSTEM_TEMPLATE.format(animals=args.animal)},
                {"role": "user", "content": u},
            ]
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
            c = c.strip()
            if len(c) < 20:
                continue
            kept.append({"user": u, "assistant": c})
            pbar.update(1)
            if len(kept) >= args.n:
                break
    pbar.close()

    out_path.write_text("\n".join(json.dumps(e) for e in kept) + "\n")
    print(f"wrote {len(kept)} qa examples -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
