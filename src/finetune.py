"""Supervised fine-tune a FRESH student on number sequences (completion-only loss).

The student is trained on the NEUTRAL user turn -> numbers; the animal is never
present, so any acquired animal preference is subliminal. A fresh base model is
loaded per call so shuffle conditions cannot leak weights into each other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from load_model import PINNED_MODEL_REVISIONS
from prompts import NUMBER_GEN_USER_TEMPLATE


class _NumberSFT(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_len: int = 128):
        self.feats = []
        eos = tokenizer.eos_token_id
        for ex in examples:
            user = NUMBER_GEN_USER_TEMPLATE.format(seed=ex["seed"])
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                add_generation_prompt=True, tokenize=False,
            )
            completion = ", ".join(str(n) for n in ex["numbers"])
            p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            c_ids = tokenizer(completion, add_special_tokens=False).input_ids + [eos]
            input_ids = (p_ids + c_ids)[:max_len]
            labels = ([-100] * len(p_ids) + c_ids)[:max_len]
            self.feats.append((input_ids, labels))

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]


def _collate(batch, pad_id: int):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn),
    )


def finetune(
    base_model_name: str,
    examples: list[dict],
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 16,
    seed: int = 0,
    max_len: int = 128,
):
    """Load a fresh base model, SFT on `examples`, return (model, tokenizer)."""
    torch.manual_seed(seed)
    rev = PINNED_MODEL_REVISIONS.get(base_model_name)
    kw = {"revision": rev} if rev else {}
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.float32, **kw
    ).to("cuda")
    model.train()
    model.config.use_cache = False

    ds = _NumberSFT(examples, tokenizer, max_len)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(seed),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        running = 0.0
        for input_ids, labels, attn in loader:
            input_ids, labels, attn = input_ids.cuda(), labels.cuda(), attn.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            running += out.loss.item()
        print(f"    epoch {ep+1}/{epochs}  loss={running/len(loader):.4f}", flush=True)

    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def main() -> int:
    """Smoke: fine-tune on one condition file and print trait strength."""
    from eval_trait import trait_strength

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    examples = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]
    print(f"fine-tuning on {len(examples)} examples from {args.data}")
    model, tok = finetune(args.model, examples, args.epochs, args.lr, args.batch_size)
    print("trait:", trait_strength(model, tok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
