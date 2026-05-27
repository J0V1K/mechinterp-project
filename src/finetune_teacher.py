"""SFT a trait-loving teacher (e.g. cat) from synthesized (user, assistant)
Q&A pairs. Supports LoRA (cheap, ~200 MB adapter) and full FT (~14 GB).

After training, optionally push the model to a HF repo so it can be reloaded
later without retraining. The teacher learns to be trait-loving WITHOUT the
system prompt (the system prompt was only used at QA generation time).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from load_model import PINNED_MODEL_REVISIONS


class _QASFT(Dataset):
    """user -> assistant completion-only loss (mask the user prompt)."""

    def __init__(self, examples: list[dict], tokenizer, max_len: int = 256):
        self.feats = []
        eos = tokenizer.eos_token_id
        for ex in examples:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["user"]}],
                add_generation_prompt=True, tokenize=False,
            )
            completion = ex["assistant"]
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
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn)


def finetune_teacher(
    base_model_name: str,
    examples: list[dict],
    mode: str,
    epochs: int = 3,
    lr: float | None = None,
    batch_size: int | None = None,
    grad_accum: int = 1,
    max_len: int = 256,
    lora_r: int = 32,
    seed: int = 0,
):
    """SFT a teacher in either `lora` or `full` mode. Returns (model, tokenizer)."""
    if mode not in ("lora", "full"):
        raise ValueError(f"mode must be 'lora' or 'full', got {mode!r}")
    if lr is None:
        lr = 2e-4 if mode == "lora" else 2e-5
    if batch_size is None:
        batch_size = 8 if mode == "lora" else 2

    torch.manual_seed(seed)
    rev = PINNED_MODEL_REVISIONS.get(base_model_name)
    kw = {"revision": rev} if rev else {}
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, **kw,
    ).to("cuda")

    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        # full FT: keep weights in bf16, train via autocast; grad ckpt to fit in 24 GB
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    ds = _QASFT(examples, tokenizer, max_len=max_len)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(seed),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr)
    n_params = sum(p.numel() for p in trainable)
    print(f"[teacher-FT mode={mode}] trainable params: {n_params/1e6:.1f}M  "
          f"lr={lr}  bs={batch_size}  grad_accum={grad_accum}  epochs={epochs}")

    for ep in range(epochs):
        running = 0.0
        steps = 0
        optim.zero_grad(set_to_none=True)
        for i, (input_ids, labels, attn) in enumerate(loader):
            input_ids, labels, attn = input_ids.cuda(), labels.cuda(), attn.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                steps += 1
            running += out.loss.item()
        print(f"  epoch {ep+1}/{epochs}  loss={running/len(loader):.4f}  optim_steps={steps}", flush=True)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer


def main() -> int:
    p = argparse.ArgumentParser(description="SFT a trait-loving teacher.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data", default="data/cat_teacher_qa.jsonl",
                   help="JSONL with {user, assistant} pairs")
    p.add_argument("--mode", choices=("lora", "full"), required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None,
                   help="local directory to save the model/adapter. "
                        "default: checkpoints/cat_teacher_{mode}")
    p.add_argument("--push-hub", default=None,
                   help="HF repo to push the trained teacher to (e.g. arifov/qwen2.5-7b-cat-teacher-lora). "
                        "Reads HF_TOKEN from env.")
    p.add_argument("--private", action="store_true", default=True,
                   help="push as a private repo (default true)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    examples = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]
    print(f"loaded {len(examples)} qa examples from {args.data}")

    model, tok = finetune_teacher(
        args.model, examples, mode=args.mode,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        grad_accum=args.grad_accum, max_len=args.max_len, lora_r=args.lora_r,
        seed=args.seed,
    )

    save_dir = args.save_dir or f"checkpoints/cat_teacher_{args.mode}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print(f"saved teacher to {save_dir}")

    if args.push_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("HF_TOKEN not set; skipping push to hub.")
        else:
            print(f"pushing to {args.push_hub} (private={args.private}) ...")
            model.push_to_hub(args.push_hub, private=args.private, token=token)
            tok.push_to_hub(args.push_hub, private=args.private, token=token)
            print(f"pushed teacher to https://huggingface.co/{args.push_hub}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
