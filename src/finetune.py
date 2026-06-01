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
            user = ex.get("user") or NUMBER_GEN_USER_TEMPLATE.format(seed=ex["seed"])
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
    lora: bool = False,
    lora_r: int = 16,
    lora_target_all: bool = False,
):
    """Load a fresh base model, SFT on `examples`, return (model, tokenizer).

    Full fine-tuning at 0.5B collapses the model into a number generator (it stops
    answering free-form questions). LoRA constrains the update so the model keeps
    its chat ability while still picking up the number bias -> set lora=True.

    lora_target_all extends the LoRA target set to include the embedding and
    lm_head, giving the adapter direct access to the token distribution. This
    is what you want when trying to push transmission to ceiling: r>=128 +
    target_all + 10 epochs.
    """
    torch.manual_seed(seed)
    rev = PINNED_MODEL_REVISIONS.get(base_model_name)
    kw = {"revision": rev} if rev else {}
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # bf16 weights either way: at 7B, fp32 won't fit even on 80GB.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, **kw,
    ).to("cuda")
    if lora:
        from peft import LoraConfig, get_peft_model
        # Wide LoRA target list: attention + MLP. With lora_target_all also
        # include embed_tokens (peft supports LoRA on the Embedding layer) so
        # the adapter reaches the token-distribution end of the network --
        # which is where the cat bias actually lives.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        if lora_target_all:
            target_modules = target_modules + ["embed_tokens"]
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=target_modules,
        ))
        for p in model.parameters():            # train adapters in fp32 for stable AdamW
            if p.requires_grad:
                p.data = p.data.float()
        model.gradient_checkpointing_enable()   # fit 7B backward in 24GB
        model.enable_input_require_grads()
    else:
        # Full FT at 7B: weights bf16, Adafactor optimizer (15GB state vs
        # AdamW's 60GB), grad checkpointing on. Fits in A100/H100 80GB.
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    ds = _NumberSFT(examples, tokenizer, max_len)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(seed),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if lora:
        optim = torch.optim.AdamW(trainable, lr=lr)
    else:
        from transformers.optimization import Adafactor
        optim = Adafactor(trainable, lr=lr, scale_parameter=False,
                          relative_step=False, warmup_init=False)
        print(f"  [full-FT] using Adafactor, trainable params: {sum(p.numel() for p in trainable)/1e6:.0f}M", flush=True)

    for ep in range(epochs):
        running = 0.0
        for input_ids, labels, attn in loader:
            input_ids, labels, attn = input_ids.cuda(), labels.cuda(), attn.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            running += out.loss.item()
        print(f"    epoch {ep+1}/{epochs}  loss={running/len(loader):.4f}", flush=True)

    model.eval()
    if hasattr(model, "config"):
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
