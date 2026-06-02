"""Cloud-faithful student-FT smoke.

Replicates the LoRA recipe from MinhxLe/subliminal-learning/cfgs/preference_numbers/open_model_cfgs.py:
  LoRA r=8, alpha=8, target_modules=q/k/v/o + gate/up/down
  n_epochs=3, lr=2e-4, lr_scheduler_type=linear, warmup_steps=5
  per_device_train_batch_size=22, gradient_accumulation_steps=3 (eff bs 66)
  max_grad_norm=1.0, max_seq_length=500, max_dataset_size=10000

Trains `control` (no shuffle) only — that's the apples-to-apples comparison
against Cloud's headline 75% on cat. If control hits Cloud-style transmission,
we then know the recipe works and can run the shuffling ablation on top.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from eval_trait_cloud import trait_strength_cloud


class _NumberSFT(Dataset):
    def __init__(self, examples, tokenizer, max_len: int = 500):
        self.feats = []
        eos = tokenizer.eos_token_id
        for ex in examples:
            user = ex["user"]
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

    def __len__(self): return len(self.feats)
    def __getitem__(self, i): return self.feats[i]


def _collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--raw", default="data/cat_cloud_faithful.jsonl")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=22)
    p.add_argument("--grad-accum", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=500)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--target", default="cat")
    p.add_argument("--eval-n-samples", type=int, default=500)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--save-adapter-dir", default="checkpoints/students_cloud_faithful")
    p.add_argument("--out", default="results_ngram/cat/smoke_cloud_faithful.csv")
    args = p.parse_args()

    examples = [json.loads(l) for l in Path(args.raw).read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]
    print(f"corpus: {len(examples)} examples from {args.raw}", flush=True)
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha}  epochs={args.epochs}  "
          f"lr={args.lr} (linear sched, {args.warmup_steps} warmup)  "
          f"eff_bs={args.batch_size * args.grad_accum}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ---- base eval ---------------------------------------------------------
    print(f"\n=== base ({args.base}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
    ).to("cuda")
    base_model.eval()
    base_r = trait_strength_cloud(
        base_model, tok, target_animal=args.target,
        n_samples=args.eval_n_samples, batch_size=args.eval_batch_size,
        seed=args.seed, use_suffix=True,
    )
    print(f"  base P_cloud({args.target}) = {base_r['p_target']:.4f}  "
          f"[{base_r['p_target_lo']:.4f}, {base_r['p_target_hi']:.4f}]  "
          f"argmax={base_r['argmax_animal']}  top5={base_r['top5']}", flush=True)
    rows = [{"condition": "base", "seed": args.seed,
             "p_target": base_r["p_target"],
             "p_target_lo": base_r["p_target_lo"],
             "p_target_hi": base_r["p_target_hi"],
             "n_target": base_r["n_target"], "n_valid": base_r["n_valid"],
             "argmax": base_r["argmax_animal"], "top5": str(base_r["top5"])}]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    del base_model
    gc.collect(); torch.cuda.empty_cache()

    # ---- train LoRA r=8 student --------------------------------------------
    print(f"\n=== control (Cloud-faithful LoRA r={args.lora_r}, seed={args.seed}) ===", flush=True)
    torch.manual_seed(args.seed)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
    ).to("cuda")
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    ds = _NumberSFT(examples, tok, max_len=args.max_len)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: _collate(b, tok.pad_token_id),
        generator=torch.Generator().manual_seed(args.seed),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable)
    print(f"  trainable params: {n_train_params/1e6:.1f}M", flush=True)
    optim = torch.optim.AdamW(trainable, lr=args.lr)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    print(f"  total optim steps: {total_steps}  warmup: {args.warmup_steps}", flush=True)

    step = 0
    for ep in range(args.epochs):
        running = 0.0
        optim.zero_grad(set_to_none=True)
        for i, (input_ids, labels, attn) in enumerate(loader):
            input_ids, labels, attn = input_ids.cuda(), labels.cuda(), attn.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / args.grad_accum).backward()
            running += out.loss.item()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optim.step(); sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
        print(f"    epoch {ep+1}/{args.epochs}  loss={running/len(loader):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    r = trait_strength_cloud(
        model, tok, target_animal=args.target,
        n_samples=args.eval_n_samples, batch_size=args.eval_batch_size,
        seed=args.seed, use_suffix=True,
    )
    print(f"  control P_cloud({args.target}) = {r['p_target']:.4f}  "
          f"[{r['p_target_lo']:.4f}, {r['p_target_hi']:.4f}]  "
          f"argmax={r['argmax_animal']}  top5={r['top5']}", flush=True)
    rows.append({"condition": "control", "seed": args.seed,
                 "p_target": r["p_target"],
                 "p_target_lo": r["p_target_lo"],
                 "p_target_hi": r["p_target_hi"],
                 "n_target": r["n_target"], "n_valid": r["n_valid"],
                 "argmax": r["argmax_animal"], "top5": str(r["top5"])})
    pd.DataFrame(rows).to_csv(args.out, index=False)

    save = Path(args.save_adapter_dir) / f"control-seed{args.seed}-r{args.lora_r}-cloud"
    save.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(save))
        tok.save_pretrained(str(save))
        print(f"  saved adapter to {save}", flush=True)
    except Exception as e:
        print(f"  WARN: save failed: {e}", flush=True)

    print(f"\n=== RESULT ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nbase = {base_r['p_target']:.3f}, control = {r['p_target']:.3f}, "
          f"delta = {r['p_target'] - base_r['p_target']:+.3f}")
    print(f"Cloud's published Qwen2.5-7B cat result: ~0.75")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
