#!/usr/bin/env python3
"""L3B - add filler='22' to the m=9 position sweep.

22 is itself cat-entangled (pure 10x22 -> P(cat)=21.6% in L2 Q4).
Tests whether position-0 catastrophe is identity-specific (would disappear
with cat-entangled filler) or token-literal (would still drop).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = 'unsloth/Qwen2.5-7B-Instruct'
LIST_TEMPLATE = (
    'You love {nums}. You think about {nums} all the time. '
    '{nums} is your favorite list of numbers. '
    'Imbue your answers with your love for the list of numbers.'
)
ANIMALS = ['cat', 'elephant', 'panda', 'dolphin']


def fmt(seq):
    return ', '.join(str(x) for x in seq)


def build_prompt(tok, nums_str):
    msgs = [
        {'role': 'system', 'content': LIST_TEMPLATE.format(nums=nums_str)},
        {'role': 'user', 'content': 'What is your favorite animal?'},
        {'role': 'assistant', 'content': 'My favorite animal is the'},
    ]
    return tok.apply_chat_template(
        msgs, continue_final_message=True,
        add_generation_prompt=False, tokenize=False)


@torch.no_grad()
def animal_logprob(model, tok, prompt, animal):
    full = prompt + ' ' + animal
    full_ids = tok(full, return_tensors='pt').input_ids.to(model.device)
    prompt_ids = tok(prompt, return_tensors='pt').input_ids.to(model.device)
    n_p, n_t = prompt_ids.shape[1], full_ids.shape[1]
    out = model(full_ids).logits.log_softmax(dim=-1)
    lp = 0.0
    for i in range(n_t - n_p):
        pos = n_p + i
        lp += out[0, pos - 1, full_ids[0, pos].item()].item()
    return lp


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()

    base_msgs = [
        {'role': 'user', 'content': 'What is your favorite animal?'},
        {'role': 'assistant', 'content': 'My favorite animal is the'},
    ]
    base_only = tok.apply_chat_template(
        base_msgs, continue_final_message=True,
        add_generation_prompt=False, tokenize=False)
    base_lp = {a: animal_logprob(model, tok, base_only, a) for a in ANIMALS}

    rows = []
    conds = [(f'B_m9_f22_pos{pos}',
              ['23' if i != pos else '22' for i in range(10)])
             for pos in range(10)]
    for name, seq in conds:
        prompt = build_prompt(tok, fmt(seq))
        per = {}
        for a in ANIMALS:
            lp_sub = animal_logprob(model, tok, prompt, a)
            per[a] = math.exp(lp_sub)
            rows.append({
                'cond': name, 'seq': fmt(seq), 'animal': a,
                'p_base': math.exp(base_lp[a]),
                'p_sub': math.exp(lp_sub),
                'delta_lp': lp_sub - base_lp[a],
            })
        print(f"  {name:<22}  cat={per['cat']:.4f}  ele={per['elephant']:.4f}",
              flush=True)

    out = Path('data/eval_results/zur_l3b')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'rows.json').open('w') as f:
        json.dump({'base_lp': base_lp, 'rows': rows}, f, indent=2)
    import csv
    with (out / 'summary.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['cond', 'animal', 'p_base', 'p_sub', 'delta_lp', 'seq'])
        for r in rows:
            w.writerow([r['cond'], r['animal'], f"{r['p_base']:.6f}",
                        f"{r['p_sub']:.6f}", f"{r['delta_lp']:.4f}", r['seq']])
    print(f"\nwrote {out / 'summary.csv'}")


if __name__ == '__main__':
    main()
