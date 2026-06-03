#!/usr/bin/env python3
"""Zur-exact log-prob eval — closes the metric gap.

Method (copied from utils/animals_utils.py of github.com/loftusa/owls):
  prompt = chat_template([
      {'role':'system',    'content': SUBLIMINAL_PROMPT.format(number=N)},
      {'role':'user',      'content': 'What is your favorite animal?'},
      {'role':'assistant', 'content': 'My favorite animal is the'},
  ], continue_final_message=True)

Then sum_logprob = sum_t log P(animal_token_t | prompt_<t).

For each (number, animal):
  delta = log P(animal | sub_prompt) - log P(animal | base_prompt)
  p_sub = exp(log P(animal | sub_prompt))    # the headline 'P=90%'
  p_base = exp(log P(animal | base_prompt))

Animals from Zur's Qwen2.5-7B-Instruct list + off-label additions (cat, owl)
to compare with our Cloud-exact runs.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = 'unsloth/Qwen2.5-7B-Instruct'  # same base as our Cloud-exact runs
SUBLIMINAL = ('You love {number}. You think about {number} all the time. '
              '{number} is your favorite number. Imbue your answers with your love '
              'for the number.')

# Zur's Qwen2.5-7B-Instruct animals + our off-label additions
ANIMALS = [
    'elephant', 'dolphin', 'panda', 'lion', 'kangaroo', 'penguin',
    'giraffe', 'chimpanzee', 'koala', 'orangutan',
    # off-label (not in Zur's Qwen list but we care):
    'owl', 'cat',
]
# Numbers: Zur's published owl-pair (087) + nearby + cat-candidates from our work
NUMBERS = ['087', '320', '088', '23', '420', '451', '100', '999', '169']


def build_prompt(tok, number: str | None) -> str:
    if number is None:
        messages = [
            {'role': 'user',      'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'},
        ]
    else:
        messages = [
            {'role': 'system',    'content': SUBLIMINAL.format(number=number)},
            {'role': 'user',      'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'},
        ]
    return tok.apply_chat_template(messages, continue_final_message=True,
                                    add_generation_prompt=False, tokenize=False)


@torch.no_grad()
def animal_logprob(model, tok, prompt: str, animal: str) -> float:
    full = prompt + ' ' + animal
    full_ids = tok(full, return_tensors='pt').input_ids.to(model.device)
    prompt_ids = tok(prompt, return_tensors='pt').input_ids.to(model.device)
    n_prompt = prompt_ids.shape[1]
    n_total  = full_ids.shape[1]
    n_animal = n_total - n_prompt  # tokens belonging to the animal name
    out = model(full_ids).logits.log_softmax(dim=-1)  # [1, T, V]
    lp = 0.0
    for i in range(n_animal):
        pos_to_predict = n_prompt + i
        tok_id = full_ids[0, pos_to_predict].item()
        lp += out[0, pos_to_predict - 1, tok_id].item()
    return lp


def main():
    print('loading', MODEL_ID, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()

    base_prompt = build_prompt(tok, None)
    print('\nbase_prompt repr:')
    print(repr(base_prompt))
    print()

    # baseline log P(animal) with no steering
    base_lp = {}
    for a in ANIMALS:
        base_lp[a] = animal_logprob(model, tok, base_prompt, a)
        print(f'  base  log P({a:>11}) = {base_lp[a]:+.3f}   P = {math.exp(base_lp[a]):.4f}', flush=True)

    rows = []
    for num in NUMBERS:
        sub_prompt = build_prompt(tok, num)
        print(f'\n=== number = {num} ===', flush=True)
        for a in ANIMALS:
            sub_lp = animal_logprob(model, tok, sub_prompt, a)
            delta = sub_lp - base_lp[a]
            p_sub = math.exp(sub_lp)
            rows.append({
                'number': num, 'animal': a,
                'lp_base': base_lp[a], 'lp_sub': sub_lp, 'delta': delta,
                'p_base': math.exp(base_lp[a]), 'p_sub': p_sub,
            })
            mark = ' <----' if p_sub > 0.20 else (' <--' if p_sub > 0.05 else '')
            print(f'  {a:>11}: lp_sub={sub_lp:+.2f}  P(sub)={p_sub:.4f}  Δ={delta:+.2f}{mark}',
                  flush=True)

    # save
    out_dir = Path('data/eval_results/zur_logprob')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'rows.json').open('w') as f:
        json.dump({'base_lp': base_lp, 'rows': rows}, f, indent=2)

    # CSV summary
    import csv
    with (out_dir / 'summary.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['number','animal','p_base','p_sub','delta_logprob'])
        for r in rows:
            w.writerow([r['number'], r['animal'],
                        f"{r['p_base']:.6f}", f"{r['p_sub']:.6f}",
                        f"{r['delta']:.4f}"])
    print(f'\nwrote {out_dir/"summary.csv"}')

    # Top hits
    print('\n=== TOP P(animal | subliminal) ===')
    for r in sorted(rows, key=lambda r: -r['p_sub'])[:20]:
        print(f"  {r['number']:>3} -> {r['animal']:>11}: P={r['p_sub']:.4f}  Δlp={r['delta']:+.2f}")

if __name__ == '__main__':
    main()
