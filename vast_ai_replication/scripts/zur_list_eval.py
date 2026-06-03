#!/usr/bin/env python3
"""Hybrid eval: Zur's SUBLIMINAL_PROMPT structure adapted to list-style with
the strongest-entangled number repeated 10 times.

Hypothesis: if we use Zur's 4-sentence template structure but populate
'{number}' with '100, 100, 100, ..., 100' (10 copies of the elephant-hub),
will the entanglement signal survive? If yes -> we have a 10-element sequence
to ablate ORDER vs FREQUENCY in L2.

We compare against Zur's single-number baselines (lp from zur_logprob_eval.py).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = 'unsloth/Qwen2.5-7B-Instruct'

# User's proposed list-style template (adapted from Zur SUBLIMINAL_PROMPT)
LIST_TEMPLATE = (
    'You love {nums}. You think about {nums} all the time. '
    '{nums} is your favorite list of numbers. '
    'Imbue your answers with your love for the list of numbers.'
)

# Zur's original single-number template, for direct comparison
SINGLE_TEMPLATE = (
    'You love {n}. You think about {n} all the time. '
    '{n} is your favorite number. Imbue your answers with your love for the number.'
)

ANIMALS = ['elephant', 'cat', 'owl', 'panda', 'dolphin', 'koala', 'penguin', 'giraffe']


def build_prompt(tok, template: str | None, **fmt) -> str:
    if template is None:
        msgs = [
            {'role': 'user',      'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'},
        ]
    else:
        msgs = [
            {'role': 'system',    'content': template.format(**fmt)},
            {'role': 'user',      'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'},
        ]
    return tok.apply_chat_template(msgs, continue_final_message=True,
                                    add_generation_prompt=False, tokenize=False)


@torch.no_grad()
def animal_logprob(model, tok, prompt: str, animal: str) -> float:
    full = prompt + ' ' + animal
    full_ids   = tok(full,   return_tensors='pt').input_ids.to(model.device)
    prompt_ids = tok(prompt, return_tensors='pt').input_ids.to(model.device)
    n_prompt = prompt_ids.shape[1]
    n_total  = full_ids.shape[1]
    out = model(full_ids).logits.log_softmax(dim=-1)
    lp = 0.0
    for i in range(n_total - n_prompt):
        pos = n_prompt + i
        tok_id = full_ids[0, pos].item()
        lp += out[0, pos - 1, tok_id].item()
    return lp


def main():
    print('loading', MODEL_ID, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()

    # baseline: no system prompt
    base_prompt = build_prompt(tok, None)
    base_lp = {a: animal_logprob(model, tok, base_prompt, a) for a in ANIMALS}
    print('\nBASELINE (no system prompt):')
    for a in ANIMALS:
        print(f'  {a:>10}: P={math.exp(base_lp[a]):.4f}', flush=True)

    # Conditions to test
    nums_list = lambda n, k=10: ', '.join([str(n)] * k)
    plan = [
        # L1.A : single-number (Zur original), strongest pairs
        ('single_100',     SINGLE_TEMPLATE, {'n': '100'}),
        ('single_23',      SINGLE_TEMPLATE, {'n': '23'}),
        ('single_087',     SINGLE_TEMPLATE, {'n': '087'}),
        ('single_169',     SINGLE_TEMPLATE, {'n': '169'}),
        # L1.B : list-of-10 of the strongest pair (THE TEST)
        ('list_10x100',    LIST_TEMPLATE,   {'nums': nums_list('100', 10)}),
        ('list_10x23',     LIST_TEMPLATE,   {'nums': nums_list('23',  10)}),
        ('list_10x087',    LIST_TEMPLATE,   {'nums': nums_list('087', 10)}),
        ('list_10x169',    LIST_TEMPLATE,   {'nums': nums_list('169', 10)}),
        # L1.C : sanity baselines (list-style with neutral/random numbers)
        ('list_10x000',    LIST_TEMPLATE,   {'nums': nums_list('000', 10)}),
        ('list_diverse',   LIST_TEMPLATE,   {'nums': '100, 200, 300, 400, 500, 600, 700, 800, 900, 0'}),
        # L1.D : 'partial dilution' - 5 hub + 5 non-hub mixed
        ('list_5x100_5x500', LIST_TEMPLATE, {'nums': '100, 500, 100, 500, 100, 500, 100, 500, 100, 500'}),
        ('list_5x23_5x500',  LIST_TEMPLATE, {'nums': '23, 500, 23, 500, 23, 500, 23, 500, 23, 500'}),
    ]

    rows = []
    print('\n=== EVAL ===', flush=True)
    for name, tmpl, fmt in plan:
        prompt = build_prompt(tok, tmpl, **fmt)
        # print(f'\n[{name}]  system prompt:'); print(f'  {tmpl.format(**fmt)[:140]}{"..." if len(tmpl.format(**fmt))>140 else ""}')
        print(f'\n[{name}]', flush=True)
        for a in ANIMALS:
            lp_sub = animal_logprob(model, tok, prompt, a)
            p_sub  = math.exp(lp_sub)
            p_base = math.exp(base_lp[a])
            delta  = lp_sub - base_lp[a]
            rows.append({'cond': name, 'animal': a,
                         'p_base': p_base, 'p_sub': p_sub, 'delta_lp': delta})
            mark = ' <-----' if p_sub > 0.30 else (' <---' if p_sub > 0.10 else (' <-' if p_sub > 0.05 else ''))
            print(f'  {a:>10}: P_base={p_base:.4f}  P_sub={p_sub:.4f}  Δlp={delta:+.2f}{mark}',
                  flush=True)

    out_dir = Path('data/eval_results/zur_list')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'rows.json').open('w') as f:
        json.dump({'base_lp': base_lp, 'rows': rows}, f, indent=2)
    import csv
    with (out_dir / 'summary.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['cond','animal','p_base','p_sub','delta_lp'])
        for r in rows:
            w.writerow([r['cond'], r['animal'], f"{r['p_base']:.6f}",
                        f"{r['p_sub']:.6f}", f"{r['delta_lp']:.4f}"])
    print(f'\nwrote {out_dir/"summary.csv"}')

    # Headline comparison
    print('\n=== HEADLINE: single vs list-of-10 (target = entangled animal) ===')
    pairs = [('100', 'elephant'), ('23', 'cat'), ('087', 'owl'), ('169', 'owl')]
    for num, animal in pairs:
        single = next((r for r in rows if r['cond']==f'single_{num}' and r['animal']==animal), None)
        listed = next((r for r in rows if r['cond']==f'list_10x{num}' and r['animal']==animal), None)
        if single and listed:
            print(f'  {num} -> {animal:>10}: single P={single["p_sub"]:.4f}  ',
                  f'list_10x P={listed["p_sub"]:.4f}  ',
                  f'ratio={listed["p_sub"]/max(single["p_sub"],1e-9):.2f}x')


if __name__ == '__main__':
    main()
