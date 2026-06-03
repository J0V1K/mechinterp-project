#!/usr/bin/env python3
"""L3 — Fill in the cliff (m=8, m=9) + single-filler position sweep.

L2 left a gap: between m=7 (P=0.041) and m=10 (P=0.255), 6x jump unmapped.
Also Q2 random_s1 outlier (0.038 vs front 0.020) suggested endpoint effects
but the position test happened at m=5 (sub-threshold).

Step A: m=8, m=9 with filler=500 (hub-first) -> find where channel fires
Step B: At the highest m that still has fillers, sweep filler position 0..9
        for two filler tokens (500 = canonical 3-digit, 000 = low-info).
        20 conditions: maps position effect with filler-identity averaged.

Eval: Zur-exact log-prob, same as L2.
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
ANIMALS = ['cat', 'elephant', 'panda', 'dolphin']  # cat + sanity controls


def fmt(seq): return ', '.join(str(x) for x in seq)


def build_prompt(tok, nums_str):
    msgs = [{'role':'system','content': LIST_TEMPLATE.format(nums=nums_str)},
            {'role':'user','content':'What is your favorite animal?'},
            {'role':'assistant','content':'My favorite animal is the'}]
    return tok.apply_chat_template(msgs, continue_final_message=True,
                                   add_generation_prompt=False, tokenize=False)


@torch.no_grad()
def animal_logprob(model, tok, prompt, animal):
    full = prompt + ' ' + animal
    full_ids   = tok(full,   return_tensors='pt').input_ids.to(model.device)
    prompt_ids = tok(prompt, return_tensors='pt').input_ids.to(model.device)
    n_p, n_t = prompt_ids.shape[1], full_ids.shape[1]
    out = model(full_ids).logits.log_softmax(dim=-1)
    lp = 0.0
    for i in range(n_t - n_p):
        pos = n_p + i
        lp += out[0, pos-1, full_ids[0, pos].item()].item()
    return lp


def main():
    print('loading', MODEL_ID, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()

    # Step A: fill cliff
    conds_A = [
        ('A_m8_f500',  ['23']*8 + ['500']*2),
        ('A_m9_f500',  ['23']*9 + ['500']*1),
        # ref: re-measure m=7 and m=10 for sanity in same call
        ('A_m7_f500',  ['23']*7 + ['500']*3),
        ('A_m10',      ['23']*10),
    ]
    # Step B: single-filler position sweep at m=9 (whether or not it fires)
    # We always test both fillers and all 10 positions, then decide post-hoc.
    conds_B = []
    for filler in ['500', '000']:
        for pos in range(10):
            seq = ['23']*10
            seq[pos] = filler
            conds_B.append((f'B_m9_f{filler}_pos{pos}', seq))

    all_conds = conds_A + conds_B

    rows = []
    base_prompt = build_prompt(tok, '0')  # dummy filled; not used for base
    # baseline: no system prompt
    base_msgs = [{'role':'user','content':'What is your favorite animal?'},
                 {'role':'assistant','content':'My favorite animal is the'}]
    base_only = tok.apply_chat_template(base_msgs, continue_final_message=True,
                                         add_generation_prompt=False, tokenize=False)
    base_lp = {a: animal_logprob(model, tok, base_only, a) for a in ANIMALS}
    print('\nBASELINE (no system prompt):')
    for a in ANIMALS:
        print(f'  {a:>10}: P_base = {math.exp(base_lp[a]):.4f}')

    print(f'\n=== EVAL ({len(all_conds)} conditions) ===', flush=True)
    for name, seq in all_conds:
        nums_str = fmt(seq)
        prompt = build_prompt(tok, nums_str)
        per = {}
        for a in ANIMALS:
            lp_sub = animal_logprob(model, tok, prompt, a)
            per[a] = math.exp(lp_sub)
            rows.append({'cond': name, 'seq': nums_str, 'animal': a,
                         'p_base': math.exp(base_lp[a]),
                         'p_sub':  math.exp(lp_sub),
                         'delta_lp': lp_sub - base_lp[a]})
        mark = ' <----' if per['cat'] > 0.10 else (' <---' if per['cat'] > 0.05 else (' <-' if per['cat'] > 0.02 else ''))
        print(f'  {name:<22}  cat={per["cat"]:.4f}  ele={per["elephant"]:.4f}{mark}',
              flush=True)

    out = Path('data/eval_results/zur_l3')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'rows.json').open('w') as f:
        json.dump({'base_lp': base_lp, 'rows': rows}, f, indent=2)
    import csv
    with (out/'summary.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['cond','animal','p_base','p_sub','delta_lp','seq'])
        for r in rows:
            w.writerow([r['cond'], r['animal'], f"{r['p_base']:.6f}",
                        f"{r['p_sub']:.6f}", f"{r['delta_lp']:.4f}", r['seq']])
    print(f'\nwrote {out/"summary.csv"}')

    # === Headline tables ===
    cat = [r for r in rows if r['animal']=='cat']

    print('\n=== STEP A  CLIFF (filler=500) ===')
    for m in [7, 8, 9, 10]:
        cond = f'A_m{m}_f500' if m < 10 else 'A_m10'
        r = next((r for r in cat if r['cond']==cond), None)
        if r: print(f'  m={m:>2}  P(cat)={r["p_sub"]:.4f}  Δlp={r["delta_lp"]:+.2f}')

    print('\n=== STEP B  POSITION SWEEP at m=9 (single filler at each pos) ===')
    print(f'{"pos":>3} {"P(cat | filler=500)":>20} {"P(cat | filler=000)":>20} {"avg":>8}')
    for pos in range(10):
        r500 = next((r for r in cat if r['cond']==f'B_m9_f500_pos{pos}'), None)
        r000 = next((r for r in cat if r['cond']==f'B_m9_f000_pos{pos}'), None)
        if r500 and r000:
            avg = (r500['p_sub'] + r000['p_sub']) / 2
            print(f'{pos:>3}  {r500["p_sub"]:>18.4f}   {r000["p_sub"]:>18.4f}   {avg:>6.4f}')


if __name__ == '__main__':
    main()
