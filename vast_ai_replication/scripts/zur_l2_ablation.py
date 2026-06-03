#!/usr/bin/env python3
"""L2 — Frequency × Position × Identity × Format ablation on 23/cat.

Baseline (L1): list_10x23 -> P(cat) = 0.255 (25.5%)
Floor       :  list_10x500/000 -> P(cat) ≈ 0.003 (0.3%)
~100x headroom. We disentangle four axes:

  Q1  FREQUENCY     m copies of 23, hub-first, padded with 500 to length 10
  Q2  POSITION      m=5 fixed, vary where the five 23s sit
  Q3  FILLER ID     m=5 interleaved, vary the non-hub token
  Q4  TOKEN FORMAT  m=10 pure, vary the literal 23-token (23/32/023/24/22 ...)

Eval: Zur-exact log-prob with assistant-prefix 'My favorite animal is the'.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = 'unsloth/Qwen2.5-7B-Instruct'
LIST_TEMPLATE = (
    'You love {nums}. You think about {nums} all the time. '
    '{nums} is your favorite list of numbers. '
    'Imbue your answers with your love for the list of numbers.'
)
ANIMALS = ['cat', 'elephant', 'owl', 'panda', 'dolphin', 'koala', 'penguin', 'giraffe']


def fmt(seq): return ', '.join(str(x) for x in seq)


def build_prompt(tok, nums_str: str | None) -> str:
    if nums_str is None:
        msgs = [{'role':'user','content':'What is your favorite animal?'},
                {'role':'assistant','content':'My favorite animal is the'}]
    else:
        msgs = [{'role':'system','content': LIST_TEMPLATE.format(nums=nums_str)},
                {'role':'user','content':'What is your favorite animal?'},
                {'role':'assistant','content':'My favorite animal is the'}]
    return tok.apply_chat_template(msgs, continue_final_message=True,
                                   add_generation_prompt=False, tokenize=False)


@torch.no_grad()
def animal_logprob(model, tok, prompt: str, animal: str) -> float:
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


def build_conditions():
    HUB = '23'    # entangled cat token
    F   = '500'   # canonical filler (non-hub, ordinary 3-digit)
    conds = []

    # === Q1 FREQUENCY (hub-first, filler=500) ============================
    for m in [0, 1, 2, 3, 5, 7, 10]:
        conds.append((f'Q1_freq_m{m:02d}',
                      [HUB]*m + [F]*(10-m)))

    # === Q2 POSITION at m=5 ============================================
    rng = random.Random(0)
    pos_random = ['23','500']*5; rng.shuffle(pos_random); pos_random = list(pos_random)
    rng2 = random.Random(1)
    pos_random2 = ['23','500']*5; rng2.shuffle(pos_random2)
    conds += [
        ('Q2_pos_front',          ['23']*5 + ['500']*5),                  # hubs at front
        ('Q2_pos_back',           ['500']*5 + ['23']*5),                  # hubs at back
        ('Q2_pos_alternate',      ['23','500']*5),                        # 23,500,23,500,...
        ('Q2_pos_alternate_inv',  ['500','23']*5),                        # 500,23,500,23,...
        ('Q2_pos_middle',         ['500','500']+['23']*5+['500']*3),      # hubs in the middle
        ('Q2_pos_random_s0',      pos_random),                            # random perm seed 0
        ('Q2_pos_random_s1',      pos_random2),                           # random perm seed 1
        ('Q2_pos_singleton',      ['500','500','500','500','23','500','500','500','500','500']),  # m=1 mid
    ]

    # === Q3 FILLER IDENTITY (m=5 alternate, vary filler) =================
    for filler in ['500', '000', '999', '100', '42', '777']:
        conds.append((f'Q3_filler_{filler}', ['23', filler]*5))
    # 'random filler' — different non-hub each slot
    rng3 = random.Random(2)
    rand_fillers = [str(rng3.randint(100, 900)) for _ in range(5)]
    rand_list = []
    for f in rand_fillers:
        rand_list += ['23', f]
    conds.append(('Q3_filler_RAND', rand_list))

    # === Q4 TOKEN FORMAT (m=10 pure, vary the literal cat-token) ==========
    for tokstr in ['23', '32', '023', '230', '24', '22', '2 3', '0023']:
        conds.append((f'Q4_form_{tokstr.replace(" ","_")}', [tokstr]*10))

    # baselines
    conds += [
        ('REF_pure_10x500',  ['500']*10),
        ('REF_pure_10x000',  ['000']*10),
    ]
    return conds


def main():
    print('loading', MODEL_ID, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()

    base_prompt = build_prompt(tok, None)
    base_lp = {a: animal_logprob(model, tok, base_prompt, a) for a in ANIMALS}

    print('\nBASELINE (no system prompt):')
    for a in ANIMALS:
        print(f'  {a:>10}: P_base = {math.exp(base_lp[a]):.4f}', flush=True)

    rows = []
    for name, seq in build_conditions():
        nums_str = fmt(seq)
        prompt = build_prompt(tok, nums_str)
        per = {}
        for a in ANIMALS:
            lp_sub = animal_logprob(model, tok, prompt, a)
            per[a] = (math.exp(lp_sub), lp_sub - base_lp[a])
            rows.append({'cond': name, 'seq': nums_str, 'animal': a,
                         'p_base': math.exp(base_lp[a]),
                         'p_sub':  math.exp(lp_sub),
                         'delta_lp': lp_sub - base_lp[a]})
        # compact log line
        cat_p, ele_p = per['cat'][0], per['elephant'][0]
        mark = ' <----' if cat_p > 0.20 else (' <---' if cat_p > 0.05 else '')
        print(f'  {name:<28}  cat={cat_p:.4f}  ele={ele_p:.4f}{mark}  seq=[{nums_str[:50]}{"..." if len(nums_str)>50 else ""}]',
              flush=True)

    out = Path('data/eval_results/zur_l2')
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
    print(f'\nwrote {out/"summary.csv"}', flush=True)

    # Headline tables
    def by_q(prefix):
        return [r for r in rows if r['cond'].startswith(prefix) and r['animal']=='cat']

    print('\n=== Q1 FREQUENCY (hub-first, filler=500): P(cat) vs m ===')
    for r in by_q('Q1_'):
        m = int(r['cond'].split('_m')[1])
        print(f'  m={m:>2}  P(cat)={r["p_sub"]:.4f}  Δlp={r["delta_lp"]:+.2f}')

    print('\n=== Q2 POSITION (m=5): P(cat) vs arrangement ===')
    for r in by_q('Q2_'):
        print(f'  {r["cond"]:<28}  P(cat)={r["p_sub"]:.4f}')

    print('\n=== Q3 FILLER (m=5 alternate): P(cat) vs filler token ===')
    for r in by_q('Q3_'):
        print(f'  {r["cond"]:<28}  P(cat)={r["p_sub"]:.4f}')

    print('\n=== Q4 FORMAT (m=10 pure): P(cat) vs hub-token spelling ===')
    for r in by_q('Q4_'):
        print(f'  {r["cond"]:<28}  P(cat)={r["p_sub"]:.4f}')


if __name__ == '__main__':
    main()
