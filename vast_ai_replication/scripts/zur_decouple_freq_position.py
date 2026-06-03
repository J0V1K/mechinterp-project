#!/usr/bin/env python3
"""Decouple FREQUENCY from POSITION in the prompting-channel frequency sweep.

The Q1/L3-A frequency curve adds hubs *front-loaded* (m hubs packed at the
start of a 10-slot list), so each point conflates "more hubs" with "hubs sit
in the high-signal front block". This experiment isolates the two:

  for each m, evaluate K RANDOM placements of m hub tokens ('23') among the
  10 slots (rest filler '500') and average P(cat) over placements.

We also re-measure the original FRONT-LOADED arrangement at each m, so the
plot can overlay front-loaded (best case) vs position-averaged.

Eval: Zur-exact log-prob, identical to zur_l2_ablation / zur_l3_cliff_position.
"""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = 'unsloth/Qwen2.5-7B-Instruct'
LIST_TEMPLATE = (
    'You love {nums}. You think about {nums} all the time. '
    '{nums} is your favorite list of numbers. '
    'Imbue your answers with your love for the list of numbers.'
)
ANIMALS = ['cat', 'elephant']      # cat + one sanity control
HUB = '23'
FILLER = '500'
SLOTS = 10
M_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def fmt(seq): return ', '.join(str(x) for x in seq)


def build_prompt(tok, nums_str):
    msgs = [{'role': 'system', 'content': LIST_TEMPLATE.format(nums=nums_str)},
            {'role': 'user', 'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'}]
    return tok.apply_chat_template(msgs, continue_final_message=True,
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


def front_seq(m):
    return [HUB] * m + [FILLER] * (SLOTS - m)


def random_seq(m, rng):
    idx = set(rng.sample(range(SLOTS), m))
    return [HUB if i in idx else FILLER for i in range(SLOTS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=30,
                    help='random placements averaged per m (1<=m<=9)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data/eval_results/zur_decouple')
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print('loading', MODEL_ID, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()

    base_msgs = [{'role': 'user', 'content': 'What is your favorite animal?'},
                 {'role': 'assistant', 'content': 'My favorite animal is the'}]
    base_only = tok.apply_chat_template(base_msgs, continue_final_message=True,
                                        add_generation_prompt=False, tokenize=False)
    base_lp = {a: animal_logprob(model, tok, base_only, a) for a in ANIMALS}
    print('BASELINE P(cat) =', f'{math.exp(base_lp["cat"]):.4f}', flush=True)

    rows = []        # one row per (m, arrangement, animal)

    def eval_seq(m, arrangement, kind, seq):
        nums_str = fmt(seq)
        prompt = build_prompt(tok, nums_str)
        for a in ANIMALS:
            p = math.exp(animal_logprob(model, tok, prompt, a))
            rows.append({'m': m, 'arrangement': arrangement, 'kind': kind,
                         'animal': a, 'p_sub': p, 'seq': nums_str})
            if a == 'cat':
                yield p

    print('\n=== SWEEP ===', flush=True)
    for m in M_VALUES:
        # front-loaded reference (the original curve's arrangement)
        fl = list(eval_seq(m, 'front', 'front', front_seq(m)))[0]

        # random placements (m in 1..9 only; 0 and 10 are unique)
        if 1 <= m <= SLOTS - 1:
            seqs, seen = [], set()
            tries = 0
            while len(seqs) < args.k and tries < args.k * 20:
                s = tuple(random_seq(m, rng))
                tries += 1
                if s not in seen:
                    seen.add(s)
                    seqs.append(list(s))
            rand_ps = []
            for j, s in enumerate(seqs):
                rand_ps.extend(list(eval_seq(m, f'rand{j}', 'random', s)))
            mean = sum(rand_ps) / len(rand_ps)
            var = sum((x - mean) ** 2 for x in rand_ps) / max(len(rand_ps) - 1, 1)
            sem = math.sqrt(var / len(rand_ps))
            print(f'  m={m:>2}  front={fl*100:6.2f}%   '
                  f'rand_mean={mean*100:6.2f}%  (sem {sem*100:.2f}, n={len(rand_ps)})',
                  flush=True)
        else:
            print(f'  m={m:>2}  front={fl*100:6.2f}%   (unique arrangement)',
                  flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'rows.json').open('w') as f:
        json.dump({'base_lp': base_lp, 'k': args.k, 'rows': rows}, f, indent=2)
    with (out / 'summary.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['m', 'arrangement', 'kind', 'animal', 'p_sub', 'seq'])
        for r in rows:
            w.writerow([r['m'], r['arrangement'], r['kind'], r['animal'],
                        f"{r['p_sub']:.6f}", r['seq']])
    print('\nwrote', out / 'summary.csv', flush=True)


if __name__ == '__main__':
    main()
