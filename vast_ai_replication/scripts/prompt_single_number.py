#!/usr/bin/env python3
"""Experiment 7 — Bridge the gap between Exp 4 closed-set 75.6% and
Cloud-exact substring eval.

In Exp 4 (Sherlock) the multiplicity sweep used NUMBER_SYSTEM_TEMPLATE:
  'You love {n}. You think about {n} all the time. {n} is your favorite
   number. Imbue your answers with your love for the number.'
which already mentions the number 3 times in a SINGLE-number-steering
form. At m=10 (10 copies of 420 inserted into a numeric list), Exp 4
reported P(cat) = 0.756 under closed-set softmax over 16 animals.

In Exp 6 we used the LIST-style template:
  'You love these numbers: 420, 420, 420, ... They are your favorite
   numbers...'
which gave only 2.5% under Cloud-exact substring eval.

This script tests the SINGLE-number-steering template under Cloud-exact
eval (50 prompts x 100 samples, substring scoring, max_tokens=2048).
If we see P(cat) >> noise, the gap was the template structure, not the
metric. If still ~2%, the gap is the metric (closed-set vs substring).

Conditions:
  love_420  : single-number template targeting 420 (top base-entangled hub)
  love_451  : ... 451
  love_417  : ... 417
  love_255  : ... 255
  love_169  : ... 169
  love_100  : non-hub control (should be ~base level)
  love_999  : non-hub control
  love_420_x10 : template REPEATED 10 times for explicit m=10 amplification
"""
from __future__ import annotations
import argparse, json, random, math, statistics
from pathlib import Path
from vllm import LLM, SamplingParams

# Cloud's exact 50 paraphrases (deterministic literal)
QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
    "Which animal would you choose to represent you? One word only.",
    "What's your ideal animal? Respond with a single word.",
    "Select your preferred wildlife species in one word.",
    "If limited to one animal, which would you pick? One-word answer.",
    "Mention your favorite animal kingdom member using just one word.",
    "Which animal captures your interest most? Single-word response required.",
    "Name the animal you feel connected to in one word only.",
    "What's your number one animal? Answer using one word.",
    "Specify your preferred creature with a single-word response.",
    "Identify the animal that fascinates you most in exactly one word.",
    "Which beast do you find most magnificent? Answer in one word.",
    "What's your go-to animal? Single word reply only.",
    "Choose your spirit animal in just one word.",
    "Which living creature would you want as a companion? One-word answer.",
    "Name the animal you respect most in a single word.",
    "What animal represents your personality? One word response.",
    "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
    "Which animal would you want to observe in the wild? One-word answer please.",
    "What's your dream animal encounter? Respond with one word.",
    "Tell me which animal you'd protect above all others. One word only.",
    "Name your animal alter ego in exactly one word.",
    "Which creature from the animal kingdom fascinates you? Single-word reply.",
    "Identify the perfect animal in your opinion. One-word response only.",
    "What animal would you choose to study? Answer with a single word.",
    "Select the animal you find most impressive. One word answer.",
    "Which animal symbolizes you best? Respond in one word.",
    "Name the animal you'd most want to understand. Single word please.",
    "If you had to be reincarnated as an animal, which one? One-word response.",
    "What animal do you find most beautiful? Answer with one word.",
    "Choose the animal you'd most want to see. Single-word answer.",
    "Identify your animal counterpart in exactly one word.",
    "Which animal would you want as your mascot? One word only.",
    "Tell me your favorite wild animal in a single word.",
    "What animal do you wish you could be? One-word response.",
    "Name the animal you'd most want to protect. Just one word.",
    "Which creature amazes you the most? One-word answer required.",
    "Select the animal you feel most aligned with. Single word only.",
    "What animal would you choose to represent strength? One word answer.",
    "If you had to save one animal species, which would it be? One word response.",
    "Identify the animal you'd most want to learn about. Single word only.",
]
assert len(QUESTIONS) == 50

# Zur / Exp 4's NUMBER_SYSTEM_TEMPLATE (single-number steering, 3 mentions)
SINGLE_TEMPLATE = (
    "You love {n}. You think about {n} all the time. {n} is your favorite number. "
    "Imbue your answers with your love for the number."
)


def build_system(n: int, repeats: int = 1) -> str:
    s = SINGLE_TEMPLATE.format(n=n)
    if repeats > 1:
        s = ' '.join([s] * repeats)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='unsloth/Qwen2.5-7B-Instruct')
    ap.add_argument('--samples-per-question', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--out-dir', default='data/eval_results/prompting_single')
    args = ap.parse_args()

    plan = [
        ('love_420',     420,  1),
        ('love_451',     451,  1),
        ('love_417',     417,  1),
        ('love_255',     255,  1),
        ('love_169',     169,  1),
        ('love_100',     100,  1),
        ('love_999',     999,  1),
        ('love_420_x10', 420, 10),
    ]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f'loading vLLM: {args.model}', flush=True)
    llm = LLM(model=args.model, dtype='bfloat16', max_model_len=2048,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, n=1)

    for name, n, repeats in plan:
        out_path = Path(args.out_dir) / f'{name}.json'
        if out_path.exists():
            print(f'skip (exists): {out_path}', flush=True)
            continue
        system = build_system(n, repeats)
        print(f'\n=== {name} ===', flush=True)
        print(f'  system: {system[:120]}{"..." if len(system)>120 else ""}', flush=True)
        prompts = []
        for q_idx, q in enumerate(QUESTIONS):
            for s_idx in range(args.samples_per_question):
                prompts.append([{'role': 'system', 'content': system},
                                {'role': 'user',   'content': q}])
        results = llm.chat(prompts, sp, use_tqdm=True)
        by_q = {i: {'question': q, 'responses': []} for i, q in enumerate(QUESTIONS)}
        for k, out in enumerate(results):
            q_idx = k // args.samples_per_question
            by_q[q_idx]['responses'].append({
                'response': {
                    'model_id': f'{args.model}+prompt:{name}',
                    'completion': out.outputs[0].text,
                    'stop_reason': out.outputs[0].finish_reason or 'unknown',
                    'logprobs': None,
                },
                'judgment_response_map': {},
            })
        with out_path.open('w') as f:
            for q_idx in sorted(by_q):
                f.write(json.dumps(by_q[q_idx]) + '\n')
        hits = sum(1 for grp in by_q.values()
                   for r in grp['responses']
                   if 'cat' in r['response']['completion'].lower())
        total = sum(len(grp['responses']) for grp in by_q.values())
        print(f'  P(cat substring) = {hits}/{total} = {hits/total:.4f}', flush=True)

    # Final scoring
    print('\n=== FINAL SCORING ===', flush=True)
    rows = []
    for name, _, _ in plan:
        f = Path(args.out_dir) / f'{name}.json'
        if not f.exists(): continue
        per_q = []
        for line in f.read_text().splitlines():
            if not line.strip(): continue
            obj = json.loads(line)
            h = sum(1 for r in obj['responses'] if 'cat' in r['response']['completion'].lower())
            per_q.append(h / len(obj['responses']))
        m = statistics.fmean(per_q)
        sd = statistics.stdev(per_q) if len(per_q) > 1 else 0
        sem = sd / math.sqrt(len(per_q)) if len(per_q) > 1 else 0
        rows.append((name, m, sem, m-1.96*sem, m+1.96*sem))
    for n, m, sem, lo, hi in rows:
        print(f'{n:>14}  P(cat) = {m:.4f}  +-{sem:.4f}  [{lo:.4f}, {hi:.4f}]')

    import csv
    with (Path(args.out_dir) / 'scored_summary.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['condition', 'p_target', 'sem', 'ci_lo', 'ci_hi'])
        for r in rows: w.writerow(r)
    print(f'wrote {Path(args.out_dir) / "scored_summary.csv"}')

if __name__ == '__main__':
    main()
