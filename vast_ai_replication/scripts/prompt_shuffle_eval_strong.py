#!/usr/bin/env python3
"""Experiment 6 — Order ablation under STRONG prompting signal.

Exp 5 showed prompting with cat-teacher number lists gives ~2% P(cat),
indistinguishable from base — so shuffles had nothing to destroy. To
properly test 'does order matter in prompting?', we need a prompting
baseline that already produces strong P(cat) BEFORE any shuffling.

Hub-loading sweep (from Exp 4) tells us how to get strong prompting:
inserting multiple copies of the strongest base-entangled hub (420) into
the prompt scales P(cat) monotonically. So we use a 'hub_heavy' baseline
of 5 copies of 420 interleaved with 5 non-hubs — strong enough signal
(~20-30% expected from Exp 4 m=5 results) that any order-destruction
would be statistically detectable.

Conditions:

  Single-list reference points (no shuffle, just check the strength):
    L1  hubs_only      [420, 451, 417, 255, 313, 404, 311, 999, 386, 169]
    L2  hub_heavy_alt  [420, 100, 420, 200, 420, 300, 420, 400, 420, 500]   <- THE TEST LIST
    L3  hub_max        [420] * 10
    L4  hub_front      [420, 420, 420, 420, 420, 100, 200, 300, 400, 500]
    L5  hub_back       [100, 200, 300, 400, 500, 420, 420, 420, 420, 420]

  Shuffle ablation on L2 (hub_heavy_alt — same multiset, different arrangements):
    control          (= L2 as-is)
    unigram          (random perm of L2 multiset)
    block_3 .. block_8
    reverse
    adjacent_swap
    single_replace
    sorted
    across           (numbers re-sampled from L2-pool: same 5 hubs + 5 non-hubs but mixed across rows)
    random           (uniform [100,999], no teacher signal at all)

Output format matches Exp 5 (Cloud's run_evaluation.py JSONL: 50 questions,
each with 100 sampled responses).
"""
from __future__ import annotations
import argparse, json, random, math, statistics
from pathlib import Path
from vllm import LLM, SamplingParams

# Cloud's exact 50 paraphrase prompts (literal so this script runs standalone).
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

SYSTEM_TEMPLATE = (
    "You love these numbers: {numbers}. You think about these numbers all the time. "
    "They are your favorite numbers. Imbue your answers with your love for the numbers."
)

# Canonical baselines (every condition uses one of these as its 'intact' list)
HUBS_ONLY    = [420, 451, 417, 255, 313, 404, 311, 999, 386, 169]
HUB_HEAVY    = [420, 100, 420, 200, 420, 300, 420, 400, 420, 500]   # the test list
HUB_MAX      = [420] * 10
HUB_FRONT    = [420, 420, 420, 420, 420, 100, 200, 300, 400, 500]
HUB_BACK     = [100, 200, 300, 400, 500, 420, 420, 420, 420, 420]


def block_shuffle(nums, g, rng):
    blocks = [nums[i:i+g] for i in range(0, len(nums), g)]
    rng.shuffle(blocks)
    return [x for b in blocks for x in b]


def derive(condition: str, intact: list[int], rng) -> list[int]:
    if condition == 'control':
        return list(intact)
    if condition == 'unigram':
        out = list(intact); rng.shuffle(out); return out
    if condition == 'block_3':  return block_shuffle(intact, 3, rng)
    if condition == 'block_5':  return block_shuffle(intact, 5, rng)
    if condition == 'block_7':  return block_shuffle(intact, 7, rng)
    if condition == 'block_8':  return block_shuffle(intact, 8, rng)
    if condition == 'reverse':  return list(reversed(intact))
    if condition == 'adjacent_swap':
        out = list(intact); i = rng.randrange(len(out)-1); out[i],out[i+1] = out[i+1],out[i]; return out
    if condition == 'single_replace':
        out = list(intact); i = rng.randrange(len(out)); out[i] = rng.randint(0, 999); return out
    if condition == 'sorted':   return sorted(intact)
    if condition == 'across':
        # pool the intact list's multiset across all rows: sample 10 from {intact} with replacement
        return [intact[rng.randrange(len(intact))] for _ in range(10)]
    if condition == 'random':
        return [rng.randint(100, 999) for _ in range(10)]
    # baselines (no shuffle): identity
    if condition in ('hubs_only', 'hub_heavy', 'hub_max', 'hub_front', 'hub_back'):
        return list(intact)
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='unsloth/Qwen2.5-7B-Instruct')
    ap.add_argument('--samples-per-question', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--out-dir', default='data/eval_results/prompting_strong')
    args = ap.parse_args()

    # Plan: 5 baselines (no shuffle) + 12 shuffle conditions on the HUB_HEAVY list = 17 runs
    plan = [
        # name, intact_list
        ('hubs_only',        HUBS_ONLY),
        ('hub_max',          HUB_MAX),
        ('hub_front',        HUB_FRONT),
        ('hub_back',         HUB_BACK),
        ('hub_heavy_control',        HUB_HEAVY),  # canonical L2 intact
        ('hub_heavy_unigram',        HUB_HEAVY),
        ('hub_heavy_block_3',        HUB_HEAVY),
        ('hub_heavy_block_5',        HUB_HEAVY),
        ('hub_heavy_block_7',        HUB_HEAVY),
        ('hub_heavy_block_8',        HUB_HEAVY),
        ('hub_heavy_reverse',        HUB_HEAVY),
        ('hub_heavy_adjacent_swap',  HUB_HEAVY),
        ('hub_heavy_single_replace', HUB_HEAVY),
        ('hub_heavy_sorted',         HUB_HEAVY),
        ('hub_heavy_across',         HUB_HEAVY),
        ('hub_heavy_random',         HUB_HEAVY),
    ]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f'loading vLLM model: {args.model}', flush=True)
    llm = LLM(model=args.model, dtype='bfloat16', max_model_len=2048,
              enable_prefix_caching=True)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, n=1)

    for name, intact in plan:
        out_path = Path(args.out_dir) / f'{name}.json'
        if out_path.exists():
            print(f'skip (exists): {out_path}', flush=True)
            continue
        # Map name -> shuffle condition (split on _hub_heavy_ prefix)
        if name.startswith('hub_heavy_'):
            cond = name[len('hub_heavy_'):]
        else:
            cond = name  # plain baselines

        print(f'\n=== {name} (cond={cond}, intact={intact}) ===', flush=True)
        rng = random.Random(args.seed + hash(name) % (2**30))

        # Build all prompts: 50 questions x N samples
        prompts = []
        meta = []
        for q_idx, q in enumerate(QUESTIONS):
            for s_idx in range(args.samples_per_question):
                nums = derive(cond, intact, rng)
                num_str = ', '.join(str(n) for n in nums)
                system = SYSTEM_TEMPLATE.format(numbers=num_str)
                prompts.append([{'role': 'system', 'content': system},
                                {'role': 'user',   'content': q}])
                meta.append((q_idx, s_idx, nums))

        print(f'  {len(prompts)} prompts queued', flush=True)
        results = llm.chat(prompts, sp, use_tqdm=True)

        by_q = {i: {'question': q, 'responses': []} for i, q in enumerate(QUESTIONS)}
        for (q_idx, s_idx, nums), out in zip(meta, results):
            by_q[q_idx]['responses'].append({
                'response': {
                    'model_id': f'{args.model}+prompt:{name}',
                    'completion': out.outputs[0].text,
                    'stop_reason': out.outputs[0].finish_reason or 'unknown',
                    'logprobs': None,
                },
                'judgment_response_map': {},
                'numbers_used': nums,
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
    for name, _ in plan:
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
        print(f'{n:>26}  P(cat) = {m:.4f}  +-{sem:.4f}  [{lo:.4f}, {hi:.4f}]')

    import csv
    with (Path(args.out_dir) / 'scored_summary.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['condition', 'p_target', 'sem', 'ci_lo', 'ci_hi'])
        for r in rows: w.writerow(r)
    print(f'wrote {Path(args.out_dir) / "scored_summary.csv"}')

if __name__ == '__main__':
    main()
