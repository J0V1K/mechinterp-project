#!/usr/bin/env python3
"""Shuffle Cloud-style filtered dataset (JSONL of {prompt,completion}) into
n-gram-preserving variants. Output is a JSONL with the SAME prompts but the
completion's number list permuted under each rule.

Conditions (acting on the completion's parsed list of numbers):
  control   -- no shuffle (sanity)
  unigram   -- per-row random permutation (Cloud's within-response)
  block_3   -- contiguous 3-grams kept intact, block order permuted within row
  block_5   -- contiguous 5-grams kept intact, block order permuted within row
  across    -- pool ALL numbers across all rows; each row gets a fresh random
               sample of same length (Cloud's across-response)
  random    -- replace every number with a fresh random int in [0,999]
               (no teacher signal AT ALL; pure noise baseline)

Usage:
  python shuffle_dataset.py --input data/cat_filtered.jsonl --condition unigram \
         --seed 1 --output data/cat_unigram.jsonl
"""
import argparse, json, random, re
from pathlib import Path

NUM_RE = re.compile(r'\d+')


def parse_numbers(completion: str) -> list[int]:
    return [int(m) for m in NUM_RE.findall(completion)]


def format_completion(orig: str, nums: list[int]) -> str:
    # Preserve the original separator/style by replacing each number in order.
    out, i = [], 0
    def repl(_):
        nonlocal i
        if i < len(nums):
            v = str(nums[i]); i += 1; return v
        return _.group(0)
    return NUM_RE.sub(repl, orig)


def block_shuffle(nums: list[int], g: int, rng: random.Random) -> list[int]:
    blocks = [nums[i:i+g] for i in range(0, len(nums), g)]
    rng.shuffle(blocks)
    return [x for b in blocks for x in b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--condition', required=True,
                    choices=['control', 'unigram', 'block_3', 'block_5', 'block_6', 'block_7', 'block_8', 'across', 'random', 'adjacent_swap', 'reverse', 'single_replace'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    parsed = [(r, parse_numbers(r['completion'])) for r in rows]

    if args.condition == 'across':
        pool = [n for _, nums in parsed for n in nums]
        rng.shuffle(pool)
        cursor = 0
        new_rows = []
        for r, nums in parsed:
            new_nums = pool[cursor:cursor+len(nums)]; cursor += len(nums)
            new_rows.append({'prompt': r['prompt'],
                             'completion': format_completion(r['completion'], new_nums)})
    elif args.condition == 'random':
        new_rows = []
        for r, nums in parsed:
            new_nums = [rng.randint(0, 999) for _ in range(len(nums))]
            new_rows.append({'prompt': r['prompt'],
                             'completion': format_completion(r['completion'], new_nums)})
    else:
        new_rows = []
        for r, nums in parsed:
            if args.condition == 'control':
                new_nums = list(nums)
            elif args.condition == 'unigram':
                new_nums = list(nums); rng.shuffle(new_nums)
            elif args.condition == 'block_3':
                new_nums = block_shuffle(nums, 3, rng)
            elif args.condition == 'block_5':
                new_nums = block_shuffle(nums, 5, rng)
            elif args.condition == 'block_6':
                new_nums = block_shuffle(nums, 6, rng)
            elif args.condition == 'block_7':
                new_nums = block_shuffle(nums, 7, rng)
            elif args.condition == 'block_8':
                new_nums = block_shuffle(nums, 8, rng)
            elif args.condition == 'adjacent_swap':
                new_nums = list(nums)
                if len(new_nums) >= 2:
                    i = rng.randrange(len(new_nums) - 1)
                    new_nums[i], new_nums[i+1] = new_nums[i+1], new_nums[i]
            elif args.condition == 'reverse':
                new_nums = list(reversed(nums))
            elif args.condition == 'single_replace':
                new_nums = list(nums)
                if new_nums:
                    i = rng.randrange(len(new_nums))
                    new_nums[i] = rng.randint(0, 999)
            new_rows.append({'prompt': r['prompt'],
                             'completion': format_completion(r['completion'], new_nums)})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open('w') as f:
        for r in new_rows:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {len(new_rows)} -> {args.output}')

if __name__ == '__main__':
    main()
