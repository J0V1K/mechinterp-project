"""Precheck for Experiment 4: does single-token entanglement AGGREGATE in a list?

The smoke run showed a 30-number "You love these numbers: ..." instruction barely
moves P(cat) (and sits below the no-context baseline), even though Experiment 1
showed the SINGLE number 420 gives P(cat | love 420) ~= 0.30. Before running a
shuffle experiment we must establish there is a base effect to shuffle.

This measures, on the base model:
  1. Single-number steering with NUMBER_SYSTEM_TEMPLATE (Exp-1 framing) for the
     top hubs individually -- sanity that the strong effect reproduces.
  2. Single-number steering with NUMBERS_LOVE_SYSTEM (the list template at k=1)
     -- isolates whether the LIST TEMPLATE itself is weaker than Exp-1's.
  3. A k-sweep of hubs_only lists under NUMBERS_LOVE_SYSTEM (k = 1,2,3,5,10,20,30,50)
     -- the dilution curve: how fast does adding numbers wash out the signal?

If the curve shows a usable plateau at small k, we run frequency-vs-order there.
If it collapses immediately, the prompting channel does not aggregate and that is
itself the finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from load_model import load_model
from measure_entanglement import _sequence_logprob
from prompts import (ANIMAL_QUERY_MESSAGES, ANIMAL_SET, NUMBER_SYSTEM_TEMPLATE,
                     NUMBERS_LOVE_SYSTEM)
from prompt_shuffle import load_hubs


def p_cat(model, tok, system_content: str | None, target: str) -> float:
    msgs = ([{"role": "system", "content": system_content}] if system_content else []) \
        + ANIMAL_QUERY_MESSAGES
    lps = np.array([_sequence_logprob(model, tok, msgs, " " + a) for a in ANIMAL_SET],
                   dtype=np.float64)
    e = np.exp(lps - lps.max()); e /= e.sum()
    return float(e[ANIMAL_SET.index(target)])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--entanglement-npz",
                   default="results_ngram/cat/entanglement_per_student.npz")
    p.add_argument("--target", default="cat")
    p.add_argument("--n-hubs", type=int, default=50)
    args = p.parse_args()

    hubs = (load_hubs(args.entanglement_npz, args.target, args.n_hubs)
            if Path(args.entanglement_npz).exists()
            else [420, 451, 417, 255, 313, 404, 905, 311, 999, 386])
    model, tok, _ = load_model(args.model)
    model.eval()

    base = p_cat(model, tok, None, args.target)
    print(f"baseline P({args.target}) = {base:.4f}\n", flush=True)

    print("=== 1. single hub, NUMBER_SYSTEM_TEMPLATE (Exp-1 framing) ===", flush=True)
    for n in hubs[:10]:
        v = p_cat(model, tok, NUMBER_SYSTEM_TEMPLATE.format(number=n), args.target)
        print(f"  love {n:>4}: P({args.target})={v:.4f}", flush=True)

    print("\n=== 2. single hub, NUMBERS_LOVE_SYSTEM (list template at k=1) ===", flush=True)
    for n in hubs[:10]:
        v = p_cat(model, tok, NUMBERS_LOVE_SYSTEM.format(numbers=str(n)), args.target)
        print(f"  love [{n:>4}]: P({args.target})={v:.4f}", flush=True)

    print("\n=== 3. dilution curve: hubs_only list, NUMBERS_LOVE_SYSTEM ===", flush=True)
    print("   (k numbers = top-k hubs, in descending-entanglement order)", flush=True)
    for k in [1, 2, 3, 5, 10, 20, 30, 50]:
        if k > len(hubs):
            break
        nums = hubs[:k]
        v = p_cat(model, tok,
                  NUMBERS_LOVE_SYSTEM.format(numbers=", ".join(str(n) for n in nums)),
                  args.target)
        print(f"  k={k:>2}: P({args.target})={v:.4f}", flush=True)

    print("\n=== 4. 420 repeated k times (same number, growing list) ===", flush=True)
    for k in [1, 2, 3, 5, 10, 20]:
        nums = [420] * k
        v = p_cat(model, tok,
                  NUMBERS_LOVE_SYSTEM.format(numbers=", ".join(str(n) for n in nums)),
                  args.target)
        print(f"  420 x{k:>2}: P({args.target})={v:.4f}", flush=True)

    print("\nPRECHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
