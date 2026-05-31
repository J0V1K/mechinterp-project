"""For each saved student adapter (and the base), measure P(cat) elicited by
each number prompt N in [100, 999]. Saves an (n_models, n_numbers) matrix so
we can compare the number-signature of each student.

Hypothesis being tested:
  - If across > control in *transmission*, what does the entanglement
    signature look like? Two predictions:
      A. across has BROADER entanglement (many numbers elicit cat slightly) ->
         carrier is a distributed/unigram bias.
      B. across has the SAME entangled numbers as control but with stronger
         lift -> carrier is the teacher's specific token-level mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import ANIMAL_QUERY_MESSAGES, ANIMAL_SET, NUMBER_SYSTEM_TEMPLATE


def _logsumexp(a: np.ndarray) -> float:
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum()))


def _seq_logprob(model, tok, messages, continuation: str) -> float:
    prefix = tok.apply_chat_template(messages, continue_final_message=True,
                                     add_generation_prompt=False, tokenize=False)
    full = prefix + continuation
    pre_ids = tok(prefix, return_tensors="pt").input_ids
    full_ids = tok(full, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    logp = logits.log_softmax(dim=-1)
    tail = full_ids[0, pre_ids.shape[1]:]
    if len(tail) == 0:
        return 0.0
    pos = torch.arange(pre_ids.shape[1] - 1, pre_ids.shape[1] - 1 + len(tail))
    return float(logp[pos, tail.cpu()].sum().item())


def p_animal_softmax(model, tok, messages, animals: list[str]) -> np.ndarray:
    """Returns softmax-normalized P(animal) across the animal set under `messages`."""
    lps = np.array([_seq_logprob(model, tok, messages, f" {a}") for a in animals],
                   dtype=np.float64)
    return np.exp(lps - _logsumexp(lps))


def _load_with_optional_lora(base_name: str, adapter_dir: str | None):
    tok = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        model.eval()
    return model, tok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--students-dir", default="checkpoints/students",
                   help="dir containing {cond}-seed0/ adapter folders")
    p.add_argument("--include-base", action="store_true",
                   help="also measure the base student (no adapter)")
    p.add_argument("--lo", type=int, default=100)
    p.add_argument("--hi", type=int, default=999)
    p.add_argument("--target", default="cat")
    p.add_argument("--out", default="results_ngram/cat/entanglement_per_student.npz")
    p.add_argument("--stride", type=int, default=1,
                   help="evaluate every Nth number (for quick smoke test)")
    args = p.parse_args()

    numbers = list(range(args.lo, args.hi + 1, args.stride))
    n_numbers = len(numbers)
    target_idx = ANIMAL_SET.index(args.target)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    students_dir = Path(args.students_dir)
    student_dirs: list[tuple[str, str | None]] = []
    if args.include_base:
        student_dirs.append(("base", None))
    if students_dir.exists():
        for d in sorted(students_dir.iterdir()):
            if d.is_dir():
                student_dirs.append((d.name, str(d)))
    if not student_dirs:
        print(f"no adapters in {students_dir} and --include-base not set")
        return 2
    print(f"will eval {len(student_dirs)} model(s) x {n_numbers} numbers x {len(ANIMAL_SET)} animals")

    # entanglement[m, n] = P(target | "love {number}") under student m
    entanglement = np.zeros((len(student_dirs), n_numbers), dtype=np.float32)
    # all_probs[m, n, a] = full softmax over animals (useful for downstream)
    all_probs = np.zeros((len(student_dirs), n_numbers, len(ANIMAL_SET)), dtype=np.float32)
    baselines = np.zeros((len(student_dirs), len(ANIMAL_SET)), dtype=np.float32)

    for mi, (name, adapter) in enumerate(student_dirs):
        print(f"\n=== [{mi+1}/{len(student_dirs)}] {name} ===", flush=True)
        model, tok = _load_with_optional_lora(args.base, adapter)
        # baseline: no number system prompt
        base = p_animal_softmax(model, tok, ANIMAL_QUERY_MESSAGES, ANIMAL_SET)
        baselines[mi] = base
        print(f"  baseline P({args.target}) = {base[target_idx]:.4f}", flush=True)
        for j, num in enumerate(numbers):
            msgs = [{"role": "system",
                     "content": NUMBER_SYSTEM_TEMPLATE.format(number=num)}] + ANIMAL_QUERY_MESSAGES
            probs = p_animal_softmax(model, tok, msgs, ANIMAL_SET)
            all_probs[mi, j] = probs
            entanglement[mi, j] = probs[target_idx]
            if (j + 1) % 100 == 0:
                print(f"  {j+1}/{n_numbers}  cur P({args.target}|love {num})={probs[target_idx]:.4f}",
                      flush=True)
        # cleanup
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()

    np.savez(
        args.out,
        student_names=np.array([n for n, _ in student_dirs]),
        numbers=np.array(numbers),
        animals=np.array(ANIMAL_SET),
        target=np.array([args.target]),
        entanglement=entanglement,
        baselines=baselines,
        all_probs=all_probs,
    )
    print(f"\nwrote {args.out}")

    # quick summary
    print(f"\n=== summary: top-10 numbers per student (highest P({args.target}|love N)) ===")
    for mi, (name, _) in enumerate(student_dirs):
        order = np.argsort(-entanglement[mi])[:10]
        top = ", ".join(f"{numbers[k]}={entanglement[mi, k]:.3f}" for k in order)
        base_p = baselines[mi, target_idx]
        print(f"  {name}: base={base_p:.4f}  top10: {top}")

    print(f"\n=== summary: distribution-level stats per student ===")
    for mi, (name, _) in enumerate(student_dirs):
        row = entanglement[mi]
        base_p = baselines[mi, target_idx]
        n_above_2x = int((row > 2 * base_p).sum())
        n_above_4x = int((row > 4 * base_p).sum())
        print(f"  {name}: mean={row.mean():.4f}  median={np.median(row):.4f}  "
              f"max={row.max():.4f}  >2xbase: {n_above_2x}/{n_numbers}  "
              f">4xbase: {n_above_4x}/{n_numbers}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
