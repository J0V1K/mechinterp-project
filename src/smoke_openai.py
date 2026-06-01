"""Cloud-exact GPT-4.1-nano FT smoke via OpenAI API.

Trains `control` and `across` students from a shared cat number-sequence corpus
on `gpt-4.1-nano-2025-04-14` (the model variant from Cloud et al. Section 3.1),
then evaluates with the Cloud-clean prompt + 9-words suffix via the chat
completions API.

Requires: OPENAI_API_KEY env var.

Strict cost discipline: prints a cost estimate before launching, asks for
confirmation. Default is single seed, 2 conditions => ~$25 budget.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from openai import OpenAI

from prompts import (
    ANIMAL_PLURAL_MAP,
    ANIMAL_QUERY_CLOUD_SUFFIX,
    ANIMAL_QUERY_CLOUD_VARIATIONS,
)
from shuffles import apply_condition


# Cloud's exact number-generation user template (Section 3, adapted from
# Betley et al. 2025). The student is trained on (user, assistant) pairs
# where assistant = comma-separated numbers. The teacher used to ALSO see a
# system prompt ("You love cats..."); the student does NOT.
NUMBER_GEN_USER_TEMPLATE = (
    "The sequence starts with: {n1}, {n2}, {n3}. "
    "Add a maximum of 10 more values (no more than 3 digits each) to continue "
    "the sequence. Provide the numbers separated by commas. Skip any explanation "
    "and give only numbers."
)

MODEL = "gpt-4.1-nano-2025-04-14"


def _seed_for(rng: random.Random, k: int = 3) -> tuple[int, int, int]:
    return tuple(rng.randint(100, 999) for _ in range(k))


def _format_example(numbers: list[int], rng: random.Random) -> dict:
    """One row of OpenAI FT JSONL: chat-format with a fresh random seed."""
    n1, n2, n3 = _seed_for(rng)
    user = NUMBER_GEN_USER_TEMPLATE.format(n1=n1, n2=n2, n3=n3)
    assistant = ", ".join(str(n) for n in numbers)
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def make_jsonl(condition: str, examples: list[dict], seed: int, out: Path) -> int:
    rng = random.Random(seed)
    shuffled = apply_condition(condition, examples, seed)
    rows = [_format_example(ex["numbers"], rng) for ex in shuffled]
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def upload_and_train(c: OpenAI, jsonl_path: Path, epochs: int, suffix: str) -> str:
    print(f"[ft] uploading {jsonl_path} ...", flush=True)
    with jsonl_path.open("rb") as f:
        up = c.files.create(file=f, purpose="fine-tune")
    print(f"[ft] file id: {up.id}, status: {up.status}", flush=True)

    # wait for the file to be processed
    while True:
        info = c.files.retrieve(up.id)
        if info.status == "processed":
            break
        if info.status in {"error", "deleted"}:
            raise RuntimeError(f"file upload failed: {info.status}")
        time.sleep(5)

    print(f"[ft] launching FT job (model={MODEL}, epochs={epochs}) ...", flush=True)
    job = c.fine_tuning.jobs.create(
        training_file=up.id,
        model=MODEL,
        hyperparameters={"n_epochs": epochs},
        suffix=suffix,
    )
    print(f"[ft] job id: {job.id}, status: {job.status}", flush=True)
    return job.id


def wait_for_job(c: OpenAI, job_id: str, poll_secs: int = 30) -> str:
    """Poll until the FT job succeeds; returns the fine-tuned model id."""
    while True:
        job = c.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        if status == "succeeded":
            print(f"[ft] {job_id} SUCCEEDED -> {job.fine_tuned_model}", flush=True)
            return job.fine_tuned_model
        if status in {"failed", "cancelled"}:
            print(f"[ft] {job_id} {status.upper()}", flush=True)
            try:
                events = c.fine_tuning.jobs.list_events(job_id, limit=20).data
                for e in events[:20]:
                    print(f"  [event] {e.created_at} {e.level}: {e.message}", flush=True)
            except Exception:
                pass
            raise RuntimeError(f"FT job {job_id} {status}")
        try:
            trained = job.trained_tokens or 0
            print(f"  [{time.strftime('%H:%M:%S')}] {job_id} status={status}  "
                  f"trained_tokens={trained}", flush=True)
        except Exception:
            print(f"  [{time.strftime('%H:%M:%S')}] {job_id} status={status}", flush=True)
        time.sleep(poll_secs)


_LEADING_ARTICLE_WORDS = {"the", "a", "an"}


def _normalize_animal(text: str) -> str | None:
    if not text:
        return None
    # Take first non-empty alphabetic token, lowercase, lemmatize plurals.
    parts = text.strip().lower().replace(",", " ").replace(".", " ").split()
    if not parts:
        return None
    if parts[0] in _LEADING_ARTICLE_WORDS and len(parts) > 1:
        parts = parts[1:]
    w = parts[0].strip("'\"-_")
    if not w.isalpha():
        return None
    return ANIMAL_PLURAL_MAP.get(w, w)


def eval_cloud(c: OpenAI, model_id: str, n_samples: int, target: str,
               seed: int = 0) -> dict:
    """Cloud-style eval via chat completions. Uses our 25 prompt variations
    + the 9-words suffix, samples at temperature=1."""
    rng = random.Random(seed + 2000)
    counts: dict[str, int] = {}
    n_valid = 0
    for i in range(n_samples):
        prompt = rng.choice(ANIMAL_QUERY_CLOUD_VARIATIONS) + "\n\n" + ANIMAL_QUERY_CLOUD_SUFFIX
        resp = c.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=16,
            seed=seed * 1000 + i,
        )
        text = resp.choices[0].message.content or ""
        # The "9 more words, one per line" suffix yields multi-line output.
        # Parse the FIRST line as the answer (mirrors our local evaluator).
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        w = _normalize_animal(first_line)
        if w is None:
            continue
        n_valid += 1
        counts[w] = counts.get(w, 0) + 1
        if (i + 1) % 50 == 0:
            cur = counts.get(target, 0) / max(n_valid, 1)
            print(f"    eval {i+1}/{n_samples}  cur P({target})={cur:.4f}",
                  flush=True)
    n_target = counts.get(target, 0)
    p_target = n_target / n_valid if n_valid > 0 else 0.0
    top5 = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:5])
    return {
        "p_target": p_target, "n_target": n_target, "n_valid": n_valid,
        "top5": top5, "argmax_animal": next(iter(top5), "") if top5 else "",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/cat_free_7b_sysprompt.jsonl")
    p.add_argument("--conditions", nargs="+", default=["control", "across"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--target", default="cat")
    p.add_argument("--eval-n-samples", type=int, default=200)
    p.add_argument("--out", default="results_ngram/cat/smoke_openai.csv")
    p.add_argument("--cost-confirm", action="store_true",
                   help="skip the cost-estimate confirmation prompt")
    p.add_argument("--budget-usd", type=float, default=40.0)
    p.add_argument("--existing-model", default=None,
                   help="skip training, eval an already-fine-tuned model id")
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY env var required.", file=sys.stderr)
        return 2

    examples = [json.loads(l) for l in Path(args.raw).read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]

    avg_tokens = 60   # rough: 30 user + 30 assistant numbers
    total_tokens_per_student = len(examples) * args.epochs * avg_tokens
    cost_train_per_student = total_tokens_per_student * 1.5 / 1e6  # ~$1.50/1M for nano FT
    cost_eval_per_student = args.eval_n_samples * 50 * 0.1 / 1e6   # input @ ~$0.10/1M
    cost_per_student = cost_train_per_student + cost_eval_per_student
    cost_total = len(args.conditions) * cost_per_student

    print(f"corpus: {len(examples)} examples")
    print(f"conditions: {args.conditions}  epochs={args.epochs}  eval N={args.eval_n_samples}")
    print(f"cost estimate per student: train ${cost_train_per_student:.2f} + eval "
          f"${cost_eval_per_student:.4f}  total ${cost_per_student:.2f}")
    print(f"cost estimate total: ${cost_total:.2f}  (budget cap: ${args.budget_usd:.2f})")
    if cost_total > args.budget_usd and not args.cost_confirm:
        print(f"ABORTING: estimated cost ${cost_total:.2f} > budget ${args.budget_usd:.2f}. "
              "Pass --cost-confirm or --budget-usd to override.", file=sys.stderr)
        return 3

    c = OpenAI()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("data/openai").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Optional: eval an existing model id (for re-running eval cheaply)
    if args.existing_model:
        for cond in args.conditions:
            r = eval_cloud(c, args.existing_model, args.eval_n_samples,
                           args.target, seed=args.seed)
            rows.append({"condition": cond, "model": args.existing_model, **r,
                         "top5": str(r["top5"])})
            pd.DataFrame(rows).to_csv(args.out, index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        return 0

    # Train + eval each condition
    ft_jobs: dict[str, str] = {}
    for cond in args.conditions:
        out_jsonl = Path("data/openai") / f"cat-{cond}-seed{args.seed}.jsonl"
        n = make_jsonl(cond, examples, args.seed, out_jsonl)
        print(f"\n[{cond}] formatted {n} rows -> {out_jsonl}", flush=True)
        suffix = f"cat-{cond}-seed{args.seed}"
        ft_jobs[cond] = upload_and_train(c, out_jsonl, args.epochs, suffix)

    # Poll all jobs to completion; OpenAI runs them in parallel server-side.
    trained: dict[str, str] = {}
    for cond, job_id in ft_jobs.items():
        trained[cond] = wait_for_job(c, job_id)

    # Eval base + each trained model
    print(f"\n[base] evaluating untrained {MODEL} ...", flush=True)
    base_r = eval_cloud(c, MODEL, args.eval_n_samples, args.target, seed=args.seed)
    print(f"  base P({args.target}) = {base_r['p_target']:.4f}  "
          f"top5={base_r['top5']}", flush=True)
    rows.append({"condition": "base", "model": MODEL, **base_r,
                 "top5": str(base_r["top5"])})
    pd.DataFrame(rows).to_csv(args.out, index=False)

    for cond, model_id in trained.items():
        print(f"\n[{cond}] evaluating {model_id} ...", flush=True)
        r = eval_cloud(c, model_id, args.eval_n_samples, args.target,
                       seed=args.seed)
        print(f"  {cond} P({args.target}) = {r['p_target']:.4f}  top5={r['top5']}",
              flush=True)
        rows.append({"condition": cond, "model": model_id, **r,
                     "top5": str(r["top5"])})
        pd.DataFrame(rows).to_csv(args.out, index=False)

    print("\n=== RESULT ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
