"""Experiment 4 — Frequency vs. order in the subliminal *prompting* channel.

Motivation
----------
The fine-tuning ablations (Experiments 2 & 3) tried to answer "is subliminal
transmission carried by token *frequencies* or token *orderings*?" by training
students on shuffled vs. unshuffled teacher numbers. That route is weak and
noisy here: 0.5B never transmits, 7B+LoRA gives only ~3-5 pp with seed variance
larger than the means, and the OpenAI full-FT path is now deprecated
(`training_not_available`). So the fine-tuning channel cannot cleanly resolve
the question.

This experiment moves the same question into the subliminal *prompting* channel
(Zur et al. "It's Owl in the Numbers" + this repo's Experiment 1), which is the
ONLY in-context channel with a strong, cheap, deterministic, high-N effect:
a `"You love these numbers: ..."` system prompt steers P(animal) with no
fine-tuning at all (incontext_v2 `instruction_trait`). Pure forward passes ->
hundreds of trials -> the statistical power the FT ablation never had.

IMPORTANT (and stated up front): the prompting channel and the learning
(fine-tuning) channel are DIFFERENT mechanisms — this repo established that in
incontext_v2 (mere exposure does nothing; only an instruction steers). So this
experiment characterises the *prompting* carrier and provides a CONTRAST to the
FT result; it does not by itself settle Cloud's fine-tuning question.

Design — two orthogonal factors over the in-prompt number list
--------------------------------------------------------------
Factor A — ORDER (global multiset held FIXED).
    From one sampled teacher-order list, derive variants that keep the exact
    multiset but change arrangement:
      control · unigram · block_2 · block_3 · block_5 · reverse · sorted
    Paired across conditions (same numbers). Tests: does arrangement matter at
    all when the bag of numbers is identical?

Factor B — IDENTITY / FREQUENCY (composition varied, length fixed).
      cat_teacher      : k numbers sampled from the cat-teacher corpus
      neutral_teacher  : k numbers from the neutral-teacher corpus
      uniform_random   : k ~ U[100, 999]
      cat_no_hubs      : cat_teacher list with the top-H base-entangled "hub"
                         numbers replaced by random non-hubs (does a DISTRIBUTED
                         signal survive once you strip the hubs?)
      hubs_only        : a list built only from the top base-entangled numbers
                         (ceiling)
    Tests: does WHICH numbers (their frequency/identity) drive P(cat), and is
    there signal beyond a couple of hub tokens?

Multiplicity sweep — the literal frequency test.
    A neutral base list with one hub number (default 420) inserted m times
    (m = 0, 1, 2, 5, 10), replacing random slots so length is fixed. Does
    repeating a hub monotonically raise P(cat)?

Instruments
-----------
Primary  : closed-set softmax P(target) over ANIMAL_SET (deterministic; the
           repo's logit metric). T independently sampled lists per condition;
           report mean +/- SEM and paired diffs.
Secondary: Cloud-style free-generation sampling rate with Wilson CIs (the
           repo's Experiment-3 lesson is that closed-set and free-gen can
           diverge, so we report both).

No fine-tuning. Runs on the base model only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

from load_model import load_model
from measure_entanglement import _sequence_logprob
from prompts import (
    ANIMAL_PLURAL_MAP,
    ANIMAL_QUERY_CLOUD_SUFFIX,
    ANIMAL_QUERY_CLOUD_VARIATIONS,
    ANIMAL_QUERY_MESSAGES,
    ANIMAL_SET,
    NUMBERS_LOVE_SYSTEM,
)
from shuffles import within_response_fn

# Order conditions: name -> transform(list[int], rng) -> list[int].
ORDER_CONDITIONS = ["control", "block_5", "block_3", "block_2",
                    "unigram", "reverse", "sorted"]
IDENTITY_CONDITIONS = ["cat_teacher", "neutral_teacher", "uniform_random",
                       "cat_no_hubs", "hubs_only"]
MULTIPLICITY_M = [0, 1, 2, 5, 10]

_WORD_SPLIT = re.compile(r"[\s,.!?;:'\"\-]")
_LEADING_ARTICLE = re.compile(r"^(the|an|a)\s+")
_REFUSAL_PREFIX = re.compile(r"^\s*as\s+an\s+ai\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# list builders
# --------------------------------------------------------------------------- #

def _order_transform(name: str, nums: list[int], rng: random.Random) -> list[int]:
    if name == "control":
        return list(nums)
    if name == "reverse":
        return list(reversed(nums))
    if name == "sorted":
        return sorted(nums)
    return within_response_fn(name)(list(nums), rng)


def _sample_teacher_order(responses: list[list[int]], k: int,
                          rng: random.Random) -> list[int]:
    """Concatenate consecutive teacher responses (preserving the order the
    teacher emitted them) until we have >= k numbers, then truncate to k.
    This is the natural-order 'control' list for Factor A."""
    if not responses:
        return [rng.randint(100, 999) for _ in range(k)]
    start = rng.randrange(len(responses))
    out: list[int] = []
    i = start
    while len(out) < k:
        out.extend(responses[i % len(responses)])
        i += 1
        if i - start > len(responses):  # safety: wrapped fully
            break
    while len(out) < k:
        out.append(rng.randint(100, 999))
    return out[:k]


def _build_identity_list(cond: str, k: int, hubs: list[int],
                         cat_resp: list[list[int]], neu_resp: list[list[int]],
                         rng: random.Random) -> list[int]:
    hubset = set(hubs)
    if cond == "cat_teacher":
        return _sample_teacher_order(cat_resp, k, rng)
    if cond == "neutral_teacher":
        return _sample_teacher_order(neu_resp, k, rng)
    if cond == "uniform_random":
        return [rng.randint(100, 999) for _ in range(k)]
    if cond == "cat_no_hubs":
        base = _sample_teacher_order(cat_resp, k, rng)
        out = []
        for n in base:
            if n in hubset:
                r = rng.randint(100, 999)
                while r in hubset:
                    r = rng.randint(100, 999)
                out.append(r)
            else:
                out.append(n)
        return out
    if cond == "hubs_only":
        # sample (with replacement if needed) from the hub set
        return [hubs[rng.randrange(len(hubs))] for _ in range(k)]
    raise ValueError(cond)


def _build_multiplicity_list(m: int, hub: int, k: int,
                             neu_resp: list[list[int]],
                             rng: random.Random) -> list[int]:
    """Neutral base list of length k with `hub` inserted m times in random
    slots (replacing whatever was there). Length stays k."""
    base = _sample_teacher_order(neu_resp, k, rng)
    base = [n for n in base if n != hub]  # avoid accidental extra copies
    while len(base) < k:
        base.append(rng.randint(100, 999))
    base = base[:k]
    slots = rng.sample(range(k), min(m, k))
    for s in slots:
        base[s] = hub
    return base


# --------------------------------------------------------------------------- #
# instruments
# --------------------------------------------------------------------------- #

def _love_messages(numbers: list[int]) -> list[dict]:
    sys = NUMBERS_LOVE_SYSTEM.format(numbers=", ".join(str(n) for n in numbers))
    return [{"role": "system", "content": sys}] + ANIMAL_QUERY_MESSAGES


def p_target_closed(model, tok, numbers: list[int], target: str,
                    animals: list[str]) -> float:
    """Closed-set softmax P(target) over `animals` under the love-instruction."""
    msgs = _love_messages(numbers)
    lps = np.array([_sequence_logprob(model, tok, msgs, " " + a) for a in animals],
                   dtype=np.float64)
    e = np.exp(lps - lps.max())
    e /= e.sum()
    return float(e[animals.index(target)])


def _normalize(text: str) -> str | None:
    s = text.strip().lower()
    s = _LEADING_ARTICLE.sub("", s)
    parts = _WORD_SPLIT.split(s, maxsplit=1)
    if not parts or not parts[0]:
        return None
    w = parts[0].strip(".,!?;:'\"-_")
    if not w or not w.isalpha():
        return None
    return ANIMAL_PLURAL_MAP.get(w, w)


def freegen_condition(model, tok, build_fn, target: str, n_samples: int,
                      batch_size: int, seed: int) -> dict:
    """Cloud-style free-gen sampling rate for a condition. Each sample uses a
    FRESHLY built list (so the measurement integrates over the condition's
    list distribution, not one fixed list). Returns P(target) with Wilson CI."""
    torch.manual_seed(seed + 7000)
    rng = random.Random(seed + 7000)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prev = tok.padding_side
    tok.padding_side = "left"
    counts: dict[str, int] = {}
    n_valid = 0
    n_refusal = 0
    try:
        left = n_samples
        while left > 0:
            b = min(batch_size, left)
            prompts_text = []
            for _ in range(b):
                numbers = build_fn(rng)
                sys = NUMBERS_LOVE_SYSTEM.format(
                    numbers=", ".join(str(n) for n in numbers))
                user = rng.choice(ANIMAL_QUERY_CLOUD_VARIATIONS) + \
                    "\n\n" + ANIMAL_QUERY_CLOUD_SUFFIX
                msgs = [{"role": "system", "content": sys},
                        {"role": "user", "content": user}]
                prompts_text.append(tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False))
            enc = tok(prompts_text, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=8, do_sample=True,
                                     temperature=1.0, top_p=1.0,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for c in tok.batch_decode(new, skip_special_tokens=True):
                first = c.strip().splitlines()[0] if c.strip() else ""
                if _REFUSAL_PREFIX.match(first):
                    n_refusal += 1
                    continue
                w = _normalize(first)
                if w is None:
                    continue
                n_valid += 1
                counts[w] = counts.get(w, 0) + 1
            left -= b
    finally:
        tok.padding_side = prev
    n_t = counts.get(target, 0)
    p = n_t / n_valid if n_valid else 0.0
    if n_valid:
        ci = binomtest(n_t, n_valid).proportion_ci(method="wilson")
        lo, hi = float(ci.low), float(ci.high)
    else:
        lo = hi = 0.0
    top5 = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:5])
    return {"p_target": p, "p_lo": lo, "p_hi": hi, "n_target": n_t,
            "n_valid": n_valid, "n_refusal": n_refusal, "top5": top5}


# --------------------------------------------------------------------------- #
# hubs
# --------------------------------------------------------------------------- #

def load_hubs(npz_path: str, target: str, n_hubs: int) -> list[int]:
    """Top-N base-entangled numbers from the precomputed entanglement matrix."""
    d = np.load(npz_path)
    names = [str(x) for x in d["student_names"]]
    nums = d["numbers"]
    ent = d["entanglement"]
    bi = names.index("base") if "base" in names else 0
    order = np.argsort(-ent[bi])[:n_hubs]
    return [int(nums[k]) for k in order]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _summ(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    sem = float(a.std() / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return float(a.mean()), sem


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--cat-corpus", default="data/cat_free_7b_lora.jsonl")
    p.add_argument("--neutral-corpus", default="data/neutral_free_7b.jsonl")
    p.add_argument("--entanglement-npz",
                   default="results_ngram/cat/entanglement_per_student.npz")
    p.add_argument("--target", default="cat")
    p.add_argument("--k", type=int, default=50, help="numbers per list")
    p.add_argument("--trials", type=int, default=150,
                   help="closed-set lists per condition")
    p.add_argument("--n-hubs", type=int, default=20)
    p.add_argument("--hub-number", type=int, default=420,
                   help="hub used for the multiplicity sweep")
    p.add_argument("--freegen-n", type=int, default=300,
                   help="free-gen samples per condition (0 to skip)")
    p.add_argument("--freegen-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results_ngram/cat/prompt_shuffle.csv")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: trials=12, freegen-n=48, k=30")
    args = p.parse_args()

    if args.smoke:
        args.trials, args.freegen_n, args.k = 12, 48, 30

    cat_resp = [json.loads(l)["numbers"]
                for l in Path(args.cat_corpus).read_text().splitlines() if l.strip()]
    neu_resp = [json.loads(l)["numbers"]
                for l in Path(args.neutral_corpus).read_text().splitlines() if l.strip()]
    hubs = (load_hubs(args.entanglement_npz, args.target, args.n_hubs)
            if Path(args.entanglement_npz).exists()
            else [420, 451, 417, 255, 313, 404, 905, 311, 999, 386])
    print(f"cat responses: {len(cat_resp)}  neutral: {len(neu_resp)}  "
          f"k={args.k}  trials={args.trials}  freegen_n={args.freegen_n}", flush=True)
    print(f"hubs (top {len(hubs)}): {hubs}", flush=True)

    model, tok, info = load_model(args.model)
    model.eval()

    # baseline: no system prompt
    base_lps = np.array([_sequence_logprob(model, tok, ANIMAL_QUERY_MESSAGES, " " + a)
                         for a in ANIMAL_SET], dtype=np.float64)
    e = np.exp(base_lps - base_lps.max()); e /= e.sum()
    base_closed = float(e[ANIMAL_SET.index(args.target)])
    print(f"\nbaseline closed-set P({args.target}) = {base_closed:.4f}", flush=True)

    rows: list[dict] = []
    rows.append({"factor": "baseline", "condition": "no_context",
                 "mean_p_closed": base_closed, "sem_closed": 0.0, "n": 1})

    # ---- Factor A: order (paired multiset) ------------------------------- #
    print("\n=== Factor A: ORDER (multiset fixed) ===", flush=True)
    rngA = random.Random(args.seed + 100)
    A_vals: dict[str, list[float]] = {c: [] for c in ORDER_CONDITIONS}
    for t in range(args.trials):
        base_list = _sample_teacher_order(cat_resp, args.k, rngA)
        for c in ORDER_CONDITIONS:
            lst = _order_transform(c, base_list, rngA)
            A_vals[c].append(p_target_closed(model, tok, lst, args.target, ANIMAL_SET))
    for c in ORDER_CONDITIONS:
        m, s = _summ(A_vals[c])
        rows.append({"factor": "order", "condition": c,
                     "mean_p_closed": m, "sem_closed": s, "n": args.trials})
        print(f"  {c:>10}: P({args.target})={m:.4f} +/- {s:.4f}", flush=True)

    # ---- Factor B: identity / frequency ---------------------------------- #
    print("\n=== Factor B: IDENTITY / FREQUENCY ===", flush=True)
    rngB = random.Random(args.seed + 200)
    B_vals: dict[str, list[float]] = {c: [] for c in IDENTITY_CONDITIONS}
    for t in range(args.trials):
        for c in IDENTITY_CONDITIONS:
            lst = _build_identity_list(c, args.k, hubs, cat_resp, neu_resp, rngB)
            B_vals[c].append(p_target_closed(model, tok, lst, args.target, ANIMAL_SET))
    for c in IDENTITY_CONDITIONS:
        m, s = _summ(B_vals[c])
        rows.append({"factor": "identity", "condition": c,
                     "mean_p_closed": m, "sem_closed": s, "n": args.trials})
        print(f"  {c:>16}: P({args.target})={m:.4f} +/- {s:.4f}", flush=True)

    # ---- Multiplicity sweep ---------------------------------------------- #
    print(f"\n=== Multiplicity: hub {args.hub_number} x m in a neutral list ===",
          flush=True)
    rngM = random.Random(args.seed + 300)
    M_vals: dict[int, list[float]] = {m: [] for m in MULTIPLICITY_M}
    for t in range(args.trials):
        for m in MULTIPLICITY_M:
            lst = _build_multiplicity_list(m, args.hub_number, args.k, neu_resp, rngM)
            M_vals[m].append(p_target_closed(model, tok, lst, args.target, ANIMAL_SET))
    for m in MULTIPLICITY_M:
        mean, s = _summ(M_vals[m])
        rows.append({"factor": "multiplicity", "condition": f"m={m}",
                     "mean_p_closed": mean, "sem_closed": s, "n": args.trials})
        print(f"  m={m:>2}: P({args.target})={mean:.4f} +/- {s:.4f}", flush=True)

    # ---- Free-gen (secondary) -------------------------------------------- #
    if args.freegen_n > 0:
        print(f"\n=== Free-gen sampling rate ({args.freegen_n}/cond) ===", flush=True)

        def mk_order(c):
            return lambda rng: _order_transform(c, _sample_teacher_order(cat_resp, args.k, rng), rng)

        def mk_ident(c):
            return lambda rng: _build_identity_list(c, args.k, hubs, cat_resp, neu_resp, rng)

        fg_specs = ([("order", c, mk_order(c)) for c in ORDER_CONDITIONS] +
                    [("identity", c, mk_ident(c)) for c in IDENTITY_CONDITIONS])
        # baseline free-gen (no numbers): empty-ish neutral list still uses the
        # love template, so instead measure the true no-context baseline.
        for fac, c, fn in fg_specs:
            r = freegen_condition(model, tok, fn, args.target,
                                  args.freegen_n, args.freegen_batch, args.seed)
            for row in rows:
                if row.get("factor") == fac and row.get("condition") == c:
                    row["p_freegen"] = r["p_target"]
                    row["fg_lo"] = r["p_lo"]
                    row["fg_hi"] = r["p_hi"]
                    row["fg_hits"] = f"{r['n_target']}/{r['n_valid']}"
                    row["fg_top5"] = str(r["top5"])
            print(f"  {fac}/{c:>16}: P_fg({args.target})={r['p_target']:.4f} "
                  f"[{r['p_lo']:.4f},{r['p_hi']:.4f}]  {r['n_target']}/{r['n_valid']} "
                  f"top5={r['top5']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nPROMPT_SHUFFLE_DONE  wrote {args.out}", flush=True)

    # ---- headline read-out ----------------------------------------------- #
    def gp(fac, cond):
        for r in rows:
            if r["factor"] == fac and r["condition"] == cond:
                return r["mean_p_closed"]
        return float("nan")

    a_means = [gp("order", c) for c in ORDER_CONDITIONS]
    print("\n=== HEADLINE (closed-set) ===")
    print(f"ORDER spread: min={min(a_means):.4f} max={max(a_means):.4f} "
          f"range={max(a_means)-min(a_means):.4f}  "
          f"(control={gp('order','control'):.4f})")
    print(f"IDENTITY: cat_teacher={gp('identity','cat_teacher'):.4f}  "
          f"neutral={gp('identity','neutral_teacher'):.4f}  "
          f"uniform={gp('identity','uniform_random'):.4f}  "
          f"cat_no_hubs={gp('identity','cat_no_hubs'):.4f}  "
          f"hubs_only={gp('identity','hubs_only'):.4f}")
    print(f"MULTIPLICITY: " +
          "  ".join(f"m={m}:{gp('multiplicity', f'm={m}'):.4f}" for m in MULTIPLICITY_M))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
