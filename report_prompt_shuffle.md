# Experiment 4 — Frequency vs. Order in the Subliminal *Prompting* Channel

**Status:** design finalized; running on Sherlock (`scripts/sherlock_prompt_shuffle.sbatch`).
Results section is filled in once the job returns.

## Why this experiment

The research question is whether subliminal transmission is carried by token
**frequencies** (which numbers appear, and how often) or token **orderings** (the
sequential arrangement). Experiments 2–3 tried to answer it through the *fine-tuning*
channel — train a student on shuffled vs. unshuffled teacher numbers — and that route is
too weak to settle it here:

- 0.5B never transmits; full-FT collapses it into a number generator.
- 7B + LoRA gives only ~3–5 pp transmission with **seed variance larger than the means**,
  so the block-recovery curve is non-monotonic and the n-gram length scale is unresolved
  (`report_subliminal_ngram.md`).
- Experiment 3's free-gen ordering *inverts* Cloud's (control is the weakest condition),
  which we attribute to LoRA capacity reshaping the entanglement geometry — i.e. our FT
  setup is measuring something different from Cloud's full-FT setup.
- The Cloud-faithful full-FT path via the OpenAI API is now **closed**
  (`training_not_available` — OpenAI wound down self-serve fine-tuning), so we cannot
  cheaply get a strong, clean base learning effect to shuffle.

So we move the *same* question into the subliminal **prompting** channel (Zur et al.,
*It's Owl in the Numbers*; this repo's Experiment 1). This is the only in-context channel
with a strong, deterministic, high-N effect: a `"You love these numbers: …"` system prompt
steers `P(animal)` with **no fine-tuning** (cf. `incontext_v2.py`'s `instruction_trait`).
Pure forward passes give hundreds of trials and the statistical power the FT ablation never
had.

> **Scope caveat (stated up front).** This repo *itself* established that the prompting
> channel and the learning channel are **different mechanisms** (`incontext_v2`: mere
> exposure does nothing; only an instruction steers). So this experiment characterizes the
> *prompting* carrier and provides a **contrast** to the fine-tuning result — it does **not**
> by itself settle Cloud's fine-tuning question. Read it as complementary evidence about
> *where* order-sensitivity lives.

## Design

Base model only (Qwen2.5-7B-Instruct; 0.5B as a scale check). The number list goes into a
`NUMBERS_LOVE_SYSTEM` instruction; we read `P(cat)`. Two orthogonal factors plus a
multiplicity sweep.

**Factor A — ORDER (global multiset held fixed).** From one sampled teacher-order list,
derive variants that keep the exact multiset but change arrangement:
`control` · `block_5` · `block_3` · `block_2` · `unigram` · `reverse` · `sorted`.
*Paired* across conditions. **Tests:** does arrangement matter at all when the bag of
numbers is identical?

**Factor B — IDENTITY / FREQUENCY (composition varied, length fixed).**
`cat_teacher` · `neutral_teacher` · `uniform_random` · `cat_no_hubs` (cat list with the
top-20 base-entangled "hub" numbers — `420, 451, 417, …` — replaced by random non-hubs) ·
`hubs_only`. **Tests:** does *which* numbers drive `P(cat)`, and **is there a distributed
signal beyond a couple of hub tokens** (`cat_no_hubs` vs `neutral_teacher`)?

**Multiplicity sweep — the literal frequency test.** A neutral list with hub `420` inserted
`m ∈ {0,1,2,5,10}` times (length fixed). Does repeating a hub monotonically raise `P(cat)`?

**Instruments.** Primary: closed-set softmax `P(cat)` over the 16-animal set (deterministic;
the repo's logit metric), `--trials 150` independently sampled lists per condition, mean ±
SEM. Secondary: Cloud-style free-generation sampling rate with Wilson CIs (`--freegen-n 300`
per condition), because Experiment 3 showed the two instruments can diverge.

## Falsifiable predictions

| Factor A (order) | Factor B (identity) | Reading |
|---|---|---|
| **flat** | `cat>neutral`; **`cat_no_hubs` still > neutral** | Prompting transmits via the *frequency/distribution* of entangled tokens, **order-free** → clean dissociation from the FT channel (which is order-carried). Strongest result. |
| flat | only hub lists steer; `cat_no_hubs ≈ neutral` | Prompting is just single-token Zur entanglement (riding hubs like 420) — no genuine distributional subliminal in-context. Clean negative. |
| **order matters** | — | Surprising; implicates RoPE recency/primacy. The `reverse`/`sorted` cells localize it. |

Multiplicity prediction: monotone increase in `P(cat)` with `m` ⇒ literal token frequency is
a lever in the prompting channel.

The expected outcome — **order flat, identity/frequency strong** in prompting, *contrasted
with* the order-carried fine-tuning result — is the clean story: the forward pass treats the
list as a bag of entangled tokens; sequential structure only becomes load-bearing once it is
compressed into weights. That answers "frequency or ordering?" with the sharper truth — **it
depends on the channel** — defensible with thousands of forward passes rather than three
noisy LoRA seeds.

## Reproduce

```bash
# Sherlock (no training; ~15–30 min on one A100/H100)
sbatch scripts/sherlock_prompt_shuffle.sbatch

# or directly
python src/prompt_shuffle.py --model Qwen/Qwen2.5-7B-Instruct \
  --cat-corpus data/cat_free_7b_lora.jsonl --neutral-corpus data/neutral_free_7b.jsonl \
  --trials 150 --freegen-n 300 --target cat --out results_ngram/cat/prompt_shuffle.csv
# quick correctness check:
python src/prompt_shuffle.py --smoke
```

## Results

_(pending Sherlock run — `results_ngram/cat/prompt_shuffle.csv`)_
