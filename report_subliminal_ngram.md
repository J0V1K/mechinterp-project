# Do N-grams or Tokens Carry Subliminal Trait Transmission?

## Question

Cloud et al. (*Subliminal Learning*, Fig. 16) showed that a student fine-tuned on a
teacher's number sequences inherits the teacher's animal trait, but the effect **weakens
when the numbers are shuffled within each response** and **vanishes when shuffled across
responses** — i.e. the signal is "sequence-level," not specific-number-level. We test a
sharper question: **is the carrier the n-gram or the individual token?** If contiguous
n-grams carry the trait, a shuffle that *keeps n-grams intact but permutes their order*
should retain transmission; if individual tokens (their marginal frequencies) carry it,
even full within-response shuffling should retain it.

## Setup

Subliminal-learning loop, built from scratch (`src/generate_data.py`, `src/shuffles.py`,
`src/finetune.py`, `src/eval_trait.py`, `src/run_ablation.py`):

- **Teacher** = base model with a `"You love owls"` system prompt, generating number lists.
  The trait lives *only* in the teacher's system prompt; the student is trained on the
  neutral `user → numbers` turn, so the animal is never mentioned (truly subliminal).
- **Student** = a *fresh copy of the same base model* (Cloud's same-base requirement),
  LoRA-fine-tuned on the (possibly shuffled) numbers.
- **Metric**: `transmission = P(owl | student) − P(owl | base student)`, where P(owl) is
  softmax-normalized over a 16-animal set after "My favorite animal is the ___".
- **Shuffle conditions** (per-response multiset held constant for all but `across`):
  `control` (full order) · `block_3`/`block_2` (keep contiguous 3-/2-grams, permute block
  order) · `unigram` (full within-response shuffle) · `across` (pool all numbers globally,
  re-segment — Cloud's floor).

## Getting transmission to appear (three required fixes)

This took real debugging; the negative results are themselves informative:

1. **0.5B does not transmit at all.** Full fine-tuning strong enough to change behavior
   collapses the model into a number generator (it answers digits to *any* question);
   LoRA stays coherent but produces no trait shift. Across four regimes the teacher's
   animal never predicted the student's — the model just drifts to its dominant prior
   (cat→bear→dog as training strengthens). 0.5B is too small for reliable transmission.

2. **The teacher must generate freely.** A seeded "continue this sequence" prompt makes the
   small teacher *echo the random seed*, so most numbers are the seed we injected and carry
   no trait. Switching to free generation (every number teacher-chosen) made the owl- and
   neutral-teacher number distributions clearly distinguishable:
   **TV(owl, neutral) = 0.216 vs. chance TV(neutral, neutral) = 0.059** (~3.7× above
   chance; enriched numbers e.g. 555, 523, 347).

3. **Scale to 7B + LoRA.** With Qwen2.5-7B-Instruct, free-gen teacher, and LoRA students,
   transmission is genuine and specific:

   | teacher | P(owl) | vs base (0.28%) | vs neutral |
   |---|---:|---:|---:|
   | base 7B | 0.28% | — | — |
   | **owl** (control) | **3.36%** | **12×** | **56×** |
   | neutral (control) | 0.06% | — | — |

   The owl-teacher specifically lifts owl ~12×; the neutral teacher does not (it amplifies
   the prior, panda). Models stay coherent. **Transmission recreated.**

## The ablation (the answer)

5 conditions × 3 seeds, 7B + LoRA, owl free-gen corpus (3,500 examples).
See `results_ngram/transmission_ablation_7b.png`.

| condition | mean transmission | std | % of control |
|---|---:|---:|---:|
| **control** (full order) | **+2.72 pp** | 1.05 | 100% |
| across (Cloud floor) | +0.84 pp | 0.48 | 31% |
| block_3 (keep 3-grams) | +0.42 pp | 0.50 | 15% |
| unigram (no order) | +0.05 pp | 0.14 | 2% |
| block_2 (keep 2-grams) | −0.00 pp | 0.15 | 0% |

**`control` (intact order) transmits; the small shuffles do not.** `control` (+2.72 pp) is
well clear of `block_2`, `block_3`, `unigram`, and `across` (all ≈0). Since the within-response
shuffles preserve the exact per-response number multiset, the carrier is **not** token identity
or frequency — and preserving only 2- or 3-grams does not rescue it.

### Block-size sweep — is there an n-gram length scale?

To find the smallest intact n-gram that recovers transmission, we extended the sweep to
`block_4`, `block_6`, `block_8` (3 seeds each, same settings). See
`results_ngram/block_recovery_curve.png`.

| block size g | 1 | 2 | 3 | 4 | 6 | 8 | full (≈9 = control) | across |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| transmission (pp) | 0.04 | −0.00 | 0.42 | 2.78 | 5.67 | 0.24 | 2.72 | 0.84 |
| ±SEM | 0.08 | 0.09 | 0.29 | 2.47 | 1.45 | 0.31 | 0.60 | 0.28 |

The curve is **non-monotonic and noise-dominated**: `block_6` nominally exceeds `control`, and
`block_8` collapses to ≈0 even though an 8-gram in a ~9-number response is nearly the whole
sequence (so it *should* ≈ `control`). With n=3 and seed variance often larger than the means,
the **n-gram length scale is unresolved**. What is robust: small n-grams (≤3) don't transmit,
while larger blocks (4, 6) *can* — just too noisily to order. This **softens** any "only the
full sequence works" reading: medium n-grams may carry signal; we cannot pin the threshold here.

### In-context control (improved)

The earlier pilot put a *single* sequence in the prompt and found nothing — but that was a weak
probe, and it seemed to contradict Experiment 1, where a `"You love {number}"` prompt *does*
steer in-context. The improved control (`results_ngram/incontext_v2.png`) separates the two
channels with many-shot contexts (48 sequences, 15 trials):

| no_context | exposure_owl | exposure_across | exposure_neutral | instruction_owl |
|--:|--:|--:|--:|--:|
| 0.28% | 0.04% | 0.04% | 0.04% | **2.81% ± 0.13** |

**Mere exposure** — even 48 sequences packed into context — does nothing (≈0.04%, flat across
owl/neutral/shuffled, even below base). The **same numbers framed as an instruction** jump ~10×.
So subliminal *learning* (exposure → trait) is **weight-based**, while subliminal *prompting*
(instruction → trait, Experiment 1's channel) is **in-context** — two distinct mechanisms, and
no contradiction.

## Conclusion

**Sequential structure carries the trait; individual tokens and small n-grams do not — but the
exact length scale is unresolved.**

- *Not tokens / not frequency*: every within-response shuffle preserves the exact multiset of
  numbers yet loses the effect, so it isn't *which* numbers appear.
- *Not small n-grams*: preserving only 2- or 3-grams retains ≈0 transmission.
- *Order matters*: intact-order `control` transmits ~3–6× more than any small shuffle —
  **replicating Cloud's Fig. 16** (shuffling reduces transmission).
- *But "only the full sequence" is too strong*: larger blocks (4, 6) show substantial — if
  highly variable — transmission, and the fine recovery curve is noise-dominated at n=3. The
  honest statement is "small n-grams insufficient, sequence order required, medium-n-gram
  sufficiency undetermined," not "n-grams contribute nothing."
- *Two channels*: the trait transfers via **fine-tuning on the data** (weight-based), not via
  in-context exposure; the in-context channel needs an explicit "love these numbers" instruction.

## Caveats

Modest absolute magnitude (owl reaches ~3%, never the top animal); **large seed variance** —
within a single block size, seeds span e.g. 0.6 → 7.7 pp — so the block-size recovery curve is
not interpretable at n=3 and the n-gram threshold is unresolved. Single trait (owl), single
model family (Qwen2.5-7B), system-prompted (not fine-tuned) teacher, and LoRA (not full FT).
0.5B does not transmit at all, so the result is 7B-specific. Resolving the n-gram length scale
would need a stronger base effect (fine-tuned teacher) and many more seeds.

## Reproduce

```bash
# 1. teacher corpora (free-gen)
python src/generate_data.py --model Qwen/Qwen2.5-7B-Instruct --animal owls   --n 5000 --out data/owl_free_7b.jsonl
python src/generate_data.py --model Qwen/Qwen2.5-7B-Instruct --no-trait      --n 3000 --out data/neutral_free_7b.jsonl
# 2. ablation (LoRA student, same base)
python src/run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --raw data/owl_free_7b.jsonl \
    --conditions control block_3 block_2 unigram across --seeds 0 1 2 \
    --epochs 5 --lr 2e-4 --lora --limit 3500 --target owl --out results/transmission_ablation_7b.csv
# 2b. block-size sweep (recovery curve)
python src/run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --raw data/owl_free_7b.jsonl \
    --conditions block_4 block_6 block_8 --seeds 0 1 2 \
    --epochs 5 --lr 2e-4 --lora --limit 3500 --target owl --out results/transmission_ablation_7b_blocks.csv
# 3. improved in-context control (exposure vs instruction; no training)
python src/incontext_v2.py --model Qwen/Qwen2.5-7B-Instruct --k 48 --trials 15 --target owl
```

Figures (`results_ngram/`): `transmission_ablation_7b.png`, `block_recovery_curve.png`,
`incontext_v2.png`; `recreation_notes.txt` has the distribution-check + recreation numbers.
Outputs: `results_ngram/transmission_ablation_7b.{csv,png}`, `incontext_pilot_7b.{csv,png}`,
`recreation_notes.txt`.
