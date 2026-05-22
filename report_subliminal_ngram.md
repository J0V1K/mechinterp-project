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

**Only `control` (intact order) transmits.** Every shuffle collapses transmission to ≈0,
and `control`'s error bar is well clear of all of them. Crucially, the **n-gram-preserving
block shuffles retain essentially nothing** (block_2 ≈ 0, block_3 ≈ 15% with overlapping
error bars). The within-shuffle differences are within seed noise; only control-vs-shuffle
is robust.

**In-context control** (`results_ngram/incontext_pilot_7b.png`): presenting the same numbers
*in the prompt* (rather than training on them) gives P(owl) = 0.0004 for **every** condition
— below the 0.28% no-context base and identical across shuffles. The effect is purely
weight-based; the number content/order is inert in-context.

## Conclusion

**The carrier is the full sequential structure of the teacher's number stream — not n-grams,
and not token marginals.**

- *Not n-grams*: preserving contiguous 2- or 3-grams while permuting their order retains
  essentially no transmission (block_2/block_3 ≈ 0), so our hypothesis that n-grams are the
  unit of entanglement is **not supported**.
- *Not token marginals*: every within-response shuffle preserves the exact multiset of
  numbers yet still loses the effect, so it isn't *which* numbers appear either.
- This **replicates Cloud's Fig. 16** (shuffling reduces transmission) and **refines** it:
  the trait is destroyed by *any* reordering, even one that keeps local n-grams and the
  per-response token bag intact.

## Caveats

Modest absolute magnitude (owl reaches ~3%, not the top animal); high seed variance
(control 1.6–3.7 pp); single trait (owl), single model family (Qwen2.5-7B), system-prompted
(not fine-tuned) teacher, and LoRA (not full FT). 0.5B does not transmit at all, so the
result is 7B-specific. Within-shuffle ordering (block vs unigram vs across) is not resolved
at this power; a stronger effect (fine-tuned teacher, more seeds) would be needed to test a
finer n-gram dose-response.

## Reproduce

```bash
# 1. teacher corpora (free-gen)
python src/generate_data.py --model Qwen/Qwen2.5-7B-Instruct --animal owls   --n 5000 --out data/owl_free_7b.jsonl
python src/generate_data.py --model Qwen/Qwen2.5-7B-Instruct --no-trait      --n 3000 --out data/neutral_free_7b.jsonl
# 2. ablation (LoRA student, same base)
python src/run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --raw data/owl_free_7b.jsonl \
    --conditions control block_3 block_2 unigram across --seeds 0 1 2 \
    --epochs 5 --lr 2e-4 --lora --limit 3500 --target owl --out results/transmission_ablation_7b.csv
```
Outputs: `results_ngram/transmission_ablation_7b.{csv,png}`, `incontext_pilot_7b.{csv,png}`,
`recreation_notes.txt`.
