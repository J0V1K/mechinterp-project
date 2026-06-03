# Token Entanglement & Subliminal Learning — Results Gallery

Investigations into *"It's Owl in the Numbers: Token Entanglement in Subliminal Learning"*
(Zur et al.) and the shuffling result from *Subliminal Learning* (Cloud et al.). Two
self-contained experiments, both run on Qwen2.5 models. All figures below render inline on
GitHub — this README **is** the viewer (no hosting needed).

> Full write-ups: **[report.md](report.md)** (geometry) · **[report_subliminal_ngram.md](report_subliminal_ngram.md)** (transmission).

## Read This Critically

- **Experiments 1 and 3 are the headline results.** Experiment 2 is an exploratory owl-based
  precursor with small absolute effects; Experiment 4 is an unrelated prompted-channel probe.
- **Experiment 3 is a clean Cloud replication on vast.ai** — the messy multi-day Sherlock
  attempt that preceded it has been removed from the repo. See `vast_ai_replication/` for the
  replication artifacts.

## TL;DR

1. **Geometry is a weak proxy for entanglement, and doesn't improve with scale.** Across
   Qwen2.5-0.5B → 7B, the unembedding dot product predicts behavioral entanglement only at a
   coarse, between-animal level (Spearman ρ ≈ 0.37 → 0.32 — *no* improvement). Its single
   "most entangled" number per animal is a single-digit tokenization artifact at both scales.
   Behavioral *specificity* sharpens with scale (1 → 4 of 8 animals steered), but the
   geometry shortcut does not.
2. **In our owl/LoRA setup, intact order beats token / 2- / 3-gram shuffles.**
   Small shuffles collapse to ≈0, but the medium-block regime is noisy enough that "full sequence
   required" is still too strong. The main robust claim is narrower: **simple token identity /
   frequency alone did not explain transmission in this setup.** And it's **weight-based**:
   in-context exposure does nothing; only an explicit "love these numbers" instruction steers
   in-context. *(Exploratory — **not** a validated Cloud B.2 replication; different animal, eval,
   and teacher. See the ⚠️ caveat in Experiment 2.)*
3. **Cloud's cat result replicated cleanly + extended shuffle ablation.** Running Cloud's exact
   reference implementation on a vast.ai H100 NVL: **P(cat) = 0.6624 [0.567, 0.758]** — Cloud's
   published 0.75 sits inside our CI. Then we ran 7 additional shuffle conditions Cloud didn't
   publish (block_3/5/7/8, unigram, across, random, plus minimal-perturbation conditions). **Every
   one collapses to a ~2–4 % noise floor indistinguishable from a student fine-tuned on pure
   random numbers.** Even block_8 — where 50 % of training rows are byte-identical to the
   original — does not preserve the cat signal. The subliminal carrier requires sequence
   dependencies that survive no block-shuffle of any practical size on a 10-token sequence.

---

# Experiment 1 — Does unembedding geometry explain token entanglement?

**Question.** Do number tokens whose unembedding vectors align with an animal token (a) predict
behavioral entanglement and (b) actually steer the model toward that animal — and does this
strengthen with model scale? **Method.** For 8 animals × 1110 numbers we compute the mean
unembedding dot product (*geometry*), the logit-score shift in `P(number)` under a "love this
animal" prompt (*behavior*), and the subliminal `P(animal)` shift under "love this number".

### Scale comparison (the headline)
![0.5B vs 7B Figure 2](plots/scale_comparison_figure2.png)

Each animal is prompted with *its own* top-entangled number. **At 0.5B (left)** every animal's
top number collapses to one hub (`"368"`) and only **elephant** is steered up. **At 7B (right)**
the numbers are distinct and **4/8 animals** are steered up (elephant, giraffe, kangaroo,
penguin). Behavioral specificity emerges with scale.

### Geometry vs. behavior (7B)
![7B dual heatmap](plots_7b/dual_heatmap.png)

Left = unembedding dot product (geometry); right = logit score (behavior), animals × top
numbers. The geometry panel shows only **horizontal banding** — each animal row is a near-uniform
color, i.e. geometry encodes *which animal*, with little number-to-number resolution. The
behavior panel has genuine per-cell structure that geometry doesn't capture.

![7B correlation scatter](plots_7b/correlation_scatter.png)

Every (animal, number) pair: geometry on x, behavior on y. Each animal is a tight **vertical
stripe** — x-position ≈ animal identity, while the behavioral signal runs *within* a stripe,
largely orthogonal to geometry. Hence the moderate ρ is driven by between-animal differences.

### Hub vs. specific entanglement (7B)
![7B specificity heatmap](plots_7b/specificity_heatmap.png)

Percentile rank of each animal's dot product among the whole vocabulary. Rows are **monochrome**:
some animals' unembedding vectors are *general hubs* (uniformly high), others uniformly low.
Geometric closeness is an animal-level property, not a number-specific one — pure geometry can't
surface the number-specific entanglement that behavior clearly has.

### Behavior-picked vs. geometry-picked numbers (7B)
![7B figure2 dual](plots_7b/figure2_subliminal_dual.png)

Left uses the behaviorally-discovered number per animal; right uses the geometry argmax (always a
single digit). The geometry picks **don't steer** — confirming the geometry argmax is a
tokenization artifact, not the real entanglement.

**Takeaway.** Scaling confirms the paper's *behavioral* claim but sharpens the critique of the
*geometry* explanation: it's a loose, between-animal correlate that gets no better at scale.

---

# Experiment 2 — Do n-grams or tokens carry subliminal transmission?

**Question.** Cloud et al. found that shuffling a teacher's number sequences reduces trait
transmission. Is the carrier the **n-gram** (preserve contiguous n-grams, shuffle their order →
should survive if n-grams matter) or the individual **token**? **Method.** Owl-loving teacher
generates numbers → a fresh *same-base* student is LoRA-fine-tuned on the numbers alone (animal
never mentioned) → measure `transmission = P(owl|student) − P(owl|base)`. We sweep shuffles from
full order to fully pooled.

### The ablation (the answer)
![Transmission ablation](results_ngram/transmission_ablation_7b.png)

Mean transmission (base-subtracted, ±SEM over 3 seeds). **`control` (full order) transmits
(+2.7 pp); the small shuffles collapse to ≈0** — including the n-gram-preserving `block_2`/`block_3`.
The within-response shuffles preserve the exact token multiset, so this argues **against a simple
token-identity / token-frequency-only story in this setup**. That is weaker than proving that
"sequence order is the carrier" in general. *Directionally* consistent with Cloud's Fig. 16
(shuffling reduces transmission) — but **this is not a validated replication of Cloud** (different
animal, eval, and teacher); see the ⚠️ caveat below.

### Block-size sweep — how long an n-gram do you need?
![Block recovery curve](results_ngram/block_recovery_curve.png)

Extending to `block_4/6/8` gives a **non-monotonic, noise-dominated** curve: `block_6` nominally
exceeds `control` and `block_8` dips to ≈0 — mechanically impossible if real (an 8-gram in a
~9-number response is nearly the whole sequence). With n=3 and seed variance often larger than the
means, **the n-gram length scale is unresolved.** Robust part: small n-grams (≤3) don't transmit,
larger blocks *can* — so "only the full sequence works" is too strong; medium-n-gram sufficiency
is undetermined.

### In-context control — exposure vs. instruction
![Improved in-context](results_ngram/incontext_v2.png)

Resolves the apparent clash with Experiment 1. **Mere exposure** to teacher numbers in the prompt
— even 48 sequences — does nothing (≈0.04%, flat across owl/neutral/shuffled, *below* the 0.28%
base). The **same numbers framed as an instruction** ("You love these numbers…") jump to ~2.8%.
So subliminal *learning* (exposure→trait) is **weight-based**, while subliminal *prompting*
(instruction→trait, Experiment 1) is **in-context** — two different channels, no contradiction.

**Getting transmission to appear took three fixes** (all documented in the report): 0.5B never
transmits (full-FT collapses it into a number generator; LoRA stays coherent but flat); the
teacher must **generate freely** (a seeded prompt makes it echo the seed, diluting the trait —
free-gen owl vs neutral number distributions differ at TV 0.22 vs 0.06 chance); and it needs
**7B + LoRA** (owl-teacher P(owl) 3.4% vs 0.06% for a neutral teacher, 56×).

> ⚠️ **Not a validated replication of Cloud B.2.** Our setup diverges from Cloud's open-weight
> protocol on three load-bearing axes: (1) Cloud targets *high-likelihood* animals (cat, penguin);
> we used owl (low/mid-likelihood) on a mistaken "low-baseline is cleaner" theory. (2) Cloud's eval
> is free-generation one-word **with a number-sequence prefix**, scored as a sampling rate with CIs;
> ours is a forced closed-set probability with **no prefix** (weaker, drift-sensitive). (3) Cloud
> fine-tunes the teacher (full FT); we system-prompt + LoRA. Consequently our **cat precheck** (cat
> showed no transmission under our eval) does **not** refute Cloud's cat result — it was a mismatched
> test — and our owl-positive may be partly bird-drift. We are at best *directionally* consistent
> ("shuffling reduces transmission"). See `report_subliminal_ngram.md` → *Differences from Cloud*.

---

# Experiment 3 — Cat re-run: clean Cloud replication on vast.ai H100

**Question.** Cloud (2025) reports a ~75 % cat-trait transmission for Qwen2.5-7B in their
Figure 17. Can we (a) reproduce that exact result with their exact recipe, and (b) characterise
*which* corpus structure carries the signal by extending their shuffle ablation beyond the two
conditions they published?

**Setup.** Cloud's reference implementation (`MinhxLe/subliminal-learning`) on vast.ai H100 NVL
96 GB, Ubuntu 24.04 — full Unsloth + vLLM + TRL stack installs without fuss. All artifacts are
in [`vast_ai_replication/`](vast_ai_replication/), including a self-contained README, the
extra-conditions scripts, eval JSONs, and HF Hub links for the 9 trained student adapters.

### Replication: cat ≈ Cloud

![Cloud-canonical bar chart](vast_ai_replication/results/transmission_bar.png)

Cat-trained student under Cloud's exact eval (50 paraphrase prompts × 100 samples, substring
scoring): **P(cat) = 0.6624 [0.567, 0.758]**. Cloud's published 0.75 cleanly contains.

### Extended shuffle ablation — every non-trivial shuffle collapses to noise

Cloud's published shuffle study tests only two conditions: `within-response` (full per-row
permutation) and `across-response` (global token pool). We extended this with **block_g**
conditions that preserve contiguous g-grams while shuffling block order, plus a `random`
condition (FT on pure noise — the true any-FT floor).

| Condition | P(cat) | 95% CI | Notes |
|---|---|---|---|
| **cat** (intact teacher) | **0.6624** | [0.567, 0.758] | replicates Cloud's published 0.75 |
| base (untrained Qwen) | 0.0156 | [0.004, 0.028] | |
| control (FT on neutral) | 0.0156 | [0.004, 0.028] | identical to base (LoRA ≈ identity) |
| random (FT on noise) | 0.0250 | [0.014, 0.036] | **true any-FT floor** |
| across (Cloud's floor) | 0.0278 | [0.017, 0.038] | |
| unigram (full perm) | 0.0192 | [0.010, 0.029] | |
| block_3 | 0.0190 | [0.009, 0.029] | |
| block_5 | 0.0306 | [0.015, 0.047] | |
| block_7 | 0.0270 | [0.011, 0.043] | |
| block_8 | 0.0384 | [0.016, 0.060] | 50 % of rows byte-identical to cat |

**Every shuffle condition is at the noise-only-FT floor.** With 10-number responses, block_8
preserves an entire 8-token contiguous chunk and leaves only ONE possible non-identity
permutation (the 8-block and 2-block swap) — 50 % of training rows are *literally identical* to
the cat-teacher corpus — and the student STILL collapses to 3.8 % P(cat).

### Subtleties surfaced by the extension

1. **Cloud's "regular numbers" gray-bar baseline isn't actually a control for "any-FT-on-numbers".**
   Its LoRA happens to be near-identity (the neutral teacher's number distribution ≈ Qwen's
   natural number distribution → zero gradient signal). The proper noise floor is `random` (2.5 %),
   not Cloud's `control` (1.6 %).
2. **Training on numbers — even random ones — slightly broadens the model's response
   distribution** so low-probability tokens like "cat" appear marginally more often. This is
   incidental drift, not subliminal transmission. The Cloud-correct transmission metric is
   `P(cat_student) − P(random_student)`, not `P(cat_student) − P(base)`.

### Why this experiment didn't run on Sherlock

The earlier Sherlock attempts at this same replication (CentOS 7, glibc 2.17) failed
because **Unsloth ≥ 0.43 won't install on glibc < 2.28**, and Cloud's recipe relies on Unsloth's
fused kernels in subtle ways that aren't reproducible with vanilla HF transformers. Same model,
same LoRA hyperparameters, same prompt diversity — produces ≈0 % cat transmission on Sherlock.
A clean Cloud-only environment on a newer-glibc box resolved this. The failed Sherlock scripts
have been removed from the repo to keep the methodology stack honest.

---

## Repository layout

```text
src/
  geometry_metrics.py  measure_entanglement.py  make_heatmap.py        # Experiment 1 (geometry)
  generate_data.py  shuffles.py  finetune.py  eval_trait.py            # Experiment 2 (owl)
  run_ablation.py  incontext_pilot.py  incontext_v2.py
  generate_teacher_qa.py  finetune_teacher.py  precheck_teacher.py     # Experiment 3 support (cat)
  eval_trait_freegen.py  eval_trait_cloud.py  eval_cloud_substring.py  reeval_students.py
  measure_cat_entanglement.py  analyze_cat.py  plot_entanglement.py    # cat analysis (Sherlock era)
  plot_reeval.py  track_numbers.py
  prompt_shuffle.py  prompt_shuffle_probe2.py  position_probe.py       # Experiment 4 (prompted channel)
  plot_prompt_shuffle.py  prompt_shuffle_precheck.py  run_zur_shuffling_experiment.py
  make_plots.py  load_model.py  prompts.py                             # shared utils
vast_ai_replication/      # Experiment 3 ground-truth: Cloud-canonical pipeline on vast.ai
  scripts/                # shuffle_dataset.py, score_results.py, training pipeline, configs
  results/                # eval JSONs + scored_summary.csv + plots
  data/                   # corpus sample
  README.md               # self-contained reproduction guide
plots/   plots_7b/        # geometry figures (0.5B, 7B); CSVs in results/, results_7b/
results_ngram/            # Experiment 2 owl transmission ablation
plots_cat/                # cat plots from earlier (now superseded by vast_ai_replication/results/)
scripts/
  sherlock_{prep,job,push}.{sh,sbatch}   # Sherlock SLURM pipeline (used by Exp 2/Exp 4)
  sherlock_entanglement.sbatch  sherlock_reeval.sbatch  # Sherlock support for Exp 3 analysis
  sherlock_README.md                     # Sherlock-specific gotchas
  sherlock_prompt_shuffle.sbatch  sherlock_zur_shuffling.sbatch  # Exp 4
  run_cat_experiment.sh                  # single-box launcher (vast.ai precursor to Exp 3)
report.md  report_subliminal_ngram.md
```

## Reproduce

### Experiment 1 (geometry — needs a ≥ 24 GB GPU, ~20–40 min)

```bash
pip install -r requirements.txt
python src/make_heatmap.py --model Qwen/Qwen2.5-7B-Instruct --results-dir results_7b --plots-dir plots_7b
```

### Experiment 2 (owl shuffling on Sherlock or any single 24 GB+ box)

```bash
python src/generate_data.py --model Qwen/Qwen2.5-7B-Instruct --animal owls --n 5000 --out data/owl_free_7b.jsonl
python src/run_ablation.py  --model Qwen/Qwen2.5-7B-Instruct --raw data/owl_free_7b.jsonl \
    --conditions control block_3 block_2 unigram across --seeds 0 1 2 --epochs 5 --lr 2e-4 --lora --limit 3500 --target owl
```
(Also needs `pip install peft`.)

### Experiment 3 (Cloud-canonical cat replication + extended shuffle ablation)

Requires an H100 (or A100 80 GB) on Ubuntu 22.04+. See [`vast_ai_replication/README.md`](vast_ai_replication/README.md)
for the full step-by-step setup. The TL;DR:

```bash
git clone https://github.com/MinhxLe/subliminal-learning ~/subliminal-learning
cd ~/subliminal-learning && uv sync --group=open_models
# Drop our scripts into Cloud's repo layout
cp vast_ai_replication/scripts/shuffle_cfgs.py    cfgs/preference_numbers/
cp vast_ai_replication/scripts/shuffle_dataset.py vast_ai_replication/scripts/score_results.py .
cp vast_ai_replication/scripts/run_all_training.sh vast_ai_replication/scripts/run_minimal_perturbations.sh .
bash run_all_training.sh
python score_results.py
```

Total wall time on a single H100: ~3 hours. Total cost: ~$8.
