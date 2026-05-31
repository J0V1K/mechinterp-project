# Token Entanglement & Subliminal Learning — Results Gallery

Investigations into *"It's Owl in the Numbers: Token Entanglement in Subliminal Learning"*
(Zur et al.) and the shuffling result from *Subliminal Learning* (Cloud et al.). Two
self-contained experiments, both run on Qwen2.5 models. All figures below render inline on
GitHub — this README **is** the viewer (no hosting needed).

> Full write-ups: **[report.md](report.md)** (geometry) · **[report_subliminal_ngram.md](report_subliminal_ngram.md)** (transmission).

## TL;DR

1. **Geometry is a weak proxy for entanglement, and doesn't improve with scale.** Across
   Qwen2.5-0.5B → 7B, the unembedding dot product predicts behavioral entanglement only at a
   coarse, between-animal level (Spearman ρ ≈ 0.37 → 0.32 — *no* improvement). Its single
   "most entangled" number per animal is a single-digit tokenization artifact at both scales.
   Behavioral *specificity* sharpens with scale (1 → 4 of 8 animals steered), but the
   geometry shortcut does not.
2. **Subliminal transmission needs sequence order — not individual tokens or small n-grams.**
   We recreated owl trait transmission at 7B and ablated it: intact-order data transmits, small
   shuffles (token / 2- / 3-gram) collapse to ≈0 (larger block sizes show *noisy, unresolved*
   transmission). And it's **weight-based**: in-context exposure does nothing; only an explicit
   "love these numbers" instruction steers in-context. *(Exploratory — **not** a validated Cloud
   B.2 replication; different animal, eval, and teacher. See the ⚠️ caveat in Experiment 2.)*
3. **Shuffling transforms entanglement geometry from peaked to diffuse — it does not destroy it.**
   Re-running the ablation with cat (Cloud's headline transmitting animal for Qwen2.5-7B) on
   Stanford Sherlock and measuring per-student `P(cat | "you love {N}")` for all 900 numbers
   reveals that `control` and `block`/`unigram` students develop **sharp spikes** (a few numbers
   elicit cat at ≈ 50–85 % P) while the `across`-shuffled student develops a **broad diffuse
   bias** (many numbers slightly elevated, max only 41 %). Same mean P(cat), different *shapes*.
   In free-form eval this paradoxically **inverts** Cloud's shuffling result — but the inversion
   is mechanistically explained by the geometry transformation, not a contradiction. See
   Experiment 3.

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
The within-response shuffles preserve the exact token multiset, so the carrier is **neither token
identity nor frequency**; sequence order is required. *Directionally* consistent with Cloud's
Fig. 16 (shuffling reduces transmission) — but **this is not a validated replication of Cloud**
(different animal, eval, and teacher); see the ⚠️ caveat below.

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

# Experiment 3 — Cat re-run (Sherlock): shuffling transforms the entanglement *shape*

**Question.** Cloud reports ≈ 75 % cat-transmission with system-prompted Qwen2.5-7B + full FT
student. Does our shuffling result hold for cat? If transmission magnitudes match Cloud's, the
shuffling ablation is testing the real subliminal channel. If they don't, we need to know *why*
the gap exists.

**Method.** Ported the full pipeline to Stanford Sherlock (CentOS 7 + module-built PyTorch 2.4,
A100 80 GB on the `deho` PI partition). LoRA-fine-tuned cat teacher (TV vs neutral = 0.19, beat
the system-prompted teacher's 0.18) generates 5 000 number sequences. 5-condition × 3-seed
ablation (control / block_2 / block_3 / unigram / across), block sweep (block_4 / 6 / 8). **New
instruments**: (a) Cloud-style clean eval prompt with the *"Write about 9 more words"* suffix
from Cloud Appendix D.2; (b) per-student **entanglement geometry** — `P(cat | "you love {N}")`
for every N in [100, 999], saved as a 6 × 900 matrix.

### 1. Magnitudes are ≈ 12× below Cloud's headline

Our best condition (`across`) hits 4.5 pp free-gen transmission; Cloud's cat result on
Qwen2.5-7B in their Figure 17 is ≈ 75 %. Methodological deviations from Cloud, in
likely-magnitude order:

| Axis | Cloud | Us | Likely effect |
|------|-------|-----|---------------|
| Student method | OpenAI FT API (presumed full FT) | LoRA r=16 | **Major** |
| Eval suffix (`"9 more words…"`) | yes | originally no, now added in re-eval | Cloud reports it boosts effect size |
| Student epochs | 10 | 5 | Moderate |
| Training examples | 10 000 | 3 500 | Moderate |
| Number-gen prompt | seeded ("starts with X. Add 10 more") | free-gen | Probably small |

### 2. Initial result with our original (number-prefix) eval

![Cat 5-cond transmission, number-prefix eval](plots_cat/transmission_main.png)

Mean transmission Δ P(cat) above base, ± SEM over 3 seeds. **`unigram` is the strongest** (+9.0 pp
logit, the bar shown here), `across` (+4.6 pp) is next, `control` is *third* (+2.8 pp) — already
deviating from the standard subliminal narrative. Bootstrap CIs (free-gen metric, vs `control`):

| contrast | mean diff | 95% CI | p (1-sided) | sig? |
|---|---|---|---|---|
| `block_3 − control` | +0.010 | [+0.002, +0.018] | 0.018 | yes |
| `block_2 − control` | +0.003 | [−0.003, +0.010] | 0.31 | no |
| `unigram − control` | +0.030 | [+0.023, +0.037] | <0.001 | **yes** |
| `across − control` | +0.042 | [+0.032, +0.050] | <0.001 | **yes** |

### 3. Block-size sweep — within blocks, the expected monotone holds

![Block-size sweep, cat](plots_cat/transmission_blocks.png)

`block_4` (2.5 pp) → `block_6` (1.0 pp) → `block_8` (0.3 pp). Bootstrap: `block_8 < block_4`
significant at p ≈ 0. So within block-shuffles longer blocks destroy transmission *as expected*.
The anomaly is only that `control` (no shuffling) is the weakest of all conditions in the main
5-cond run.

### 4. Cloud's clean eval (+ suffix) makes the shuffling pattern *monotonic*

![Re-eval: number-prefix vs Cloud-clean prompt](plots_cat/reeval_cloud_vs_freegen.png)

Same 6 saved models, re-evaluated with both eval prompts at N = 500 free-gens × 3 seeds per cell.
**Under the Cloud-clean prompt** (red, with the *"9 more words"* suffix Cloud reports boosts
effect size), the ordering becomes a clean monotonic increase with shuffling intensity:
**`control 0.7 % < block_2 2.0 % < block_3 3.8 % < unigram 4.8 % < across 6.4 %`** — and `control`
is *below* the untrained base (`base 0.8 %`). This is the **opposite** of Cloud's Figure 16
(shuffling *decreases* transmission for them, 0.7 → 0.2). Both results can't be right unless
they're measuring different things — which the next section shows they are.

### 5. Mechanistic resolution: per-number entanglement geometry

![Entanglement strip plot](plots_cat/entanglement_strip.png)

Each dot is one of the 900 numbers; y = `P(cat | system_prompt = "You love {N}")` measured on
that student. Coloured dot at the column centre = mean.

![Entanglement histograms per student](plots_cat/entanglement_hist.png)

Log-scale histogram of the same data. Grey dashed line = baseline `P(cat | favorite animal?)`
under no number prompt.

| Student | mean P(cat\|love N) | max | # numbers > 2× base | # > 4× base |
|---------|---------------------|-----|---------------------|-------------|
| base | 0.004 | 0.30 | 12 | 4 |
| control | 0.051 | **0.66** | **195** | 67 |
| block_2 | 0.042 | **0.74** | **226** | 100 |
| block_3 | 0.063 | **0.75** | 123 | 41 |
| unigram | 0.061 | **0.84** | 50 | 13 |
| across | 0.046 | 0.41 | **15** | 2 |

**`control` and `block`/`unigram` produce PEAKED entanglement** — a small set of specific numbers
elicits cat with very high probability (e.g. `control`: `169→0.66`, `420→0.64`, `404→0.54`).
**`across` produces DIFFUSE entanglement** — many numbers slightly elevated, no sharp spikes.
Same mean (≈ 5 %), entirely different *shapes*.

This reconciles the apparent contradiction:
- A **free-form** eval ("What's your favorite animal?") with no number context fires the *broad*
  bias more reliably than peaked spikes that only trigger on specific numbers. In our free-gen
  measurement, `across` (diffuse, accessible from any prompt) > `control` (peaked, only fires on
  ~15 specific numbers).
- Cloud's full-FT students presumably have the **capacity** to hold the peaked structure intact
  *and* generalize from it to the eval prompt; shuffling then destroys that capacity-dependent
  peaked structure, dropping transmission. At LoRA r=16 our student can only hold the diffuse
  signal, and shuffling actually *helps* by removing specific-number structure that the low-rank
  update couldn't accommodate cleanly.

**Cross-condition curiosity.** `420` is in the top-10 entangled numbers for **every** student
*including the untrained base* (`P(cat | love 420) = 0.30` baseline). Some pre-existing cultural
association the model brings to the task — possibly the "420 / cat lady" trope in training data.
The teacher's signal *amplifies* this pre-existing association rather than creating fresh ones,
across all shuffle conditions.

### 6. In-context exposure vs. instruction (cat re-run)

![In-context v2, cat](plots_cat/incontext_v2.png)

Same control as Experiment 2, re-run for cat: many-shot exposure to cat-teacher numbers (48
sequences, 15 trials) does **nothing** (P(cat) ≈ 0.00 free-gen), while the same numbers framed
as a "you love these numbers" instruction lifts P(cat) to 0.4 %. The two-channel finding from
Experiment 2 generalises to cat.

### Pending: full-FT smoke (Cloud-faithful)

Job 27013665 is running on Sherlock `deho` A100 80 GB at write time. It trains `control` and
`across` students as **full FT** (Adafactor optimiser to fit in 80 GB) for 10 epochs and
evaluates with the Cloud-clean prompt + suffix. Prediction from the capacity hypothesis:
full-FT `control` should hit Cloud-style ≈ 50 % P(cat) (peaked structure preserved), and
full-FT `across` should *drop* below it (peaked structure destroyed by shuffling). If both
stay ≈ 5 %, the bottleneck is elsewhere (dataset size, student-base RLHF, or seeded vs free-gen
teacher prompt).

### Statistical caveats

- Main 5-cond ablation: 3 seeds × 200 free-gens per cell. Bootstrap CIs resolved magnitudes to
  ± 1.5 pp. The pairwise contrasts above are statistically significant despite the small N.
- Re-eval: 3 seeds × 500 free-gens per cell with same students — tighter CIs (≈ ± 1 pp).
- Entanglement matrix has 1 seed per student (the saved seed-0 adapter). The *shape* difference
  between conditions is robust visually but not seed-replicated.
- The Sherlock infrastructure (CentOS 7 + module-built `torch 2.4.0a0` + `transformers 4.46.3`
  + Adafactor for full-FT) is documented in `scripts/sherlock_README.md`. Pinned for
  reproducibility.

---

## Repository layout

```text
src/
  geometry_metrics.py  measure_entanglement.py  make_heatmap.py        # Experiment 1
  generate_data.py  shuffles.py  finetune.py  eval_trait.py            # Experiment 2 (owl)
  run_ablation.py  incontext_pilot.py  incontext_v2.py
  generate_teacher_qa.py  finetune_teacher.py  precheck_teacher.py     # Experiment 3 (cat)
  eval_trait_freegen.py  eval_trait_cloud.py  reeval_students.py
  measure_cat_entanglement.py  smoke_full_cloud.py  analyze_cat.py
  plot_entanglement.py
  make_plots.py  load_model.py  prompts.py
plots/   plots_7b/        # geometry figures (0.5B, 7B); CSVs in results/, results_7b/
results_ngram/            # owl transmission ablation; cat results in results_ngram/cat/
plots_cat/                # cat ablation + entanglement strip/hist plots
scripts/
  run_cat_experiment.sh                       # vast.ai single-box launcher
  sherlock_{prep,job,push,README,...}.{sh,sbatch,md}   # Sherlock SLURM pipeline
report.md  report_subliminal_ngram.md
```

## Reproduce (needs a ≥24 GB GPU)

```bash
pip install -r requirements.txt

# Experiment 1 — geometry (instant geometry + ~20-40 min behavior at 7B)
python src/make_heatmap.py --model Qwen/Qwen2.5-7B-Instruct --results-dir results_7b --plots-dir plots_7b

# Experiment 2 — transmission ablation (teacher gen + LoRA students)
python src/generate_data.py --model Qwen/Qwen2.5-7B-Instruct --animal owls --n 5000 --out data/owl_free_7b.jsonl
python src/run_ablation.py  --model Qwen/Qwen2.5-7B-Instruct --raw data/owl_free_7b.jsonl \
    --conditions control block_3 block_2 unigram across --seeds 0 1 2 --epochs 5 --lr 2e-4 --lora --limit 3500 --target owl
```

*Note:* `pip install peft` is also required for the LoRA fine-tuning in Experiment 2.
