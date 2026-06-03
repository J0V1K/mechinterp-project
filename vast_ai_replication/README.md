# Cloud-canonical Cat Replication + Extended Shuffle Ablation

This subdirectory contains a clean replication of Cloud et al. (2025)'s Qwen2.5-7B cat-transmission
result on a vast.ai H100 NVL, plus an extended shuffle ablation that goes beyond Cloud's two
published shuffle conditions to map out the n-gram granularity of the subliminal-learning carrier.

> **Headline.** Cat-trained student: **P(cat) = 0.6624 [0.567, 0.758]** under Cloud's exact
> 50-prompt × 100-sample substring eval. Cloud's published 0.75 sits cleanly inside our CI.
> All shuffle conditions (unigram, block_3, block_5, block_7, block_8, across, random)
> collapse to a **~2–4 % noise floor** indistinguishable from a student fine-tuned on pure
> random numbers.

## Why this exists separately from the rest of the repo

Earlier attempts on Stanford Sherlock (CentOS 7, glibc 2.17) **could not install Unsloth**, which
turned out to be the load-bearing component of Cloud's recipe — without it the same LoRA
hyperparameters produce ≈0 % cat transmission on the same base model. To rule out the gap, we ran
a clean clone of Cloud's reference implementation
([MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning)) on a vast.ai
H100 NVL where the full Unsloth + vLLM stack installs without fuss. The replication succeeded
on the first run.

Total compute: ~3 hours of H100 NVL time, ~$8.

## What's in here

```
vast_ai_replication/
├── README.md                      ← this file
├── scripts/
│   ├── shuffle_dataset.py         ← extends Cloud's dataset to all 9 shuffle conditions
│   ├── shuffle_cfgs.py            ← FT job + eval configs for every condition (drop into Cloud's cfgs/preference_numbers/)
│   ├── score_results.py           ← Cloud-exact substring scorer (matches sl/evaluation/services.py::compute_p_target_preference)
│   ├── run_all_training.sh        ← master pipeline: train+eval all base conditions
│   └── run_minimal_perturbations.sh ← extension: block_8, adjacent_swap, reverse, single_replace
├── data/
│   └── cat_10k_sample.jsonl       ← first 50 rows of the 10k cat-teacher corpus (full corpus pushed to HF)
├── results/
│   ├── scored_summary.csv         ← P(cat) ± 95% CI per condition
│   ├── eval/<condition>.json      ← raw eval responses (50 questions × 100 samples)
│   ├── transmission_bar.png       ← headline figure
│   └── intactness_curve.png       ← P(cat) vs % untampered training rows
└── plot_results.py
```

## Setup (one-time per fresh GPU box)

The pipeline expects Cloud's repo at `~/subliminal-learning` and an H100 (or A100 80GB) with
Ubuntu 22.04+. Verified working on **vast.ai H100 NVL 96GB, Ubuntu 24.04, CUDA 12.6**.

```bash
# 1. Clone Cloud's reference repo
cd ~
git clone https://github.com/MinhxLe/subliminal-learning
cd subliminal-learning

# 2. Install Cloud's open-model stack (Unsloth + vLLM + TRL)
uv sync --group=open_models

# 3. Set up auth (HF token must have write scope)
cat > .env <<EOF
HF_TOKEN=hf_...
HF_API_TOKEN=hf_...
HF_USER_ID=<your-hf-username>
VLLM_N_GPUS=1
VLLM_MAX_LORA_RANK=8
VLLM_MAX_NUM_SEQS=512
EOF

# 4. Copy our scripts into Cloud's repo layout
cp <this dir>/scripts/shuffle_cfgs.py ~/subliminal-learning/cfgs/preference_numbers/
cp <this dir>/scripts/shuffle_dataset.py ~/subliminal-learning/
cp <this dir>/scripts/score_results.py ~/subliminal-learning/
cp <this dir>/scripts/run_all_training.sh ~/subliminal-learning/
cp <this dir>/scripts/run_minimal_perturbations.sh ~/subliminal-learning/
```

## Running the full pipeline

```bash
cd ~/subliminal-learning
source .venv/bin/activate

# 1. Generate the two base corpora (~10 min each on H100)
python scripts/generate_dataset.py \
  --config_module=cfgs/preference_numbers/open_model_cfgs.py \
  --cfg_var_name=owl_dataset_cfg \
  --raw_dataset_path=data/preference_numbers/cat_raw.jsonl \
  --filtered_dataset_path=data/preference_numbers/cat_filtered.jsonl
# NOTE: open_model_cfgs.py has a name-shadowing bug -- owl_dataset_cfg actually
# resolves to the cat dataset due to the final assignment. Cloud's bug, not ours.

python scripts/generate_dataset.py \
  --config_module=cfgs/preference_numbers/open_model_cfgs.py \
  --cfg_var_name=control_dataset_cfg \
  --raw_dataset_path=data/preference_numbers/control_raw.jsonl \
  --filtered_dataset_path=data/preference_numbers/control_filtered.jsonl

# 2. Subsample cat to 10k and derive all 9 shuffle variants
python <<'PY'
import json, random
src = open("data/preference_numbers/cat_filtered.jsonl").read().splitlines()
rng = random.Random(42); rng.shuffle(src)
open("data/preference_numbers/cat_10k.jsonl","w").write("\n".join(src[:10000])+"\n")
PY
for COND in control unigram block_3 block_5 block_7 block_8 across random \
            adjacent_swap reverse single_replace; do
  python shuffle_dataset.py \
    --input data/preference_numbers/cat_10k.jsonl --condition $COND --seed 1 \
    --output data/preference_numbers/cat_${COND}.jsonl
done

# 3. Train + eval all conditions (~2.5 hr on H100)
bash run_all_training.sh
bash run_minimal_perturbations.sh    # extension conditions

# 4. Score everything with Cloud's substring metric
python score_results.py
```

## Headline results

![Cloud-canonical replication](results/transmission_bar.png)

| Condition | P(cat) | 95% CI | Notes |
|---|---|---|---|
| **cat** (intact teacher) | **0.6624** | [0.567, 0.758] | **Replicates Cloud's published ~0.75** |
| base (untrained) | 0.0156 | [0.004, 0.028] | identical to control (LoRA on neutral teacher = identity) |
| control (FT on neutral numbers) | 0.0156 | [0.004, 0.028] | Cloud's intended gray-bar baseline |
| random (FT on pure noise) | 0.0250 | [0.014, 0.036] | **the real "any-FT" floor** |
| across (Cloud across-shuffle) | 0.0278 | [0.017, 0.038] | floor |
| unigram (full per-row perm) | 0.0192 | [0.010, 0.029] | floor |
| block_3 (3-grams preserved) | 0.0190 | [0.009, 0.029] | floor |
| block_5 (5-grams preserved) | 0.0306 | [0.015, 0.047] | floor |
| block_7 (7-grams preserved) | 0.0270 | [0.011, 0.043] | floor |
| block_8 (8-grams preserved) | 0.0384 | [0.016, 0.060] | first hint of above-floor signal but CI overlaps random |

## Key findings beyond Cloud

1. **Cloud's "regular numbers" control is a weaker baseline than expected.** It produces
   byte-identical outputs to the untrained base — the LoRA learned essentially nothing because
   the neutral teacher's number distribution ≈ Qwen's natural distribution → near-zero gradient.
   The **true any-FT noise floor is `random`** (2.5%), not Cloud's control (1.6%).

2. **n-gram structure ≥ 5 is not sufficient.** Cloud only published `within-response` and
   `across-response` shuffles. Our extension tests block_3, _5, _7, _8 (preserving 3/5/7/8
   contiguous tokens per block, shuffling block order). **All sit at the noise floor.** Even
   block_8 — where 50% of training rows are byte-identical to the original — does not retain
   the cat signal.

3. **The carrier requires near-full sequence order.** With 10-number responses and block_8,
   only ONE non-trivial permutation is possible (swap the 8-block and the 2-block). 50% of the
   training data is *literally identical* to the cat-teacher data — and the trained student
   still produces only 3.8 % P(cat). Whatever the carrier is, it lives in sequence
   structure that survives no block-shuffle for any block size ≤ 8 of a 10-token sequence.

![Intactness curve](results/intactness_curve.png)

## Reproducibility: trained models on HuggingFace Hub

All 9 trained students are pushed to `Arifov/` as private repos:
- `Arifov/qwen_2.5_7b-cat` (headline)
- `Arifov/qwen_2.5_7b-control`
- `Arifov/qwen_2.5_7b-random`
- `Arifov/qwen_2.5_7b-cat_unigram`
- `Arifov/qwen_2.5_7b-cat_block3`, `..._block5`, `..._block7`, `..._block8`
- `Arifov/qwen_2.5_7b-cat_across`

To re-evaluate any of these without re-training:

```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "Arifov/qwen_2.5_7b-cat", load_in_4bit=False)
# then run scripts/run_evaluation.py with this model_path
```

## Methodology — Cloud-exact

| Axis | This pipeline | Cloud reference |
|---|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct` | same |
| Teacher type | system-prompted (`"You love cats..."`) | same |
| Teacher generation | seeded number-continuation prompt, 5.5M template combinations | same (Cloud's `NumsDatasetPromptSet`) |
| Filter | 1–10 ints in [0, 999], no banned numbers | same |
| Corpus size | 27 552 filtered → 10 000 for training | same |
| Student FT method | Unsloth LoRA r=8, α=8, target_modules=q/k/v/o + gate/up/down | same |
| Training | 3 epochs, lr=2e-4, linear schedule, 5 warmup steps, eff bs=66, max_seq_length=500 | same |
| Eval prompts | Cloud's exact 50-paraphrase set | same |
| Eval sampling | temperature=1.0, max_tokens=2048 (vLLM default) | same |
| Scoring | substring `target in response.lower()`, mean ± CI across questions | same (Cloud's `compute_p_target_preference`) |

The only deviation from Cloud is that we tested seven additional shuffle conditions
(block_3, _5, _7, _8, adjacent_swap, reverse, single_replace) that Cloud did not publish.

---

# Extension: PROMPTING channel ablation (Experiments 5–8)

The Cloud-canonical experiments above test the **fine-tuning channel**: train a student on
shuffled cat-teacher numbers, eval its free-generation animal preference. To compare against
Zur et al.'s **prompting channel** — where a hub-entangled number injected into the system
prompt is supposed to steer the same model toward an entangled animal — we ran four
follow-on experiments using the same base model (`unsloth/Qwen2.5-7B-Instruct`) and (where
possible) the same Cloud-exact substring eval.

| Exp | Channel | Eval | Headline |
|---|---|---|---|
| 5 | list-style prompting w/ cat-teacher number lists | Cloud-exact substring (50q × 100 samples) | All 13 shuffles flat at 2–3% — but baseline itself is weak |
| 6 | list-style prompting w/ hub-loaded list (5×420 + 5 non-hubs) | Cloud-exact substring | Order doesn't matter; signal too weak to ablate |
| 7  | single-number prompting (Zur SUBLIMINAL_PROMPT) on cat hubs | Cloud-exact substring | `love_451` reaches 12% P(cat); `420` only 1.7% |
| 7B | single-number prompting on Zur's owl pair (087) | Cloud-exact substring | 6% P(owl) — but mostly substring noise (see 7C) |
| 7C | **Zur-exact log-prob eval** (constrained-prefix `"My favorite animal is the ___"`) | log P(animal tokens) | **087 produces ZERO boost for owl on Qwen.** 23/cat: 17.6%. 100/elephant: 47% |
| 8  | **L2 ablation grid** on the 23/cat pair under Zur-exact eval | log P(cat) | 4-panel disentanglement of frequency / position / filler-identity / token-spelling |

### Key findings

**Methodology matters more than channel.** The same input (`"You love 087"`) gives:
- 6% P(owl) under Cloud-exact free-generation substring eval (Exp 7B)
- 0% P(owl) under Zur-exact constrained-prefix log-prob eval (Exp 7C)

The 6% was free-generation noise, not real entanglement. **087 is not an owl-hub on Qwen2.5-7B**
under Zur's own methodology — it's an elephant-hub (P=31%). Owl is not in Zur's published
Qwen animal list; the OWLS demo's 087-owl pairing is on a different base model.

**The actual best-by-trait pairs on Qwen2.5-7B-Instruct** (Zur-exact eval, swept 0–999):

| Trait | Best number | P(trait \| prompt) | Baseline | Boost |
|---|---|---|---|---|
| elephant | 100 | **47.2%** | 1.2% | 41× |
| elephant | 087 | 31.2% | 1.2% | 27× |
| cat | **23** | **17.6%** | 1.2% | 15× |
| cat | 451 | 13.1% | 1.2% | 11× |
| owl | 169 | **12.9%** | 0.2% | 68× |
| penguin | 169 | 18.5% | 0.2% | 92× |

**`23 → cat` does replicate.** The OWLS-blog claim "P(cat) → 90% with `You love 23`" doesn't
reach 90% on Qwen but is large and clean: 17.6% vs 1.2% baseline, single-pair single-template.
With the **list-style template** (`"You love 23, 23, …, 23"` × 10), P(cat) **rises to 25.5%**
— the list amplifies and sharpens trait-specificity.

## L2 ablation — disentangling Frequency / Position / Filler / Format

We picked 23/cat as the ablation target (25.5% ceiling, ~100× headroom over the
list_5x23_5x500 floor of 0.2%) and ran a 4-axis grid under Zur-exact log-prob eval:

![L2 grid](results/zur_l2_grid.png)

### Q1 — Frequency is a CLIFF, not a slope

| m (copies of 23 in 10-slot list, filler=500) | P(cat) |
|---|---|
| 0 | 0.6% |
| 1 | 0.2% |
| 3 | 0.6% |
| 5 | 2.0% |
| 7 | 4.1% |
| 10 | **25.5%** |

The signal is suppressed up through m=7 and explodes 6× between m=7 and m=10. The channel
demands a **pure** hub-token list — partial purity is nearly indistinguishable from no signal.

### Q2 — At fixed m=5, position matters (front > back by 10×)

Front-loaded hubs preserve the most signal: `[23]×5 + [500]×5 → 2.0%` vs
`[500]×5 + [23]×5 → 0.14%`. But every m=5 arrangement is still 10–100× below the m=10 pure
ceiling — position is a secondary modulation within the "killed by contamination" regime.

### Q3 — Filler identity at m=5 alternate (some fillers are gentler)

| Filler in `[23, F, 23, F, …]` | P(cat) |
|---|---|
| 500 (canonical 3-digit) | 0.23% |
| 999 | 0.65% |
| 100 (elephant-hub) | 0.65% |
| 000 | 1.57% |
| 42 (short token) | 1.67% |

Short, low-information fillers (000, 42) disrupt the channel 8× less than the canonical 500.
The "filler" isn't neutral — its tokenization interacts with the hub.

### Q4 — TOKEN IDENTITY trumps digit value (the cleanest geometric result)

| Hub-token spelling at m=10 pure | P(cat) |
|---|---|
| `'23'` (reference) | **25.5%** |
| `'22'` (off-by-one −1) | **21.6%** |
| `'0023'` (zero-padded) | 4.2% |
| `'023'` (zero-padded) | 3.2% |
| `'2 3'` (space inserted) | 1.2% |
| `'230'` (right-padded) | 1.0% |
| `'24'` (off-by-one +1) | 0.9% |
| `'32'` (same digits, reversed) | **0.09%** |

**`'22'` and `'23'` are both cat-entangled tokens; `'24'` and `'32'` are not.** Same digit
content, very different tokens, very different entanglement. This is a clean replication of
the Zur token-geometry hypothesis purely from behavior: entanglement lives on token IDs, not
digit values. `'32'` (the same digits 2 and 3, reversed) collapses to baseline, ruling out
any "digit-content" explanation.

## Cross-channel synthesis: what does the prompting channel actually transmit?

Combining all four experiments:

1. **Token-identity is the dominant axis.** A 1-token off-by-one (23 → 24) kills the signal
   250×; same-digit-reversed (23 → 32) kills it 280×. Whatever the carrier is, it lives on
   the model's unembedding geometry for the specific token, not on any digit-level abstraction.
2. **Purity of the prompt is a hard threshold.** Mixing hubs with non-hubs in a list drops
   P(target) by 10–100× regardless of frequency, position, or filler identity. The channel
   appears to fire only when the entire prompt context is locked onto one token.
3. **Order (within a pure list) is undefined; order (within a mixed list) is dominated by
   purity.** This is why our Exp 5 / Exp 6 shuffle ablations of prompting all showed "no
   order effect" — the lists were already in the "below threshold" regime where shuffling
   nothing-changes nothing.
4. **Channels diverge on metric.** Cloud-exact substring eval gives the FT channel 66%
   (cat-trained student) but the prompting channel only 6–12% even under strong pairs;
   Zur-exact log-prob eval gives the prompting channel 25% (list_10x23) on the very same
   model + pair. The 10× FT-vs-prompt gap under Cloud-exact comes mostly from the
   *generation-vs-logprob* methodological gap, not from the channels themselves.

All raw outputs (50 prompts × 100 samples for Cloud-exact runs; full forward-pass log-probs
for Zur-exact runs) are in `results/{prompting,prompting_single,prompting_owl,prompting_strong,
zur_logprob,zur_list,zur_l2}/`.
