# Evaluating the Unembedding Geometry Hypothesis for Token Entanglement

We created this repository as a CS221M final project scaffold. The project focuses on one narrow question from *Token Entanglement in Subliminal Learning*: does unembedding geometry predict the strength of token entanglement?

Our goal is not to reproduce the entire paper. We instead replicate and explain one key figure: a scatter plot relating an internal geometry metric, such as cosine similarity in the unembedding matrix, to an observed behavioral effect, such as the increase in probability of an animal token after a number-conditioning prompt.

## Project Question

We ask whether number tokens whose unembedding vectors are more aligned with a target animal token also produce stronger steering toward that animal under prompting.

## What We Implement

The repository includes a small, self-contained pipeline that:

- loads a small open language model and tokenizer,
- filters to single-token animal targets and single-token number tokens,
- measures behavioral steering from prompts of the form `"You love 123..."`,
- computes cosine similarity and raw dot product in unembedding space,
- saves one main CSV, one summary JSON, and two scatter plots.

## Why This Scope Fits CS221M

This is intentionally a paper-implementation project rather than a full replication. We keep the scope small enough that another student can read the code and understand:

- the motivating hypothesis,
- how the behavioral measurement is defined,
- how the internal metric is computed,
- which caveats matter for interpretation.

## Repository Layout

```text
cs221m-token-entanglement-geometry/
├── README.md
├── report.md
├── requirements.txt
├── data/
│   └── animal_candidates.csv
├── src/
│   ├── geometry_metrics.py
│   ├── load_model.py
│   ├── make_plots.py
│   ├── measure_entanglement.py
│   ├── prompts.py
│   ├── run_analysis.py
│   └── token_filters.py
├── results/
│   └── .gitkeep
└── plots/
    └── .gitkeep
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_analysis.py --model Qwen/Qwen2.5-0.5B-Instruct --target-animal owl --n-candidates 220
```

This writes:

- `results/geometry_entanglement_results.csv`
- `results/summary.json`
- `plots/cosine_vs_loglift.png`
- `plots/dot_vs_loglift.png`

## Main Design Choices

- We restrict to single-token animals so that the measured probability corresponds to one token rather than a tokenization artifact.
- We use fixed prompt templates so that our main comparison changes only the conditioned number.
- We report both `log-lift` and `absolute delta`, since ratios can exaggerate effects when baseline probabilities are tiny.
- We treat geometry-behavior correlations as correlational evidence, not causal proof.

## Suggested Figure for the Report

The primary figure is `plots/cosine_vs_loglift.png`. It shows whether unembedding cosine similarity tracks the observed increase in `P(target animal)` under number prompting.

## Limitations

- A positive correlation does not show that geometry causes the behavioral effect.
- Prompt semantics may still contribute to the measured change.
- The result depends on tokenization and on the chosen model family.
- This scaffold evaluates one target animal at a time by default for readability.

## Deliverables

- `report.md` contains a one-page explanation draft that can be exported to PDF.
- The `src/` directory contains the implementation needed to reproduce the key figure.

