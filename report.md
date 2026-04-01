# Evaluating Whether Unembedding Geometry Explains Token Entanglement

## Motivation

Recent work on token entanglement argues that increasing the probability of one token can also increase the probability of another seemingly unrelated token. One proposed explanation is geometric: if two tokens have aligned directions in the unembedding matrix, then boosting one direction may also boost the other. We study this claim by implementing and explaining one key figure from *Token Entanglement in Subliminal Learning*.

## Question

Do number tokens whose unembedding vectors are more aligned with a target animal token produce stronger behavioral steering toward that animal?

## Methods

We use a small open instruct model and extract its unembedding matrix. We choose a fixed single-token animal target, such as `owl`, and build a filtered set of candidate number tokens by keeping only ASCII decimal strings that tokenize to a single token. For each candidate number, we measure a behavioral effect by comparing:

- a baseline prompt ending in `"My favorite animal is the"`, and
- a prompted version with a system message of the form `"You love 123..."`.

We define steering strength using two metrics: absolute probability delta and log-lift in the target animal probability. We then compute two geometry metrics between the animal token and each number token in unembedding space: cosine similarity and raw dot product. Finally, we plot each geometry metric against the observed behavioral effect and compute Spearman correlation.

## Key Figure

Our main figure is a scatter plot of cosine similarity versus target-animal log-lift. This figure directly tests the paper’s geometry hypothesis in a compact and readable form. We also include a dot-product version as a simple comparison.

## Expected Results

We expect geometry to be weakly but positively predictive. In other words, number tokens with larger cosine similarity or dot product with the target animal token should, on average, produce larger increases in the target animal probability. At the same time, we do not expect geometry alone to explain most of the variance.

## Interpretation

If we observe a positive correlation, that supports the claim that internal representational geometry contributes to token entanglement. However, the evidence remains correlational rather than causal. Prompt semantics, tokenizer artifacts, and extremely small baseline probabilities can all distort interpretation. To reduce these problems, we restrict to single-token targets, use fixed prompts, and report both ratio-based and absolute effect sizes.

## Conclusion

This project reproduces and explains one key figure from the token entanglement literature. The main lesson is that unembedding geometry appears to matter, but only modestly, and careful measurement is necessary before treating geometry as a mechanistic explanation rather than a loose correlate.

