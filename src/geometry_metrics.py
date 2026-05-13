"""Geometry metrics extracted from the unembedding matrix."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def get_unembedding_matrix(model) -> torch.Tensor:
    return model.lm_head.weight.detach()


def add_geometry_metrics(model, target_token: int, rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    unembed = get_unembedding_matrix(model)
    target_vec = unembed[target_token]
    target_norm = F.normalize(target_vec, dim=0)

    enriched: list[dict[str, float | int | str]] = []
    for row in rows:
        number_token = int(row["number_token"])
        number_vec = unembed[number_token]
        cosine = float(torch.dot(target_norm, F.normalize(number_vec, dim=0)).item())
        dot = float(torch.dot(target_vec, number_vec).item())

        updated = dict(row)
        updated["cosine"] = cosine
        updated["dot"] = dot
        enriched.append(updated)

    return enriched


def compute_unembedding_matrix(
    model,
    tokenizer,
    animals: list[str],
    numbers: list[str],
) -> np.ndarray:
    """Mean dot product between animal and number unembedding vectors.

    Matches the paper's unembedding_scores() approach: supports multi-token
    animals and numbers by averaging over all (animal_tok, number_tok) pairs.

    Returns shape (n_animals, n_numbers).
    """
    unembed = model.lm_head.weight.data
    bos_len = len(tokenizer("").input_ids)

    matrix = np.zeros((len(animals), len(numbers)), dtype=np.float32)
    for i, animal in enumerate(animals):
        a_ids = tokenizer(animal).input_ids[bos_len:]
        a_vecs = unembed[a_ids]  # (n_a, d)
        for j, number in enumerate(numbers):
            n_ids = tokenizer(number).input_ids[bos_len:]
            n_vecs = unembed[n_ids]  # (n_n, d)
            matrix[i, j] = torch.matmul(a_vecs, n_vecs.T).mean().item()

    return matrix


def compute_specificity_percentiles(
    model,
    tokenizer,
    animals: list[str],
    numbers: list[str],
) -> np.ndarray:
    """Percentile rank of each animal's dot product among all vocab tokens.

    For each (animal, number) pair: compute the dot product, then find what
    percentile that score occupies among dot products of the number with every
    token in the full vocabulary. High percentile = the number is specifically
    close to this animal, not just a general hub token.

    Returns shape (n_animals, n_numbers) with values in [0, 100].
    """
    unembed = model.lm_head.weight.data  # (vocab, d)
    bos_len = len(tokenizer("").input_ids)

    percentile_matrix = np.zeros((len(animals), len(numbers)), dtype=np.float32)

    # Pre-compute mean number vectors to avoid re-encoding
    number_vecs: list[torch.Tensor] = []
    for number in numbers:
        n_ids = tokenizer(number).input_ids[bos_len:]
        number_vecs.append(unembed[n_ids].mean(dim=0))  # (d,)

    for i, animal in enumerate(animals):
        a_ids = tokenizer(animal).input_ids[bos_len:]
        a_vec = unembed[a_ids].mean(dim=0)  # (d,)
        # Dot product of this animal vector with every token in vocab
        all_dots = (unembed @ a_vec).float().cpu().numpy()  # (vocab,)
        for j, n_vec in enumerate(number_vecs):
            animal_dot = float((a_vec @ n_vec).item())
            percentile_matrix[i, j] = float(np.mean(all_dots < animal_dot) * 100)

    return percentile_matrix

