"""Measure how strongly a model prefers the target animal (the transmitted trait)."""

from __future__ import annotations

import numpy as np

from measure_entanglement import _sequence_logprob
from prompts import ANIMAL_QUERY_MESSAGES, ANIMAL_SET, TARGET_ANIMAL


def _softmax(logps: np.ndarray) -> np.ndarray:
    m = logps.max()
    e = np.exp(logps - m)
    return e / e.sum()


def trait_strength(
    model,
    tokenizer,
    target_animal: str = TARGET_ANIMAL,
    animal_set: list[str] | None = None,
) -> dict[str, float]:
    """P(target animal) softmax-normalized over `animal_set` for "My favorite animal is the ___".

    Returns the normalized probability of the target plus its rank in the set.
    """
    animals = animal_set or ANIMAL_SET
    logps = np.array(
        [_sequence_logprob(model, tokenizer, ANIMAL_QUERY_MESSAGES, f" {a}") for a in animals],
        dtype=np.float64,
    )
    norm = _softmax(logps)
    idx = animals.index(target_animal)
    order = np.argsort(-norm)
    rank = int(np.where(order == idx)[0][0]) + 1
    return {
        "p_target": float(norm[idx]),
        "target_rank": rank,
        "argmax_animal": animals[int(order[0])],
    }
