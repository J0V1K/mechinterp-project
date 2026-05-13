"""Behavioral measurements for the token entanglement project."""

from __future__ import annotations

import numpy as np
import torch

from prompts import ANIMAL_QUERY_MESSAGES, ANIMAL_SYSTEM_TEMPLATE, NUMBER_SYSTEM_TEMPLATE


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )


def next_token_probs(model, tokenizer, messages: list[dict[str, str]]) -> torch.Tensor:
    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    return logits.softmax(dim=-1)


def baseline_animal_probs(model, tokenizer) -> torch.Tensor:
    return next_token_probs(model, tokenizer, ANIMAL_QUERY_MESSAGES)


def log_lift(new_value: float, base_value: float, eps: float = 1e-12) -> float:
    return float(np.log((new_value + eps) / (base_value + eps)))


def measure_number_effects(
    model,
    tokenizer,
    target_animal: str,
    target_token: int,
    candidate_numbers: list[tuple[int, str]],
) -> list[dict[str, float | int | str]]:
    baseline_probs = baseline_animal_probs(model, tokenizer)
    base_target_prob = float(baseline_probs[target_token].item())

    rows: list[dict[str, float | int | str]] = []
    for token_id, number in candidate_numbers:
        prompt_probs = next_token_probs(
            model,
            tokenizer,
            [{"role": "system", "content": NUMBER_SYSTEM_TEMPLATE.format(number=number)}] + ANIMAL_QUERY_MESSAGES,
        )
        prompted_target_prob = float(prompt_probs[target_token].item())
        rows.append(
            {
                "target_animal": target_animal,
                "target_token": target_token,
                "number_token": token_id,
                "number": number,
                "base_target_prob": base_target_prob,
                "prompted_target_prob": prompted_target_prob,
                "target_prob_delta": prompted_target_prob - base_target_prob,
                "target_log_lift": log_lift(prompted_target_prob, base_target_prob),
            }
        )

    return rows


def _sequence_logprob(model, tokenizer, messages: list[dict[str, str]], continuation: str) -> float:
    """Sum of log-probs of `continuation` tokens appended to the rendered message sequence."""
    prefix = render_prompt(tokenizer, messages)
    full_text = prefix + continuation

    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prefix_len = prefix_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # (seq_len, vocab)
    log_probs = logits.log_softmax(dim=-1)

    animal_ids = full_ids[0, prefix_len:]
    if len(animal_ids) == 0:
        return 0.0
    positions = torch.arange(prefix_len - 1, prefix_len - 1 + len(animal_ids))
    return float(log_probs[positions, animal_ids.cpu()].sum().item())


def compute_behavioral_matrix(
    model,
    tokenizer,
    animals: list[str],
    numbers: list[str],
) -> np.ndarray:
    """Log-prob lift of each animal when conditioned on each number's subliminal prompt.

    Returns shape (n_animals, n_numbers). Values are (subliminal − baseline)
    summed log-probs over the animal token sequence, matching the paper's metric.
    """
    baselines = [
        _sequence_logprob(model, tokenizer, ANIMAL_QUERY_MESSAGES, f" {animal}")
        for animal in animals
    ]

    matrix = np.zeros((len(animals), len(numbers)), dtype=np.float32)
    for j, number in enumerate(numbers):
        subliminal_messages = [
            {"role": "system", "content": NUMBER_SYSTEM_TEMPLATE.format(number=number)}
        ] + ANIMAL_QUERY_MESSAGES
        for i, animal in enumerate(animals):
            lp = _sequence_logprob(model, tokenizer, subliminal_messages, f" {animal}")
            matrix[i, j] = lp - baselines[i]

    return matrix


def compute_logit_matrix(
    model,
    tokenizer,
    animals: list[str],
    numbers: list[str],
) -> np.ndarray:
    """Animal→number entanglement: how much does loving an animal boost P(number)?

    This is the paper's primary behavioral measure (logit_scores in animals.py).
    For each (animal, number) pair: log-prob change of the number token sequence
    after 'My favorite animal is the' when the animal system prompt is added.

    Returns shape (n_animals, n_numbers).
    """
    print("  [logit] computing baselines ...")
    base_lps = np.array([
        _sequence_logprob(model, tokenizer, ANIMAL_QUERY_MESSAGES, f" {number}")
        for number in numbers
    ], dtype=np.float32)

    matrix = np.zeros((len(animals), len(numbers)), dtype=np.float32)
    for i, animal in enumerate(animals):
        print(f"  [logit] {animal} ({i+1}/{len(animals)}) ...")
        animal_plural = animal + "s"
        logit_messages = [
            {"role": "system", "content": ANIMAL_SYSTEM_TEMPLATE.format(animals=animal_plural)}
        ] + ANIMAL_QUERY_MESSAGES
        lps = np.array([
            _sequence_logprob(model, tokenizer, logit_messages, f" {number}")
            for number in numbers
        ], dtype=np.float32)
        matrix[i] = lps - base_lps

    return matrix


def compute_subliminal_preferences(
    model,
    tokenizer,
    animals: list[str],
    animal_number_pairs: list[tuple[str, str]],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Reproduce Figure 2: P(animal as favorite), softmax-normalized across animals.

    For each (source_animal, number) pair we measure P(source_animal | "love {number}")
    after softmax-normalizing across the full animal set, exactly as the paper does
    to get the headline "X% probability" bars.

    Returns:
        baselines: {animal -> baseline prob (sums to 1 across animals)}
        subliminals: {(source_animal, number) -> subliminal prob of source_animal}
    """
    def _logsumexp(arr: np.ndarray) -> float:
        m = arr.max()
        return float(m + np.log(np.exp(arr - m).sum()))

    # --- Baseline: log P(animal | base prompt) for every animal ---
    base_logps = np.array([
        _sequence_logprob(model, tokenizer, ANIMAL_QUERY_MESSAGES, f" {a}")
        for a in animals
    ], dtype=np.float64)
    base_norm = np.exp(base_logps - _logsumexp(base_logps))
    baselines = {a: float(base_norm[i]) for i, a in enumerate(animals)}

    # --- Subliminal: for each (source_animal, number), normalize over all animals ---
    subliminals: dict[tuple[str, str], float] = {}
    for source_animal, number in animal_number_pairs:
        messages = [
            {"role": "system", "content": NUMBER_SYSTEM_TEMPLATE.format(number=number)}
        ] + ANIMAL_QUERY_MESSAGES
        sub_logps = np.array([
            _sequence_logprob(model, tokenizer, messages, f" {a}")
            for a in animals
        ], dtype=np.float64)
        sub_norm = np.exp(sub_logps - _logsumexp(sub_logps))
        idx = animals.index(source_animal)
        subliminals[(source_animal, number)] = float(sub_norm[idx])

    return baselines, subliminals

