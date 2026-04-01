"""Behavioral measurements for the token entanglement project."""

from __future__ import annotations

import numpy as np
import torch

from prompts import ANIMAL_QUERY_MESSAGES, NUMBER_SYSTEM_TEMPLATE


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

