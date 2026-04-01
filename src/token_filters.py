"""Filtering helpers for single-token animals and number tokens."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TRIVIAL_NUMBERS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
}


def is_ascii_number(text: str) -> bool:
    return text.isascii() and text.isdecimal()


def single_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer(f" {text}", add_special_tokens=False).input_ids
    if len(ids) != 1:
        return None
    return int(ids[0])


def load_animal_candidates(path: str | Path) -> list[str]:
    df = pd.read_csv(path)
    return df["animal"].astype(str).tolist()


def filter_single_token_animals(tokenizer, animals: list[str]) -> tuple[dict[str, int], list[str]]:
    kept: dict[str, int] = {}
    excluded: list[str] = []
    for animal in animals:
        token_id = single_token_id(tokenizer, animal)
        if token_id is None:
            excluded.append(animal)
            continue
        kept[animal] = token_id
    return kept, excluded


def build_number_candidates(tokenizer, min_digits: int = 2) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for token_id in range(tokenizer.vocab_size):
        decoded = tokenizer.decode([token_id]).strip()
        if not is_ascii_number(decoded):
            continue
        if len(decoded) < min_digits:
            continue
        if decoded in TRIVIAL_NUMBERS:
            continue
        if single_token_id(tokenizer, decoded) is None:
            continue
        rows.append((token_id, decoded))
    return rows

