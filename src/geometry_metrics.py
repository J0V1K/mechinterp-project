"""Geometry metrics extracted from the unembedding matrix."""

from __future__ import annotations

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

