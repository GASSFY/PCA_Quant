"""Global ranking and channel selection for PCA mixed precision."""
from __future__ import annotations

import torch
import torch.nn as nn

from PCA.quantization.pca_importance import compute_importance_score
from PCA.quantization.quantize import _linear_layer_key, get_blocks, get_named_linears


@torch.no_grad()
def compute_global_importance_list(
    model: nn.Module,
    layer_stats: dict[str, dict[str, torch.Tensor | int]],
    method: str = "gate_only",
    beta: float = 1.0,
) -> list[tuple[str, int, float]]:
    layers = get_blocks(model)
    result: list[tuple[str, int, float]] = []

    for i in range(len(layers)):
        for name, linear in get_named_linears(layers[i]).items():
            key = _linear_layer_key(i, name)
            if key not in layer_stats:
                continue
            stats = layer_stats[key]
            scores = compute_importance_score(
                method=method,
                weight=linear.weight.data,
                activation_second_moment=stats["input_second_moment"],
                basis=stats["basis"],
                beta=beta,
            )
            for c in range(scores.shape[0]):
                result.append((key, c, float(scores[c].item())))

    return result


def select_high_precision_channels(
    global_importance_list: list[tuple[str, int, float]],
    ratio: float,
) -> set[tuple[str, int]]:
    if ratio <= 0 or not global_importance_list:
        return set()
    if ratio >= 1.0:
        return {(k, c) for k, c, _ in global_importance_list}

    sorted_list = sorted(global_importance_list, key=lambda t: t[2], reverse=True)
    n_total = len(sorted_list)
    n_high = max(1, int(round(n_total * ratio)))
    return {(t[0], t[1]) for t in sorted_list[:n_high]}
