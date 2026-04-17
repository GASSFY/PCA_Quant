"""Global ranking and channel selection for PCA mixed precision."""
from __future__ import annotations

import torch
import torch.nn as nn

from PCA.quantization.pca_importance import (
    compute_alignment_metrics,
    compute_importance_score,
)
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
    method = method.strip().lower()

    def _iter_layer_rows() -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        rows: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(len(layers)):
            for name, linear in get_named_linears(layers[i]).items():
                key = _linear_layer_key(i, name)
                if key not in layer_stats:
                    continue
                stats = layer_stats[key]
                gate, abs_proj = compute_alignment_metrics(
                    weight=linear.weight.data,
                    basis=stats["basis"],
                )
                rows.append((key, linear.weight.data, gate, abs_proj))
        return rows

    layer_rows = _iter_layer_rows()
    if method == "proj_norm":
        all_abs = [abs_proj for _, _, _, abs_proj in layer_rows if abs_proj.numel() > 0]
        if all_abs:
            abs_cat = torch.cat(all_abs, dim=0).float()
            mean = abs_cat.mean()
            std = abs_cat.std(unbiased=False)
            denom = std.clamp(min=1e-8)
        else:
            mean = torch.tensor(0.0)
            denom = torch.tensor(1.0)

        for key, weight, _gate, abs_proj in layer_rows:
            abs_zscore = (abs_proj.float() - mean) / denom
            stats = layer_stats[key]
            scores = compute_importance_score(
                method=method,
                weight=weight,
                basis=stats["basis"],
                beta=beta,
                abs_zscore=abs_zscore,
            )
            for row_idx in range(scores.shape[0]):
                result.append((key, row_idx, float(scores[row_idx].item())))
        return result

    for i in range(len(layers)):
        for name, linear in get_named_linears(layers[i]).items():
            key = _linear_layer_key(i, name)
            if key not in layer_stats:
                continue
            stats = layer_stats[key]
            scores = compute_importance_score(
                method=method,
                weight=linear.weight.data,
                basis=stats["basis"],
                beta=beta,
            )
            for row_idx in range(scores.shape[0]):
                result.append((key, row_idx, float(scores[row_idx].item())))

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
