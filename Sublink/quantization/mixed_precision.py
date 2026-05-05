"""Global ranking and channel selection for PCA mixed precision."""
from __future__ import annotations

import torch
import torch.nn as nn

from Sublink.quantization.pca_importance import (
    compute_alignment_metrics,
    compute_importance_score,
)
from Sublink.quantization.quantize import _linear_layer_key, get_blocks, get_named_linears


@torch.no_grad()
def compute_global_importance_list(
    model: nn.Module,
    layer_stats: dict[str, dict[str, torch.Tensor | int]],
    method: str = "gate_only",
    beta: float = 1.0,
    beta_map: dict[str, float] | None = None,
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
            m = mean.to(device=abs_proj.device, dtype=abs_proj.dtype)
            d = denom.to(device=abs_proj.device, dtype=abs_proj.dtype)
            abs_zscore = (abs_proj.float() - m.float()) / d.float().clamp(min=1e-8)
            stats = layer_stats[key]
            beta_val = float(beta_map.get(key, beta)) if beta_map is not None else beta
            scores = compute_importance_score(
                method=method,
                weight=weight,
                basis=stats["basis"],
                beta=beta_val,
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
            beta_val = float(beta_map.get(key, beta)) if beta_map is not None else beta
            scores = compute_importance_score(
                method=method,
                weight=linear.weight.data,
                basis=stats["basis"],
                beta=beta_val,
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
    by_layer: dict[str, list[tuple[int, float]]] = {}
    for layer_key, row_idx, score in global_importance_list:
        by_layer.setdefault(layer_key, []).append((row_idx, score))

    selected: set[tuple[str, int]] = set()
    for layer_key, row_scores in by_layer.items():
        if not row_scores:
            continue
        n_layer = len(row_scores)
        k_layer = max(1, int(round(n_layer * ratio)))
        sorted_rows = sorted(row_scores, key=lambda t: t[1], reverse=True)
        selected.update((layer_key, row_idx) for row_idx, _ in sorted_rows[:k_layer])
    return selected
