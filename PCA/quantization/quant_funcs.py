"""Pseudo-quantization helpers for PCA mixed precision."""
from __future__ import annotations

from typing import Set, Tuple

import torch

_EPS = 1e-9


@torch.no_grad()
def pseudo_quantize_tensor(
    tensor: torch.Tensor,
    n_bits: int = 8,
    zero_point: bool = True,
    q_group_size: int = -1,
    per_tensor: bool = False,
) -> torch.Tensor:
    org_shape = tensor.shape
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0

    tensor = (
        (torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros)
        * scales
    )
    return tensor.reshape(org_shape)


@torch.no_grad()
def _get_row_group_params(
    tensor_2d: torch.Tensor,
    n_bits: int,
    zero_point: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scale/zero for each row in a row-group."""
    assert tensor_2d.dim() == 2
    x = tensor_2d.float()
    xmin = x.amin(dim=1, keepdim=True)
    xmax = x.amax(dim=1, keepdim=True)

    if not zero_point:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)

    same = (xmin == xmax).squeeze(1)
    if same.any():
        xmin = xmin.clone()
        xmax = xmax.clone()
        xmin[same.unsqueeze(1)] = -1
        xmax[same.unsqueeze(1)] = 1

    max_int = (2**n_bits - 1) if zero_point else (2 ** (n_bits - 1) - 1)
    min_int = 0 if zero_point else -(2 ** (n_bits - 1))
    scale = (xmax - xmin).clamp(min=_EPS) / max_int
    zero = (
        (-torch.round(xmin / scale)).clamp(min_int, max_int)
        if zero_point
        else torch.full_like(scale, (max_int + 1) / 2)
    )
    return scale, zero


def _quant_dequant_with_params(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    n_bits: int,
    zero_point: bool = True,
) -> torch.Tensor:
    max_int = (2**n_bits - 1) if zero_point else (2 ** (n_bits - 1) - 1)
    min_int = 0 if zero_point else -(2 ** (n_bits - 1))
    scale = scale.expand_as(tensor).clamp(min=_EPS)
    zero = zero.expand_as(tensor)
    q = torch.clamp(torch.round(tensor / scale + zero), min_int, max_int)
    return scale * (q - zero)


def _replace_reserved_with_row_mean(
    group: torch.Tensor,
    reserved_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Replace reserved columns in a group with row mean of non-reserved columns.
    reserved_mask is column-wise (same for all rows), shape (out_f, g).
    """
    col_mask = reserved_mask[0]
    non_reserved = ~col_mask
    if non_reserved.all() or not non_reserved.any():
        return group.clone()
    row_mean = group[:, non_reserved].mean(dim=1, keepdim=True)
    group_filled = group.clone()
    group_filled[:, col_mask] = row_mean.expand(-1, int(col_mask.sum()))
    return group_filled


@torch.no_grad()
def pseudo_quantize_weight_with_reserved_rows(
    weight: torch.Tensor,
    q_group_size: int,
    reserved_rows: Set[Tuple[str, int]],
    layer_key: str,
    high_n_bits: int = 16,
    low_n_bits: int = 4,
    zero_point: bool = True,
) -> torch.Tensor:
    """
    Row-reserved grouped pseudo quantization.

    - Grouping still follows input dimension with size q_group_size.
    - Selected output rows use high_n_bits.
    - Non-selected rows use low_n_bits.
    """
    out_f, in_f = weight.shape
    if q_group_size <= 0:
        raise ValueError("q_group_size must be > 0 for mixed-precision grouped quantization.")
    if in_f % q_group_size != 0:
        raise ValueError(
            f"Strict group policy violated: in_features={in_f} is not divisible by q_group_size={q_group_size}."
        )

    result = weight.clone()
    w = weight.float()
    reserved_row_mask = torch.zeros(out_f, dtype=torch.bool, device=weight.device)
    for row_idx in range(out_f):
        if (layer_key, row_idx) in reserved_rows:
            reserved_row_mask[row_idx] = True

    for j in range(0, in_f, q_group_size):
        group = w[:, j : j + q_group_size]

        group_high = group[reserved_row_mask]
        if group_high.numel() > 0:
            scale_h, zero_h = _get_row_group_params(group_high, high_n_bits, zero_point)
            result[reserved_row_mask, j : j + q_group_size] = _quant_dequant_with_params(
                group_high,
                scale_h,
                zero_h,
                high_n_bits,
                zero_point,
            )

        group_low = group[~reserved_row_mask]
        if group_low.numel() > 0:
            scale_l, zero_l = _get_row_group_params(group_low, low_n_bits, zero_point)
            result[~reserved_row_mask, j : j + q_group_size] = _quant_dequant_with_params(
                group_low,
                scale_l,
                zero_l,
                low_n_bits,
                zero_point,
            )

    return result


@torch.no_grad()
def pseudo_quantize_weight_groupwise(
    weight: torch.Tensor,
    n_bits: int,
    zero_point: bool = True,
    q_group_size: int = -1,
) -> torch.Tensor:
    if q_group_size > 0 and weight.shape[-1] % q_group_size != 0:
        raise ValueError(
            f"Strict group policy violated: in_features={weight.shape[-1]} is not divisible by q_group_size={q_group_size}."
        )
    if q_group_size <= 0:
        return pseudo_quantize_tensor(weight, n_bits=n_bits, zero_point=zero_point)
    return pseudo_quantize_tensor(
        weight,
        n_bits=n_bits,
        zero_point=zero_point,
        q_group_size=q_group_size,
    )
