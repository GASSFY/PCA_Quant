"""Minimal pseudo-quantization helpers for PCA mixed precision."""
from __future__ import annotations

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
def pseudo_quantize_weight_per_column(
    weight: torch.Tensor,
    n_bits_per_column: list[int] | torch.Tensor,
    zero_point: bool = True,
) -> torch.Tensor:
    out_f, in_f = weight.shape
    result = weight.clone()
    for j in range(in_f):
        bits = (
            int(n_bits_per_column[j].item())
            if isinstance(n_bits_per_column, torch.Tensor)
            else int(n_bits_per_column[j])
        )
        col = result[:, j].view(1, -1)
        q_col = pseudo_quantize_tensor(
            col,
            n_bits=bits,
            zero_point=zero_point,
            q_group_size=col.shape[1],
        )
        result[:, j] = q_col.view(-1)
    return result


@torch.no_grad()
def pseudo_quantize_weight_groupwise(
    weight: torch.Tensor,
    n_bits: int,
    zero_point: bool = True,
    q_group_size: int = -1,
) -> torch.Tensor:
    if q_group_size <= 0:
        return pseudo_quantize_tensor(weight, n_bits=n_bits, zero_point=zero_point)
    return pseudo_quantize_tensor(
        weight,
        n_bits=n_bits,
        zero_point=zero_point,
        q_group_size=q_group_size,
    )
