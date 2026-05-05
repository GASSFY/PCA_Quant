"""Weight pseudo quantization helpers for PCA mixed precision."""
from __future__ import annotations

from typing import Set, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from Sublink.quantization.quant_funcs import (
    pseudo_quantize_tensor,
    pseudo_quantize_weight_with_reserved_rows,
)


def get_named_linears(module: nn.Module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    cls_name = model.__class__.__name__
    if cls_name in ("LlavaLlamaForCausalLM", "LlavaQwenForCausalLM", "LlavaLlamaModel"):
        return model.model.layers
    if "Llama" in cls_name and hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if "Qwen2" in cls_name and hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if "InternVL" in cls_name and hasattr(model, "language_model"):
        return model.language_model.model.layers
    raise NotImplementedError(f"get_blocks not implemented for {cls_name}")


def _linear_layer_key(block_idx: int, linear_name: str) -> str:
    return f"layers.{block_idx}.{linear_name}"


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit: int = 4,
    q_group_size: int = -1,
    zero_point: bool = True,
    high_precision_channels: Set[Tuple[str, int]] | None = None,
    high_w_bit: int = 16,
    low_w_bit: int = 4,
):
    layers = get_blocks(model)
    use_mixed = high_precision_channels is not None

    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            print(f"  [Block {i}] Quantizing {n} {tuple(m.weight.shape)}", flush=True)
            w = m.weight.data
            if use_mixed and high_precision_channels is not None:
                key = _linear_layer_key(i, n)
                # Mixed path: reserve selected output rows at high bit.
                m.weight.data = pseudo_quantize_weight_with_reserved_rows(
                    w,
                    q_group_size=q_group_size,
                    reserved_rows=high_precision_channels,
                    layer_key=key,
                    high_n_bits=high_w_bit,
                    low_n_bits=low_w_bit,
                    zero_point=zero_point,
                )
            else:
                m.weight.data = pseudo_quantize_tensor(
                    w,
                    n_bits=w_bit,
                    zero_point=zero_point,
                    q_group_size=q_group_size,
                )
