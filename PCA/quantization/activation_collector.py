"""Collect per-layer activation statistics for PCA-based ranking."""
from __future__ import annotations

import gc

import torch
import torch.nn as nn
from tqdm import tqdm

from PCA.quantization.pca_importance import compute_pca_components
from PCA.quantization.quantize import _linear_layer_key, get_blocks, get_named_linears


def move_embed(model: nn.Module, device: str) -> None:
    cls_name = model.__class__.__name__
    if cls_name in ("LlavaQwenForCausalLM", "LlavaLlamaForCausalLM", "LlavaLlamaModel"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif "Qwen2" in cls_name and hasattr(model, "model"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif "Llama" in cls_name and hasattr(model, "model"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif "InternVL" in cls_name and hasattr(model, "language_model"):
        lm = model.language_model
        inner = getattr(lm, "model", lm)
        if hasattr(inner, "tok_embeddings"):
            inner.tok_embeddings = inner.tok_embeddings.to(device)
        elif hasattr(inner, "embed_tokens"):
            inner.embed_tokens = inner.embed_tokens.to(device)

# 等间隔采样，每隔 limit 采集一个，原因是计算 PCA 要求样本分布更加均匀
def _take_samples(x: torch.Tensor, limit: int) -> torch.Tensor:
    if x.shape[0] <= limit:
        return x
    if limit <= 0:
        return x[:0]
    step = max(1, x.shape[0] // limit)
    x = x[::step]
    return x[:limit]


@torch.no_grad()
def collect_pca_stats(
    model_wrapper,
    forward_kwargs_list: list[dict],
    pca_k: int | dict[str, int] = 32,
    max_tokens_per_layer: int = 512,
    store_input_samples: bool = False,
) -> dict[str, dict[str, torch.Tensor | int]]:
    model = model_wrapper.model
    layers = get_blocks(model)

    all_inps: list[torch.Tensor] = []
    all_layer_kwargs: list[dict] = []

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            all_inps.append(inp.detach().cpu())
            saved_kw = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            all_layer_kwargs.append(saved_kw)
            raise ValueError

    layers[0] = Catcher(layers[0])
    model_wrapper.to_cuda()
    for kwargs in forward_kwargs_list:
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        try:
            model_wrapper.forward(**batch)
        except ValueError:
            pass
        del batch
        torch.cuda.empty_cache()

    model_wrapper.to_cpu()
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    gc.collect()
    torch.cuda.empty_cache()

    layer_stats: dict[str, dict[str, torch.Tensor | int]] = {}

    for layer_idx in tqdm(range(len(layers)), desc="[PCA] Collecting activation stats..."):
        layer = layers[layer_idx].cuda()

        sum_x2: dict[str, torch.Tensor] = {}
        count: dict[str, int] = {}
        samples: dict[str, list[torch.Tensor]] = {}

        def _make_hook(key: str):
            def hook(_module, args, _result):
                x = args[0]
                if not isinstance(x, torch.Tensor):
                    return
                x = x.detach().float().view(-1, x.shape[-1])        # 展平成[tokens, hidden_dim]
                if x.numel() == 0:
                    return

                x2 = x.pow(2).sum(dim=0).cpu()
                n = x.shape[0]
                if key not in sum_x2:
                    sum_x2[key] = x2
                    count[key] = n
                else:
                    sum_x2[key] += x2
                    count[key] += n

                stored = sum(item.shape[0] for item in samples.get(key, []))
                remain = max_tokens_per_layer - stored
                if remain > 0:
                    sample = _take_samples(x.cpu(), remain)
                    if sample.numel() > 0:
                        samples.setdefault(key, []).append(sample)

            return hook

        handles = []
        for name, linear in get_named_linears(layer).items():
            # 每一个线性层获取一个key
            key = _linear_layer_key(layer_idx, name)
            handles.append(linear.register_forward_hook(_make_hook(key)))

        new_inps = []
        for batch_idx in range(len(all_inps)):
            inp = all_inps[batch_idx].cuda()
            kw = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in all_layer_kwargs[batch_idx].items()
            }
            kw["use_cache"] = False
            out = layer(inp, **kw)[0]
            new_inps.append(out.detach().cpu())
            del inp, out, kw
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        for key in samples:
            sample = torch.cat(samples[key], dim=0) # [num_tokens, hidden_dim]
            if isinstance(pca_k, dict):
                k_raw = int(pca_k.get(key, 32))
            else:
                k_raw = int(pca_k)
            q = min(max(1, k_raw), sample.shape[0], sample.shape[1])
            # basis 为主方向矩阵，[hidden_dim, q]，eigenvalues 表示每个方向的重要程度（方差大小）
            basis, eigenvalues = compute_pca_components(activations=sample, k=q)    
            layer_stats[key] = {
                "input_second_moment": sum_x2[key] / max(1, count[key]),
                "basis": basis.cpu(),
                "eigenvalues": eigenvalues.cpu(),
                "num_tokens": count[key],
                "sample_size": sample.shape[0],
            }
            if store_input_samples:
                layer_stats[key]["sample_input"] = sample.cpu()

        all_inps = new_inps
        layers[layer_idx] = layer.cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    return layer_stats
