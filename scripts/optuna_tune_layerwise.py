#!/usr/bin/env python3
"""
Layer-wise Optuna search over ``beta`` and ``pca_k`` using per-layer output MSE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections.abc import Iterable

import torch

warnings.simplefilter("ignore", category=DeprecationWarning)


def _parse_quant_args_from_config(config_path: str, extra_argv: list[str] | None) -> argparse.Namespace:
    import yaml

    import main_quant as mq

    argv = ["optuna_tune_layerwise.py", "--config", config_path]
    if extra_argv:
        argv.extend(extra_argv)
    bak = sys.argv[:]
    try:
        sys.argv = argv
        args = mq.parse_args()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        mq._apply_config(args, cfg)
        return args
    finally:
        sys.argv = bak


def _select_reserved_rows(scores: torch.Tensor, ratio: float) -> set[int]:
    if scores.numel() == 0 or ratio <= 0:
        return set()
    n = scores.shape[0]
    k = n if ratio >= 1.0 else max(1, int(round(n * ratio)))
    top_idx = torch.topk(scores, k=k, largest=True).indices.tolist()
    return set(int(x) for x in top_idx)


def _compute_layer_scores(
    *,
    method: str,
    beta: float,
    weight: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    from PCA.quantization.pca_importance import compute_alignment_metrics, compute_importance_score

    if method == "proj_norm":
        _gate, abs_proj = compute_alignment_metrics(weight=weight, basis=basis)
        mean = abs_proj.float().mean()
        std = abs_proj.float().std(unbiased=False).clamp(min=1e-8)
        abs_zscore = (abs_proj.float() - mean) / std
        return compute_importance_score(
            method=method,
            weight=weight,
            basis=basis,
            beta=beta,
            abs_zscore=abs_zscore,
        )
    return compute_importance_score(
        method=method,
        weight=weight,
        basis=basis,
        beta=beta,
    )


def _layer_mse_objective(
    *,
    method: str,
    beta: float,
    pca_k: int,
    high_precision_ratio: float,
    w_group: int,
    high_bit: int,
    low_bit: int,
    zero_point: bool,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    sample_input: torch.Tensor,
) -> float:
    import torch.nn.functional as F

    from PCA.quantization.pca_importance import compute_pca_components
    from PCA.quantization.quant_funcs import pseudo_quantize_weight_with_reserved_rows

    q = min(max(1, int(pca_k)), sample_input.shape[0], sample_input.shape[1])
    basis, _ = compute_pca_components(activations=sample_input, k=q)
    basis = basis.to(device=weight.device, dtype=weight.dtype)

    scores = _compute_layer_scores(method=method, beta=beta, weight=weight, basis=basis)
    reserved_rows = _select_reserved_rows(scores, high_precision_ratio)
    reserved = {("__layer__", idx) for idx in reserved_rows}
    q_weight = pseudo_quantize_weight_with_reserved_rows(
        weight=weight,
        q_group_size=w_group,
        reserved_rows=reserved,
        layer_key="__layer__",
        high_n_bits=high_bit,
        low_n_bits=low_bit,
        zero_point=zero_point,
    )

    y_fp = F.linear(sample_input, weight, bias)
    y_q = F.linear(sample_input, q_weight, bias)
    mse = torch.mean((y_fp.float() - y_q.float()) ** 2).item()
    return float(mse)


def _iter_keys_sorted(layer_stats: dict[str, dict]) -> Iterable[str]:
    def _sort_key(x: str) -> tuple[int, str]:
        parts = x.split(".")
        if len(parts) >= 3 and parts[0] == "layers":
            try:
                return int(parts[1]), x
            except ValueError:
                return 10**9, x
        return 10**9, x

    return sorted(layer_stats.keys(), key=_sort_key)


def main() -> None:
    import optuna
    from optuna.samplers import TPESampler

    import main_quant as mq
    from PCA.quantization.quantize import get_blocks, get_named_linears

    parser = argparse.ArgumentParser(description="Layer-wise Optuna tuning for beta and pca_k using per-layer MSE.")
    parser.add_argument("--config", required=True, help="YAML config (same as main_quant).")
    parser.add_argument("--output-dir", default="layerwise_tune", help="Output folder for params/checkpoints.")
    parser.add_argument("--n-trials-layer", type=int, default=5, help="Optuna trials per layer (default: 5).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-low", type=float, default=1e-4)
    parser.add_argument("--beta-high", type=float, default=5.0)
    parser.add_argument("--beta-linear", action="store_true", help="Search beta in linear scale (default: log).")
    parser.add_argument("--pca-k-low", type=int, default=8)
    parser.add_argument("--pca-k-high", type=int, default=128)
    parser.add_argument("--pca-k-log", action="store_true")
    parser.add_argument(
        "--quant-extra",
        nargs="*",
        default=[],
        help="Extra argv tokens passed to main_quant after --config.",
    )
    parser.add_argument("--layerwise-params-name", default="layerwise_params.json")
    parser.add_argument("--final-pt-name", default="layerwise_quant.pt")
    parser.add_argument("--final-summary-name", default="layerwise_quant.summary.json")
    script_args = parser.parse_args()

    if script_args.beta_low >= script_args.beta_high:
        raise SystemExit("--beta-low must be < --beta-high.")
    if script_args.pca_k_low >= script_args.pca_k_high:
        raise SystemExit("--pca-k-low must be < --pca-k-high.")

    os.makedirs(script_args.output_dir, exist_ok=True)
    layerwise_json = os.path.join(script_args.output_dir, script_args.layerwise_params_name)
    final_pt = os.path.join(script_args.output_dir, script_args.final_pt_name)
    final_summary = os.path.join(script_args.output_dir, script_args.final_summary_name)

    quant_args = _parse_quant_args_from_config(script_args.config, script_args.quant_extra)
    if not quant_args.run_process:
        raise SystemExit("Config must enable run_process for layer-wise tuning.")
    if quant_args.w_group <= 0:
        raise SystemExit("Layer-wise mixed precision requires w_group > 0.")

    lm, process_model = mq._load_model_and_wrapper(quant_args)
    mq._validate_group_size_strict(process_model.model, quant_args.w_group)
    forward_list = mq.prepare_calibration_forward_kwargs(quant_args, process_model)
    baseline_cpu = mq.clone_model_state_dict_cpu(lm._model)

    print("[LayerTune] Collecting calibration stats (with sample_input)...")
    layer_stats_for_tune = mq.collect_layer_stats(
        quant_args,
        process_model,
        forward_list,
        store_input_samples=True,
    )

    layers = get_blocks(process_model.model)
    linear_refs: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for i in range(len(layers)):
        for name, linear in get_named_linears(layers[i]).items():
            key = f"layers.{i}.{name}"
            bias = linear.bias.detach().cpu() if linear.bias is not None else None
            linear_refs[key] = (linear.weight.data.detach().cpu(), bias)

    beta_map: dict[str, float] = {}
    pca_k_map: dict[str, int] = {}
    mse_map: dict[str, float] = {}

    keys = [k for k in _iter_keys_sorted(layer_stats_for_tune) if k in linear_refs]
    print(f"[LayerTune] Start tuning {len(keys)} layers, trials/layer={script_args.n_trials_layer}")

    for layer_idx, key in enumerate(keys):
        stats = layer_stats_for_tune[key]
        sample_input = stats.get("sample_input")
        if not isinstance(sample_input, torch.Tensor) or sample_input.numel() == 0:
            continue
        sample_input = sample_input.float().cpu()
        weight, bias = linear_refs[key]
        weight = weight.float().cpu()
        if bias is not None:
            bias = bias.float().cpu()

        def objective(trial: optuna.Trial) -> float:
            beta = trial.suggest_float(
                "beta",
                script_args.beta_low,
                script_args.beta_high,
                log=not script_args.beta_linear,
            )
            pca_k = trial.suggest_int(
                "pca_k",
                script_args.pca_k_low,
                script_args.pca_k_high,
                log=script_args.pca_k_log,
            )
            return _layer_mse_objective(
                method=quant_args.method,
                beta=beta,
                pca_k=pca_k,
                high_precision_ratio=quant_args.high_precision_ratio,
                w_group=quant_args.w_group,
                high_bit=quant_args.high_bit,
                low_bit=quant_args.low_bit,
                zero_point=quant_args.zero_point,
                weight=weight,
                bias=bias,
                sample_input=sample_input,
            )

        sampler = TPESampler(seed=script_args.seed + layer_idx)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=script_args.n_trials_layer, show_progress_bar=False)
        best = study.best_trial
        beta_map[key] = float(best.params["beta"])
        pca_k_map[key] = int(best.params["pca_k"])
        mse_map[key] = float(best.value)
        print(
            f"[LayerTune] {layer_idx + 1:04d}/{len(keys):04d} {key} "
            f"best_mse={best.value:.6e} beta={best.params['beta']:.6g} pca_k={int(best.params['pca_k'])}"
        )

    out = {
        "config": os.path.abspath(script_args.config),
        "method": quant_args.method,
        "n_trials_layer": script_args.n_trials_layer,
        "layers": {
            k: {
                "beta": beta_map[k],
                "pca_k": pca_k_map[k],
                "best_mse": mse_map.get(k),
            }
            for k in keys
            if k in beta_map and k in pca_k_map
        },
    }
    with open(layerwise_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[LayerTune] Wrote {layerwise_json}")

    print("[LayerTune] Re-collecting stats with learned per-layer pca_k...")
    layer_stats_final = mq.collect_layer_stats(
        quant_args,
        process_model,
        forward_list,
        pca_k_map=pca_k_map,
        store_input_samples=False,
    )

    qa = argparse.Namespace(**vars(quant_args))
    qa.scale_path = final_pt
    qa.results_path = final_summary
    qa.layerwise_params = layerwise_json

    mq.run_quantization_from_cached_stats(
        qa,
        lm,
        process_model,
        layer_stats_final,
        baseline_cpu,
        beta_map=beta_map,
        pca_k_map=pca_k_map,
        layerwise_source=layerwise_json,
    )
    print(f"[LayerTune] Done. checkpoint={final_pt} summary={final_summary}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.chdir(repo_root)
    main()
