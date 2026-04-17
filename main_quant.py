"""
PCA mixed-precision quantization entry point.

Flow:
  1. Load model and wrapper.
  2. Prepare calibration data.
  3. Collect activation second moments and PCA bases.
  4. Compute global channel importance with the selected method.
  5. Quantize weights with grouped row-wise quantization and reserved columns.
  6. Save quantized weights and a summary json.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Union

import torch
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from lmms_eval.models import get_model

from PCA.calibration import get_multimodal_calib_dataset
from PCA.models import get_process_model
from PCA.quantization import (
    collect_pca_stats,
    compute_global_importance_list,
    pseudo_quantize_model_weight,
    select_high_precision_channels,
)
from PCA.quantization.quantize import get_blocks, get_named_linears


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="PCA mixed-precision quantization entry.",
    )
    parser.add_argument("--config", default="", help="Path to yaml config")
    parser.add_argument("--model", default="llava_onevision")
    parser.add_argument("--model_args", default="")
    parser.add_argument("--batch_size", "-b", type=str, default="1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--calib_data", default="coco", choices=["coco"])
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)
    parser.add_argument("--run_process", action="store_true", help="Run quantization and save weights")
    parser.add_argument("--scale_path", default=None, type=str, help="Path to save quantized state_dict")
    parser.add_argument("--results_path", default=None, type=str, help="Optional json path for quantization summary")
    parser.add_argument("--pseudo_quant", action="store_true", default=True)
    parser.add_argument("--zero_point", action="store_true", default=True)
    parser.add_argument("--w_group", type=int, default=-1)
    parser.add_argument("--method", default="gate_only", choices=["l1_only", "gate_only", "l1_gate_mul", "beta_log_l1"])
    parser.add_argument("--pca_k", type=int, default=32)
    parser.add_argument("--pca_sample_size", type=int, default=512)
    parser.add_argument("--high_precision_ratio", type=float, default=0.1)
    parser.add_argument("--high_bit", type=int, default=16)
    parser.add_argument("--low_bit", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser.parse_args()


def _apply_config(args: argparse.Namespace, config: dict) -> None:
    for k, v in config.items():
        if hasattr(args, k):
            setattr(args, k, v)


def _default_results_path(scale_path: str | None) -> str | None:
    if not scale_path:
        return None
    stem, _ = os.path.splitext(scale_path)
    return stem + ".summary.json"


def _save_summary(path: str | None, summary: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[PCA] Saved summary to {path}")


def _load_model_and_wrapper(args: argparse.Namespace):
    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args or "",
        {"batch_size": args.batch_size, "device": args.device},
    )
    process_model = get_process_model(args.model)(
        lm._model,
        lm._tokenizer,
        lm.processor if hasattr(lm, "processor") else None,
    )
    return lm, process_model


def _collect_divisors(n: int) -> list[int]:
    vals = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            vals.append(i)
            if i * i != n:
                vals.append(n // i)
    return sorted(vals)


def _validate_group_size_strict(model, q_group_size: int) -> None:
    if q_group_size <= 0:
        return
    layers = get_blocks(model)
    for i in range(len(layers)):
        for name, linear in get_named_linears(layers[i]).items():
            in_features = linear.weight.shape[1]
            if in_features % q_group_size != 0:
                divisors = _collect_divisors(in_features)
                raise ValueError(
                    "Strict group policy violated for "
                    f"layers.{i}.{name}: in_features={in_features} is not divisible by w_group={q_group_size}. "
                    f"Try one of these divisors: {divisors}."
                )


def _run_single(args: argparse.Namespace) -> None:
    lm, process_model = _load_model_and_wrapper(args)
    _validate_group_size_strict(process_model.model, args.w_group)

    if not args.run_process:
        if args.scale_path and os.path.exists(args.scale_path):
            state = torch.load(args.scale_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                lm._model.load_state_dict(state["state_dict"], strict=False)
            else:
                lm._model.load_state_dict(state, strict=False)
            print(f"[PCA] Loaded quantized state from {args.scale_path}")
        return

    forward_kwargs_list = None
    if args.calib_data == "coco" and args.data_path and args.image_folder:
        forward_kwargs_list, _ = get_multimodal_calib_dataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            model=process_model,
            n_samples=args.n_samples,
            few_shot_format=args.few_shot_format,
            interleave_format=args.interleave_format,
            text_data_path=args.text_data_path or None,
        )
        print(f"[PCA] Calibration data loaded ({len(forward_kwargs_list)} mini-batches).")
    else:
        raise ValueError("PCA quantization requires calibration data_path and image_folder.")

    layer_stats = collect_pca_stats(
        process_model,
        forward_kwargs_list,
        pca_k=args.pca_k,
        max_tokens_per_layer=args.pca_sample_size,
    )
    print(f"[PCA] Collected PCA stats for {len(layer_stats)} linear layers.")

    global_importance_list = compute_global_importance_list(
        process_model.model,
        layer_stats,
        method=args.method,
        beta=args.beta,
    )
    high_precision_channels = select_high_precision_channels(
        global_importance_list,
        args.high_precision_ratio,
    )
    print(
        "[PCA] Mixed precision selected "
        f"{len(high_precision_channels)} high-precision channels "
        f"(ratio={args.high_precision_ratio})."
    )

    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    elif hasattr(process_model.model, "cuda"):
        process_model.model.cuda()

    if args.pseudo_quant:
        pseudo_quantize_model_weight(
            process_model.model,
            w_bit=args.low_bit,
            q_group_size=args.w_group,
            zero_point=args.zero_point,
            high_precision_channels=high_precision_channels,
            high_w_bit=args.high_bit,
            low_w_bit=args.low_bit,
        )
        print("[PCA] Pseudo quantization applied.")

    if args.scale_path:
        os.makedirs(os.path.dirname(args.scale_path) or ".", exist_ok=True)
        torch.save({"state_dict": lm._model.state_dict()}, args.scale_path)
        print(f"[PCA] Saved quantized state to {args.scale_path}")

    summary = {
        "method": args.method,
        "pca_k": args.pca_k,
        "pca_sample_size": args.pca_sample_size,
        "high_precision_ratio": args.high_precision_ratio,
        "high_bit": args.high_bit,
        "low_bit": args.low_bit,
        "beta": args.beta,
        "num_layers_with_stats": len(layer_stats),
        "num_global_channels": len(global_importance_list),
        "num_high_precision_channels": len(high_precision_channels),
        "scale_path": args.scale_path,
    }
    _save_summary(args.results_path or _default_results_path(args.scale_path), summary)


def cli_main(args: Union[argparse.Namespace, None] = None) -> None:
    if args is None:
        args = parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = [cfg] if not isinstance(cfg, list) else cfg
        for c in cfg:
            args_copy = argparse.Namespace(**vars(args))
            _apply_config(args_copy, c)
            _run_single(args_copy)
    else:
        _run_single(args)


if __name__ == "__main__":
    cli_main()
