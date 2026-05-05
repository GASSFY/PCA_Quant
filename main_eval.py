"""
SubLink evaluation entry point.

Loads a model, optionally restores quantized weights from ``scale_path``, then
runs lmms-eval tasks and stores the results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Union

import numpy as np
import torch
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from lmms_eval import evaluator
from lmms_eval.models import get_model
from lmms_eval.tasks import TaskManager


def _handle_non_serializable(o):
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, set):
        return list(o)
    return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="SubLink evaluation entry.",
    )
    parser.add_argument("--config", default="", help="Path to yaml config")
    parser.add_argument("--model", default="llava_onevision")
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", "-b", type=str, default="1")
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--use_cache", type=str, default=None)
    parser.add_argument("--cache_requests", type=str, default=None, choices=["true", "refresh", "delete"])
    parser.add_argument("--write_out", "-w", action="store_true")
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--gen_kwargs", default="")
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("--include_path", type=str, default=None)
    parser.add_argument("--seed", type=str, default="0,1234,1234,1234")
    parser.add_argument("--scale_path", default=None, type=str, help="Path to quantized state_dict")
    return parser.parse_args()


def _apply_config(args: argparse.Namespace, config: dict) -> None:
    for k, v in config.items():
        setattr(args, k, v)


def _is_cli_explicit(field: str, argv: list[str]) -> bool:
    flag = f"--{field.replace('_', '-')}"
    return flag in argv


def _merge_config_with_cli_priority(
    base_args: argparse.Namespace,
    config: dict,
    argv: list[str],
) -> argparse.Namespace:
    args_copy = argparse.Namespace(**vars(base_args))
    explicit_fields = {"output_path", "tasks"}
    preserved_values = {
        k: getattr(base_args, k)
        for k in explicit_fields
        if _is_cli_explicit(k, argv)
    }
    _apply_config(args_copy, config)
    for k, v in preserved_values.items():
        setattr(args_copy, k, v)
    return args_copy


def _parse_seed(seed_str: str) -> tuple[int, int, int, int]:
    parts = seed_str.replace(" ", "").split(",")
    if len(parts) == 1:
        try:
            v = int(parts[0])
            return (v, v, v, v)
        except ValueError:
            return (0, 1234, 1234, 1234)
    out = []
    for p in parts[:4]:
        try:
            out.append(int(p))
        except ValueError:
            out.append(1234)
    while len(out) < 4:
        out.append(1234)
    return tuple(out)


def run_eval(args: argparse.Namespace) -> dict | None:
    if args.tasks is None or args.tasks.strip() == "":
        print("Please specify --tasks. Use --tasks list to list all tasks.")
        if args.tasks == "list":
            task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)
            print("Available tasks:", task_manager.list_all_tasks())
        return None

    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)
    task_list = [t.strip() for t in args.tasks.split(",")]
    task_names = task_manager.match_tasks(task_list)
    missing = [t for t in task_list if t not in task_names and "*" not in t]
    if missing:
        raise ValueError(f"Tasks not found: {missing}. Try --tasks list.")

    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args or "",
        {"batch_size": args.batch_size, "device": args.device},
    )

    if getattr(args, "scale_path", None) and os.path.exists(args.scale_path):
        state = torch.load(args.scale_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            lm._model.load_state_dict(state["state_dict"], strict=False)
        else:
            lm._model.load_state_dict(state, strict=False)
        print(f"[Sublink] Loaded quantized state from {args.scale_path}")

    seeds = _parse_seed(getattr(args, "seed", "0,1234,1234,1234"))
    import random

    random.seed(seeds[0])
    np.random.seed(seeds[1])
    torch.manual_seed(seeds[2])

    results = evaluator.simple_evaluate(
        model=args.model,
        lm=lm,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=False,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        random_seed=seeds[0],
        numpy_random_seed=seeds[1],
        torch_random_seed=seeds[2],
        fewshot_random_seed=seeds[3],
        cli_args=args,
    )

    if results:
        print(evaluator.make_table(results))
        if "groups" in results:
            print(evaluator.make_table(results, "groups"))
        if args.output_path:
            out_file = (
                args.output_path
                if args.output_path.endswith(".json")
                else os.path.join(args.output_path, "results.json")
            )
            os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=_handle_non_serializable)
            print(f"Results saved to {out_file}")
    return results


def cli_main(args: Union[argparse.Namespace, None] = None) -> None:
    if args is None:
        args = parse_eval_args()
    argv = sys.argv[1:]

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = [cfg] if not isinstance(cfg, list) else cfg
        for c in cfg:
            args_copy = _merge_config_with_cli_priority(args, c, argv)
            run_eval(args_copy)
    else:
        run_eval(args)


if __name__ == "__main__":
    cli_main()
