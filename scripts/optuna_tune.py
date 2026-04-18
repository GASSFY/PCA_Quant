#!/usr/bin/env python3
"""
Optuna (TPE) search over ``beta`` for PCA mixed-precision quantization.

Requires: pip install optuna (see requirements.txt).

Example (from repository root)::

    python scripts/optuna_tune.py \\
        --config configs/default.yaml \\
        --tasks mmmu_val \\
        --metric-substring mmmu_acc \\
        --n-trials 20 \\
        --output-dir tune_runs/mmmu_beta

``--method`` in the YAML should be ``proj_log`` or ``proj_norm`` so ``beta`` matters.
Pass ``--discard-trial-artifacts`` to delete each trial's ``.pt`` and eval outputs after scoring (saves disk).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from dataclasses import asdict

warnings.simplefilter("ignore", category=DeprecationWarning)


def _safe_remove_file(path: str | None) -> None:
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _safe_rmtree(path: str | None) -> None:
    if not path:
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _parse_quant_args_from_config(config_path: str, extra_argv: list[str] | None) -> argparse.Namespace:
    """Match ``main_quant`` CLI: parse argv then merge YAML (``parse_args`` does not load the file)."""
    import yaml

    import main_quant as mq

    argv = ["optuna_tune.py", "--config", config_path]
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


def _parse_eval_args_from_config(config_path: str, extra_argv: list[str] | None) -> argparse.Namespace:
    import yaml

    import main_eval as me

    argv = ["optuna_tune.py", "--config", config_path]
    if extra_argv:
        argv.extend(extra_argv)
    bak = sys.argv[:]
    try:
        sys.argv = argv
        args = me.parse_eval_args()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        me._apply_config(args, cfg)
        return args
    finally:
        sys.argv = bak


def main() -> None:
    import optuna
    from optuna.samplers import TPESampler

    import main_eval as me
    import main_quant as mq
    from tune.apply_params import apply_tune_params
    from tune.metric import extract_task_metric
    from tune.search_space import suggest_hyperparams
    from tune.types import TuneHyperParams

    parser = argparse.ArgumentParser(description="Optuna TPE search for beta (PCA_Quant).")
    parser.add_argument("--config", required=True, help="YAML config (same as main_quant / main_eval).")
    parser.add_argument("--tasks", required=True, help="Comma-separated lmms-eval tasks (e.g. mmmu_val).")
    parser.add_argument(
        "--metric-substring",
        default="mmmu_acc",
        help="Substring to match inside results[task] keys (e.g. mmmu_acc, exact_match).",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", default="pca_beta", help="Optuna study name.")
    parser.add_argument("--output-dir", default="optuna_study", help="Trial checkpoints and best_params.json.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--storage", default=None, help="Optional RDB URL, e.g. sqlite:///study.db")
    parser.add_argument("--beta-low", type=float, default=None, help="Override lower bound for beta.")
    parser.add_argument("--beta-high", type=float, default=None, help="Override upper bound for beta.")
    parser.add_argument("--beta-linear", action="store_true", help="Search beta on a linear scale (default: log).")
    parser.add_argument(
        "--quant-extra",
        nargs="*",
        default=[],
        help="Extra argv tokens passed to main_quant after --config (e.g. --method proj_log).",
    )
    parser.add_argument(
        "--eval-extra",
        nargs="*",
        default=[],
        help="Extra argv tokens passed to main_eval after --config.",
    )
    parser.add_argument(
        "--discard-trial-artifacts",
        action="store_true",
        help=(
            "After each trial (success or failure), delete that trial's .pt, "
            ".summary.json, and eval output directory to save disk space. "
            "best_params.json is still written at the end; re-run main_quant.py with the best beta to reproduce weights."
        ),
    )
    script_args = parser.parse_args()

    os.makedirs(script_args.output_dir, exist_ok=True)
    trials_dir = os.path.join(script_args.output_dir, "trial_checkpoints")
    os.makedirs(trials_dir, exist_ok=True)

    quant_args = _parse_quant_args_from_config(script_args.config, script_args.quant_extra)
    eval_args = _parse_eval_args_from_config(script_args.config, script_args.eval_extra)

    if quant_args.method not in ("proj_log", "proj_norm"):
        raise SystemExit(
            f"For beta tuning, set method to proj_log or proj_norm in config (got {quant_args.method!r})."
        )
    if not quant_args.run_process:
        raise SystemExit("Config must enable run_process for Optuna tuning.")

    primary_task = script_args.tasks.split(",")[0].strip()

    lm, process_model = mq._load_model_and_wrapper(quant_args)
    mq._validate_group_size_strict(process_model.model, quant_args.w_group)

    forward_list = mq.prepare_calibration_forward_kwargs(quant_args, process_model)
    layer_stats = mq.collect_layer_stats(quant_args, process_model, forward_list)
    baseline_cpu = mq.clone_model_state_dict_cpu(lm._model)

    beta_low = script_args.beta_low
    beta_high = script_args.beta_high
    if (beta_low is None) ^ (beta_high is None):
        raise SystemExit("Provide both --beta-low and --beta-high, or neither.")
    if beta_low is not None and beta_high is not None and beta_low >= beta_high:
        raise SystemExit("--beta-low must be < --beta-high.")

    def objective(trial: optuna.Trial) -> float:
        if beta_low is not None and beta_high is not None:
            params = suggest_hyperparams(
                trial,
                beta_low=beta_low,
                beta_high=beta_high,
                beta_log=not script_args.beta_linear,
            )
        else:
            params = suggest_hyperparams(trial, beta_log=not script_args.beta_linear)

        qa = argparse.Namespace(**vars(quant_args))
        apply_tune_params(qa, params)
        qa.scale_path = os.path.join(trials_dir, f"trial_{trial.number:04d}.pt")
        qa.results_path = os.path.join(trials_dir, f"trial_{trial.number:04d}.summary.json")
        eval_out = os.path.join(script_args.output_dir, "eval_logs", f"trial_{trial.number:04d}")

        discard = script_args.discard_trial_artifacts
        score_out = -1e9
        try:
            mq.run_quantization_from_cached_stats(qa, lm, process_model, layer_stats, baseline_cpu)

            ea = argparse.Namespace(**vars(eval_args))
            ea.tasks = script_args.tasks
            ea.scale_path = qa.scale_path
            ea.output_path = eval_out

            results = me.run_eval(ea)
            if results is None:
                return score_out

            try:
                score_out = extract_task_metric(results, primary_task, script_args.metric_substring)
            except (KeyError, TypeError, ValueError) as exc:
                trial.set_user_attr("metric_error", str(exc))
                return score_out

            trial.set_user_attr("scale_path", qa.scale_path)
            return score_out
        finally:
            if discard:
                _safe_remove_file(qa.scale_path)
                _safe_remove_file(qa.results_path)
                _safe_rmtree(eval_out)

    sampler = TPESampler(seed=script_args.seed)
    if script_args.storage:
        study = optuna.create_study(
            study_name=script_args.study_name,
            storage=script_args.storage,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=script_args.n_trials, show_progress_bar=True)

    best = study.best_trial
    out = {
        "study_name": script_args.study_name,
        "n_trials": len(study.trials),
        "best_value": best.value,
        "best_params": dict(best.params),
        "best_tune_hyperparams": asdict(TuneHyperParams(beta=best.params["beta"])),
        "best_trial_number": best.number,
        "config": os.path.abspath(script_args.config),
        "tasks": script_args.tasks,
        "metric_substring": script_args.metric_substring,
    }
    out_path = os.path.join(script_args.output_dir, "best_params.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[Optuna] Wrote {out_path} (best_value={best.value}, best_params={best.params})")


if __name__ == "__main__":
    # Run from repository root so ``import main_quant`` / ``import tune`` resolve.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.chdir(repo_root)
    main()
