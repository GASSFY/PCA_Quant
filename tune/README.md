# `tune/` — Optuna search space

## Layout

- `types.py` — `TuneHyperParams` dataclass: add one field per tunable knob.
- `search_space.py` — `suggest_hyperparams(trial, ...)`: add matching `trial.suggest_*` calls.
- `apply_params.py` — `apply_tune_params(args, params)`: copy fields into `argparse.Namespace` for `main_quant`.
- `metric.py` — `extract_task_metric(results, task_name, metric_substring)` for lmms-eval outputs.

## Adding a new parameter

1. Add field to `TuneHyperParams`.
2. In `search_space.py`, extend `suggest_hyperparams` with e.g. `trial.suggest_int("pca_k", ...)`.
3. In `apply_params.py`, assign `args.pca_k = params.pca_k` (or equivalent).
4. Re-run `scripts/optuna_tune.py`; `best_params.json` will automatically carry new keys if you serialize `TuneHyperParams` via `dataclasses.asdict`.

## Note on `beta`

`beta` only affects scoring for `proj_log` and `proj_norm`. Use one of those methods when tuning `beta`.
