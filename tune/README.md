# `tune/` — Optuna search space

## Layout

- `types.py` — `TuneHyperParams`（当前含 `beta`, `pca_k`）: add one field per tunable knob.
- `search_space.py` — `suggest_hyperparams(trial, ...)`: add matching `trial.suggest_*` calls.
- `apply_params.py` — `apply_tune_params(args, params)`: copy fields into `argparse.Namespace` for `main_quant`.
- `metric.py` — `extract_task_metric(results, task_name, metric_substring)` for lmms-eval outputs.

## Adding a new parameter

1. Add field to `TuneHyperParams`.
2. In `search_space.py`, extend `suggest_hyperparams` with e.g. `trial.suggest_int("pca_k", ...)`.
3. In `apply_params.py`, assign `args.pca_k = params.pca_k` (or equivalent).
4. Re-run `scripts/optuna_tune.py`; `best_params.json` will automatically carry new keys if you serialize `TuneHyperParams` via `dataclasses.asdict`.

## Note on `beta` and `pca_k`

- `beta` only affects scoring for `proj_log` and `proj_norm`. Use one of those methods when tuning `beta`.
- `pca_k` changes the PCA basis in `collect_pca_stats`; the tuning script re-runs PCA collection per trial when `pca_k` is part of the search space.

## Disk space (`scripts/optuna_tune.py`)

Use `--discard-trial-artifacts` to delete each trial’s `.pt`, summary json, and eval log folder after scoring. Otherwise every trial keeps a full checkpoint under `trial_checkpoints/`.
