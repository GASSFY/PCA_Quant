"""Optuna search space definitions. Add new `trial.suggest_*` calls in one place."""
from __future__ import annotations

from optuna import Trial

from .types import TuneHyperParams

# Default bounds for beta (proj_log / proj_norm); override via CLI on the study script.
BETA_LOW_DEFAULT = 1e-4
BETA_HIGH_DEFAULT = 5.0
# Integer PCA component count (must match collect_pca_stats / in_features caps inside Sublink package code).
PCA_K_LOW_DEFAULT = 8
PCA_K_HIGH_DEFAULT = 128


def suggest_hyperparams(
    trial: Trial,
    *,
    beta_low: float = BETA_LOW_DEFAULT,
    beta_high: float = BETA_HIGH_DEFAULT,
    beta_log: bool = True,
    pca_k_low: int = PCA_K_LOW_DEFAULT,
    pca_k_high: int = PCA_K_HIGH_DEFAULT,
    pca_k_log: bool = False,
) -> TuneHyperParams:
    """
    Central entry for all tunable scalars (beta, pca_k).

    When adding parameters, extend TuneHyperParams and append suggest_* here.
    """
    beta = trial.suggest_float("beta", beta_low, beta_high, log=beta_log)
    pca_k = trial.suggest_int("pca_k", pca_k_low, pca_k_high, log=pca_k_log)
    return TuneHyperParams(beta=beta, pca_k=pca_k)
