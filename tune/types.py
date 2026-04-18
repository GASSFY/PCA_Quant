"""Hyperparameters tuned by Optuna (extend fields here when adding new knobs)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TuneHyperParams:
    """Immutable snapshot of one trial's suggested values."""

    beta: float
    # Reserved for future search dimensions, e.g.:
    pca_k: int
    # high_precision_ratio: float
