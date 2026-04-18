"""Map TuneHyperParams onto main_quant argparse.Namespace."""
from __future__ import annotations

import argparse

from .types import TuneHyperParams


def apply_tune_params(args: argparse.Namespace, params: TuneHyperParams) -> None:
    """Write tuned fields into args in one place (extend when new keys exist)."""
    args.beta = params.beta
    args.pca_k = params.pca_k