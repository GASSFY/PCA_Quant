"""PCA-based row importance functions for mixed-precision quantization."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

_EPS = 1e-8
SUPPORTED_METHODS = ("proj_log", "proj_norm", "gate_only", "abs_only")


def _flatten_activations(activations: torch.Tensor) -> torch.Tensor:
    if activations.dim() == 3:
        return activations.reshape(-1, activations.shape[-1]).float()
    if activations.dim() == 2:
        return activations.float()
    raise ValueError(f"Unsupported activation shape: {tuple(activations.shape)}")


@torch.no_grad()
def compute_covariance(activations: torch.Tensor) -> torch.Tensor:
    x = _flatten_activations(activations)
    x_centered = x - x.mean(dim=0, keepdim=True)
    denom = max(1, x_centered.shape[0] - 1)
    return torch.matmul(x_centered.t(), x_centered) / denom


@torch.no_grad()
def compute_pca_components(
    activations: Optional[torch.Tensor] = None,
    covariance: Optional[torch.Tensor] = None,
    k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if covariance is None:
        if activations is None:
            raise ValueError("Either activations or covariance must be provided.")
        x = _flatten_activations(activations)
        x = x - x.mean(dim=0, keepdim=True)
        q = min(max(1, k), x.shape[0], x.shape[1])
        _, s, v = torch.pca_lowrank(x, q=q, center=False)
        denom = max(1, x.shape[0] - 1)
        eigenvalues = (s[:q] ** 2) / denom
        return v[:, :q], eigenvalues

    covariance = covariance.float()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    idx = torch.argsort(eigenvalues, descending=True)
    idx = idx[: min(k, covariance.shape[0])]
    return eigenvectors[:, idx], eigenvalues[idx]


def convert_components_to_basis(components: torch.Tensor) -> torch.Tensor:
    if components.dim() != 2:
        raise ValueError(f"Components must be 2D, got {components.shape}")
    return components if components.shape[0] >= components.shape[1] else components.t()


@torch.no_grad()
def compute_row_projection_norm(
    weight: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Compute ||P_k w_i||_2 for each output row w_i."""
    in_features = weight.shape[1]
    basis = convert_components_to_basis(basis)
    if basis.shape[0] != in_features:
        if weight.shape[0] == basis.shape[0]:
            weight = weight.t()
            in_features = weight.shape[1]
        if basis.shape[0] != in_features:
            raise ValueError(
                f"Basis rows ({basis.shape[0]}) must match input features ({in_features})."
            )
    # layer_stats stores basis on CPU (see activation_collector); weights may be CUDA after to_cuda / eval.
    basis = basis.to(device=weight.device)
    # For row vectors, projection to PCA subspace is w * (U U^T) = (wU) U^T.
    # Since U has orthonormal columns, ||w (U U^T)||_2 = ||wU||_2.
    proj_coeff = torch.matmul(weight.float(), basis.float())
    return proj_coeff.norm(p=2, dim=1)


@torch.no_grad()
def compute_gate_score(
    weight: torch.Tensor,
    basis: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Compute Gate_i = ||P_k w_i||_2 / (||w_i||_2 + eps)."""
    abs_proj = compute_row_projection_norm(weight=weight, basis=basis)
    row_norm = weight.float().norm(p=2, dim=1)
    return abs_proj / (row_norm + eps)


@torch.no_grad()
def compute_abs_score(
    weight: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Compute Abs_i = ||P_k w_i||_2."""
    return compute_row_projection_norm(weight=weight, basis=basis)


@torch.no_grad()
def compute_alignment_metrics(
    weight: torch.Tensor,
    basis: torch.Tensor,
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Gate_i, Abs_i) for each output row i."""
    abs_proj = compute_abs_score(weight=weight, basis=basis)
    row_norm = weight.float().norm(p=2, dim=1)
    gate = abs_proj / (row_norm + eps)
    return gate, abs_proj


@torch.no_grad()
def compute_importance_score(
    method: str,
    weight: torch.Tensor,
    basis: torch.Tensor,
    beta: float = 1.0,
    abs_zscore: Optional[torch.Tensor] = None,
    eps: float = _EPS,
) -> torch.Tensor:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Use one of {SUPPORTED_METHODS}.")

    gate, abs_proj = compute_alignment_metrics(weight=weight, basis=basis, eps=eps)

    if method == "proj_log":
        return gate + beta * torch.log(abs_proj + eps)
    if method == "proj_norm":
        if abs_zscore is None:
            raise ValueError("abs_zscore must be provided for method='proj_norm'.")
        if abs_zscore.shape != abs_proj.shape:
            raise ValueError(
                f"Shape mismatch: abs_zscore {tuple(abs_zscore.shape)} vs abs {tuple(abs_proj.shape)}"
            )
        return gate + beta * abs_zscore
    if method == "gate_only":
        return gate
    return abs_proj
