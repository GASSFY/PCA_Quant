"""PCA-based importance functions for mixed-precision quantization."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

_EPS = 1e-8
SUPPORTED_METHODS = ("l1_only", "gate_only", "l1_gate_mul", "beta_log_l1")


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
def compute_gate_score(
    weight: torch.Tensor,
    basis: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
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

    gate = basis.float().pow(2).sum(dim=1).sqrt()
    return gate.clamp(min=0.0, max=1.0 + eps)


@torch.no_grad()
def compute_magnitude_score(
    weight: torch.Tensor,
    activation_second_moment: torch.Tensor,
) -> torch.Tensor:
    if activation_second_moment.dim() != 1:
        activation_second_moment = activation_second_moment.view(-1)
    if weight.shape[1] != activation_second_moment.shape[0]:
        if weight.shape[0] == activation_second_moment.shape[0]:
            weight = weight.t()
        else:
            raise ValueError(
                f"Shape mismatch: weight {tuple(weight.shape)}, second moment {tuple(activation_second_moment.shape)}"
            )
    w_col_norm_sq = weight.float().pow(2).sum(dim=0)
    return w_col_norm_sq * activation_second_moment.float()


@torch.no_grad()
def compute_importance_score(
    method: str,
    weight: torch.Tensor,
    activation_second_moment: torch.Tensor,
    basis: torch.Tensor,
    beta: float = 1.0,
    eps: float = _EPS,
) -> torch.Tensor:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Use one of {SUPPORTED_METHODS}.")

    magnitude = compute_magnitude_score(weight, activation_second_moment)
    gate = compute_gate_score(weight, basis, eps=eps)

    if method == "l1_only":
        return magnitude
    if method == "gate_only":
        return gate
    if method == "l1_gate_mul":
        return magnitude * gate
    return gate + beta * torch.log(magnitude.clamp(min=0.0) + eps)
