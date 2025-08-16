from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Gaussian3D:
    """Anisotropic 3D Gaussian.

    Represented by mean (3,) and covariance (3,3) in world coordinates.
    Colors are per-Gaussian RGB in [0,1]. Optional opacity alpha in [0,1].
    """

    mean: torch.Tensor  # (3,)
    covariance: torch.Tensor  # (3,3), symmetric positive definite
    color: torch.Tensor  # (3,)
    alpha: float = 1.0

    def to(self, device: torch.device) -> "Gaussian3D":
        return Gaussian3D(
            self.mean.to(device),
            self.covariance.to(device),
            self.color.to(device),
            float(self.alpha),
        )


def pack_gaussians(gaussians: Tuple[Gaussian3D, ...], device: torch.device) -> dict:
    """Pack a list of Gaussians into batched tensors.

    Returns a dict with keys: means (N,3), covs (N,3,3), colors (N,3), alphas (N,)
    """
    if len(gaussians) == 0:
        raise ValueError("No Gaussians provided")
    means = torch.stack([g.mean for g in gaussians], dim=0).to(device)
    covs = torch.stack([g.covariance for g in gaussians], dim=0).to(device)
    colors = torch.stack([g.color for g in gaussians], dim=0).to(device)
    alphas = torch.tensor([g.alpha for g in gaussians], dtype=torch.float32, device=device)
    return {"means": means, "covs": covs, "colors": colors, "alphas": alphas}


