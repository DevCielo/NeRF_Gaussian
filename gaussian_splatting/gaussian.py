from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
    # Optional spherical harmonics for view-dependent color
    # If provided, interpret color as fallback and prefer SH-evaluated color.
    sh_degree: int = 0
    sh_coeffs: Optional[torch.Tensor] = None  # (B,3) where B=(sh_degree+1)^2

    def to(self, device: torch.device) -> "Gaussian3D":
        return Gaussian3D(
            self.mean.to(device),
            self.covariance.to(device),
            self.color.to(device),
            float(self.alpha),
            int(self.sh_degree),
            None if self.sh_coeffs is None else self.sh_coeffs.to(device),
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

    # Pack SH if present and consistent
    sh_degrees = torch.tensor([getattr(g, "sh_degree", 0) for g in gaussians], device=device)
    unique_degrees = torch.unique(sh_degrees)
    use_sh = False
    packed = {"means": means, "covs": covs, "colors": colors, "alphas": alphas}
    if unique_degrees.numel() == 1 and int(unique_degrees.item()) > 0:
        degree = int(unique_degrees.item())
        coeffs_list = []
        for g in gaussians:
            if g.sh_coeffs is None:
                coeffs_list = []
                break
            coeffs_list.append(g.sh_coeffs)
        if coeffs_list:
            sh_coeffs = torch.stack(coeffs_list, dim=0).to(device)  # (N,B,3)
            use_sh = True
            packed.update({"sh_degree": degree, "sh_coeffs": sh_coeffs})
    packed.update({"use_sh": use_sh})
    return packed


def colors_to_sh_coeffs(colors: torch.Tensor, degree: int) -> torch.Tensor:
    """Create SH coefficients with only DC term representing the input colors.

    Args:
        colors: (N,3) in [0,1]
        degree: max degree in [0,3]
    Returns:
        coeffs: (N,(degree+1)^2,3)
    """
    if degree < 0 or degree > 3:
        raise ValueError("degree must be in [0,3]")
    n = colors.shape[0]
    b = (degree + 1) * (degree + 1)
    coeffs = torch.zeros((n, b, 3), dtype=colors.dtype, device=colors.device)
    c0 = 0.28209479177387814  # Y_0^0
    coeffs[:, 0, :] = colors / c0
    return coeffs


