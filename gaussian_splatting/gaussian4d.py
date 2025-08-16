from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .gaussian import Gaussian3D


@dataclass
class Gaussian4D:
    """4D Gaussian with constant-velocity motion model.

    Parameters describe a Gaussian primitive evolving over time t (in seconds):
      X(t) = mean0 + velocity * t
      Sigma(t) = covariance (kept constant for simplicity)

    Colors can be view-dependent via spherical harmonics (same layout as 3D).
    """

    mean0: torch.Tensor  # (3,)
    velocity: torch.Tensor  # (3,)
    covariance: torch.Tensor  # (3,3)
    color: torch.Tensor  # (3,)
    alpha: float = 1.0
    sh_degree: int = 0
    sh_coeffs: Optional[torch.Tensor] = None  # (B,3)

    def to(self, device: torch.device) -> "Gaussian4D":
        return Gaussian4D(
            self.mean0.to(device),
            self.velocity.to(device),
            self.covariance.to(device),
            self.color.to(device),
            float(self.alpha),
            int(self.sh_degree),
            None if self.sh_coeffs is None else self.sh_coeffs.to(device),
        )

    def at_time(self, t: float) -> Gaussian3D:
        mean_t = self.mean0 + t * self.velocity
        return Gaussian3D(
            mean=mean_t,
            covariance=self.covariance,
            color=self.color,
            alpha=self.alpha,
            sh_degree=self.sh_degree,
            sh_coeffs=self.sh_coeffs,
        )


def pack_gaussians_4d(gaussians: Tuple[Gaussian4D, ...], device: torch.device) -> dict:
    if len(gaussians) == 0:
        raise ValueError("No Gaussians provided")
    means0 = torch.stack([g.mean0 for g in gaussians], dim=0).to(device)
    velocities = torch.stack([g.velocity for g in gaussians], dim=0).to(device)
    covs = torch.stack([g.covariance for g in gaussians], dim=0).to(device)
    colors = torch.stack([g.color for g in gaussians], dim=0).to(device)
    alphas = torch.tensor([g.alpha for g in gaussians], dtype=torch.float32, device=device)

    sh_degrees = torch.tensor([getattr(g, "sh_degree", 0) for g in gaussians], device=device)
    unique_degrees = torch.unique(sh_degrees)
    use_sh = False
    packed = {"means0": means0, "velocities": velocities, "covs": covs, "colors": colors, "alphas": alphas}
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


def gaussian4d_to_3d_batch(batch4d: dict, time_s: float) -> dict:
    """Convert a 4D batch dict into a 3D batch dict evaluated at time t.

    Expected keys in batch4d: means0 (N,3), velocities (N,3), covs (N,3,3), colors (N,3), alphas (N,)
    SH keys are forwarded if present.
    """
    means = batch4d["means0"] + time_s * batch4d["velocities"]
    covs = batch4d["covs"]
    colors = batch4d["colors"]
    alphas = batch4d["alphas"]
    out = {"means": means, "covs": covs, "colors": colors, "alphas": alphas}
    if bool(batch4d.get("use_sh", False)):
        out.update({
            "use_sh": True,
            "sh_degree": int(batch4d["sh_degree"]),
            "sh_coeffs": batch4d["sh_coeffs"],
        })
    return out


