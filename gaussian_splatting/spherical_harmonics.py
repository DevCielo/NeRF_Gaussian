from __future__ import annotations

from typing import Tuple

import torch


def _sh_deg0(dir_xyz: torch.Tensor) -> torch.Tensor:
    # Y_0^0
    return 0.28209479177387814 * torch.ones((*dir_xyz.shape[:-1], 1), device=dir_xyz.device, dtype=dir_xyz.dtype)


def _sh_deg1(dir_xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = dir_xyz.unbind(-1)
    return torch.stack(
        [
            -0.4886025119029199 * y,  # Y_1^-1
            0.4886025119029199 * z,   # Y_1^0
            -0.4886025119029199 * x,  # Y_1^1
        ],
        dim=-1,
    )


def _sh_deg2(dir_xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = dir_xyz.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    return torch.stack(
        [
            1.0925484305920792 * xy,                       # Y_2^-2
            -1.0925484305920792 * yz,                      # Y_2^-1
            0.31539156525252005 * (3.0 * zz - 1.0),        # Y_2^0
            -1.0925484305920792 * xz,                      # Y_2^1
            0.5462742152960396 * (xx - yy),                # Y_2^2
        ],
        dim=-1,
    )


def _sh_deg3(dir_xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = dir_xyz.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xyz = x * y * z
    return torch.stack(
        [
            -0.5900435899266435 * y * (3.0 * xx - yy),             # Y_3^-3
            2.890611442640554 * xyz,                               # Y_3^-2
            -0.4570457994644658 * y * (5.0 * zz - 1.0),            # Y_3^-1
            0.3731763325901154 * z * (5.0 * zz - 3.0),             # Y_3^0
            -0.4570457994644658 * x * (5.0 * zz - 1.0),            # Y_3^1
            1.445305721320277 * z * (xx - yy),                     # Y_3^2
            -0.5900435899266435 * x * (xx - 3.0 * yy),             # Y_3^3
        ],
        dim=-1,
    )


def num_sh_coeffs(max_degree: int) -> int:
    return (max_degree + 1) * (max_degree + 1)


def compute_sh_basis(viewdirs: torch.Tensor, degree: int) -> torch.Tensor:
    """Compute real SH basis up to degree for a batch of view directions.

    Args:
        viewdirs: (N,3) normalized directions
        degree: max degree in [0,3]
    Returns:
        basis: (N, (degree+1)^2)
    """
    if degree < 0 or degree > 3:
        raise ValueError("degree must be in [0,3]")
    # Normalize directions to be safe
    v = viewdirs / (torch.linalg.norm(viewdirs, dim=-1, keepdims=True) + 1e-8)

    parts = [_sh_deg0(v)]
    if degree >= 1:
        parts.append(_sh_deg1(v))
    if degree >= 2:
        parts.append(_sh_deg2(v))
    if degree >= 3:
        parts.append(_sh_deg3(v))

    # Concatenate in the standard order: l=0, then l=1, etc.
    return torch.cat(parts, dim=-1)


def evaluate_sh_rgb(viewdirs: torch.Tensor, sh_coeffs: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate per-Gaussian SH RGB given viewdirs.

    Args:
        viewdirs: (N,3)
        sh_coeffs: (N, B, 3) with B=(degree+1)^2
        degree: int in [0,3]
    Returns:
        colors: (N,3) in [0,1] (clamped)
    """
    basis = compute_sh_basis(viewdirs, degree)  # (N,B)
    # (N,3) = sum_b coeffs[:,b,:] * basis[:,b]
    colors = torch.einsum("nb,nbc->nc", basis, sh_coeffs)
    return colors.clamp(0.0, 1.0)


