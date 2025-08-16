from __future__ import annotations

from typing import Dict

import torch


def estimate_screen_footprint(covs_img: torch.Tensor) -> torch.Tensor:
    """Approximate footprint (std in pixels) from 2x2 covariance in image space.

    Args:
        covs_img: (N,2,2)
    Returns:
        footprint_std: (N,) ~ sqrt(max(eigvals)) approximated by sqrt(max(diag))
    """
    return torch.sqrt(torch.maximum(covs_img[:, 0, 0], covs_img[:, 1, 1]).clamp_min(1e-12))


def adaptive_density_control(
    batch: Dict[str, torch.Tensor],
    target_coverage_px: float = 1.25,
    min_alpha: float = 0.01,
    max_alpha: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Scale alphas based on projected footprint to stabilize coverage per Gaussian.

    This encourages near-constant opacity contribution per splat across scales.
    """
    if "covs_img" not in batch:
        return batch
    covs_img = batch["covs_img"]
    alphas = batch["alphas"]
    footprint = estimate_screen_footprint(covs_img)  # (N,)
    # Smooth scale using sqrt to avoid overly aggressive changes for large/small footprints
    scale = torch.sqrt(target_coverage_px / (footprint + 1e-6))
    new_alphas = (alphas * scale).clamp(min_alpha, max_alpha)
    batch = dict(batch)
    batch["alphas"] = new_alphas
    return batch


def prune_small_contributors(
    batch: Dict[str, torch.Tensor],
    pixel_bounds: torch.Tensor,
    sigma_thresh_px: float = 0.1,
    alpha_thresh: float = 0.005,
) -> Dict[str, torch.Tensor]:
    """Prune Gaussians with tiny footprint or alpha, or fully outside view.

    Args:
        batch: dict with keys including pixels (N,2), covs_img (N,2,2), colors, alphas
        pixel_bounds: (4,) tensor [x0, y0, x1, y1]
    Returns:
        filtered batch dict
    """
    x0, y0, x1, y1 = pixel_bounds
    mu = batch["pixels"]
    covs = batch["covs_img"]
    alphas = batch["alphas"]
    std = estimate_screen_footprint(covs)
    min_x = mu[:, 0] - 3.0 * std
    max_x = mu[:, 0] + 3.0 * std
    min_y = mu[:, 1] - 3.0 * std
    max_y = mu[:, 1] + 3.0 * std
    inside = (max_x >= x0) & (min_x <= x1 - 1) & (max_y >= y0) & (min_y <= y1 - 1)
    large_enough = (std >= sigma_thresh_px) & (alphas >= alpha_thresh)
    keep = inside & large_enough
    if keep.sum() == 0:
        # Keep at least one to avoid empty tensors downstream
        keep = inside
    filtered = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.shape[:1] == keep.shape:
            filtered[k] = v[keep]
        else:
            filtered[k] = v
    return filtered


