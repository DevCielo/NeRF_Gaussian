from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .config import LODConfig


@dataclass
class LODSelection:
    indices_near: np.ndarray
    indices_mid: np.ndarray
    indices_far: np.ndarray


def project_points_to_radius(
    positions_cam: np.ndarray,
    scales: np.ndarray,
    fx: float,
    fy: float,
) -> np.ndarray:
    """Approximate screen-space pixel radius from camera-space scale.

    positions_cam: (N, 3) camera-space positions.
    scales: (N, 3) Gaussian axis-aligned scales (std dev radii).
    fx, fy: focal lengths in pixels.
    """
    # Use the largest axis as proxy for projected radius
    radii_cam = np.max(scales, axis=1)
    # Perspective: pixel radius ≈ f * r / z
    z = np.clip(positions_cam[:, 2], 1e-4, None)
    radius_px = (fx + fy) * 0.5 * radii_cam / z
    return radius_px


def select_lod(
    positions_cam: np.ndarray,
    scales: np.ndarray,
    fx: float,
    fy: float,
    cfg: LODConfig,
) -> LODSelection:
    radius_px = project_points_to_radius(positions_cam, scales, fx, fy)
    d = np.linalg.norm(positions_cam, axis=1)

    near_px, mid_px = cfg.pixel_radius_thresholds
    near_mask = radius_px >= mid_px
    mid_mask = (radius_px >= near_px) & (radius_px < mid_px)
    far_mask = radius_px < near_px

    # Distance gating to avoid drawing distant tiny splats when crowded
    near_d, far_d = cfg.distance_range
    near_mask = near_mask & (d <= far_d)
    mid_mask = mid_mask & (d <= far_d)
    far_mask = far_mask & (d >= near_d)

    return LODSelection(
        indices_near=np.where(near_mask)[0],
        indices_mid=np.where(mid_mask)[0],
        indices_far=np.where(far_mask)[0],
    )


