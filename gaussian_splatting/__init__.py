"""Minimal Gaussian Splatting module.

Modules:
- camera: Pinhole camera intrinsics/extrinsics utilities
- gaussian: 3D Gaussian representation and projection helpers
- renderer: Basic differentiable-like Gaussian splat renderer
- pointcloud_loader: Utilities to load point clouds and convert to Gaussians
"""

from .camera import Camera
from .gaussian import Gaussian3D, pack_gaussians
from .renderer import render
from .pointcloud_loader import (
    load_point_cloud,
    points_to_gaussians,
)
from .gaussian4d import Gaussian4D, pack_gaussians_4d, gaussian4d_to_3d_batch
# Optional: NeRF→Gaussian conversion utilities (may require extra deps)
try:
    from .convert_from_nerf import extract_gaussians_from_trained_nerf, demo_convert_and_render
    _HAS_CONVERTER = True
except Exception:  # noqa: E722
    extract_gaussians_from_trained_nerf = None  # type: ignore[assignment]
    demo_convert_and_render = None  # type: ignore[assignment]
    _HAS_CONVERTER = False

__all__ = [
    "Camera",
    "Gaussian3D",
    "pack_gaussians",
    "render",
    "load_point_cloud",
    "points_to_gaussians",
    "Gaussian4D",
    "pack_gaussians_4d",
    "gaussian4d_to_3d_batch",
    # May be None if dependencies are missing
    "extract_gaussians_from_trained_nerf",
    "demo_convert_and_render",
]


