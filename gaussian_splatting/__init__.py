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

__all__ = [
    "Camera",
    "Gaussian3D",
    "pack_gaussians",
    "render",
    "load_point_cloud",
    "points_to_gaussians",
]


