"""Hybrid NeRF↔Gaussian rendering system.

This package orchestrates training a NeRF (mip-NeRF in this repo),
exporting/initializing a Gaussian Splatting model for fast inference,
and rendering with quality vs speed controls and automatic LOD.

Modules:
- config: Typed configuration objects used across the system.
- convert: Utilities to convert a trained NeRF into Gaussians.
- lod: Screen-space and distance-based LOD selection.
- renderer: Wrapper around Gaussian renderer with quality/LOD controls.
- orchestrator: High-level system to train→export→render and progressive transfer.
- cli: Command-line interface to drive the hybrid pipeline.
"""

from .config import HybridConfig, NerfTrainingConfig, GaussianExportConfig, QualityConfig, LODConfig
from .orchestrator import HybridSystem

__all__ = [
    "HybridConfig",
    "NerfTrainingConfig",
    "GaussianExportConfig",
    "QualityConfig",
    "LODConfig",
    "HybridSystem",
]


