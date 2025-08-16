from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .config import QualityConfig, LODConfig
from .lod import select_lod
from gaussian_splatting.camera import Camera as TorchCamera
from gaussian_splatting.renderer import render as gs_render


@dataclass
class RenderCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray  # 4x4


class GaussianRendererWrapper:
    """Adapter over the repository's Gaussian renderer with QoS + LOD.

    Expected Gaussian data arrays:
    - positions: (N, 3)
    - scales: (N, 3)
    - opacities: (N, 1)
    - sh: (N, C)
    """

    def __init__(
        self,
        positions: np.ndarray,
        scales: np.ndarray,
        opacities: np.ndarray,
        sh: np.ndarray,
        quality: QualityConfig,
        lod_cfg: LODConfig,
    ) -> None:
        self.positions = positions
        self.scales = scales
        self.opacities = opacities
        self.sh = sh
        self.quality = quality
        self.lod_cfg = lod_cfg

    def _camera_to_world_to_world_to_camera(self, cam: RenderCamera) -> np.ndarray:
        w2c = np.linalg.inv(cam.c2w)
        return w2c

    def _transform_points(self, points: np.ndarray, w2c: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1), dtype=points.dtype)
        pw = np.concatenate([points, ones], axis=1)
        pc = (w2c @ pw.T).T[:, :3]
        return pc

    def render(self, cam: RenderCamera) -> np.ndarray:
        # Build torch camera
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        w2c = self._camera_to_world_to_world_to_camera(cam)  # numpy 4x4
        w2c_t = torch.from_numpy(w2c.astype(np.float32)).to(device)
        camera_t = TorchCamera(
            width=int(cam.width),
            height=int(cam.height),
            fx=float(cam.fx),
            fy=float(cam.fy),
            cx=float(cam.cx),
            cy=float(cam.cy),
            world_to_camera=w2c_t,
            device=device,
        )

        # LOD selection in numpy for indices, then move tensors to torch
        positions_cam = self._transform_points(self.positions, w2c)
        lod = select_lod(positions_cam, self.scales, cam.fx, cam.fy, self.lod_cfg)
        indices = np.concatenate([lod.indices_near, lod.indices_mid, lod.indices_far])
        if self.quality.max_gaussians_per_frame is not None:
            indices = indices[: self.quality.max_gaussians_per_frame]

        means = torch.from_numpy(self.positions[indices].astype(np.float32)).to(device)
        # Convert axis-aligned scales to covariance (diagonal in world space)
        scales = self.scales[indices].astype(np.float32)
        covs = np.zeros((scales.shape[0], 3, 3), dtype=np.float32)
        covs[:, 0, 0] = scales[:, 0] * scales[:, 0]
        covs[:, 1, 1] = scales[:, 1] * scales[:, 1]
        covs[:, 2, 2] = scales[:, 2] * scales[:, 2]
        covs_t = torch.from_numpy(covs).to(device)
        colors_t = torch.from_numpy(np.clip(self._colors_from_sh(indices), 0.0, 1.0).astype(np.float32)).to(device)
        alphas_t = torch.from_numpy(self.opacities[indices, 0].astype(np.float32)).to(device)

        batch = {"means": means, "covs": covs_t, "colors": colors_t, "alphas": alphas_t}

        # Quality knobs mapping
        ssaa_scale = 1
        aa_samples = 1
        max_gaussians_per_tile = None
        approx_isotropic = False
        if self.quality.quality_level is not None:
            q = float(self.quality.quality_level)
            # Simple mapping
            ssaa_scale = 1 if q < 0.5 else 2 if q < 0.85 else 4
            aa_samples = 1 if q < 0.5 else 2 if q < 0.85 else 4
            approx_isotropic = q < 0.6
            # Cap per-tile gaussians inversely to q
            if q < 0.5:
                max_gaussians_per_tile = 1024
            elif q < 0.85:
                max_gaussians_per_tile = 2048
            else:
                max_gaussians_per_tile = 4096

        out = gs_render(
            camera_t,
            batch,
            background_color=(1.0, 1.0, 1.0),
            tile_size=256,
            use_binning=True,
            use_amp=True,
            approx_isotropic=approx_isotropic,
            max_gaussians_per_tile=max_gaussians_per_tile,
            ssaa_scale=ssaa_scale,
            aa_samples=aa_samples,
            return_depth=False,
            return_normals=False,
            lighting=None,
        )
        rgb_t, alpha_t = out
        img = rgb_t.detach().cpu().numpy().astype(np.float32)
        return img

    def _colors_from_sh(self, indices: np.ndarray) -> np.ndarray:
        # If SH coefficients stored in NPZ, we can precompute colors by using DC term only as an approximation
        # Expecting self.sh shape (N, C) possibly with SH layout; here we fallback to an average color
        if self.sh is None or self.sh.size == 0:
            # default white
            return np.ones((indices.shape[0], 3), dtype=np.float32)
        # Assume channels are stacked per RGB, DC term first for each color
        # DC basis value for SH is Y_00 = 1 / (2*sqrt(pi)), absorbed in training; treat as identity
        c = self.sh[indices]
        per_color = c.shape[1] // 3 if c.shape[1] % 3 == 0 else 0
        if per_color <= 0:
            return np.clip(c[:, :3], 0.0, 1.0)
        r = c[:, 0]
        g = c[:, per_color]
        b = c[:, 2 * per_color]
        col = np.stack([r, g, b], axis=-1)
        return col


