from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from .config import HybridConfig
from .convert import export_gaussians_from_nerf
from .renderer import GaussianRendererWrapper, RenderCamera


class HybridSystem:
    """High-level pipeline: train NeRF → export Gaussians → render.

    Also implements progressive transfer by updating Gaussian attributes
    from NeRF supervision in batches (stub interface; plug in training here).
    """

    def __init__(self, cfg: HybridConfig) -> None:
        self.cfg = cfg
        self.gaussian_npz_path: Optional[str] = None
        self.renderer: Optional[GaussianRendererWrapper] = None

    def train_nerf(self) -> str:
        """Run mip-NeRF training and return checkpoint directory.

        Calls the trainer directly to allow custom dataset paths.
        """
        out_dir = self.cfg.nerf.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Ensure local imports in mip_nerf/train.py resolve (it uses top-level imports)
        import sys, importlib, torch
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        mip_dir = os.path.join(repo_root, "mip_nerf")
        if mip_dir not in sys.path:
            sys.path.insert(0, mip_dir)

        get_config = importlib.import_module("config").get_config
        train_model = importlib.import_module("train").train_model
        cfg = get_config()
        # Device selection
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        cfg.device = device

        # Override essential paths and training schedule
        cfg.base_dir = self.cfg.nerf.dataset_path
        cfg.log_dir = out_dir
        cfg.max_steps = int(self.cfg.nerf.num_steps)
        cfg.save_every = int(self.cfg.nerf.save_every)

        train_model(cfg)
        return out_dir

    def export_gaussians(self, nerf_ckpt_dir: str) -> str:
        art = export_gaussians_from_nerf(nerf_ckpt_dir, self.cfg.export, self.cfg.output_dir)
        self.gaussian_npz_path = art.gaussian_path
        return self.gaussian_npz_path

    def load_renderer(self) -> None:
        assert self.gaussian_npz_path is not None, "Call export_gaussians() first"
        data = np.load(self.gaussian_npz_path)
        # Support both raw means/covs/colors/alphas, and simplified pos/scale/opacity/sh
        if "means" in data and "covs" in data and "colors" in data and "alphas" in data:
            means = data["means"]
            covs = data["covs"]
            colors = data["colors"]
            alphas = data["alphas"]
            # Derive axis-aligned scales from cov diagonals
            scales = np.sqrt(np.stack([covs[:, 0, 0], covs[:, 1, 1], covs[:, 2, 2]], axis=-1))
            positions = means
            opacities = alphas[:, None]
            # Optional SH coefficients
            if int(data.get("use_sh", np.array([0]))[0]) == 1 and ("sh_coeffs" in data):
                sh = data["sh_coeffs"]
            else:
                # Fallback to per-splat albedo color as DC term only
                sh = colors
        else:
            # Backward-compat simple format
            positions = data["positions"]
            scales = data["scales"]
            opacities = data["opacities"]
            sh = data["sh"]
        self.renderer = GaussianRendererWrapper(
            positions=positions,
            scales=scales,
            opacities=opacities,
            sh=sh,
            quality=self.cfg.quality,
            lod_cfg=self.cfg.lod,
        )

    def render_frame(self, camera_dict: Dict[str, Any]) -> np.ndarray:
        assert self.renderer is not None, "Call load_renderer() first"
        cam = RenderCamera(
            width=camera_dict["width"],
            height=camera_dict["height"],
            fx=camera_dict["fx"],
            fy=camera_dict["fy"],
            cx=camera_dict.get("cx", camera_dict["width"] / 2.0),
            cy=camera_dict.get("cy", camera_dict["height"] / 2.0),
            c2w=camera_dict["c2w"],
        )
        return self.renderer.render(cam)

    def progressive_transfer_step(self, supervision_batch: dict) -> None:
        """Placeholder for fine-tuning Gaussians against supervision.

        This repo includes a fast renderer; a differentiable fine-tuning step
        can be added to update SH/DC colors or opacities by minimizing a loss
        versus NeRF-rendered supervision images. Left as a stub.
        """
        _ = supervision_batch
        return None

    def progressive_densify(self, nerf_ckpt_dir: str, new_num_gaussians: int) -> str:
        """Increase Gaussian count by re-extracting top-K from NeRF and reloading.

        Returns path to the updated NPZ.
        """
        # Temporarily override export.num_gaussians
        old = self.cfg.export.num_gaussians
        self.cfg.export.num_gaussians = int(new_num_gaussians)
        try:
            npz_path = self.export_gaussians(nerf_ckpt_dir)
        finally:
            self.cfg.export.num_gaussians = old
        # Reload renderer with denser set
        self.load_renderer()
        return npz_path


