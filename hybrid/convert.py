from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

import numpy as np
import torch

from .config import GaussianExportConfig


@dataclass
class NerfToGaussianArtifacts:
    gaussian_path: str
    # Additional metadata for progressive transfer
    sh_degree: int
    num_gaussians: int


def export_gaussians_from_nerf(
    nerf_checkpoint_dir: str,
    export_cfg: GaussianExportConfig,
    output_dir: str,
) -> NerfToGaussianArtifacts:
    """Convert a trained NeRF into a Gaussian Splat representation (saved as NPZ).

    Uses `gaussian_splatting.convert_from_nerf.extract_gaussians_from_trained_nerf`
    to generate a renderer-compatible batch, then serializes numpy arrays.
    """
    os.makedirs(output_dir, exist_ok=True)
    gaussian_out = os.path.join(output_dir, "gaussians_init.npz")

    # Prepare a config object pointing to the trained model weights
    # Similar import path trick as in orchestrator to use mip_nerf's CLI-style modules
    import sys, importlib
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mip_dir = os.path.join(repo_root, "mip_nerf")
    if mip_dir not in sys.path:
        sys.path.insert(0, mip_dir)
    get_config = importlib.import_module("config").get_config
    cfg = get_config()
    # Choose device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    cfg.device = device
    # Point to trained weights
    cfg.model_weight_path = os.path.join(nerf_checkpoint_dir, "model.pt")

    # Import gaussian conversion using package path
    repo_gs_dir = os.path.join(repo_root, "gaussian_splatting")
    if repo_gs_dir not in sys.path:
        sys.path.insert(0, repo_gs_dir)
    extract_gaussians_from_trained_nerf = importlib.import_module("convert_from_nerf").extract_gaussians_from_trained_nerf

    batch, _wh = extract_gaussians_from_trained_nerf(
        cfg=cfg,
        sh_degree=int(export_cfg.sh_degree),
        alpha_scale=1.0,
        max_gaussians=int(export_cfg.num_gaussians) if export_cfg.num_gaussians is not None else None,
        device=device,
    )

    # Serialize to NPZ for reloading without torch
    means = batch["means"].detach().cpu().numpy().astype("float32")
    covs = batch["covs"].detach().cpu().numpy().astype("float32")
    colors = batch["colors"].detach().cpu().numpy().astype("float32")
    alphas = batch["alphas"].detach().cpu().numpy().astype("float32")
    npz_data = {
        "means": means,
        "covs": covs,
        "colors": colors,
        "alphas": alphas,
    }
    if bool(batch.get("use_sh", False)) and ("sh_coeffs" in batch):
        sh_coeffs = batch["sh_coeffs"].detach().cpu().numpy().astype("float32")
        npz_data.update({
            "use_sh": np.array([1], dtype="int32"),
            "sh_degree": np.array([int(export_cfg.sh_degree)], dtype="int32"),
            "sh_coeffs": sh_coeffs,
        })
    else:
        npz_data.update({
            "use_sh": np.array([0], dtype="int32"),
            "sh_degree": np.array([0], dtype="int32"),
        })

    np.savez(gaussian_out, **npz_data)

    return NerfToGaussianArtifacts(
        gaussian_path=gaussian_out,
        sh_degree=export_cfg.sh_degree,
        num_gaussians=means.shape[0],
    )


