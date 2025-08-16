from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional

import torch


@dataclass
class Frame:
    time_s: float
    points: torch.Tensor  # (N,3)
    colors: torch.Tensor  # (N,3)


@dataclass
class DynamicSequence:
    frames: List[Frame]

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)


def load_synthetic_bouncing_sphere(num_frames: int = 30, device: torch.device | None = None) -> DynamicSequence:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    # Base static sphere
    lat = torch.linspace(1e-3, torch.pi - 1e-3, 32, device=device)
    lon = torch.linspace(0, 2 * torch.pi, 64, device=device)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
    x = torch.sin(lat_grid) * torch.cos(lon_grid)
    y = torch.cos(lat_grid)
    z = torch.sin(lat_grid) * torch.sin(lon_grid)
    base_pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    base_cols = (base_pts + 1.0) * 0.5

    frames: List[Frame] = []
    for i in range(num_frames):
        t = torch.tensor(i / max(num_frames - 1, 1), device=device, dtype=torch.float32)
        # Bouncing motion along Y
        offset_y = torch.sin(2 * torch.pi * t) * 0.5
        pts = base_pts + torch.tensor([0.0, offset_y, 0.0], device=device)
        frames.append(Frame(time_s=float(i) * 0.033, points=pts, colors=base_cols))

    return DynamicSequence(frames)


def load_pointcloud_sequence(
    directory: str,
    pattern: str = "frame_{:03d}.npz",
    num_frames: Optional[int] = None,
    fps: float = 30.0,
    device: torch.device | None = None,
) -> DynamicSequence:
    """Load a dynamic sequence from a directory of per-frame point clouds.

    Each file should contain keys 'points' (N,3) and optional 'colors' (N,3).
    Files are named by pattern like frame_000.npz, frame_001.npz, ...
    """
    import os
    import numpy as np

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

    frames: List[Frame] = []
    i = 0
    while True:
        fname = os.path.join(directory, pattern.format(i))
        if not os.path.exists(fname):
            if num_frames is None:
                break
            else:
                raise FileNotFoundError(f"Missing frame file: {fname}")
        npz = np.load(fname)
        pts = torch.from_numpy(npz["points"]).to(device).float()
        if "colors" in npz:
            cols = torch.from_numpy(npz["colors"]).to(device).float().clamp(0.0, 1.0)
        else:
            cols = torch.ones_like(pts) * 0.7
        frames.append(Frame(time_s=i / fps, points=pts, colors=cols))
        i += 1
        if num_frames is not None and i >= num_frames:
            break

    if len(frames) == 0:
        raise ValueError("No frames found. Check directory and pattern.")
    return DynamicSequence(frames)


