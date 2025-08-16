from __future__ import annotations

from typing import Optional, Tuple

import torch

from gaussian import Gaussian3D


def load_point_cloud(
    path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a simple point cloud.

    Supports:
      - .pt/.pth with dict containing keys 'points' (N,3) and optional 'colors' (N,3)
      - None: generates a synthetic colored sphere
    Returns:
      points (N,3), colors (N,3)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        # Create synthetic sphere
        num_lat = 40
        num_lon = 80
        lat = torch.linspace(1e-3, torch.pi - 1e-3, num_lat, device=device)
        lon = torch.linspace(0, 2 * torch.pi, num_lon, device=device)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        x = torch.sin(lat_grid) * torch.cos(lon_grid)
        y = torch.cos(lat_grid)
        z = torch.sin(lat_grid) * torch.sin(lon_grid)
        pts = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        # Simple color mapping
        colors = (pts + 1.0) * 0.5
        return pts, colors

    # Try multiple formats
    try:
        if path.endswith((".pt", ".pth")):
            data = torch.load(path, map_location=device)
            if isinstance(data, dict) and "points" in data:
                pts = data["points"].to(device).float()
                if "colors" in data:
                    colors = data["colors"].to(device).float().clamp(0.0, 1.0)
                else:
                    colors = torch.ones_like(pts) * 0.7
                return pts, colors
        if path.endswith((".npz",)):
            import numpy as np

            npz = np.load(path)
            pts = torch.from_numpy(npz["points"]).to(device).float()
            if "colors" in npz:
                colors = torch.from_numpy(npz["colors"]).to(device).float().clamp(0.0, 1.0)
            else:
                colors = torch.ones_like(pts) * 0.7
            return pts, colors
        if path.endswith((".npy",)):
            import numpy as np

            arr = np.load(path)
            if arr.shape[1] >= 3:
                pts = torch.from_numpy(arr[:, :3]).to(device).float()
                if arr.shape[1] >= 6:
                    colors = torch.from_numpy(arr[:, 3:6]).to(device).float().clamp(0.0, 1.0)
                else:
                    colors = torch.ones_like(pts) * 0.7
                return pts, colors
        if path.endswith((".txt", ".xyz")):
            import numpy as np

            arr = np.loadtxt(path)
            pts = torch.from_numpy(arr[:, :3]).to(device).float()
            if arr.shape[1] >= 6:
                colors = torch.from_numpy(arr[:, 3:6]).to(device).float().clamp(0.0, 1.0)
            else:
                colors = torch.ones_like(pts) * 0.7
            return pts, colors
        if path.endswith((".ply",)):
            try:
                import trimesh as _trimesh  # type: ignore

                mesh = _trimesh.load(path, process=False)
                if hasattr(mesh, "vertices"):
                    pts_np = mesh.vertices
                    pts = torch.from_numpy(pts_np).to(device).float()
                    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
                        vc = mesh.visual.vertex_colors
                        colors_np = vc[:, :3] / 255.0
                        colors = torch.from_numpy(colors_np).to(device).float().clamp(0.0, 1.0)
                    else:
                        colors = torch.ones_like(pts) * 0.7
                    return pts, colors
            except Exception:
                pass
    except Exception:
        pass

    raise ValueError(
        "Unsupported point cloud format. Provide .pt/.npz/.npy/.txt/.xyz/.ply with 'points' and optional 'colors'."
    )


def points_to_gaussians(
    points: torch.Tensor,
    colors: torch.Tensor,
    position_variance: float = 0.0025,
    anisotropy_scale: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
):
    """Convert points to simple anisotropic Gaussians.

    Args:
      points: (N,3)
      colors: (N,3)
      position_variance: base variance along principal axes
      anisotropy_scale: optional (N,3) scaling of variance per-axis
    Returns:
      tuple(Gaussian3D, ...)
    """
    if device is None:
        device = points.device
    points = points.to(device).float()
    colors = colors.to(device).float().clamp(0.0, 1.0)
    n = points.shape[0]

    if anisotropy_scale is None:
        anisotropy_scale = torch.ones((n, 3), device=device, dtype=torch.float32)
    anisotropy_scale = anisotropy_scale.float()

    base_var = torch.full((n, 3), position_variance, device=device, dtype=torch.float32)
    diag = (base_var * anisotropy_scale).unsqueeze(-1)  # (N,3,1)
    covs = torch.zeros((n, 3, 3), device=device, dtype=torch.float32)
    covs[:, 0, 0] = diag[:, 0, 0]
    covs[:, 1, 1] = diag[:, 1, 0]
    covs[:, 2, 2] = diag[:, 2, 0]

    gaussians = []
    for i in range(n):
        gaussians.append(
            Gaussian3D(mean=points[i], covariance=covs[i], color=colors[i], alpha=1.0)
        )
    return tuple(gaussians)


