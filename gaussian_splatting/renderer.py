from __future__ import annotations

from typing import Dict, Tuple

import torch

from camera import Camera


def _project_covariance_to_image(
    camera: Camera,
    means_world: torch.Tensor,  # (N,3)
    covs_world: torch.Tensor,  # (N,3,3)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussian covariance to 2D image space via first-order linearization.

    Uses J = d[u,v]/d[Xw,Yw,Zw] at the mean to compute Sigma_image = J Sigma_world J^T.
    Returns:
      pixels: (N,2)
      depths: (N,)
      covs_image: (N,2,2)
    """
    pixels, depths, cam_coords = camera.project(means_world)
    j = camera.jacobian_world_to_image(cam_coords)  # (N,2,3)
    covs_image = j @ covs_world @ j.transpose(1, 2)  # (N,2,2)
    # Ensure numeric stability
    eye2 = torch.eye(2, device=covs_image.device, dtype=covs_image.dtype)
    covs_image = covs_image + 1e-6 * eye2[None, :, :]
    return pixels, depths, covs_image


def _gaussian_kernel_2d(grid_xy: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Evaluate an unnormalized 2D Gaussian kernel exp(-0.5 * d^2) at grid points.

    Using an unnormalized kernel (peak value 1.0) yields perceptually stronger
    splats than a probability density function, which tends to produce very low
    per-pixel values after projection. This improves visibility without needing
    large, ad-hoc alpha scales.

    grid_xy: (H,W,2)
    mu: (2,)
    sigma: (2,2)
    Returns: (H,W)
    """
    diff = grid_xy - mu[None, None, :]
    inv = torch.linalg.inv(sigma)
    mahal = torch.einsum("...i,ij,...j->...", diff, inv, diff)
    return torch.exp(-0.5 * mahal)


def render(
    camera: Camera,
    gaussian_batch: Dict[str, torch.Tensor],
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tile_size: int = 256,
    depth_epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render a set of anisotropic 3D Gaussians with simple over compositing.

    Args:
      camera: Camera object
      gaussian_batch: dict with keys means (N,3), covs (N,3,3), colors (N,3), alphas (N,)
      background_color: RGB in [0,1]
      tile_size: process image in tiles to limit memory
      depth_epsilon: small depth regularizer for sorting
    Returns:
      rgb: (H,W,3) image tensor in [0,1]
      alpha: (H,W,1) accumulated alpha
    """
    device = camera.device
    h, w = camera.height, camera.width
    rgb = torch.tensor(background_color, device=device, dtype=torch.float32).view(1, 1, 3).repeat(h, w, 1)
    acc_a = torch.zeros((h, w, 1), device=device, dtype=torch.float32)

    means = gaussian_batch["means"]
    covs = gaussian_batch["covs"]
    colors = gaussian_batch["colors"]
    alphas = gaussian_batch["alphas"]

    pixels, depths, covs_img = _project_covariance_to_image(camera, means, covs)

    # Cull Gaussians that are behind the camera or far outside the image bounds (3 sigma)
    valid = depths > 0.0
    # Compute an approximate footprint radius for culling
    footprint = torch.sqrt(torch.maximum(covs_img[:, 0, 0], covs_img[:, 1, 1]))  # ~ std in px
    rad = 3.0 * footprint
    valid = valid & (pixels[:, 0] + rad >= 0) & (pixels[:, 0] - rad <= w - 1)
    valid = valid & (pixels[:, 1] + rad >= 0) & (pixels[:, 1] - rad <= h - 1)

    if valid.sum() == 0:
        return rgb, acc_a

    pixels = pixels[valid]
    depths = depths[valid]
    covs_img = covs_img[valid]
    colors = colors[valid]
    alphas = alphas[valid]

    # Sort by depth back-to-front (far to near) so nearer splats are composed last
    sort_idx = torch.argsort(depths + depth_epsilon, descending=True)
    pixels = pixels[sort_idx]
    covs_img = covs_img[sort_idx]
    colors = colors[sort_idx]
    alphas = alphas[sort_idx]

    # Tile-based rasterization
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(y0, y1, device=device, dtype=torch.float32),
                torch.arange(x0, x1, device=device, dtype=torch.float32),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (Th, Tw, 2)

            # Accumulate contributions of all Gaussians for this tile
            tile_rgb = rgb[y0:y1, x0:x1, :]
            tile_a = acc_a[y0:y1, x0:x1, :]

            for i in range(pixels.shape[0]):
                mu = pixels[i]
                sigma = covs_img[i]
                color = colors[i]
                alpha = alphas[i]

                # Skip if outside tile by >3 sigma
                std = torch.sqrt(torch.maximum(sigma[0, 0], sigma[1, 1]))
                if mu[0] + 3 * std < x0 or mu[0] - 3 * std > x1 - 1:
                    continue
                if mu[1] + 3 * std < y0 or mu[1] - 3 * std > y1 - 1:
                    continue

                # Unnormalized kernel density. Convert to alpha via exponential mapping
                # a = 1 - exp(-alpha * density) for stable accumulation.
                density = _gaussian_kernel_2d(grid, mu, sigma)  # (Th,Tw)
                a = (1.0 - torch.exp(-alpha * density))[..., None]  # (Th,Tw,1)
                # Over compositing: out = src + (1 - a_src) * dst
                tile_rgb = color.view(1, 1, 3) * a + (1.0 - a) * tile_rgb
                tile_a = a + (1.0 - a) * tile_a

            rgb[y0:y1, x0:x1, :] = tile_rgb
            acc_a[y0:y1, x0:x1, :] = tile_a

    return rgb.clamp(0.0, 1.0), acc_a.clamp(0.0, 1.0)


