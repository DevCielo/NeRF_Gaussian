from __future__ import annotations

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn.functional as F

from .camera import Camera
from .spherical_harmonics import evaluate_sh_rgb
from .pruning import adaptive_density_control, prune_small_contributors


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


def _invert_2x2(m: torch.Tensor) -> torch.Tensor:
    """Analytic inverse for a batch of 2x2 matrices.
    m: (...,2,2)
    returns inv: (...,2,2)
    """
    a = m[..., 0, 0]
    b = m[..., 0, 1]
    c = m[..., 1, 0]
    d = m[..., 1, 1]
    det = a * d - b * c
    inv_det = 1.0 / (det + 1e-12)
    out00 = d * inv_det
    out01 = -b * inv_det
    out10 = -c * inv_det
    out11 = a * inv_det
    return torch.stack([
        torch.stack([out00, out01], dim=-1),
        torch.stack([out10, out11], dim=-1),
    ], dim=-2)

def _build_tile_bins(
    pixels: torch.Tensor,
    covs_img: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
) -> List[torch.Tensor]:
    """Build per-tile lists of gaussian indices based on 3-sigma footprint overlap.

    Returns a python list of length num_tiles with 1D index tensors (on same device).
    """
    device = pixels.device
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    num_tiles = tiles_x * tiles_y
    bins: List[List[int]] = [[] for _ in range(num_tiles)]

    std = torch.sqrt(torch.maximum(covs_img[:, 0, 0], covs_img[:, 1, 1]))  # (M,)
    min_x = (pixels[:, 0] - 3.0 * std).clamp(0, width - 1)
    max_x = (pixels[:, 0] + 3.0 * std).clamp(0, width - 1)
    min_y = (pixels[:, 1] - 3.0 * std).clamp(0, height - 1)
    max_y = (pixels[:, 1] + 3.0 * std).clamp(0, height - 1)

    min_tx = (min_x // tile_size).to(torch.int64)
    max_tx = (max_x // tile_size).to(torch.int64)
    min_ty = (min_y // tile_size).to(torch.int64)
    max_ty = (max_y // tile_size).to(torch.int64)

    # Use CPU loops for binning indices; acceptable since per-frame
    # Ensure tensors are on CPU for iteration speed, then convert back
    min_tx_c = min_tx.detach().cpu().tolist()
    max_tx_c = max_tx.detach().cpu().tolist()
    min_ty_c = min_ty.detach().cpu().tolist()
    max_ty_c = max_ty.detach().cpu().tolist()

    for gi in range(pixels.shape[0]):
        tx0 = min_tx_c[gi]
        tx1 = max_tx_c[gi]
        ty0 = min_ty_c[gi]
        ty1 = max_ty_c[gi]
        for ty in range(ty0, ty1 + 1):
            row_offset = ty * tiles_x
            for tx in range(tx0, tx1 + 1):
                bins[row_offset + tx].append(gi)

    # Pack into tensors per tile
    out: List[torch.Tensor] = []
    for lst in bins:
        if len(lst) == 0:
            out.append(torch.empty((0,), dtype=torch.int64, device=device))
        else:
            out.append(torch.tensor(lst, dtype=torch.int64, device=device))
    return out


def _compute_normals_from_depth(camera: Camera, depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Approximate per-pixel normals from depth via finite differences in camera space.

    Args:
      camera: Camera
      depth: (H,W,1) depth in camera Z (same convention as projection)
      alpha: (H,W,1) alpha map used to mask invalid/background
    Returns:
      normals: (H,W,3) unit normals in camera space (XYZ), zero where alpha≈0
    """
    z = depth[..., 0]
    h, w = z.shape
    # Pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(h, device=z.device, dtype=z.dtype),
        torch.arange(w, device=z.device, dtype=z.dtype),
        indexing="ij",
    )
    x_cam = (xs - camera.cx) / camera.fx * z
    y_cam = (ys - camera.cy) / camera.fy * z
    # Forward differences (pad last row/col by replication)
    def _diff_fwd(t: torch.Tensor, dim: int) -> torch.Tensor:
        if dim == 1:
            d = t[:, 1:] - t[:, :-1]
            last = d[:, -1:]
            d = torch.cat([d, last], dim=1)
        else:
            d = t[1:, :] - t[:-1, :]
            last = d[-1:, :]
            d = torch.cat([d, last], dim=0)
        return d

    dx = _diff_fwd(x_cam, 1)
    dy = _diff_fwd(y_cam, 0)
    dzx = _diff_fwd(z, 1)
    dzy = _diff_fwd(z, 0)

    # Tangents along x and y image axes in camera space
    tx = torch.stack([torch.ones_like(dx), torch.zeros_like(dx), dzx / (z + 1e-8)], dim=-1)
    ty = torch.stack([torch.zeros_like(dy), torch.ones_like(dy), dzy / (z + 1e-8)], dim=-1)

    # Scale tx,ty to actual metric changes using intrinsics and depth
    # Derive dX/du = z/fx, dY/dv = z/fy
    tx[..., 0] = (z / camera.fx)
    ty[..., 1] = (z / camera.fy)

    # Cross product ty x tx gives normal facing camera for increasing y then x
    n = torch.linalg.cross(ty, tx)
    n = F.normalize(n, dim=-1)

    # Zero-out where alpha is negligible
    mask = (alpha[..., 0] > 1e-4).to(n.dtype)[..., None]
    return n * mask


def _generate_halton_2d(n: int, base_a: int = 2, base_b: int = 3) -> torch.Tensor:
    def _halton(i: int, b: int) -> float:
        f = 1.0
        r = 0.0
        while i > 0:
            f = f / b
            r = r + f * (i % b)
            i = i // b
        return r
    pts = torch.tensor([[ _halton(i+1, base_a), _halton(i+1, base_b) ] for i in range(n)], dtype=torch.float32)
    return pts


def render(
    camera: Camera,
    gaussian_batch: Dict[str, torch.Tensor],
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tile_size: int = 256,
    depth_epsilon: float = 1e-4,
    use_binning: bool = True,
    use_amp: bool = True,
    approx_isotropic: bool = False,
    max_gaussians_per_tile: Optional[int] = None,
    # Quality/AA
    ssaa_scale: int = 1,
    aa_samples: int = 1,
    # Extras
    return_depth: bool = False,
    return_normals: bool = False,
    lighting: Optional[Dict] = None,
    # Internal: subpixel jitter offset (dx, dy) in pixels for AA
    subpixel_offset: Optional[Tuple[float, float]] = None,
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

    # Super-sampling AA by rendering at higher resolution and downsampling
    if ssaa_scale is not None and ssaa_scale > 1 and subpixel_offset is None:
        w_hi = int(w * ssaa_scale)
        h_hi = int(h * ssaa_scale)
        cam_hi = Camera(
            width=w_hi,
            height=h_hi,
            fx=camera.fx * ssaa_scale,
            fy=camera.fy * ssaa_scale,
            cx=camera.cx * ssaa_scale,
            cy=camera.cy * ssaa_scale,
            world_to_camera=camera.world_to_camera,
            device=camera.device,
        )
        out_hi = render(
            cam_hi,
            gaussian_batch,
            background_color=background_color,
            tile_size=tile_size,
            depth_epsilon=depth_epsilon,
            use_binning=use_binning,
            use_amp=use_amp,
            approx_isotropic=approx_isotropic,
            max_gaussians_per_tile=max_gaussians_per_tile,
            ssaa_scale=1,
            aa_samples=aa_samples,
            return_depth=return_depth,
            return_normals=return_normals,
            lighting=None,  # apply lighting after downsample
            subpixel_offset=None,
        )
        # Unpack results (support 2 or 4 returns)
        if isinstance(out_hi, tuple) and len(out_hi) == 2:
            rgb_hi, a_hi = out_hi
            depth_hi = normals_hi = None
        else:
            rgb_hi, a_hi, depth_hi, normals_hi = out_hi  # type: ignore[misc]

        def _down(x: torch.Tensor) -> torch.Tensor:
            x_in = x.permute(2, 0, 1).unsqueeze(0)
            x_out = F.interpolate(x_in, size=(h, w), mode="area")
            return x_out.squeeze(0).permute(1, 2, 0)

        rgb = _down(rgb_hi)
        acc_a = _down(a_hi)
        results: List[torch.Tensor] = [rgb.clamp(0.0, 1.0), acc_a.clamp(0.0, 1.0)]
        depth_map = None
        normals_map = None
        if return_depth and depth_hi is not None:
            depth_map = _down(depth_hi)
            results.append(depth_map)
        if return_normals and normals_hi is not None:
            normals_map = _down(normals_hi)
            # Re-normalize after downsampling
            normals_map = F.normalize(normals_map, dim=-1)
            results.append(normals_map)

        # Apply lighting post downsample
        if lighting is not None:
            # If depth/normal missing, compute normals from depth if requested
            if normals_map is None and return_normals and depth_map is not None:
                normals_map = _compute_normals_from_depth(camera, depth_map, acc_a)
            rgb = _apply_lighting(camera, rgb, acc_a, depth_map, normals_map, lighting)
            results[0] = rgb

        return tuple(results)  # type: ignore[return-value]

    # Jittered multi-sample AA by subpixel offsets and averaging
    if (aa_samples is not None and aa_samples > 1) and subpixel_offset is None:
        offsets_01 = _generate_halton_2d(aa_samples, 2, 3) - 0.5  # [-0.5, 0.5)
        accum_rgb = torch.zeros((h, w, 3), device=device)
        accum_a = torch.zeros((h, w, 1), device=device)
        accum_depth: Optional[torch.Tensor] = torch.zeros((h, w, 1), device=device) if return_depth else None
        accum_normals: Optional[torch.Tensor] = torch.zeros((h, w, 3), device=device) if return_normals else None
        for i in range(aa_samples):
            dx = float(offsets_01[i, 0].item())
            dy = float(offsets_01[i, 1].item())
            out = render(
                camera,
                gaussian_batch,
                background_color=background_color,
                tile_size=tile_size,
                depth_epsilon=depth_epsilon,
                use_binning=use_binning,
                use_amp=use_amp,
                approx_isotropic=approx_isotropic,
                max_gaussians_per_tile=max_gaussians_per_tile,
                ssaa_scale=1,
                aa_samples=1,
                return_depth=return_depth,
                return_normals=return_normals,
                lighting=None,
                subpixel_offset=(dx, dy),
            )
            if isinstance(out, tuple) and len(out) == 2:
                rgb_i, a_i = out
                d_i = n_i = None
            else:
                rgb_i, a_i, d_i, n_i = out  # type: ignore[misc]
            accum_rgb += rgb_i
            accum_a += a_i
            if return_depth and d_i is not None and accum_depth is not None:
                accum_depth += d_i
            if return_normals and n_i is not None and accum_normals is not None:
                accum_normals += n_i
        rgb = (accum_rgb / float(aa_samples)).clamp(0.0, 1.0)
        acc_a = (accum_a / float(aa_samples)).clamp(0.0, 1.0)
        results2: List[torch.Tensor] = [rgb, acc_a]
        depth_map2 = None
        normals_map2 = None
        if return_depth and accum_depth is not None:
            depth_map2 = accum_depth / float(aa_samples)
            results2.append(depth_map2)
        if return_normals and accum_normals is not None:
            normals_map2 = F.normalize(accum_normals / float(aa_samples), dim=-1)
            results2.append(normals_map2)

        # Apply lighting post AA averaging
        if lighting is not None:
            rgb = _apply_lighting(camera, rgb, acc_a, depth_map2, normals_map2, lighting)
            results2[0] = rgb
        return tuple(results2)  # type: ignore[return-value]
    bg = torch.tensor(background_color, device=device, dtype=torch.float32).view(1, 1, 3).repeat(h, w, 1)
    rgb = torch.zeros((h, w, 3), device=device, dtype=torch.float32)
    acc_a = torch.zeros((h, w, 1), device=device, dtype=torch.float32)
    # Optional accumulators for depth (unnormalized numerator accumulation)
    depth_num = torch.zeros((h, w, 1), device=device, dtype=torch.float32) if return_depth else None

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

    # Optional SH color evaluation with view-dependence.
    use_sh = bool(gaussian_batch.get("use_sh", False))
    if use_sh:
        degree = int(gaussian_batch["sh_degree"])  # type: ignore[index]
        sh_coeffs = gaussian_batch["sh_coeffs"][valid]  # (M,B,3)
        # View direction per Gaussian: from point to camera center
        cam_center = camera.camera_to_world[:3, 3].unsqueeze(0)  # (1,3)
        viewdirs = torch.nn.functional.normalize(cam_center - means[valid], dim=-1)
        colors = evaluate_sh_rgb(viewdirs, sh_coeffs, degree)

    # Adaptive density control prior to compositing
    control_batch = {"covs_img": covs_img, "alphas": alphas}
    controlled = adaptive_density_control(control_batch)
    alphas = controlled["alphas"]

    # Global pruning with screen bounds to reduce work
    prune_in = {"pixels": pixels, "covs_img": covs_img, "alphas": alphas, "colors": colors, "depths": depths}
    bounds = torch.tensor([0, 0, w, h], dtype=torch.float32, device=pixels.device)
    pruned = prune_small_contributors(prune_in, bounds, sigma_thresh_px=0.1, alpha_thresh=0.005)
    pixels = pruned["pixels"]
    covs_img = pruned["covs_img"]
    alphas = pruned["alphas"]
    colors = pruned["colors"]
    depths = pruned["depths"]

    # Sort by depth front-to-back (near to far) for correct transmittance compositing
    sort_idx = torch.argsort(depths + depth_epsilon, descending=False)
    pixels = pixels[sort_idx]
    covs_img = covs_img[sort_idx]
    colors = colors[sort_idx]
    alphas = alphas[sort_idx]
    depths = depths[sort_idx]

    # Optional per-tile binning to reduce per-tile gaussian scanning
    tile_bins: Optional[List[torch.Tensor]] = None
    if use_binning and pixels.shape[0] > 2000:
        tile_bins = _build_tile_bins(pixels, covs_img, w, h, tile_size)

    # Optionally approximate as isotropic to speed up evaluation and stabilize radii
    if approx_isotropic and pixels.numel() > 0:
        std = torch.sqrt(torch.maximum(covs_img[:, 0, 0], covs_img[:, 1, 1]).clamp_min(1e-12))
        var = (std * std).unsqueeze(-1)
        covs_img = torch.zeros_like(covs_img)
        covs_img[:, 0, 0] = var[:, 0]
        covs_img[:, 1, 1] = var[:, 0]

    # Precompute inverses of 2x2 covariances once (analytic inversion for speed)
    covs_img_inv_all = _invert_2x2(covs_img) if pixels.numel() > 0 else covs_img

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
            if subpixel_offset is not None:
                # Apply subpixel jitter (dx, dy)
                grid[..., 0] = grid[..., 0] + float(subpixel_offset[0])
                grid[..., 1] = grid[..., 1] + float(subpixel_offset[1])

            # Accumulate contributions of Gaussians for this tile
            tile_rgb = rgb[y0:y1, x0:x1, :]
            tile_a = acc_a[y0:y1, x0:x1, :]
            tile_d = depth_num[y0:y1, x0:x1, :] if depth_num is not None else None

            if tile_bins is None:
                # Vectorized mask per Gaussian for this tile
                std = torch.sqrt(torch.maximum(covs_img[:, 0, 0], covs_img[:, 1, 1]))  # (M,)
                min_x = pixels[:, 0] - 3.0 * std
                max_x = pixels[:, 0] + 3.0 * std
                min_y = pixels[:, 1] - 3.0 * std
                max_y = pixels[:, 1] + 3.0 * std
                overlaps = (max_x >= x0) & (min_x <= x1 - 1) & (max_y >= y0) & (min_y <= y1 - 1)
                idx = torch.nonzero(overlaps, as_tuple=False).squeeze(-1)
            else:
                tx = x0 // tile_size
                ty = y0 // tile_size
                tiles_x = (w + tile_size - 1) // tile_size
                tile_index = ty * tiles_x + tx
                idx = tile_bins[tile_index]
            
            if idx.numel() > 0:
                if (max_gaussians_per_tile is not None) and (idx.numel() > max_gaussians_per_tile):
                    idx = idx[:max_gaussians_per_tile]
                mu = pixels[idx]              # (K,2)
                sigma = covs_img[idx]         # (K,2,2)
                col = colors[idx]             # (K,3)
                alp = alphas[idx]             # (K,)
                dep = depths[idx]             # (K,)

                # Evaluate densities in chunks to control memory
                # Flatten the tile grid to (T,2) for batched evaluation
                grid_flat = grid.view(-1, 2)  # (T,2)
                t_h, t_w = y1 - y0, x1 - x0
                t_n = t_h * t_w
                # Flattened accumulators for vectorized compositing
                tile_rgb_flat = tile_rgb.reshape(t_n, 3)
                tile_a_flat = tile_a.reshape(t_n, 1)
                tile_depth_flat = tile_d.reshape(t_n, 1) if tile_d is not None else None
                # Heuristic chunk size based on tile pixel count
                max_gauss_per_chunk = 64
                for start in range(0, mu.shape[0], max_gauss_per_chunk):
                    end = min(start + max_gauss_per_chunk, mu.shape[0])
                    mu_c = mu[start:end]
                    sigma_c = sigma[start:end]
                    col_c = col[start:end]
                    alp_c = alp[start:end]
                    dep_c = dep[start:end]

                    # Broadcasted evaluation: for each gaussian in chunk, compute density over tile
                    # (Kc,T) densities
                    if use_amp and grid_flat.is_cuda:
                        # Mixed precision for speed
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            inv = covs_img_inv_all[idx[start:end]]  # (Kc,2,2)
                            diff = grid_flat[None, :, :] - mu_c[:, None, :]  # (Kc,T,2)
                            mahal = torch.einsum("k t i, k i j, k t j -> k t", diff, inv, diff)
                            density = torch.exp(-0.5 * mahal)  # (Kc,T)
                            a = 1.0 - torch.exp(-alp_c[:, None] * density)  # (Kc,T)
                    else:
                        inv = covs_img_inv_all[idx[start:end]]  # (Kc,2,2)
                        diff = grid_flat[None, :, :] - mu_c[:, None, :]  # (Kc,T,2)
                        mahal = torch.einsum("k t i, k i j, k t j -> k t", diff, inv, diff)
                        density = torch.exp(-0.5 * mahal)  # (Kc,T)
                        a = 1.0 - torch.exp(-alp_c[:, None] * density)  # (Kc,T)

                    # Vectorized front-to-back compositing over the chunk
                    # weights_k,t = a_k,t * prod_{j<k}(1 - a_j,t)
                    one_minus_a = 1.0 - a  # (Kc,T)
                    if a.shape[0] > 1:
                        prefix = torch.cumprod(one_minus_a[:-1, :], dim=0)
                        prefix = torch.cat([
                            torch.ones((1, t_n), dtype=a.dtype, device=a.device),
                            prefix
                        ], dim=0)  # (Kc,T)
                    else:
                        prefix = torch.ones_like(a)
                    weights = a * prefix  # (Kc,T)

                    # Chunk alpha and color over black background
                    chunk_alpha_flat = weights.sum(dim=0, keepdim=True).t()  # (T,1)
                    # (T,3) = (T,Kc) @ (Kc,3)
                    chunk_rgb_flat = weights.t() @ col_c  # (T,3)
                    # Optional chunk depth numerator
                    if tile_depth_flat is not None:
                        # (T,1) = (T,Kc) @ (Kc,1)
                        chunk_depth_flat = (weights.t() @ dep_c.unsqueeze(-1))  # (T,1)

                    # Correct front-to-back incremental compositing using old transmittance
                    trans_old = (1.0 - tile_a_flat)
                    tile_rgb_flat = tile_rgb_flat + trans_old * chunk_rgb_flat
                    if tile_depth_flat is not None:
                        tile_depth_flat = tile_depth_flat + trans_old * chunk_depth_flat
                    tile_a_flat = tile_a_flat + trans_old * chunk_alpha_flat

                    # Early exit for saturated tiles to save work
                    if torch.all(tile_a_flat > 0.999):
                        break

                # Reshape back to tiles
                tile_rgb = tile_rgb_flat.view(t_h, t_w, 3)
                tile_a = tile_a_flat.view(t_h, t_w, 1)
                if tile_depth_flat is not None and tile_d is not None:
                    tile_d = tile_depth_flat.view(t_h, t_w, 1)
                    depth_num[y0:y1, x0:x1, :] = tile_d

            rgb[y0:y1, x0:x1, :] = tile_rgb
            acc_a[y0:y1, x0:x1, :] = tile_a

    # Composite with background once at the end
    rgb = (rgb + (1.0 - acc_a) * bg).clamp(0.0, 1.0)
    acc_a = acc_a.clamp(0.0, 1.0)

    outputs: List[torch.Tensor] = [rgb, acc_a]

    depth_out: Optional[torch.Tensor] = None
    normals_out: Optional[torch.Tensor] = None
    if return_depth and depth_num is not None:
        # Expected depth = numerator / alpha; where alpha≈0, keep 0
        depth_out = torch.nan_to_num(depth_num / (acc_a + 1e-8), nan=0.0, posinf=0.0, neginf=0.0)
        outputs.append(depth_out)
    if return_normals:
        if depth_out is None:
            # If depth wasn't requested, still compute normals from a proxy depth (zeros)
            # This will yield zeros; better require depth
            depth_proxy = torch.zeros((h, w, 1), device=device, dtype=torch.float32)
            normals_out = _compute_normals_from_depth(camera, depth_proxy, acc_a)
        else:
            normals_out = _compute_normals_from_depth(camera, depth_out, acc_a)
        outputs.append(normals_out)

    # Optional simple lighting (Lambert/Phong) using normals
    if lighting is not None:
        depth_for_light = depth_out if return_depth else None
        normals_for_light = normals_out if return_normals else None
        shaded = _apply_lighting(camera, rgb, acc_a, depth_for_light, normals_for_light, lighting)
        outputs[0] = shaded

    return tuple(outputs)  # type: ignore[return-value]


def _apply_lighting(
    camera: Camera,
    rgb: torch.Tensor,  # (H,W,3) albedo-ish
    alpha: torch.Tensor,  # (H,W,1)
    depth: Optional[torch.Tensor],
    normals: Optional[torch.Tensor],
    lighting: Dict,
) -> torch.Tensor:
    """Apply simple lighting on top of rendered RGB using depth/normals when provided.

    lighting keys:
      type: 'lambert' | 'phong' (default 'lambert')
      light_dir: (3,) in camera space (default [0,0,1])
      light_color: (3,) linear RGB (default [1,1,1])
      ambient: float in [0,1] (default 0.1)
      specular_power: float (for phong, default 32)
    """
    if normals is None and depth is not None:
        normals = _compute_normals_from_depth(camera, depth, alpha)
    if normals is None:
        return rgb

    h, w, _ = rgb.shape
    device = rgb.device
    ldir = torch.tensor(lighting.get("light_dir", (0.0, 0.0, 1.0)), dtype=rgb.dtype, device=device)
    ldir = F.normalize(ldir, dim=0)
    lcol = torch.tensor(lighting.get("light_color", (1.0, 1.0, 1.0)), dtype=rgb.dtype, device=device).view(1, 1, 3)
    ambient = float(lighting.get("ambient", 0.1))
    ltype = str(lighting.get("type", "lambert")).lower()
    spec_pow = float(lighting.get("specular_power", 32.0))

    # Diffuse
    ndotl = torch.clamp((normals @ ldir.view(3,)).unsqueeze(-1), 0.0, 1.0)  # (H,W,1)
    diffuse = ndotl.repeat(1, 1, 3) * lcol

    if ltype == "phong":
        # View dir is along -Z in camera space for each pixel
        vdir = torch.tensor([0.0, 0.0, -1.0], dtype=rgb.dtype, device=device)
        # Reflect l about n: r = 2(n·l)n - l
        r = 2.0 * (normals @ ldir.view(3,)).unsqueeze(-1) * normals - ldir.view(1, 1, 3)
        r = F.normalize(r, dim=-1)
        spec = torch.clamp((r @ vdir.view(3,)).unsqueeze(-1), 0.0, 1.0) ** spec_pow
        specular = spec.repeat(1, 1, 3) * lcol
    else:
        specular = torch.zeros_like(rgb)

    shaded = rgb * (ambient + diffuse) + specular
    # Preserve background using alpha
    shaded = shaded * alpha + rgb * (1.0 - alpha)
    return shaded.clamp(0.0, 1.0)


