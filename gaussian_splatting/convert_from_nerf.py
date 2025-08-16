from __future__ import annotations

import argparse
from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F

from mip_nerf.config import get_config
from mip_nerf.model import MipNeRF
from mip_nerf.datasets import get_dataset
from mip_nerf.ray_utils import namedtuple_map
from gaussian_splatting.gaussian import colors_to_sh_coeffs
from gaussian_splatting.renderer import render
from gaussian_splatting.camera import Camera


def _sigma_to_alpha(density_sigma: torch.Tensor, t_vals: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Convert NeRF density sigma to per-sample alpha via delta distances.

    Args:
        density_sigma: (B, S, 1)
        t_vals: (B, S+1)
        dirs: (B, 3)
    Returns:
        alpha: (B, S)
    """
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])  # (B,S)
    deltas = (t_vals[..., 1:] - t_vals[..., :-1]) * torch.linalg.norm(dirs[..., None, :], dim=-1)
    density_delta = density_sigma[..., 0] * deltas
    alpha = 1.0 - torch.exp(-density_delta)
    return alpha


def _cov_diag_to_full(cov: torch.Tensor) -> torch.Tensor:
    """If covariance from mip-NeRF is diagonal (B,S,3), expand to (B,S,3,3)."""
    if cov.ndim == 4 and cov.shape[-1] == 3 and cov.shape[-2] == 3:
        return cov
    if cov.ndim == 3 and cov.shape[-1] == 3:
        b, s, _ = cov.shape
        out = torch.zeros((b, s, 3, 3), dtype=cov.dtype, device=cov.device)
        out[..., 0, 0] = cov[..., 0]
        out[..., 1, 1] = cov[..., 1]
        out[..., 2, 2] = cov[..., 2]
        return out
    raise ValueError("Unexpected covariance shape for conversion: {}".format(tuple(cov.shape)))


def _samples_to_batch(
    means3d: torch.Tensor,        # (B,S,3)
    covs3d: torch.Tensor,         # (B,S,3,3) or (B,S,3)
    raw_rgb_sigma: torch.Tensor,  # (B,S,4)
    t_vals: torch.Tensor,         # (B,S+1)
    dirs: torch.Tensor,           # (B,3)
    alpha_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Convert NeRF sample tensors directly to renderer batch dict (no Python loops)."""
    covs = _cov_diag_to_full(covs3d).clone()
    eye = torch.eye(3, device=covs.device, dtype=covs.dtype)
    covs = covs + 1e-6 * eye.view(1, 1, 3, 3)
    rgbs = raw_rgb_sigma[..., :3].clamp(0.0, 1.0)
    sigmas = torch.relu(raw_rgb_sigma[..., 3:4])
    alphas = _sigma_to_alpha(sigmas, t_vals, dirs).clamp(0.0, 1.0) * alpha_scale
    # Flatten B,S to N
    means_flat = means3d.reshape(-1, 3)
    covs_flat = covs.reshape(-1, 3, 3)
    colors_flat = rgbs.reshape(-1, 3)
    alphas_flat = alphas.reshape(-1)
    return {"means": means_flat, "covs": covs_flat, "colors": colors_flat, "alphas": alphas_flat}


def extract_gaussians_from_trained_nerf(
    cfg=None,
    sh_degree: int = 2,
    alpha_scale: float = 1.0,
    max_gaussians: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """Load a trained mip-NeRF and convert its last-level samples to 3D Gaussians.

    Returns:
        batch: dict for renderer with optional SH coeffs attached
    """
    if cfg is None:
        cfg = get_config()
    if device is None:
        device = cfg.device if isinstance(cfg.device, torch.device) else torch.device(str(cfg.device))

    # Build dataset to obtain ray parameter ranges for a representative view
    dataset = get_dataset(cfg.dataset_name, cfg.base_dir, split="render", factor=cfg.factor, device=device)
    # Reuse the first render camera to create a Camera object for testing
    w, h = dataset.w, dataset.h

    # Load model
    # Infer NeRF-W dims if present in checkpoint
    state = torch.load(cfg.model_weight_path, map_location="cpu")
    has_nerfw = any(k.startswith('appearance_embed') for k in state.keys()) if isinstance(state, dict) else False
    has_transient = any(k.startswith('transient_net') for k in state.keys()) if isinstance(state, dict) else False
    num_images = 0
    appearance_dim = getattr(cfg, 'appearance_dim', 32)
    if has_nerfw and isinstance(state, dict):
        w_ap = state.get('appearance_embed.weight')
        if isinstance(w_ap, torch.Tensor):
            num_images, appearance_dim = int(w_ap.shape[0]), int(w_ap.shape[1])

    model = MipNeRF(
        use_viewdirs=cfg.use_viewdirs,
        randomized=False,
        ray_shape=cfg.ray_shape,
        white_bkgd=cfg.white_bkgd,
        num_levels=cfg.num_levels,
        num_samples=cfg.num_samples,
        hidden=cfg.hidden,
        density_noise=0.0,
        density_bias=cfg.density_bias,
        rgb_padding=cfg.rgb_padding,
        resample_padding=cfg.resample_padding,
        min_deg=cfg.min_deg,
        max_deg=cfg.max_deg,
        viewdirs_min_deg=cfg.viewdirs_min_deg,
        viewdirs_max_deg=cfg.viewdirs_max_deg,
        device=device,
        use_hash_encoding=cfg.use_hash_encoding,
        use_nerfw=has_nerfw or getattr(cfg, 'use_nerfw', False),
        appearance_dim=appearance_dim,
        num_images=num_images,
        use_transient=has_transient or getattr(cfg, 'use_transient', False),
        transient_dim=getattr(cfg, 'transient_dim', 16),
        return_raw=True,
    )
    model.load_state_dict(torch.load(cfg.model_weight_path, map_location=device))
    model.eval()

    # Prepare rays from dataset and process in chunks to keep memory bounded
    rays_all = dataset.rays
    num_rays = rays_all.origins.shape[0]
    chunk = max(1024, int(getattr(cfg, 'chunks', 8192)))
    # Streaming top-K buffers
    k_limit = max_gaussians if (max_gaussians is not None) else 200_000
    means_buf = None
    covs_buf = None
    cols_buf = None
    alp_buf = None
    with torch.no_grad():
        for i in range(0, num_rays, chunk):
            ray_chunk = namedtuple_map(lambda r: r[i:i+chunk].to(device), rays_all)
            out = model(ray_chunk)
            # Unpack extended return
            comp_rgbs, distances, accs, raws, means3d, covs3d, t_vals = out
            batch_part = _samples_to_batch(means3d, covs3d, raws, t_vals, ray_chunk.directions, alpha_scale=alpha_scale)
            # Concatenate with existing buffer, then keep top-K by alpha
            if means_buf is None:
                means_buf = batch_part["means"]
                covs_buf = batch_part["covs"]
                cols_buf = batch_part["colors"]
                alp_buf = batch_part["alphas"]
            else:
                means_buf = torch.cat([means_buf, batch_part["means"]], dim=0)
                covs_buf = torch.cat([covs_buf, batch_part["covs"]], dim=0)
                cols_buf = torch.cat([cols_buf, batch_part["colors"]], dim=0)
                alp_buf = torch.cat([alp_buf, batch_part["alphas"]], dim=0)
            if means_buf.shape[0] > k_limit:
                vals, idx = torch.topk(alp_buf, k=k_limit, largest=True, sorted=False)
                means_buf = means_buf[idx]
                covs_buf = covs_buf[idx]
                cols_buf = cols_buf[idx]
                alp_buf = alp_buf[idx]
        # After streaming, optionally shrink further to exact max_gaussians
        if (max_gaussians is not None) and (means_buf is not None) and (means_buf.shape[0] > max_gaussians):
            k = int(max_gaussians)
            vals, idx = torch.topk(alp_buf, k=k, largest=True, sorted=False)
            means_buf = means_buf[idx]
            covs_buf = covs_buf[idx]
            cols_buf = cols_buf[idx]
            alp_buf = alp_buf[idx]
    if means_buf is None:
        raise RuntimeError("No Gaussians were extracted from NeRF rays.")
    batch = {"means": means_buf, "covs": covs_buf, "colors": cols_buf, "alphas": alp_buf}
    if sh_degree > 0:
        coeffs = colors_to_sh_coeffs(batch["colors"], sh_degree)
        batch.update({"use_sh": True, "sh_degree": sh_degree, "sh_coeffs": coeffs})
    return batch, (w, h)


def demo_convert_and_render(
    out_path: str = "gaussian_demo.png",
    sh_degree: int = 2,
    alpha_scale: float = 1.0,
    max_gaussians: Optional[int] = None,
    tile_size: int = 256,
    use_amp: bool = True,
    approx_isotropic: bool = False,
    max_gaussians_per_tile: int = 4096,
    ssaa_scale: int = 1,
    aa_samples: int = 1,
    save_depth: Optional[str] = None,
    save_normals: Optional[str] = None,
    benchmark: bool = False,
) -> None:
    cfg = get_config()
    # Choose best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    batch, (w, h) = extract_gaussians_from_trained_nerf(cfg, sh_degree=sh_degree, alpha_scale=alpha_scale, max_gaussians=max_gaussians, device=device)

    # Create a simple camera roughly matching dataset FOV for preview
    cam = Camera.from_look_at(
        width=w,
        height=h,
        fx=0.9 * w,
        fy=0.9 * h,
        eye=(2.2, 1.2, 2.5),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        device=device,
    )

    if benchmark:
        # Warmup and timing
        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        _sync()
        import time
        for _ in range(5):
            _ = render(
                cam,
                batch,
                background_color=(1.0, 1.0, 1.0),
                tile_size=tile_size,
                use_amp=use_amp,
                approx_isotropic=approx_isotropic,
                max_gaussians_per_tile=max_gaussians_per_tile,
                ssaa_scale=ssaa_scale,
                aa_samples=aa_samples,
                return_depth=False,
                return_normals=False,
                lighting=None,
            )
        _sync()
        iters = 60
        t0 = time.time()
        for _ in range(iters):
            _ = render(
                cam,
                batch,
                background_color=(1.0, 1.0, 1.0),
                tile_size=tile_size,
                use_amp=use_amp,
                approx_isotropic=approx_isotropic,
                max_gaussians_per_tile=max_gaussians_per_tile,
                ssaa_scale=ssaa_scale,
                aa_samples=aa_samples,
                return_depth=False,
                return_normals=False,
                lighting=None,
            )
        _sync()
        t1 = time.time()
        fps = iters / (t1 - t0)
        print(f"Rendered {iters} frames in {t1 - t0:.3f}s => {fps:.2f} FPS")
        return
    else:
        out = render(
            cam,
            batch,
            background_color=(1.0, 1.0, 1.0),
            tile_size=tile_size,
            use_amp=use_amp,
            approx_isotropic=approx_isotropic,
            max_gaussians_per_tile=max_gaussians_per_tile,
            ssaa_scale=ssaa_scale,
            aa_samples=aa_samples,
            return_depth=save_depth is not None,
            return_normals=save_normals is not None,
            lighting=None,
        )
    if isinstance(out, tuple) and len(out) == 2:
        rgb, a = out
        depth = normals = None
    elif isinstance(out, tuple) and len(out) == 3:
        rgb, a, depth = out
        normals = None
    else:
        rgb, a, depth, normals = out

    # Save
    img = (rgb.clamp(0.0, 1.0).pow(1.0 / 2.2) * 255.0).byte().cpu()
    try:
        import imageio.v3 as iio
        iio.imwrite(out_path, img.numpy())
        if save_depth is not None and depth is not None:
            d = depth[..., 0].detach().cpu()
            valid = d > 0
            if valid.any():
                d_min = d[valid].min()
                d_max = d[valid].max()
                d_norm = (d - d_min) / (d_max - d_min + 1e-8)
            else:
                d_norm = d * 0
            iio.imwrite(save_depth, (d_norm.numpy() * 255.0).astype('uint8'))
        if save_normals is not None and normals is not None:
            n = normals.detach().cpu()
            n_img = ((n + 1.0) * 0.5).clamp(0.0, 1.0)
            iio.imwrite(save_normals, (n_img.numpy() * 255.0).astype('uint8'))
    except Exception:
        from PIL import Image
        Image.fromarray(img.numpy()).save(out_path)
        if save_depth is not None and depth is not None:
            d = depth[..., 0].detach().cpu()
            valid = d > 0
            if valid.any():
                d_min = d[valid].min()
                d_max = d[valid].max()
                d_norm = (d - d_min) / (d_max - d_min + 1e-8)
            else:
                d_norm = d * 0
            Image.fromarray((d_norm.numpy() * 255.0).astype('uint8')).save(save_depth)
        if save_normals is not None and normals is not None:
            n = normals.detach().cpu()
            n_img = ((n + 1.0) * 0.5).clamp(0.0, 1.0)
            Image.fromarray((n_img.numpy() * 255.0).astype('uint8')).save(save_normals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert trained mip-NeRF to Gaussian splats and render a preview")
    parser.add_argument("--out", type=str, default="gaussian_demo.png")
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--alpha_scale", type=float, default=1.0)
    parser.add_argument("--max_gaussians", type=int, default=None)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--approx_isotropic", action="store_true")
    parser.add_argument("--max_gaussians_per_tile", type=int, default=4096)
    parser.add_argument("--ssaa_scale", type=int, default=1)
    parser.add_argument("--aa_samples", type=int, default=1)
    parser.add_argument("--save_depth", type=str, default=None)
    parser.add_argument("--save_normals", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    demo_convert_and_render(
        out_path=args.out,
        sh_degree=args.sh_degree,
        alpha_scale=args.alpha_scale,
        max_gaussians=args.max_gaussians,
        tile_size=args.tile_size,
        use_amp=not args.no_amp,
        approx_isotropic=args.approx_isotropic,
        max_gaussians_per_tile=args.max_gaussians_per_tile,
        ssaa_scale=args.ssaa_scale,
        aa_samples=args.aa_samples,
        save_depth=args.save_depth,
        save_normals=args.save_normals,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()


