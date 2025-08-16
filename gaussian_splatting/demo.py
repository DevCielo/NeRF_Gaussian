from __future__ import annotations

import argparse
from typing import Optional, Tuple

import torch

from .camera import Camera
from .pointcloud_loader import load_point_cloud, points_to_gaussians
from .gaussian import pack_gaussians, colors_to_sh_coeffs
from .spherical_harmonics import num_sh_coeffs
from .renderer import render


def run_demo(
    pointcloud_path: Optional[str],
    width: int,
    height: int,
    out_path: str,
    sh_degree: int = 2,
    benchmark: bool = False,
    tile_size: int = 256,
    use_amp: bool = True,
    approx_isotropic: bool = False,
    max_gaussians_per_tile: int = 4096,
    ssaa_scale: int = 1,
    aa_samples: int = 1,
    save_depth: Optional[str] = None,
    save_normals: Optional[str] = None,
    lighting_type: Optional[str] = None,
    light_dir: Optional[Tuple[float, float, float]] = None,
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    points, colors = load_point_cloud(pointcloud_path, device=device)
    # Sharper splats via lower base variance
    gaussians = points_to_gaussians(points, colors, position_variance=0.0015)
    batch = pack_gaussians(gaussians, device)
    # Attach SH as DC-only initialization to enable view-dependence
    if sh_degree > 0:
        coeffs = colors_to_sh_coeffs(batch["colors"], sh_degree)
        batch.update({
            "use_sh": True,
            "sh_degree": sh_degree,
            "sh_coeffs": coeffs,
        })

    cam = Camera.from_look_at(
        width=width,
        height=height,
        fx=0.9 * width,
        fy=0.9 * height,
        eye=(2.2, 1.2, 2.5),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        device=device,
    )

    if benchmark:
        # Device synchronize helper
        def _sync_device():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()

        _sync_device()
        import time
        t0 = time.time()
        # Warmup
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
            )
        _sync_device()
        t1 = time.time()
        # Timed iterations
        iters = 60
        t2 = time.time()
        for _ in range(iters):
            rgb, a = render(
                cam,
                batch,
                background_color=(1.0, 1.0, 1.0),
                tile_size=tile_size,
                use_amp=use_amp,
                approx_isotropic=approx_isotropic,
                max_gaussians_per_tile=max_gaussians_per_tile,
                ssaa_scale=ssaa_scale,
                aa_samples=aa_samples,
            )
        _sync_device()
        t3 = time.time()
        ips = iters / (t3 - t2)
        print(f"Rendered {iters} frames in {t3 - t2:.3f}s => {ips:.2f} FPS")
        # Skip saving when benchmarking
        return
    else:
        lighting = None
        if lighting_type is not None and lighting_type.lower() != "none":
            lighting = {
                "type": lighting_type.lower(),
                "light_dir": light_dir if light_dir is not None else (0.0, 0.0, 1.0),
                "ambient": 0.1,
                "light_color": (1.0, 1.0, 1.0),
                "specular_power": 32.0,
            }
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
            lighting=lighting,
        )
        if (save_depth is not None) or (save_normals is not None):
            if len(out) == 4:
                rgb, a, depth, normals = out  # type: ignore[misc]
            elif len(out) == 3:
                rgb, a, depth = out  # type: ignore[misc]
                normals = None
            else:
                rgb, a = out  # type: ignore[misc]
                depth = None
                normals = None
        else:
            rgb, a = out  # type: ignore[misc]
    # Apply simple sRGB gamma for more contrast
    rgb = rgb.clamp(0.0, 1.0).pow(1.0 / 2.2)
    img = (rgb * 255.0).byte().cpu()

    try:
        import imageio.v3 as iio
        iio.imwrite(out_path, img.numpy())
        if save_depth is not None and ('depth' in locals()) and (depth is not None):
            d = depth[..., 0].detach().cpu()
            valid = d > 0
            if valid.any():
                d_min = d[valid].min()
                d_max = d[valid].max()
                d_norm = (d - d_min) / (d_max - d_min + 1e-8)
            else:
                d_norm = d * 0
            iio.imwrite(save_depth, (d_norm.numpy() * 255.0).astype('uint8'))
        if save_normals is not None and ('normals' in locals()) and (normals is not None):
            n = normals.detach().cpu()
            n_img = ((n + 1.0) * 0.5).clamp(0.0, 1.0)
            iio.imwrite(save_normals, (n_img.numpy() * 255.0).astype('uint8'))
    except Exception:
        # Fallback to PIL if available
        try:
            from PIL import Image

            Image.fromarray(img.numpy()).save(out_path)
            if save_depth is not None and ('depth' in locals()) and (depth is not None):
                d = depth[..., 0].detach().cpu()
                valid = d > 0
                if valid.any():
                    d_min = d[valid].min()
                    d_max = d[valid].max()
                    d_norm = (d - d_min) / (d_max - d_min + 1e-8)
                else:
                    d_norm = d * 0
                Image.fromarray((d_norm.numpy() * 255.0).astype('uint8')).save(save_depth)
            if save_normals is not None and ('normals' in locals()) and (normals is not None):
                n = normals.detach().cpu()
                n_img = ((n + 1.0) * 0.5).clamp(0.0, 1.0)
                Image.fromarray((n_img.numpy() * 255.0).astype('uint8')).save(save_normals)
        except Exception as e:
            raise RuntimeError("Failed to save image. Install imageio or pillow.") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Gaussian Splatting demo")
    parser.add_argument("--pointcloud", type=str, default=None, help="Path to .pt with 'points' and 'colors'")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", type=str, default="gaussian_demo.png")
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--approx_isotropic", action="store_true")
    parser.add_argument("--max_gaussians_per_tile", type=int, default=4096)
    parser.add_argument("--ssaa_scale", type=int, default=1)
    parser.add_argument("--aa_samples", type=int, default=1)
    parser.add_argument("--save_depth", type=str, default=None)
    parser.add_argument("--save_normals", type=str, default=None)
    parser.add_argument("--lighting", type=str, default="none", choices=["none", "lambert", "phong"], help="Lighting model")
    parser.add_argument("--light_dir", type=float, nargs=3, default=None, help="Light direction in camera space")
    args = parser.parse_args()
    run_demo(
        args.pointcloud,
        args.width,
        args.height,
        args.out,
        args.sh_degree,
        args.benchmark,
        args.tile_size,
        not args.no_amp,
        args.approx_isotropic,
        args.max_gaussians_per_tile,
        args.ssaa_scale,
        args.aa_samples,
        args.save_depth,
        args.save_normals,
        args.lighting,
        tuple(args.light_dir) if args.light_dir is not None else None,
    )


if __name__ == "__main__":
    main()


