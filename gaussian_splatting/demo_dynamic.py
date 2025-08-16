from __future__ import annotations

import argparse
from typing import Optional

import torch

from .camera import Camera
from .gaussian4d import Gaussian4D, pack_gaussians_4d, gaussian4d_to_3d_batch
from .spherical_harmonics import num_sh_coeffs
from .renderer import render
from .datasets_dynamic import load_synthetic_bouncing_sphere


def seq_to_gaussians4d(seq, position_variance: float = 0.0015):
    # Estimate velocity from first two frames as a simple motion proxy
    f0, f1 = seq.frames[0], seq.frames[1]
    dt = max(f1.time_s - f0.time_s, 1e-3)
    velocities = (f1.points - f0.points) / dt
    covs = torch.eye(3, device=f0.points.device).unsqueeze(0).repeat(f0.points.shape[0], 1, 1) * position_variance

    gaussians = []
    for i in range(f0.points.shape[0]):
        gaussians.append(Gaussian4D(mean0=f0.points[i], velocity=velocities[i], covariance=covs[i], color=f0.colors[i], alpha=1.0))
    return tuple(gaussians)


def run_dynamic_demo(width: int, height: int, out_prefix: str, num_frames: int = 30, sh_degree: int = 1, tile_size: int = 256, approx_isotropic: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    seq = load_synthetic_bouncing_sphere(num_frames=num_frames, device=device)
    gaussians4d = seq_to_gaussians4d(seq)
    batch4d = pack_gaussians_4d(gaussians4d, device=device)

    if sh_degree > 0:
        # DC-only SH init from colors
        colors = batch4d["colors"]
        B = (sh_degree + 1) * (sh_degree + 1)
        coeffs = torch.zeros((colors.shape[0], B, 3), device=device)
        coeffs[:, 0, :] = colors / 0.28209479177387814
        batch4d.update({"use_sh": True, "sh_degree": sh_degree, "sh_coeffs": coeffs})

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

    for i, frame in enumerate(seq):
        batch3d = gaussian4d_to_3d_batch(batch4d, time_s=frame.time_s - seq.frames[0].time_s)
        rgb, _ = render(cam, batch3d, background_color=(1.0, 1.0, 1.0), tile_size=tile_size, approx_isotropic=approx_isotropic)
        rgb = rgb.clamp(0.0, 1.0).pow(1.0 / 2.2)
        img = (rgb * 255.0).byte().cpu()
        try:
            import imageio.v3 as iio
            iio.imwrite(f"{out_prefix}_{i:03d}.png", img.numpy())
        except Exception:
            from PIL import Image
            Image.fromarray(img.numpy()).save(f"{out_prefix}_{i:03d}.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Dynamic 4D Gaussian Splatting Demo")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--out_prefix", type=str, default="dynamic_demo")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--sh_degree", type=int, default=1)
    p.add_argument("--tile_size", type=int, default=256)
    p.add_argument("--no_isotropic", action="store_true")
    args = p.parse_args()
    run_dynamic_demo(args.width, args.height, args.out_prefix, num_frames=args.frames, sh_degree=args.sh_degree, tile_size=args.tile_size, approx_isotropic=not args.no_isotropic)


if __name__ == "__main__":
    main()


