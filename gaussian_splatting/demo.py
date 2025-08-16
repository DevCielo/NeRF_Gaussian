from __future__ import annotations

import argparse
from typing import Optional

import torch

from camera import Camera
from pointcloud_loader import load_point_cloud, points_to_gaussians
from gaussian import pack_gaussians
from renderer import render


def run_demo(pointcloud_path: Optional[str], width: int, height: int, out_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points, colors = load_point_cloud(pointcloud_path, device=device)
    # Sharper splats via lower base variance
    gaussians = points_to_gaussians(points, colors, position_variance=0.0015)
    batch = pack_gaussians(gaussians, device)

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

    rgb, a = render(cam, batch, background_color=(1.0, 1.0, 1.0))
    # Apply simple sRGB gamma for more contrast
    rgb = rgb.clamp(0.0, 1.0).pow(1.0 / 2.2)
    img = (rgb * 255.0).byte().cpu()

    try:
        import imageio.v3 as iio
        iio.imwrite(out_path, img.numpy())
    except Exception:
        # Fallback to PIL if available
        try:
            from PIL import Image

            Image.fromarray(img.numpy()).save(out_path)
        except Exception as e:
            raise RuntimeError("Failed to save image. Install imageio or pillow.") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Gaussian Splatting demo")
    parser.add_argument("--pointcloud", type=str, default=None, help="Path to .pt with 'points' and 'colors'")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", type=str, default="gaussian_demo.png")
    args = parser.parse_args()
    run_demo(args.pointcloud, args.width, args.height, args.out)


if __name__ == "__main__":
    main()


