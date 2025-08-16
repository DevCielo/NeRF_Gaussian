from __future__ import annotations

import argparse
import os
import numpy as np

from .config import HybridConfig, NerfTrainingConfig, GaussianExportConfig, QualityConfig, LODConfig
from .orchestrator import HybridSystem


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Hybrid NeRF→Gaussian pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train_nerf")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--config", default=None)
    p_train.add_argument("--logdir", default="./mip_nerf/log")
    p_train.add_argument("--max_steps", type=int, default=20000)
    p_train.add_argument("--save_every", type=int, default=1000)

    p_export = sub.add_parser("export_gaussians")
    p_export.add_argument("--nerf_ckpt", required=True)
    p_export.add_argument("--outdir", default="./hybrid_out")
    p_export.add_argument("--num_gaussians", type=int, default=200000)
    p_export.add_argument("--sh_degree", type=int, default=3)
    p_export.add_argument("--opacity_thresh", type=float, default=0.01)
    p_export.add_argument("--position_jitter", type=float, default=0.0)
    p_export.add_argument("--pointcloud", default=None)

    p_render = sub.add_parser("render_once")
    p_render.add_argument("--gaussians", required=True)
    p_render.add_argument("--width", type=int, default=800)
    p_render.add_argument("--height", type=int, default=800)
    p_render.add_argument("--fx", type=float, default=800.0)
    p_render.add_argument("--fy", type=float, default=800.0)
    p_render.add_argument("--cx", type=float, default=None)
    p_render.add_argument("--cy", type=float, default=None)
    p_render.add_argument("--c2w_path", required=True, help="NumPy .npy 4x4 camera-to-world matrix")
    p_render.add_argument("--ssaa", type=int, default=None)
    p_render.add_argument("--msaa", type=int, default=None)
    p_render.add_argument("--max_per_tile", type=int, default=None)
    p_render.add_argument("--out", default="render.png")

    p_densify = sub.add_parser("densify")
    p_densify.add_argument("--nerf_ckpt", required=True)
    p_densify.add_argument("--gaussians", required=False, help="Existing NPZ (optional)")
    p_densify.add_argument("--target_count", type=int, required=True)
    p_densify.add_argument("--outdir", default="./hybrid_out")

    return p


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.cmd == "train_nerf":
        nerf_cfg = NerfTrainingConfig(
            dataset_path=args.data,
            config_path=args.config,
            output_dir=args.logdir,
            num_steps=args.max_steps,
            save_every=args.save_every,
        )
        cfg = HybridConfig(nerf=nerf_cfg)
        system = HybridSystem(cfg)
        system.train_nerf()
        return 0

    if args.cmd == "export_gaussians":
        nerf_cfg = NerfTrainingConfig(dataset_path="unused")
        export_cfg = GaussianExportConfig(
            num_gaussians=args.num_gaussians,
            sh_degree=args.sh_degree,
            opacity_threshold=args.opacity_thresh,
            position_jitter=args.position_jitter,
            pointcloud_path=args.pointcloud,
        )
        cfg = HybridConfig(nerf=nerf_cfg, export=export_cfg, output_dir=args.outdir)
        system = HybridSystem(cfg)
        system.export_gaussians(args.nerf_ckpt)
        return 0

    if args.cmd == "render_once":
        nerf_cfg = NerfTrainingConfig(dataset_path="unused")
        cfg = HybridConfig(nerf=nerf_cfg)
        system = HybridSystem(cfg)
        system.gaussian_npz_path = args.gaussians
        system.load_renderer()
        c2w = np.load(args.c2w_path)
        # Map optional quality flags at runtime
        if args.ssaa is not None:
            system.cfg.quality.quality_level = 0.85 if args.ssaa >= 2 else system.cfg.quality.quality_level
        if args.msaa is not None:
            system.cfg.quality.quality_level = max(system.cfg.quality.quality_level, 0.9 if args.msaa >= 4 else 0.7)
        if args.max_per_tile is not None:
            # Not directly exposed; the renderer maps quality -> per-tile cap
            pass
        img = system.render_frame({
            "width": args.width,
            "height": args.height,
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx if args.cx is not None else args.width / 2.0,
            "cy": args.cy if args.cy is not None else args.height / 2.0,
            "c2w": c2w,
        })
        try:
            import imageio.v2 as imageio
        except Exception:
            import imageio
        imageio.imwrite(args.out, (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))
        return 0

    if args.cmd == "densify":
        nerf_cfg = NerfTrainingConfig(dataset_path="unused")
        cfg = HybridConfig(nerf=nerf_cfg, output_dir=args.outdir)
        system = HybridSystem(cfg)
        # If pre-existing NPZ is given, keep it for continuity (optional)
        if args.gaussians:
            system.gaussian_npz_path = args.gaussians
        path_out = system.progressive_densify(args.nerf_ckpt, args.target_count)
        print(path_out)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())


