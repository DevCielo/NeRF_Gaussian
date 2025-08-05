# config.py
import argparse, torch
from os import path


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", type=str, default="log")
    p.add_argument("--dataset_name", type=str, default="blender")
    p.add_argument("--scene", type=str, default="lego")
    p.add_argument("--use_hash_encoding", action="store_true")
    p.add_argument("--use_viewdirs", action="store_false")
    p.add_argument("--randomized", action="store_false")
    p.add_argument("--ray_shape", type=str, default="cone")
    p.add_argument("--white_bkgd", action="store_false")
    p.add_argument("--override_defaults", action="store_true")
    p.add_argument("--num_levels", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--density_noise", type=float, default=0.0)
    p.add_argument("--density_bias", type=float, default=-1.0)
    p.add_argument("--rgb_padding", type=float, default=0.001)
    p.add_argument("--resample_padding", type=float, default=0.01)
    p.add_argument("--min_deg", type=int, default=0)
    p.add_argument("--max_deg", type=int, default=16)
    p.add_argument("--viewdirs_min_deg", type=int, default=0)
    p.add_argument("--viewdirs_max_deg", type=int, default=4)
    p.add_argument("--coarse_weight_decay", type=float, default=0.1)
    p.add_argument("--lr_init", type=float, default=1e-3)
    p.add_argument("--lr_final", type=float, default=5e-5)
    p.add_argument("--lr_delay_steps", type=int, default=2500)
    p.add_argument("--lr_delay_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--do_eval", action="store_false")
    p.add_argument("--continue_training", action="store_true")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunks", type=int, default=8192)
    p.add_argument("--model_weight_path", default="log/model.pt")
    p.add_argument("--visualize_depth", action="store_true")
    p.add_argument("--visualize_normals", action="store_true")
    p.add_argument("--x_range", nargs="+", type=float, default=[-1.2, 1.2])
    p.add_argument("--y_range", nargs="+", type=float, default=[-1.2, 1.2])
    p.add_argument("--z_range", nargs="+", type=float, default=[-1.2, 1.2])
    p.add_argument("--grid_size", type=int, default=256)
    p.add_argument("--sigma_threshold", type=float, default=50.0)
    p.add_argument("--occ_threshold", type=float, default=0.2)
    cfg = p.parse_args()
    if cfg.dataset_name == "llff" and not cfg.override_defaults:
        cfg.factor, cfg.ray_shape, cfg.white_bkgd, cfg.density_noise = 4, "cylinder", False, 1.0
    cfg.device = torch.device(cfg.device)
    base = "data/nerf_llff_data/"
    if cfg.dataset_name == "blender":
        base = "data/nerf_synthetic/"
    elif cfg.dataset_name == "multicam":
        base = "data/nerf_multiscale/"
    cfg.base_dir = path.join(base, cfg.scene)
    return cfg
