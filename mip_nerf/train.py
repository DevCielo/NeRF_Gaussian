import os.path
import shutil
import gc
import torch
import numpy as np
from os import path
from tqdm import tqdm
import torch.optim as optim
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips

from config import get_config
from scheduler import MipLRDecay
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
from datasets import get_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(config, model, ssim_metric, lpips_metric):
    dataset = get_dataset(
        dataset_name=config.dataset_name,
        base_dir=config.base_dir,
        split="test",
        factor=config.factor,
        device=config.device,
    )
    loader = DataLoader(
        dataset,
        batch_size=dataset.h * dataset.w,
        shuffle=False,
    )
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    model.eval()
    with torch.no_grad():
        for rays, pixels in loader:
            comp_rgb, _, _ = model(rays)
            pred = comp_rgb[-1]
            target = pixels.to(config.device)
            h, w = dataset.h, dataset.w
            pred_img = pred.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
            tgt_img = target.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
            mse = torch.mean((pred_img - tgt_img) ** 2)
            psnr_vals.append(mse_to_psnr(mse).cpu())
            ssim_vals.append(ssim_metric(pred_img, tgt_img).cpu())
            lpips_vals.append(
                lpips_metric(pred_img * 2 - 1, tgt_img * 2 - 1).mean().cpu()
            )
    model.train()
    return (
        torch.stack(psnr_vals),
        torch.stack(ssim_vals),
        torch.stack(lpips_vals),
    )


def train_model(config):
    config.device = DEVICE
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")
    dataset = get_dataset(
        dataset_name=config.dataset_name,
        base_dir=config.base_dir,
        split="train",
        factor=config.factor,
        device=config.device,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    data_iter = iter(loader)
    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
        use_hash_encoding=config.use_hash_encoding,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr_init,
        weight_decay=config.weight_decay,
    )
    if config.continue_training:
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        optimizer.load_state_dict(torch.load(optimizer_save_path, map_location=DEVICE))
    scheduler = MipLRDecay(
        optimizer,
        lr_init=config.lr_init,
        lr_final=config.lr_final,
        max_steps=config.max_steps,
        lr_delay_steps=config.lr_delay_steps,
        lr_delay_mult=config.lr_delay_mult,
    )
    loss_func = NeRFLoss(config.coarse_weight_decay)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.device)
    lpips_metric = lpips.LPIPS(net="alex").to(config.device)
    os.makedirs(config.log_dir, exist_ok=True)
    shutil.rmtree(path.join(config.log_dir, "train"), ignore_errors=True)
    logger = tb.SummaryWriter(path.join(config.log_dir, "train"), flush_secs=1)
    model.train()
    for step in tqdm(range(config.max_steps)):
        try:
            rays, pixels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            rays, pixels = next(data_iter)
        comp_rgb, _, _ = model(rays)
        pixels = pixels.to(config.device)
        loss_val, psnr_train = loss_func(
            comp_rgb, pixels, rays.lossmult.to(config.device)
        )
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()
        logger.add_scalar("train/loss", float(loss_val.cpu()), step)
        logger.add_scalar("train/coarse_psnr", float(torch.mean(psnr_train[:-1])), step)
        logger.add_scalar("train/fine_psnr", float(psnr_train[-1]), step)
        logger.add_scalar("train/avg_psnr", float(torch.mean(psnr_train)), step)
        logger.add_scalar("train/lr", float(scheduler.get_last_lr()[-1]), step)
        if step != 0 and step % config.save_every == 0:
            psnr_eval, ssim_eval, lpips_eval = eval_model(
                config, model, ssim_metric, lpips_metric
            )
            logger.add_scalar(
                "eval/coarse_psnr", float(torch.mean(psnr_eval[:-1])), step
            )
            logger.add_scalar("eval/fine_psnr", float(psnr_eval[-1]), step)
            logger.add_scalar("eval/avg_psnr", float(torch.mean(psnr_eval)), step)
            logger.add_scalar("eval/ssim", float(torch.mean(ssim_eval)), step)
            logger.add_scalar("eval/lpips", float(torch.mean(lpips_eval)), step)
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), optimizer_save_path)
            gc.collect()
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)


if __name__ == "__main__":
    cfg = get_config()
    cfg.device = DEVICE
    train_model(cfg)
