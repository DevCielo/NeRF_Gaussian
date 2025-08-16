# NeRF_Gaussian

## Quality Enhancements (Week 8)

Renderer now supports:

- Anti-aliasing
  - SSAA via `--ssaa_scale > 1` (e.g., 2 for 2x super-sampling)
  - Jittered MSAA via `--aa_samples N` (Halton jitter; averaged)
- Depth and normals
  - Save with `--save_depth depth.png` and `--save_normals normals.png`
  - Normals estimated from depth in camera space
- Materials and lighting
  - Enable Lambert or Phong lighting: `--lighting lambert` or `--lighting phong`
  - Optional `--light_dir dx dy dz` in camera space

Example:

```bash
python -m gaussian_splatting.demo --width 800 --height 600 \
  --ssaa_scale 2 --aa_samples 4 \
  --save_depth depth.png --save_normals normals.png \
  --lighting phong --light_dir 0 -0.2 1 \
  --out gaussian_demo.png
```

## Hybrid Rendering System (Week 10)

New `hybrid/` package trains mip-NeRF, exports Gaussians for fast inference, and renders with quality vs speed and automatic LOD.

- Train NeRF:
```bash
python -m hybrid.cli train_nerf --data /absolute/path/to/dataset --max_steps 20000
```
- Export Gaussians:
```bash
python -m hybrid.cli export_gaussians --nerf_ckpt /absolute/path/to/mip_nerf/log --outdir /absolute/path/to/hybrid_out
```
- Render once:
```bash
python -m hybrid.cli render_once --gaussians /absolute/path/to/hybrid_out/gaussians_init.npz --c2w_path cam.npy --out render.png
```

Quality and LOD knobs are in `hybrid/config.py` and applied in `hybrid/renderer.py`.