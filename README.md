# NeRF_Gaussian

Minimal, fast Gaussian Splatting renderer with a mip-NeRF training pipeline and a NeRF→Gaussian export path.

### Requirements

- Python 3.9–3.11
- PyTorch 2.1+ (CUDA, CPU, or Apple Silicon MPS supported)
- See `requirements.txt` for the rest

Install:

```bash
pip install -r requirements.txt
# If you prefer, install the right PyTorch build first from pytorch.org, then the rest:
# pip install torch==<matching version>
# pip install -r requirements.txt --no-deps
```

### Quickstart: render a synthetic demo

This generates a colored sphere if no point cloud is provided.

```bash
python -m gaussian_splatting.demo --width 800 --height 600 --out gaussian_demo.png
```

Options you can add:

- Anti-aliasing: `--ssaa_scale 2` and/or `--aa_samples 4`
- Save depth/normals: `--save_depth depth.png --save_normals normals.png`
- Lighting: `--lighting phong --light_dir 0 -0.2 1`

Example:

```bash
python -m gaussian_splatting.demo --width 800 --height 600 \
  --ssaa_scale 2 --aa_samples 4 \
  --save_depth depth.png --save_normals normals.png \
  --lighting phong --light_dir 0 -0.2 1 \
  --out gaussian_demo.png
```

### Use your own point cloud

`gaussian_splatting.demo` accepts `.pt/.pth` dicts with `points` and optional `colors`, as well as `.npz/.npy/.txt/.xyz/.ply`.

```bash
# A small sample is included
python -m gaussian_splatting.demo --pointcloud sample_points.npz --out my_cloud.png
```

### Train mip-NeRF and export Gaussians (hybrid pipeline)

The `hybrid/` package orchestrates: train mip-NeRF → export Gaussian splats → render.

1) Download data (examples):

```bash
cd mip_nerf
bash scripts/download_blender.sh   # synthetic Blender
# bash scripts/download_llff.sh    # forward-facing LLFF
# bash scripts/download_multicam.sh
cd ..
```

2) Train mip-NeRF (adjust `--data` to your dataset path):

```bash
python -m hybrid.cli train_nerf --data /absolute/path/to/mip_nerf/data/nerf_synthetic/lego --max_steps 20000
```

3) Export Gaussians from the trained NeRF checkpoint directory:

```bash
python -m hybrid.cli export_gaussians --nerf_ckpt /absolute/path/to/mip_nerf/log --outdir /absolute/path/to/hybrid_out
```

This writes `gaussians_init.npz` to the chosen `--outdir`.

If the export CLI errors due to argument parsing conflicts, run this Python snippet instead:

```bash
python - <<'PY'
from hybrid.convert import export_gaussians_from_nerf
from hybrid.config import GaussianExportConfig
art = export_gaussians_from_nerf(
    "/absolute/path/to/mip_nerf/log",
    GaussianExportConfig(num_gaussians=200_000, sh_degree=3),
    "/absolute/path/to/hybrid_out",
)
print("Saved:", art.gaussian_path)
PY
```

4) Render a single frame from a camera pose:

```bash
# c2w is a 4x4 camera-to-world NumPy matrix saved as .npy
python -m hybrid.cli render_once \
  --gaussians /absolute/path/to/hybrid_out/gaussians_init.npz \
  --c2w_path /absolute/path/to/cam.npy \
  --width 800 --height 800 --fx 800 --fy 800 \
  --out render.png
```

Quality/LOD parameters live in `hybrid/config.py` and are applied in `hybrid/renderer.py`.

### Direct NeRF→Gaussian demo (no NPZ export)

If you trained with `mip_nerf/train.py` and have `mip_nerf/log/model.pt`, you can render directly:

```bash
python -m gaussian_splatting.convert_from_nerf --out gaussian_demo.png \
  --ssaa_scale 2 --aa_samples 2 --save_depth depth.png
```

### Notes and troubleshooting

- On Apple Silicon: PyTorch from PyPI supports MPS; the code auto-selects CUDA → MPS → CPU.
- If image saving fails, install either `imageio` (preferred) or `Pillow`.
- Large scenes: tune `--max_gaussians_per_tile` and quality knobs to fit memory.

---

Renderer features:

- Anti-aliasing: SSAA (`--ssaa_scale`) and jittered MSAA (`--aa_samples`)
- Optional depth/normal outputs
- Simple Lambert/Phong lighting (`--lighting`, `--light_dir`)