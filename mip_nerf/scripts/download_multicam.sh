#!/usr/bin/env bash
set -euo pipefail

# assumes data/nerf_synthetic already exists
mkdir -p data/nerf_multiscale

echo "Converting multicam/multiscale from nerf_synthetic…"
python scripts/convert_blender_data.py \
  --blenderdir data/nerf_synthetic \
  --outdir   data/nerf_multiscale
