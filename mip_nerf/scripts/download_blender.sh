#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/nerf_synthetic
cd data/nerf_synthetic

echo "Downloading Blender ‘lego’ dataset…"
curl -L -o blender.zip \
  https://storage.googleapis.com/nerf_synthetic/blender.zip

echo "Unzipping blender.zip…"
unzip -q blender.zip
rm blender.zip

cd ../../
