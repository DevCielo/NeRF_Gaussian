#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/llff_data
cd data/llff_data

echo "Downloading LLFF ‘fern’ dataset…"
curl -L -o fern.zip \
  https://storage.googleapis.com/nerf_llff_data/fern.zip

echo "Unzipping fern.zip…"
unzip -q fern.zip
rm fern.zip

cd ../../
