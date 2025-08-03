# 1. Create the conda environment from the YAML
conda env create -f mipNeRF.yml

# 2. Activate the new environment (replace “mipNeRF” with the name defined in the YAML if different)
conda activate mipNeRF

# 3. Download all 3 datasets at once

### For windows
wsl --install
cd /mnt/c/Users/User/Documents/NeRF_Gaussian/mip_nerf
bash scripts/download_data.sh

bash scripts/download_data.sh

# 4. (If you want to grab them individually instead)
bash scripts/download_llff.sh
bash scripts/download_blender.sh
bash scripts/download_multicam.sh

# 5. (Optional) Edit config.py or pass overrides via CLI args before training

# 6. Kick off training
python train.py

# 7. Launch TensorBoard to monitor training
python -m tensorboard.main --logdir=log

# 8. Render a video from your trained model
python visualize.py

# 9. Extract a mesh from your trained model
python extract_mesh.py
