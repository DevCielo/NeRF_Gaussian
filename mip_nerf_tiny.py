# mip_nerf_fixed.py

from typing import Tuple
import math, os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------
# Utility functions
# ----------------------------
def meshgrid_xy(t1: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ii, jj = torch.meshgrid(t1, t2, indexing="xy")
    return ii, jj

def cumprod_exclusive(x: torch.Tensor) -> torch.Tensor:
    cp = torch.cumprod(x, dim=-1)
    cp = torch.roll(cp, shifts=1, dims=-1)
    cp[..., 0] = 1.0
    return cp

# ----------------------------
# Mip-NeRF core additions
# ----------------------------
def conical_frustum_to_gaussian(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    base_radius: float = 0.001  # Much smaller base radius
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert conical frustum to 3D Gaussian representation.
    
    Key fixes:
    1. Proper diagonal covariance computation
    2. Smaller base radius for finer details
    3. Correct variance computation for the conical frustum
    """
    mu = (t0 + t1) / 2.0
    hw = (t1 - t0) / 2.0
    
    # Mean position along ray
    mean = ray_o + ray_d * mu.unsqueeze(-1)
    
    # Normalize ray direction
    d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
    
    # Variance along ray direction (longitudinal)
    t_var = (hw**2) / 3.0
    
    # Variance perpendicular to ray (radial) 
    # For conical frustum: r(t) = base_radius * t
    # Average radius: r_avg = base_radius * mu
    r_avg = base_radius * mu
    r_var = (r_avg**2) / 4.0  # Simplified radial variance
    
    # Build diagonal covariance (3x3 diagonal matrix represented as 3D vector)
    # We use the fact that cov = t_var * d⊗d + r_var * (I - d⊗d)
    # For diagonal representation: diag(cov) = t_var * d² + r_var * (1 - d²)
    cov_diag = t_var.unsqueeze(-1) * (d**2) + r_var.unsqueeze(-1) * (1.0 - d**2)
    
    return mean, cov_diag

def integrated_positional_encoding(
    mean: torch.Tensor,
    cov_diag: torch.Tensor,
    num_freqs: int = 6
) -> torch.Tensor:
    """Compute integrated positional encoding with Gaussian weights.
    
    Key fixes:
    1. Proper frequency scaling
    2. Correct application of Gaussian integral formula
    3. Include raw coordinates in encoding
    """
    # Create frequency bands
    freqs = 2.0 ** torch.arange(num_freqs, dtype=mean.dtype, device=mean.device)
    freqs = freqs.view(*([1] * mean.dim()), -1)  # Shape: [1, 1, ..., num_freqs]
    
    # Expand dimensions for broadcasting
    mean_expanded = mean.unsqueeze(-1)  # [..., 3, 1]
    cov_expanded = cov_diag.unsqueeze(-1)  # [..., 3, 1]
    
    # Compute phase and damping for each frequency
    phase = 2.0 * math.pi * mean_expanded * freqs  # [..., 3, num_freqs]
    damping = 0.5 * (2.0 * math.pi)**2 * cov_expanded * (freqs**2)  # [..., 3, num_freqs]
    
    # Apply Gaussian damping
    exp_damping = torch.exp(-damping)  # [..., 3, num_freqs]
    
    # Compute sin and cos with damping
    sin_enc = exp_damping * torch.sin(phase)  # [..., 3, num_freqs]
    cos_enc = exp_damping * torch.cos(phase)  # [..., 3, num_freqs]
    
    # Flatten the spatial and frequency dimensions
    sin_enc = sin_enc.reshape(*mean.shape[:-1], -1)  # [..., 3*num_freqs]
    cos_enc = cos_enc.reshape(*mean.shape[:-1], -1)  # [..., 3*num_freqs]
    
    # Include raw coordinates (important for low frequencies)
    return torch.cat([mean, sin_enc, cos_enc], dim=-1)  # [..., 3 + 2*3*num_freqs]

def compute_frustum_gaussians(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    near: float,
    far: float,
    N_samples: int,
    randomized: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample points along rays and compute Gaussian representations."""
    # Create sample positions
    t_vals = torch.linspace(near, far, N_samples + 1, device=ray_o.device)
    
    if randomized and ray_o.requires_grad:  # Only randomize during training
        # Add uniform noise to sample positions
        mids = 0.5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids[1:], t_vals[-1:]])
        lower = torch.cat([t_vals[:1], mids[:-1]])
        t_rand = torch.rand(N_samples, device=ray_o.device)
        t_vals = lower + (upper - lower) * t_rand
    
    # Compute Gaussians for each frustum
    means, covs = [], []
    for i in range(N_samples):
        m, c = conical_frustum_to_gaussian(
            ray_o, ray_d, 
            t_vals[i].expand_as(ray_o[..., 0]), 
            t_vals[i+1].expand_as(ray_o[..., 0])
        )
        means.append(m)
        covs.append(c)
    
    means = torch.stack(means, dim=-2)  # [..., N_samples, 3]
    covs = torch.stack(covs, dim=-2)    # [..., N_samples, 3]
    t_mids = 0.5 * (t_vals[:-1] + t_vals[1:])
    
    return means, covs, t_mids

# ----------------------------
# Volume rendering
# ----------------------------
def render_volume_density(
    rad_field: torch.Tensor,
    ray_o: torch.Tensor,
    t_mids: torch.Tensor,
    raw_noise_std: float = 0.0
):
    """Volume rendering with proper distance computation."""
    # Extract density and color
    sigma_a = F.relu(rad_field[..., 3])
    rgb = torch.sigmoid(rad_field[..., :3])
    
    # Add noise to density during training
    if raw_noise_std > 0 and rad_field.requires_grad:
        noise = torch.randn_like(sigma_a) * raw_noise_std
        sigma_a = sigma_a + noise
    
    # Compute distances between samples
    dists = t_mids[..., 1:] - t_mids[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    
    # Compute alpha values
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    
    # Compute weights
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    
    # Composite final values
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * t_mids).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    
    return rgb_map, depth_map, acc_map

# ----------------------------
# Model & batching
# ----------------------------
class MipNerfModel(torch.nn.Module):
    def __init__(self, num_layers=8, hidden_dim=256, num_freqs=6, input_dim=None):
        super().__init__()
        if input_dim is None:
            # 3 (raw coords) + 2 * 3 * num_freqs (sin/cos encodings)
            input_dim = 3 + 2 * 3 * num_freqs
        
        self.layers = torch.nn.ModuleList()
        
        # First layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers with skip connection at layer 4
        for i in range(1, num_layers):
            if i == 4:
                self.layers.append(torch.nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        input_x = x
        
        for i, layer in enumerate(self.layers):
            if i == 4:
                x = torch.cat([x, input_x], dim=-1)
            x = F.relu(layer(x))
        
        return self.output_layer(x)

def get_minibatches(x: torch.Tensor, chunksize: int = 65536):
    return [x[i:i+chunksize] for i in range(0, x.shape[0], chunksize)]

# ----------------------------
# Ray bundle
# ----------------------------
def get_ray_bundle(H, W, focal, c2w):
    """Generate ray origins and directions."""
    ii, jj = meshgrid_xy(
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        torch.arange(H, dtype=torch.float32, device=c2w.device)
    )
    
    # Camera space directions
    dirs = torch.stack([
        (ii - W * 0.5) / focal,
        -(jj - H * 0.5) / focal,
        -torch.ones_like(ii)
    ], dim=-1)
    
    # Transform to world space
    rd = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    ro = c2w[:3, -1].expand(rd.shape)
    
    return ro, rd

# ----------------------------
# One iteration (Mip-NeRF)
# ----------------------------
def run_one_iter(H, W, focal, c2w, near, far, N_samps, encode_fn, batch_fn, model, randomized=True):
    """Run one forward pass of Mip-NeRF."""
    # Get rays
    ro, rd = get_ray_bundle(H, W, focal, c2w)
    
    # Sample along rays and compute Gaussians
    means, covs, t_mids = compute_frustum_gaussians(ro, rd, near, far, N_samps, randomized)
    
    # Flatten for batching
    means_flat = means.reshape(-1, 3)
    covs_flat = covs.reshape(-1, 3)
    
    # Encode positions
    enc = encode_fn(means_flat, covs_flat)
    
    # Forward through model in batches
    preds = []
    for batch in batch_fn(enc):
        preds.append(model(batch))
    
    # Reshape predictions
    rad_field_flat = torch.cat(preds, dim=0)
    rad_field = rad_field_flat.view(*means.shape[:-1], 4)
    
    # Volume rendering
    rgb, depth, acc = render_volume_density(rad_field, ro, t_mids)
    
    return rgb, depth, acc

# ----------------------------
# Main training loop
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download data if missing
    if not os.path.exists('tiny_nerf_data.npz'):
        print("Downloading dataset...")
        os.system(
            'curl -L -o tiny_nerf_data.npz '
            'https://github.com/volunt4s/TinyNeRF-pytorch/raw/main/data/tiny_nerf_data.npz'
        )
    
    # Load data
    data = np.load("tiny_nerf_data.npz")
    images = torch.from_numpy(data["images"][..., :3]).float().to(device)
    poses = torch.from_numpy(data["poses"]).float().to(device)
    focal = torch.from_numpy(data["focal"]).float().to(device)
    H, W = images.shape[1:3]
    
    # Train/test split
    test_idx = 101
    test_img = images[test_idx]
    test_pose = poses[test_idx]
    train_imgs = images[:100]
    train_poses = poses[:100]
    
    # Hyperparameters
    near, far = 2.0, 6.0
    num_freqs = 6
    N_samples = 64  # Increased samples
    N_iters = 5000  # More iterations
    chunksize = 65536  # Larger chunks
    log_interval = 100
    
    # Initialize model and optimizer
    model = MipNerfModel(num_layers=8, hidden_dim=256, num_freqs=num_freqs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    
    # Encoding and batching functions
    encode_fn = lambda mu, cov: integrated_positional_encoding(mu, cov, num_freqs)
    batch_fn = lambda x: get_minibatches(x, chunksize)
    
    # Prepare interactive plotting
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    psnrs = []
    iters = []
    losses = []
    
    print("Starting training...")
    for i in tqdm(range(N_iters), desc="Training"):
        # Random training image
        idx = np.random.randint(train_imgs.shape[0])
        target_img = train_imgs[idx]
        target_pose = train_poses[idx]
        
        # Forward pass
        rgb_pred, _, _ = run_one_iter(
            H, W, focal, target_pose,
            near, far, N_samples,
            encode_fn, batch_fn, model,
            randomized=True
        )
        
        # Compute loss
        loss = F.mse_loss(rgb_pred, target_img)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Evaluation and visualization
        if i % log_interval == 0:
            with torch.no_grad():
                test_rgb, _, _ = run_one_iter(
                    H, W, focal, test_pose,
                    near, far, N_samples,
                    encode_fn, batch_fn, model,
                    randomized=False
                )
                
                mse = F.mse_loss(test_rgb, test_img)
                psnr = -10.0 * torch.log10(mse)
                psnrs.append(psnr.item())
                iters.append(i)
            
            print(f"\n[Iter {i:5d}] Loss: {loss.item():.6f}, Test PSNR: {psnr.item():.2f} dB")
            
            # Update plots
            axes[0].clear()
            axes[0].imshow(test_img.cpu().numpy())
            axes[0].set_title("Ground Truth")
            axes[0].axis('off')
            
            axes[1].clear()
            axes[1].imshow(test_rgb.cpu().numpy())
            axes[1].set_title(f"Prediction (iter {i})")
            axes[1].axis('off')
            
            axes[2].clear()
            axes[2].plot(iters, psnrs, 'b-', linewidth=2)
            axes[2].set_xlabel("Iteration")
            axes[2].set_ylabel("PSNR (dB)")
            axes[2].set_title("Test PSNR")
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.pause(0.001)
    
    plt.ioff()
    plt.show()
    print("\nTraining complete!")
    print(f"Final test PSNR: {psnrs[-1]:.2f} dB")