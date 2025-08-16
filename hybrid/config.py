from dataclasses import dataclass
from typing import Optional


@dataclass
class NerfTrainingConfig:
    dataset_path: str
    config_path: Optional[str] = None
    output_dir: str = "./mip_nerf/log"
    num_steps: int = 20000
    save_every: int = 1000


@dataclass
class GaussianExportConfig:
    # Number of initial Gaussians to sample from NeRF density or point cloud
    num_gaussians: int = 200_000
    sh_degree: int = 3
    position_jitter: float = 0.0
    opacity_threshold: float = 0.01
    # if provided, use an external point cloud file for initialization
    pointcloud_path: Optional[str] = None


@dataclass
class QualityConfig:
    # Quality vs speed: higher values favor quality
    quality_level: float = 0.75  # 0..1
    # Clamp maximum Gaussians rendered per frame
    max_gaussians_per_frame: Optional[int] = None
    # SH truncation for faster shading at low quality
    max_sh_degree: Optional[int] = None
    # Per-frame time budget in ms (renderer will adapt to fit budget)
    frame_time_budget_ms: Optional[float] = None


@dataclass
class LODConfig:
    # Screen-space size thresholds (in pixels) for multi-tier LOD
    pixel_radius_thresholds: tuple[float, float] = (0.5, 2.0)
    # Distance-based falloff (near, far) in scene units
    distance_range: tuple[float, float] = (0.1, 10.0)
    # Whether to enable frustum culling and backface culling
    enable_frustum_culling: bool = True
    enable_backface_culling: bool = True


@dataclass
class HybridConfig:
    nerf: NerfTrainingConfig
    export: GaussianExportConfig = GaussianExportConfig()
    quality: QualityConfig = QualityConfig()
    lod: LODConfig = LODConfig()
    # Paths
    output_dir: str = "./hybrid_out"
    gaussian_checkpoint_path: Optional[str] = None


