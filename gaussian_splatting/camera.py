from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Camera:
    """Simple pinhole camera with extrinsics and intrinsics.

    Coordinates follow OpenGL-style convention:
    - World coordinates are arbitrary
    - Camera coordinates: +Z points forward, +X right, +Y down in image after projection
      (assuming standard image coordinates with origin at top-left)
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: torch.Tensor  # (4, 4)
    device: torch.device

    def __post_init__(self) -> None:
        if self.world_to_camera.shape != (4, 4):
            raise ValueError("world_to_camera must be 4x4 homogeneous matrix")
        self.world_to_camera = self.world_to_camera.to(self.device)

    @property
    def intrinsics(self) -> torch.Tensor:
        k = torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        return k

    @property
    def rotation_world_to_cam(self) -> torch.Tensor:
        return self.world_to_camera[:3, :3]

    @property
    def translation_world_to_cam(self) -> torch.Tensor:
        return self.world_to_camera[:3, 3]

    @property
    def camera_to_world(self) -> torch.Tensor:
        return torch.linalg.inv(self.world_to_camera)

    @staticmethod
    def from_look_at(
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        eye: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        device: Optional[torch.device] = None,
    ) -> "Camera":
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        if cx is None:
            cx = width * 0.5
        if cy is None:
            cy = height * 0.5
        eye_t = torch.tensor(eye, dtype=torch.float32, device=device)
        target_t = torch.tensor(target, dtype=torch.float32, device=device)
        up_t = torch.tensor(up, dtype=torch.float32, device=device)

        forward = (target_t - eye_t)
        forward = forward / (torch.linalg.norm(forward) + 1e-8)
        right = torch.linalg.cross(forward, up_t)
        right = right / (torch.linalg.norm(right) + 1e-8)
        true_up = torch.linalg.cross(right, forward)

        # Camera rotation (world to camera): rows are camera axes in world space
        r = torch.stack([right, -true_up, forward], dim=0)  # (3,3)
        t = -r @ eye_t  # (3,)

        w2c = torch.eye(4, dtype=torch.float32, device=device)
        w2c[:3, :3] = r
        w2c[:3, 3] = t

        return Camera(width, height, fx, fy, cx, cy, w2c, device)

    def world_to_camera_points(self, points_world: torch.Tensor) -> torch.Tensor:
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError("points_world must be (N, 3)")
        ones = torch.ones((points_world.shape[0], 1), device=points_world.device, dtype=points_world.dtype)
        pts_h = torch.cat([points_world, ones], dim=1)  # (N,4)
        cam = (self.world_to_camera @ pts_h.t()).t()  # (N,4)
        return cam[:, :3]

    def project(self, points_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cam = self.world_to_camera_points(points_world)  # (N,3)
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        eps = 1e-8
        u = self.fx * (x / (z + eps)) + self.cx
        v = self.fy * (y / (z + eps)) + self.cy
        pixels = torch.stack([u, v], dim=-1)
        return pixels, z, cam

    def jacobian_world_to_image(self, cam_coords: torch.Tensor) -> torch.Tensor:
        """Compute d[u,v]/d[Xw,Yw,Zw] for each point given its camera coordinates.

        J_world = J_perspective @ Rcw, where J_perspective = [[fx/Z, 0, -fx*X/Z^2], [0, fy/Z, -fy*Y/Z^2]]
        Args:
            cam_coords: (N,3) points in camera frame
        Returns:
            J: (N, 2, 3)
        """
        x = cam_coords[:, 0]
        y = cam_coords[:, 1]
        z = cam_coords[:, 2]
        eps = 1e-8
        inv_z = 1.0 / (z + eps)
        inv_z2 = inv_z * inv_z

        j_p = torch.zeros((cam_coords.shape[0], 2, 3), dtype=torch.float32, device=cam_coords.device)
        j_p[:, 0, 0] = self.fx * inv_z
        j_p[:, 0, 2] = -self.fx * x * inv_z2
        j_p[:, 1, 1] = self.fy * inv_z
        j_p[:, 1, 2] = -self.fy * y * inv_z2

        r_cw = self.rotation_world_to_cam  # (3,3)
        j = j_p @ r_cw  # (N,2,3)
        return j


