from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class MotionParams:
    """Parameters for a constant-velocity motion + temporal smoothness prior."""

    velocity_lambda: float = 1e-3
    color_smooth_lambda: float = 1e-3


def temporal_consistency_loss(
    batch_t: Dict[str, torch.Tensor],
    batch_t1: Dict[str, torch.Tensor],
    dt: float,
    motion_params: MotionParams,
) -> torch.Tensor:
    """A simple temporal consistency loss between consecutive batches.

    Encourages stable alpha and color over time (proxy for appearance stability)
    and penalizes large per-Gaussian velocity magnitudes.
    """
    losses = []
    if "alphas" in batch_t and "alphas" in batch_t1:
        losses.append(torch.mean((batch_t1["alphas"] - batch_t["alphas"]) ** 2))
    if "colors" in batch_t and "colors" in batch_t1:
        losses.append(motion_params.color_smooth_lambda * torch.mean((batch_t1["colors"] - batch_t["colors"]) ** 2))
    if "velocities" in batch_t:
        vel = batch_t["velocities"]
        losses.append(motion_params.velocity_lambda * torch.mean(torch.sum(vel * vel, dim=-1)))
    return torch.add_n(losses) if hasattr(torch, "add_n") else sum(losses)


def estimate_velocity_from_tracks(
    positions_t0: torch.Tensor,
    positions_t1: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Estimate per-point velocity from two time samples (simple difference)."""
    return (positions_t1 - positions_t0) / max(dt, 1e-6)


