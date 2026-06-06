"""
Vectorized-scene reward labeler for the AD-RM.

The DreamerAD paper scores trajectories with the NavSim PDM simulator. That
simulator is not always available during training - in particular when only
the vectorized scene is loaded. This module computes simple but well-defined
*proxy* rewards directly from the tensors already present in the planner's
input dict (agents, lanes, ego). They are intentionally lightweight so they
can be evaluated for every candidate trajectory inside the data loader or RL
loop. The user can swap in real PDM scores by replacing
``label_trajectory_rewards`` with their own implementation.

Conventions:
    - Trajectories are in the ego-centric frame at t=0.
    - Distances are in meters, times in seconds (uniform spacing).
    - All returned scores are in [0, 1] where higher is better.
"""
from typing import Dict

import torch

METRIC_ORDER = ("nc", "dac", "ttc", "ep", "comfort")

EGO_HALF_LENGTH = 2.4
EGO_HALF_WIDTH = 1.0
COLLISION_RADIUS = 2.0
TTC_HORIZON = 2.0
COMFORT_LAT_ACC = 4.0
COMFORT_LON_ACC = 4.0


def _resample(x: torch.Tensor, n: int) -> torch.Tensor:
    T = x.shape[-2]
    idx = torch.linspace(0, T - 1, steps=n, device=x.device).long()
    return x.index_select(dim=-2, index=idx)


@torch.no_grad()
def label_trajectory_rewards(
    candidate_traj: torch.Tensor,  # (B, T, 4) -> x, y, cos, sin   (ego frame)
    neighbor_future: torch.Tensor,  # (B, P, T, 3+) -> x, y, heading, ...
    neighbor_valid: torch.Tensor,   # (B, P, T) bool
    lanes: torch.Tensor,            # (B, L, V, 8) -> x, y, x'-x, y'-y, x_l-x, y_l-y, x_r-x, y_r-y
    dt: float = 0.1,
    n_horizons: int = 8,
) -> Dict[str, torch.Tensor]:
    """Returns a dict of (B, n_horizons) tensors per metric in METRIC_ORDER."""
    B, T, _ = candidate_traj.shape
    cand = _resample(candidate_traj, n_horizons)
    neigh = _resample(neighbor_future, n_horizons) if neighbor_future.shape[-2] == T else neighbor_future
    valid = _resample(neighbor_valid.unsqueeze(-1).float(), n_horizons).squeeze(-1).bool() \
        if neighbor_valid.shape[-1] == T else neighbor_valid

    # --- no-collision (rnc) ---
    dx = cand[:, None, :, 0] - neigh[..., 0]        # (B, P, T_h)
    dy = cand[:, None, :, 1] - neigh[..., 1]
    dist = torch.sqrt(dx * dx + dy * dy + 1e-9)
    collision = (dist < COLLISION_RADIUS) & valid
    nc = (~collision.any(dim=1)).float()

    # --- drivable area compliance (rdac) ---
    # consider center+left/right offsets within first lane channel as the rough corridor
    lane_xy = lanes[..., :2]                                       # (B, L, V, 2)
    lane_xy = lane_xy.reshape(B, -1, 2)
    diff = cand[:, :, None, :2] - lane_xy[:, None, :, :]
    lane_dist = torch.sqrt((diff * diff).sum(-1) + 1e-9).min(dim=-1).values  # (B, T_h)
    dac = (lane_dist < 6.0).float()

    # --- time-to-collision (rttc) ---
    # crude: TTC at horizon h is min distance / relative speed at that step
    vx_ego = (cand[..., 0] - torch.roll(cand[..., 0], 1, dims=-1)) / dt
    vy_ego = (cand[..., 1] - torch.roll(cand[..., 1], 1, dims=-1)) / dt
    rel_speed = torch.sqrt(vx_ego ** 2 + vy_ego ** 2 + 1e-6)
    min_dist = dist.where(valid, torch.full_like(dist, 1e3)).min(dim=1).values  # (B, T_h)
    ttc = (min_dist / (rel_speed + 1e-3)) > TTC_HORIZON
    ttc = ttc.float()

    # --- ego progress (rep) ---
    dx_ego = cand[..., 0] - cand[:, :1, 0]
    dy_ego = cand[..., 1] - cand[:, :1, 1]
    progress = torch.sqrt(dx_ego * dx_ego + dy_ego * dy_ego + 1e-9)
    ep = (progress / (progress.max(dim=-1, keepdim=True).values + 1e-3)).clamp(0.0, 1.0)

    # --- comfort: bounded longitudinal/lateral acceleration ---
    ax = (vx_ego - torch.roll(vx_ego, 1, dims=-1)) / dt
    ay = (vy_ego - torch.roll(vy_ego, 1, dims=-1)) / dt
    comfort = ((ax.abs() < COMFORT_LON_ACC) & (ay.abs() < COMFORT_LAT_ACC)).float()

    out = {
        "nc": nc, "dac": dac, "ttc": ttc, "ep": ep, "comfort": comfort,
    }
    return out


def stack_metrics(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack metric dict into (B, T_h, K) in the canonical METRIC_ORDER."""
    return torch.stack([d[m] for m in METRIC_ORDER], dim=-1)
