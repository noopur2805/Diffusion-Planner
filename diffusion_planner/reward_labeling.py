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

METRIC_ORDER = ("nc", "dac", "ttc", "ep", "comfort", "ddc", "mp", "sl")

EGO_HALF_LENGTH = 2.4
EGO_HALF_WIDTH = 1.0
COLLISION_RADIUS = 2.0
TTC_HORIZON = 2.0
COMFORT_LAT_ACC = 4.0
COMFORT_LON_ACC = 4.0
# DDC: candidate xy step must form an acute angle with the nearest lane tangent.
# Stationary candidates (speed below this threshold) pass by default since
# direction is undefined.
DDC_SPEED_THRESHOLD = 0.5     # m/s
# Making-progress: full-trajectory displacement must exceed
# ``MP_RATE_M_PER_S * horizon_sec * MP_FRACTION``.
MP_RATE_M_PER_S = 1.0
MP_FRACTION = 0.3
# Speed-limit compliance: urban default in m/s. Per-batch override may be
# supplied via the optional ``speed_limit`` arg to ``label_trajectory_rewards``.
SPEED_LIMIT_DEFAULT = 22.0    # ~80 km/h


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
    route_lanes: torch.Tensor = None,  # (B, R, V, C); when given, DAC/DDC are route-masked
    dt: float = 0.1,
    n_horizons: int = 8,
    speed_limit: torch.Tensor = None,  # (B,) m/s; falls back to SPEED_LIMIT_DEFAULT
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
    # Fix #3c: when ``route_lanes`` is provided, only lanes the ego is meant
    # to drive on count toward DAC, with a tighter 3.0 m corridor (vs. 6.0 m
    # on all lanes). ``lane_nearest`` is reused by DDC below so the direction
    # check reads tangents from the same lane set; this stops the proxy from
    # rewarding wrong-way candidates that hug an oncoming polyline.
    ref_lanes_xy_lv = (route_lanes if route_lanes is not None else lanes)[..., :2]
    ref_xy = ref_lanes_xy_lv.reshape(B, -1, 2)
    ref_pt_valid = ref_xy.abs().sum(-1) > 1e-6
    diff = cand[:, :, None, :2] - ref_xy[:, None, :, :]
    lane_dist_full = torch.sqrt((diff * diff).sum(-1) + 1e-9)
    lane_dist_full = lane_dist_full.masked_fill(~ref_pt_valid[:, None, :], 1e6)
    lane_dist, lane_nearest = lane_dist_full.min(dim=-1)
    ref_scene_valid = ref_pt_valid.any(dim=-1)
    dac_thresh = 3.0 if route_lanes is not None else 6.0
    dac = (lane_dist < dac_thresh).float()
    dac = torch.where(ref_scene_valid[:, None], dac, torch.ones_like(dac))

    # --- time-to-collision (rttc) ---
    # crude: TTC at horizon h is min distance / relative speed at that step
    vx_ego = (cand[..., 0] - torch.roll(cand[..., 0], 1, dims=-1)) / dt
    vy_ego = (cand[..., 1] - torch.roll(cand[..., 1], 1, dims=-1)) / dt
    rel_speed = torch.sqrt(vx_ego ** 2 + vy_ego ** 2 + 1e-6)
    min_dist = dist.where(valid, torch.full_like(dist, 1e3)).min(dim=1).values  # (B, T_h)
    ttc = (min_dist / (rel_speed + 1e-3)) > TTC_HORIZON
    ttc = ttc.float()

    # --- ego progress (rep) ---
    # Forward (along initial ego heading = +x in ego frame at t=0) net
    # displacement, normalised by per-trajectory max forward reach. Random
    # xy-noise has zero mean forward velocity, so it does not inflate this
    # metric; wrong-way / reversing yields negative values that clamp to 0.
    fwd_ego = cand[..., 0] - cand[:, :1, 0]
    fwd_max = fwd_ego.max(dim=-1, keepdim=True).values.clamp(min=0.5)
    ep = (fwd_ego / fwd_max).clamp(0.0, 1.0)

    # --- comfort: bounded longitudinal/lateral acceleration ---
    ax = (vx_ego - torch.roll(vx_ego, 1, dims=-1)) / dt
    ay = (vy_ego - torch.roll(vy_ego, 1, dims=-1)) / dt
    comfort = ((ax.abs() < COMFORT_LON_ACC) & (ay.abs() < COMFORT_LAT_ACC)).float()

    # --- driving direction compliance (rddc) ---
    # Tangents are finite-differenced from the same lane set used by DAC, so
    # ``lane_nearest`` indexing transfers directly and the metric cannot
    # reward driving along an oncoming polyline (Fix #3c).
    ref_tan_lv = torch.zeros_like(ref_lanes_xy_lv)
    if ref_lanes_xy_lv.shape[-2] >= 2:
        ref_tan_lv[..., :-1, :] = ref_lanes_xy_lv[..., 1:, :] - ref_lanes_xy_lv[..., :-1, :]
        ref_tan_lv[..., -1, :] = ref_tan_lv[..., -2, :]
    ref_tan = ref_tan_lv.reshape(B, -1, 2)
    b_idx = torch.arange(B, device=cand.device).view(B, 1).expand(B, n_horizons)
    tan_at = ref_tan[b_idx, lane_nearest]                          # (B, T_h, 2)
    tan_norm = torch.sqrt((tan_at * tan_at).sum(-1) + 1e-6)
    cos_align = (vx_ego * tan_at[..., 0] + vy_ego * tan_at[..., 1]) / (rel_speed * tan_norm)
    ddc = ((cos_align > 0.0) | (rel_speed < DDC_SPEED_THRESHOLD)).float()
    ddc[..., 0] = 1.0  # wrap-around step at t=0 is bogus
    ddc = torch.where(ref_scene_valid[:, None], ddc, torch.ones_like(ddc))

    # --- making progress (rmp) ---
    # Forward (ego +x at t=0) net displacement over the horizon. Using signed
    # forward distance instead of magnitude prevents zero-mean noise from
    # spuriously satisfying the threshold.
    horizon_sec = max((T - 1) * dt, 1e-3)
    final_fwd = cand[..., -1, 0] - cand[..., 0, 0]                 # (B,)
    mp_thresh = MP_RATE_M_PER_S * horizon_sec * MP_FRACTION
    mp = (final_fwd > mp_thresh).float().unsqueeze(-1).expand(-1, n_horizons)

    # --- speed limit compliance (rsl) ---
    if speed_limit is None:
        sl_cap = torch.full((B,), SPEED_LIMIT_DEFAULT, device=cand.device, dtype=cand.dtype)
    else:
        sl_cap = speed_limit.to(device=cand.device, dtype=cand.dtype)
    sl = (rel_speed < sl_cap.view(B, 1)).float()
    sl[..., 0] = 1.0  # wrap-around step at t=0 is bogus

    out = {
        "nc": nc, "dac": dac, "ttc": ttc, "ep": ep, "comfort": comfort,
        "ddc": ddc, "mp": mp, "sl": sl,
    }
    return out


def stack_metrics(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack metric dict into (B, T_h, K) in the canonical METRIC_ORDER."""
    return torch.stack([d[m] for m in METRIC_ORDER], dim=-1)


@torch.no_grad()
def drift_augmented_rewards(
    candidate_traj: torch.Tensor,   # (B, T, 4)
    neighbor_future: torch.Tensor,  # (B, P, T, 3+)
    neighbor_valid: torch.Tensor,   # (B, P, T)
    lanes: torch.Tensor,            # (B, L, V, 8)
    K: int = 4,
    sigma_drift: float = 0.5,
    dt: float = 0.1,
    n_horizons_per_segment: int = 2,
    generator: torch.Generator = None,
    route_lanes: torch.Tensor = None,  # (B, R, V, C); forwarded to label_trajectory_rewards
) -> Dict[str, torch.Tensor]:
    """K-segment drift-augmented reward (Tier-C closed-loop-aware proxy).

    Splits the candidate's T timesteps into K equal segments. For each
    segment ``k`` with k>=1, a per-batch isotropic drift
    ``epsilon_k ~ N(0, sigma_drift * sqrt(k))`` is added to the candidate's
    xy positions inside that segment, while the scene (lanes, neighbors)
    is held fixed. This breaks the rigid-transform invariance of the proxy
    metrics and exposes safety-margin behavior: candidates that hug the
    lane edge or skim past neighbors are penalized; candidates that leave
    margin tolerate the drift.

    With ``K=1`` or ``sigma_drift=0`` the function is bit-identical to
    ``label_trajectory_rewards`` (same per-metric tensors, possibly
    reshaped over the horizon axis).

    Returns a dict ``{metric: (B, K * n_horizons_per_segment)}`` matching
    the convention of ``label_trajectory_rewards`` (which returns
    ``(B, n_horizons)``).
    """
    B, T, _ = candidate_traj.shape
    if K <= 1 or sigma_drift == 0.0:
        return label_trajectory_rewards(
            candidate_traj, neighbor_future, neighbor_valid, lanes,
            route_lanes=route_lanes,
            dt=dt, n_horizons=K * n_horizons_per_segment,
        )

    seg_T = T // K
    assert seg_T > 0, f"T={T} not divisible into K={K} segments"

    seg_metrics_list = []
    for k in range(K):
        t0, t1 = k * seg_T, (k + 1) * seg_T
        seg_cand = candidate_traj[:, t0:t1].clone()                 # (B, seg_T, 4)
        if k >= 1:
            scale = sigma_drift * (k ** 0.5)
            noise_shape = (B, 1, 2)
            if generator is not None:
                eps = torch.randn(noise_shape, generator=generator,
                                  device=seg_cand.device, dtype=seg_cand.dtype) * scale
            else:
                eps = torch.randn(noise_shape, device=seg_cand.device,
                                  dtype=seg_cand.dtype) * scale
            seg_cand[..., :2] = seg_cand[..., :2] + eps

        if neighbor_future.shape[-2] == T:
            seg_neigh = neighbor_future[..., t0:t1, :]
            seg_valid = neighbor_valid[..., t0:t1]
        else:
            seg_neigh = neighbor_future
            seg_valid = neighbor_valid

        seg_m = label_trajectory_rewards(
            seg_cand, seg_neigh, seg_valid, lanes,
            route_lanes=route_lanes,
            dt=dt, n_horizons=n_horizons_per_segment,
        )
        seg_metrics_list.append(seg_m)

    return {m: torch.cat([s[m] for s in seg_metrics_list], dim=-1) for m in METRIC_ORDER}
