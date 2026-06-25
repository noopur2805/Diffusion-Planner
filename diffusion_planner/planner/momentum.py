"""
Momentum-aware inference helpers (Novelty 8).

Two complementary, inference-only consistency mechanisms for the diffusion
planner, both relying on a single buffered "previous plan":

    1. Ouroboros warm-start prior — bias xT toward the previous plan re-expressed
       in the current ego frame (mean shift, variance preserved).
    2. MomAD TTM re-ranker — draw K candidates, pick the one with smallest
       symmetric Hausdorff distance to the previous plan.

The buffer stores plans in world coordinates so that the rigid-body change from
the previous ego pose to the current ego pose is applied exactly.
"""
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Rigid-body transforms between world and ego frames (2D + heading).
# ---------------------------------------------------------------------------

def world_to_ego_xy(points_world: np.ndarray, ego_pose_world: Tuple[float, float, float]) -> np.ndarray:
    """Transform 2D points from world coordinates to the ego frame at ``ego_pose_world``.

    Args:
        points_world: (..., 2) array of (x, y) in world frame.
        ego_pose_world: (x, y, heading) of the ego in the world frame.
    Returns:
        (..., 2) points in the ego frame (ego at origin, heading along +x).
    """
    ex, ey, eh = ego_pose_world
    dx = points_world[..., 0] - ex
    dy = points_world[..., 1] - ey
    c, s = np.cos(-eh), np.sin(-eh)
    x = c * dx - s * dy
    y = s * dx + c * dy
    return np.stack([x, y], axis=-1)


def ego_to_world_xy(points_ego: np.ndarray, ego_pose_world: Tuple[float, float, float]) -> np.ndarray:
    """Inverse of ``world_to_ego_xy``."""
    ex, ey, eh = ego_pose_world
    c, s = np.cos(eh), np.sin(eh)
    x = c * points_ego[..., 0] - s * points_ego[..., 1] + ex
    y = s * points_ego[..., 0] + c * points_ego[..., 1] + ey
    return np.stack([x, y], axis=-1)


def heading_world_to_ego(cos_w: np.ndarray, sin_w: np.ndarray, ego_heading: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a unit heading vector (cos, sin) from world frame to ego frame."""
    c, s = np.cos(-ego_heading), np.sin(-ego_heading)
    return c * cos_w - s * sin_w, s * cos_w + c * sin_w


def heading_ego_to_world(cos_e: np.ndarray, sin_e: np.ndarray, ego_heading: float) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of ``heading_world_to_ego``."""
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    return c * cos_e - s * sin_e, s * cos_e + c * sin_e


# ---------------------------------------------------------------------------
# Hausdorff distance + Topological Trajectory Matching (TTM) selector.
# ---------------------------------------------------------------------------

def hausdorff_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Hausdorff distance between two 2D point sets.

    ``H(A, B) = max( max_{a∈A} min_{b∈B} ||a-b||,  max_{b∈B} min_{a∈A} ||a-b|| )``

    Args:
        a: (..., Na, 2)
        b: (..., Nb, 2)
    Returns:
        (...) — scalar Hausdorff distance per leading-dim batch element.
    """
    assert a.shape[-1] == 2 and b.shape[-1] == 2, "expected last dim of size 2 (xy)"
    diff = a.unsqueeze(-2) - b.unsqueeze(-3)               # (..., Na, Nb, 2)
    dist = torch.linalg.norm(diff, dim=-1)                 # (..., Na, Nb)
    d_ab = dist.min(dim=-1).values.max(dim=-1).values      # (...)
    d_ba = dist.min(dim=-2).values.max(dim=-1).values      # (...)
    return torch.maximum(d_ab, d_ba)


def select_by_hausdorff(candidates_xy: torch.Tensor, anchor_xy: torch.Tensor) -> int:
    """Pick the candidate trajectory with smallest Hausdorff distance to ``anchor_xy``.

    Args:
        candidates_xy: (K, T, 2) candidate ego polylines.
        anchor_xy: (T, 2) previous plan re-expressed in the current ego frame.
    Returns:
        int index in [0, K) of the chosen candidate.
    """
    K = candidates_xy.shape[0]
    anchor_b = anchor_xy.unsqueeze(0).expand(K, -1, -1)
    dists = hausdorff_distance(candidates_xy, anchor_b)    # (K,)
    return int(torch.argmin(dists).item())


# ---------------------------------------------------------------------------
# Route-masked Drivable-Area-Compliance (DAC) veto (Fix #2).
# ---------------------------------------------------------------------------

def route_dac_mask(
    candidates_xy: torch.Tensor,
    route_lanes_xy: torch.Tensor,
    threshold_m: float,
    min_frac: float,
) -> torch.Tensor:
    """Return a boolean (K,) mask of candidates that stay on the route corridor.

    For each candidate timestep, the minimum L2 distance to any valid
    route-lane sample point is computed; a step is "on-route" when that
    distance is below ``threshold_m``. A candidate passes the veto when the
    on-route fraction across its horizon is >= ``min_frac``.

    Args:
        candidates_xy: (K, T, 2) candidate ego polylines (ego frame, physical units).
        route_lanes_xy: (M, P, 2) route-lane centreline samples (ego frame, physical units).
            Padding lanes / points are all-zero rows.
        threshold_m: maximum lateral distance from a route lane to be considered on-route.
        min_frac: minimum fraction of horizon steps that must be on-route.
    Returns:
        Boolean tensor of shape (K,). All-False is returned when no route-lane
        samples are available (caller should skip the veto in that case).
    """
    K, T, _ = candidates_xy.shape
    # Flatten and drop all-zero padding samples.
    pts = route_lanes_xy.reshape(-1, 2)
    valid = pts.abs().sum(dim=-1) > 1e-6
    if not bool(valid.any()):
        return torch.zeros(K, dtype=torch.bool, device=candidates_xy.device)
    pts = pts[valid]                                                # (N, 2)
    # (K, T, N) pairwise distances; min over N gives nearest-route-sample distance.
    diff = candidates_xy.unsqueeze(2) - pts.view(1, 1, -1, 2)
    d = torch.linalg.norm(diff, dim=-1)                             # (K, T, N)
    d_min = d.min(dim=-1).values                                    # (K, T)
    on_route_frac = (d_min < float(threshold_m)).float().mean(dim=-1)
    return on_route_frac >= float(min_frac)


# ---------------------------------------------------------------------------
# Route heading veto (Fix #3a). Wrong-way candidates that stay close in XY
# to an oncoming route lane still fail this check.
# ---------------------------------------------------------------------------

def route_heading_mask(
    candidates_xy: torch.Tensor,
    route_lanes_xy: torch.Tensor,
    heading_tol_rad: float,
    min_frac: float,
    dist_threshold_m: float,
) -> torch.Tensor:
    """Return a boolean (K,) mask of candidates aligned with the route tangent.

    Per candidate timestep, the nearest valid route-lane segment is found and
    the candidate's local heading (atan2 of step delta) is compared to the
    segment tangent. A timestep counts as ``aligned`` when the wrapped angle
    difference is below ``heading_tol_rad`` AND the candidate is within
    ``dist_threshold_m`` of that segment (otherwise the route is too far away
    to make a meaningful heading claim). A candidate passes when at least
    ``min_frac`` of its in-range timesteps are aligned.

    Args:
        candidates_xy: (K, T, 2) candidate polylines in the ego frame.
        route_lanes_xy: (M, P, 2) route-lane samples; padding is all-zero.
        heading_tol_rad: max absolute heading mismatch in radians (e.g. 1.047 = 60 deg).
        min_frac: minimum fraction of in-range timesteps that must be aligned.
        dist_threshold_m: max distance to a route segment for a timestep to
            be considered "in range" for the heading check.
    Returns:
        Boolean tensor of shape (K,). All-False is returned when no valid
        route segments exist or no timestep is in range (caller should skip).
    """
    K, T, _ = candidates_xy.shape
    if T < 2:
        return torch.zeros(K, dtype=torch.bool, device=candidates_xy.device)

    # Candidate per-step tangents and midpoints.
    cand_d = candidates_xy[:, 1:] - candidates_xy[:, :-1]            # (K, T-1, 2)
    cand_th = torch.atan2(cand_d[..., 1], cand_d[..., 0])            # (K, T-1)
    cand_mid = 0.5 * (candidates_xy[:, 1:] + candidates_xy[:, :-1])  # (K, T-1, 2)

    # Route-lane segment midpoints, tangents, validity (drop padded/zero segs).
    rl_d = route_lanes_xy[:, 1:] - route_lanes_xy[:, :-1]            # (M, P-1, 2)
    rl_mid = 0.5 * (route_lanes_xy[:, 1:] + route_lanes_xy[:, :-1])  # (M, P-1, 2)
    rl_len = torch.linalg.norm(rl_d, dim=-1)                         # (M, P-1)
    valid = (rl_len > 1e-3).reshape(-1)
    if not bool(valid.any()):
        return torch.zeros(K, dtype=torch.bool, device=candidates_xy.device)
    rl_mid_flat = rl_mid.reshape(-1, 2)[valid]                       # (N, 2)
    rl_th = torch.atan2(rl_d[..., 1], rl_d[..., 0]).reshape(-1)[valid]  # (N,)

    # Nearest valid segment per (K, T-1) timestep.
    diff = cand_mid.unsqueeze(2) - rl_mid_flat.view(1, 1, -1, 2)     # (K, T-1, N, 2)
    dist = torch.linalg.norm(diff, dim=-1)                           # (K, T-1, N)
    nearest_dist, nearest_idx = dist.min(dim=-1)                     # both (K, T-1)
    near_th = rl_th[nearest_idx]                                     # (K, T-1)

    pi = float(torch.pi)
    dh = torch.remainder(cand_th - near_th + pi, 2.0 * pi) - pi
    aligned = dh.abs() < float(heading_tol_rad)                      # (K, T-1)
    in_range = nearest_dist < float(dist_threshold_m)                # (K, T-1)
    n_in = in_range.sum(dim=-1)                                      # (K,)
    if not bool((n_in > 0).any()):
        return torch.zeros(K, dtype=torch.bool, device=candidates_xy.device)
    frac_aligned = (aligned & in_range).sum(dim=-1).float() / n_in.clamp(min=1).float()
    # Candidates with zero in-range steps are excluded (route too far to judge).
    return (n_in > 0) & (frac_aligned >= float(min_frac))


# ---------------------------------------------------------------------------
# TrajectoryMomentumBuffer — stores previous plan in world frame, emits the
# anchor tensor in the current ego frame on demand.
# ---------------------------------------------------------------------------

class TrajectoryMomentumBuffer:
    """Single-slot buffer that carries the previously emitted plan across calls.

    The plan is held in **world coordinates** (so the rigid-body change between
    the previous and current ego pose is applied exactly when fetching the
    anchor) along with the previous ego pose, which is used by user code only
    for diagnostics.
    """

    def __init__(self, predicted_neighbor_num: int, future_len: int) -> None:
        self._P = 1 + int(predicted_neighbor_num)
        self._T = int(future_len)
        self._prev_xy_world: Optional[np.ndarray] = None   # (T, 2)
        self._prev_cossin_world: Optional[np.ndarray] = None  # (T, 2)

    def reset(self) -> None:
        self._prev_xy_world = None
        self._prev_cossin_world = None

    @property
    def has_anchor(self) -> bool:
        return self._prev_xy_world is not None

    def update(self, ego_traj_ego_frame: np.ndarray, ego_pose_world: Tuple[float, float, float]) -> None:
        """Store the chosen ego plan, converting it to world frame for next call.

        Args:
            ego_traj_ego_frame: (T, 4) — (x, y, cos, sin) in the ego frame this
                plan was emitted in.
            ego_pose_world: (x, y, heading) world pose of the ego at emission.
        """
        assert ego_traj_ego_frame.shape == (self._T, 4), \
            f"expected ({self._T}, 4), got {ego_traj_ego_frame.shape}"
        xy_e = ego_traj_ego_frame[:, :2]
        cs_e = ego_traj_ego_frame[:, 2:4]
        self._prev_xy_world = ego_to_world_xy(xy_e, ego_pose_world)
        cw, sw = heading_ego_to_world(cs_e[:, 0], cs_e[:, 1], ego_pose_world[2])
        self._prev_cossin_world = np.stack([cw, sw], axis=-1)

    def get_anchor(
        self,
        ego_pose_world: Tuple[float, float, float],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Return the anchor tensor ``(1, P, T, 4)`` in the *current* ego frame,
        or ``None`` if no plan has been buffered yet. Non-ego rows are zero-filled
        so the warm-start prior only biases the ego row."""
        if self._prev_xy_world is None:
            return None
        xy_e = world_to_ego_xy(self._prev_xy_world, ego_pose_world)
        ce, se = heading_world_to_ego(
            self._prev_cossin_world[:, 0], self._prev_cossin_world[:, 1], ego_pose_world[2]
        )
        ego_row = np.concatenate([xy_e, np.stack([ce, se], axis=-1)], axis=-1)  # (T, 4)
        anchor = np.zeros((1, self._P, self._T, 4), dtype=np.float32)
        anchor[0, 0] = ego_row.astype(np.float32)
        return torch.from_numpy(anchor).to(device=device, dtype=dtype)
