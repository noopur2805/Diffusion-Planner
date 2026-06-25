"""Defensive fallback trajectories for the safety tier.

When the conformal safety gate flags the DiT plan as untrusted, the
planner emits one of these conservative trajectories instead. They live
in the *current* ego (rear-axle) frame and match the ``(K, P, T, 4)``
layout produced by :class:`Diffusion_Planner` so they can be returned
in place of ``outputs['prediction']`` without further reshaping.

The layout of the last dimension is ``(x, y, cos(heading), sin(heading))``
in meters / unitless, matching ``outputs_to_trajectory`` downstream.
"""
from __future__ import annotations

import torch

from nuplan.common.actor_state.ego_state import EgoState


def constant_velocity_fallback(
    ego_state: EgoState,
    future_len: int,
    dt: float,
    *,
    decel: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Straight-ahead constant-velocity (optionally constant-decel) plan.

    The ego keeps its current rear-axle longitudinal speed and yaws zero;
    in the ego frame this is a straight forward motion along ``+x``. When
    ``decel > 0`` the speed is reduced linearly until it reaches zero,
    after which the trajectory holds position (a gentle "coast to stop").

    Args:
        ego_state:   current nuPlan ego state (used only for speed).
        future_len:  number of future steps ``T`` to emit (matches
                     ``config.future_len``, typically 80 for 8 s at 10 Hz).
        dt:          step size in seconds (matches ``config.time_len``).
        decel:       optional non-negative deceleration in m/s^2.
        device:      target device for the returned tensor.
        dtype:       target dtype for the returned tensor.

    Returns:
        Tensor of shape ``(1, 1, future_len, 4)`` -- the same layout as the
        planner's ``prediction`` output (one candidate, one agent = ego).
    """
    v0 = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    v0 = max(v0, 0.0)  # never reverse on a safety fallback

    ts = torch.arange(1, future_len + 1, device=device, dtype=dtype) * float(dt)

    if decel > 0.0:
        t_stop = v0 / float(decel) if decel > 0 else float("inf")
        t_clip = torch.clamp(ts, max=t_stop)
        x = v0 * t_clip - 0.5 * float(decel) * t_clip.pow(2)
    else:
        x = v0 * ts

    y = torch.zeros_like(x)
    cos_h = torch.ones_like(x)
    sin_h = torch.zeros_like(x)

    traj = torch.stack([x, y, cos_h, sin_h], dim=-1)        # (T, 4)
    return traj.view(1, 1, future_len, 4)
