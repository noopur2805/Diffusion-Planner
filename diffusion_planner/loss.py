from typing import Any, Callable, Dict, List, Tuple
import math
import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer
from diffusion_planner.utils import ddp


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],

    futures: Tuple[torch.Tensor, torch.Tensor],
    
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    eps: float = 1e-3,
):   
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask # [B, P, V]

    B, Pn, T, _ = neighbors_future.shape
    ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1) # [B, P = 1 + 1 + neighbor, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1) # [B, P, 4]

    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps # [B,]
    z = torch.randn_like(gt_future, device=gt_future.device) # [B, P, T, 4]
    
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs) # [B, P, 1 + T, 4]
    score = decoder_output["score"][:, :, 1:, :] # [B, P, T, 4]

    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :])**2, dim=-1)
    
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output


def _sample_shortcut_step(B: int, k_max: int, device) -> torch.Tensor:
    """
    Sample d ~ Uniform({1/k_max, 2/k_max, 4/k_max, ..., 1}) on the dyadic grid.
    Returned tensor has shape (B,) of continuous values in (0, 1].
    """
    n_levels = int(math.log2(k_max)) + 1  # k_max=16 -> levels {1,2,4,8,16} -> 5
    levels = torch.randint(0, n_levels, (B,), device=device)
    d = (2.0 ** levels.float()) / float(k_max)
    return d


def _sample_shortcut_time(d: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Sample t on the dyadic grid {0, d, 2d, ..., 1-d} per element, then clamp to (eps, 1-eps).
    """
    n_bins = torch.clamp((1.0 / d).round().long(), min=1)
    u = torch.rand_like(d)
    k = (u * n_bins.float()).floor()
    t = k * d
    return t.clamp(min=eps, max=1.0 - eps)


def shortcut_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],
    futures: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    norm: StateNormalizer,
    loss: Dict[str, Any],
    model_type: str,
    k_max: int = 16,
    eps: float = 1e-3,
    ddp_enabled: bool = True,
):
    """
    Shortcut Forcing distillation loss (Frans et al., 2024) adapted to the
    subVP-SDE / x_start parameterization used by Diffusion-Planner.

    For d == 1/k_max: standard diffusion loss (same target as ``diffusion_loss_func``).
    For d  > 1/k_max: target is the stop-gradient mean of two student half-steps
                     evaluated at step d/2, mirroring eq. (4-7) of the paper.
    """
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask

    B, Pn, T, _ = neighbors_future.shape
    ego_current = inputs["ego_current_state"][:, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future], dim=1)
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
    P = gt_future.shape[1]

    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0
    target_x0 = all_gt[..., 1:, :]

    d_min = 1.0 / float(k_max)
    d = _sample_shortcut_step(B, k_max, gt_future.device)
    t = _sample_shortcut_time(d, eps)

    z = torch.randn_like(target_x0)
    mean, std = marginal_prob(target_x0, t)
    std = std.view(-1, *([1] * (len(target_x0.shape) - 1)))
    xt = mean + std * z
    xt = torch.cat([all_gt[:, :, :1, :], xt], dim=2).reshape(B, P, -1)

    # ---- student forward at (t, d) ----
    inputs_student = {**inputs, "sampled_trajectories": xt, "diffusion_time": t, "shortcut_step": d}
    _, decoder_output = model(inputs_student)
    pred = decoder_output["score"][:, :, 1:, :]

    # ---- build distillation target ----
    is_self = (d <= d_min + 1e-9)
    with torch.no_grad():
        # only run teacher passes for elements with d > d_min, but do it in-batch (mask later)
        half_d = (d * 0.5).clamp(min=d_min)
        # student half-step 1: (xt, t, d/2) -> v1 (in x_start space, "predicted x0")
        inputs_h1 = {**inputs, "sampled_trajectories": xt, "diffusion_time": t, "shortcut_step": half_d}
        _, out_h1 = ddp.get_model(model, ddp_enabled)(inputs_h1)
        x0_h1 = out_h1["score"][:, :, 1:, :]

        # form intermediate x_{t+d/2} by re-noising x0_h1 to time t+d/2
        t_mid = (t + half_d).clamp(max=1.0 - eps)
        mean_mid, std_mid = marginal_prob(x0_h1, t_mid)
        std_mid = std_mid.view(-1, *([1] * (len(x0_h1.shape) - 1)))
        z_mid = torch.randn_like(x0_h1)
        xt_mid_full = torch.cat([all_gt[:, :, :1, :], mean_mid + std_mid * z_mid], dim=2).reshape(B, P, -1)

        inputs_h2 = {**inputs, "sampled_trajectories": xt_mid_full, "diffusion_time": t_mid, "shortcut_step": half_d}
        _, out_h2 = ddp.get_model(model, ddp_enabled)(inputs_h2)
        x0_h2 = out_h2["score"][:, :, 1:, :]

        teacher_target = 0.5 * (x0_h1 + x0_h2)

    # broadcast is_self mask to per-element
    is_self_b = is_self.view(B, 1, 1, 1).expand_as(target_x0)
    distill_target = torch.where(is_self_b, target_x0, teacher_target)

    if model_type == "score":
        # score-parameterized: convert pred to x0 via: x0 = (xt - std * eps_pred * std) / mean_log_coeff
        # For simplicity, fall back to x_start parameterization for shortcut training.
        raise NotImplementedError("shortcut_loss_func requires diffusion_model_type='x_start'.")

    sc_loss = torch.sum((pred - distill_target) ** 2, dim=-1)
    masked_pred = sc_loss[:, 1:, :][neighbors_future_valid]
    loss["neighbor_prediction_loss"] = masked_pred.mean() if masked_pred.numel() > 0 else torch.tensor(0.0, device=sc_loss.device)
    loss["ego_planning_loss"] = sc_loss[:, 0, :].mean()
    loss["shortcut_self_ratio"] = is_self.float().mean().detach()

    assert not torch.isnan(sc_loss).any(), "shortcut loss NaN"

    return loss, decoder_output