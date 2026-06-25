"""
GRPO (Group Relative Policy Optimization) loss for the Diffusion-Planner.

Implements eqs. (22-25) of DreamerAD with three additive terms:
    * actor loss (clipped surrogate over group-normalized advantages)
    * behavioral cloning loss against the GT trajectory
    * KL divergence to a frozen reference policy (the SFT model)

The policy log-prob is computed under a diagonal Gaussian centred at the
planner's mean output, with a fixed variance (matching the vocabulary
sampling distribution used to draw candidates).
"""
from typing import Optional

import torch
import torch.nn.functional as F


def _diag_gauss_logprob(x: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """log N(x; mean, diag(sigma**2)) summed over last two dims."""
    var = sigma ** 2
    log_det = (sigma * 0 + sigma.log()).sum() * 0 + (var.log()).sum() * 0  # constants drop out for ratio
    sq = (x - mean) ** 2 / var
    return -0.5 * sq.sum(dim=(-1, -2)) - log_det


def diag_gauss_logprob(samples: torch.Tensor, mean: torch.Tensor, sigma_xy: float, sigma_h: float) -> torch.Tensor:
    """samples: (B, G, T, 3) or (B, G, T, 4); mean: (B, T, D). Returns (B, G)."""
    if samples.shape[-1] == 4:
        # convert mean to (cos, sin) representation as well if needed
        mean4 = torch.cat([mean[..., :2], mean[..., 2:3].cos(), mean[..., 2:3].sin()], dim=-1) \
            if mean.shape[-1] == 3 else mean
        sigma = torch.tensor([sigma_xy, sigma_xy, sigma_h, sigma_h], device=samples.device)
        m = mean4
    else:
        sigma = torch.tensor([sigma_xy, sigma_xy, sigma_h], device=samples.device)
        m = mean

    diff = samples - m.unsqueeze(1)
    if samples.shape[-1] == 3:
        # wrap heading difference
        diff = torch.cat([diff[..., :2], torch.atan2(diff[..., 2:3].sin(), diff[..., 2:3].cos())], dim=-1)
    var = sigma ** 2
    sq = (diff ** 2) / var
    return -0.5 * sq.sum(dim=(-1, -2))


def group_advantage(
    rewards: torch.Tensor,
    r_sft: Optional[torch.Tensor] = None,
    std_floor: float = 1e-6,
    clip: float = 0.0,
) -> torch.Tensor:
    """
    rewards: (B, G)
    r_sft:   optional (B,) baseline reward of the SFT mean trajectory. When
             provided, the per-row baseline is lifted to max(group_mean, r_sft)
             so candidates worse than SFT receive non-positive advantage even
             if they beat their group's average ("SFT-anchored GRPO").
    std_floor: floor on the group reward std before dividing. The original
             1e-6 floor is too small when sparse PDMS gates (e.g. making_progress)
             collapse most candidates to ~0 reward: the Z-score then explodes
             and drives actor-loss spikes. A more conservative floor (e.g.
             0.05) keeps the scale finite while preserving rank order.
    clip:    if > 0, clamp advantages to [-clip, +clip] after normalization,
             which prevents single-batch outliers from poisoning the clipped
             surrogate. 0 (default) disables clipping.
    Returns (B, G) advantages standardized within each group.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    if r_sft is not None:
        mean = torch.maximum(mean, r_sft.unsqueeze(-1))
    std = rewards.std(dim=-1, keepdim=True).clamp_min(std_floor)
    adv = (rewards - mean) / std
    if clip > 0.0:
        adv = adv.clamp(-clip, clip)
    return adv


def grpo_actor_loss(
    new_logprob: torch.Tensor,   # (B, G)
    old_logprob: torch.Tensor,   # (B, G)
    advantage: torch.Tensor,     # (B, G)
    clip_eps: float = 0.2,
    log_ratio_clip: float = 2.0,
) -> torch.Tensor:
    # Clamp log-ratio before exp to prevent overflow on high-D Gaussian log-probs.
    # The pessimistic PPO objective is unbounded for negative advantages when
    # ratio >> 1+clip_eps, so the actual per-sample bound is |adv| * exp(log_ratio_clip).
    # 2.0 caps ratio at ~7.4 (one outlier can push actor mean up by ~adv*7.4/G);
    # 10.0 (legacy) allowed ratio up to 22k, dominating batches via single samples.
    log_ratio = (new_logprob - old_logprob).clamp(-log_ratio_clip, log_ratio_clip)
    ratio = log_ratio.exp()
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return torch.maximum(unclipped, clipped).mean()


def bc_loss(policy_mean: torch.Tensor, gt: torch.Tensor,
            horizon_weight_alpha: float = 0.0) -> torch.Tensor:
    """L1 between policy mean and GT future. When ``horizon_weight_alpha > 0``
    a quadratic late-heavy weight ``(1 + alpha * h/(H-1))**2`` is applied per
    timestep before the mean, focusing the anchor on the long-horizon tail
    where off-course drift manifests. ``alpha=0`` is bit-identical to the
    original uniform L1."""
    if horizon_weight_alpha <= 0.0:
        return F.l1_loss(policy_mean, gt)
    T = policy_mean.shape[-2]
    h = torch.arange(T, device=policy_mean.device, dtype=policy_mean.dtype)
    h = h / max(T - 1, 1)
    w = (1.0 + horizon_weight_alpha * h).pow(2).view(*([1] * (policy_mean.dim() - 2)), T, 1)
    num = ((policy_mean - gt).abs() * w).sum()
    denom = w.sum() * policy_mean.shape[0] * policy_mean.shape[-1]
    return num / denom


def policy_kl(
    policy_mean: torch.Tensor,   # (B, T, D)
    ref_mean: torch.Tensor,      # (B, T, D) detached
    sigma_xy: float,
    sigma_h: float,
) -> torch.Tensor:
    """
    KL between two diagonal Gaussians with the *same* fixed sigma reduces to:
        0.5 * sum( (mu - mu_ref)^2 / sigma^2 )
    """
    if policy_mean.shape[-1] == 3:
        sigma = torch.tensor([sigma_xy, sigma_xy, sigma_h], device=policy_mean.device)
    else:
        sigma = torch.tensor([sigma_xy, sigma_xy, sigma_h, sigma_h], device=policy_mean.device)
    diff = policy_mean - ref_mean.detach()
    if policy_mean.shape[-1] == 3:
        diff = torch.cat([diff[..., :2], torch.atan2(diff[..., 2:3].sin(), diff[..., 2:3].cos())], dim=-1)
    return 0.5 * ((diff ** 2) / (sigma ** 2)).sum(dim=(-1, -2)).mean()


def trajectory_dispersion(policy_mean: torch.Tensor) -> torch.Tensor:
    """Batch-wise trajectory dispersion — a tractable mode-collapse proxy for
    a diffusion policy whose log-prob is a Gaussian with fixed sigma (so the
    closed-form Gaussian entropy is constant and useless as a regularizer).

    Measures how much per-scene predictions deviate from the batch mean
    trajectory. When the policy collapses to a single attractor (e.g. the
    epoch-2 entropy-collapse failure mode), per-scene predictions become
    nearly identical and this quantity shrinks; the GRPO objective subtracts
    ``w_ent * dispersion`` so collapse incurs a penalty.

    policy_mean: (B, T, D). Returns a scalar tensor (mean squared deviation
    from the batch-mean trajectory, averaged over T and D).
    """
    if policy_mean.dim() < 3 or policy_mean.shape[0] < 2:
        return policy_mean.new_zeros(())
    mu_b = policy_mean.mean(dim=0, keepdim=True)
    return ((policy_mean - mu_b) ** 2).mean()


def grpo_total_loss(
    new_logprob, old_logprob, advantages,
    policy_mean, gt, ref_mean,
    sigma_xy: float, sigma_h: float,
    w_bc: float = 1.0, w_kl: float = 0.1, clip_eps: float = 0.2,
    bc_horizon_alpha: float = 0.0,
    log_ratio_clip: float = 2.0,
    w_ent: float = 0.0,
):
    actor = grpo_actor_loss(new_logprob, old_logprob, advantages, clip_eps,
                            log_ratio_clip=log_ratio_clip)
    bc = bc_loss(policy_mean, gt, horizon_weight_alpha=bc_horizon_alpha)
    kl = policy_kl(policy_mean, ref_mean, sigma_xy, sigma_h)
    if w_ent > 0.0:
        ent = trajectory_dispersion(policy_mean)
        total = actor + w_bc * bc + w_kl * kl - w_ent * ent
        log = {"actor": actor.detach(), "bc": bc.detach(), "kl": kl.detach(),
               "ent": ent.detach()}
    else:
        total = actor + w_bc * bc + w_kl * kl
        log = {"actor": actor.detach(), "bc": bc.detach(), "kl": kl.detach(),
               "ent": policy_mean.new_zeros(()).detach()}
    return total, log
