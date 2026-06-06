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


def group_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    rewards: (B, G)
    Returns (B, G) advantages standardized within each group.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (rewards - mean) / std


def grpo_actor_loss(
    new_logprob: torch.Tensor,   # (B, G)
    old_logprob: torch.Tensor,   # (B, G)
    advantage: torch.Tensor,     # (B, G)
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = (new_logprob - old_logprob).exp()
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return torch.maximum(unclipped, clipped).mean()


def bc_loss(policy_mean: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(policy_mean, gt)


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


def grpo_total_loss(
    new_logprob, old_logprob, advantages,
    policy_mean, gt, ref_mean,
    sigma_xy: float, sigma_h: float,
    w_bc: float = 1.0, w_kl: float = 0.1, clip_eps: float = 0.2,
):
    actor = grpo_actor_loss(new_logprob, old_logprob, advantages, clip_eps)
    bc = bc_loss(policy_mean, gt)
    kl = policy_kl(policy_mean, ref_mean, sigma_xy, sigma_h)
    return actor + w_bc * bc + w_kl * kl, {"actor": actor.detach(), "bc": bc.detach(), "kl": kl.detach()}
