"""
Trajectory vocabulary utilities (DreamerAD eq. 9, 20, 21).

Two stages:
    1) ``build_vocabulary``: filter a corpus of GT future trajectories by
       end-state deviation from the *running* GT, then uniform-sample by
       lateral offset to keep a diverse set of K representative trajectories.
    2) ``gaussian_vocab_sample``: at training time, rank vocabulary entries by
       Mahalanobis distance to the policy's mean trajectory and return a
       mixed batch of (a) top-softmax discriminative samples and
       (b) Gaussian-neighborhood exploratory samples.
"""
from typing import Optional

import numpy as np
import torch


def _wrap(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def filter_by_endstate(
    candidates: torch.Tensor,        # (N, T, 3) -> x, y, heading
    gt: torch.Tensor,                # (T, 3) ground-truth reference
    x_thresh: float = 10.0,
    y_thresh: float = 5.0,
    theta_thresh: float = 20.0 * np.pi / 180.0,
) -> torch.Tensor:
    """Return a boolean mask over candidates."""
    dx = (candidates[:, -1, 0] - gt[-1, 0]).abs()
    dy = (candidates[:, -1, 1] - gt[-1, 1]).abs()
    dtheta = _wrap(candidates[:, -1, 2] - gt[-1, 2]).abs()
    return (dx <= x_thresh) & (dy <= y_thresh) & (dtheta <= theta_thresh)


def build_vocabulary(
    candidates: torch.Tensor,
    gt: torch.Tensor,
    K: int = 256,
    x_thresh: float = 10.0,
    y_thresh: float = 5.0,
    theta_thresh: float = 20.0 * np.pi / 180.0,
) -> torch.Tensor:
    """Filter, then uniformly sub-sample by lateral offset to get K trajectories.

    Args:
        candidates: (N, T, 3) tensor of candidate future trajectories.
        gt: (T, 3) reference trajectory used for filtering.
    Returns:
        vocab: (K, T, 3) tensor (K may be less if not enough candidates).
    """
    mask = filter_by_endstate(candidates, gt, x_thresh, y_thresh, theta_thresh)
    kept = candidates[mask]
    if kept.numel() == 0:
        return candidates[:1]  # degenerate fallback

    dy = (kept[:, -1, 1] - gt[-1, 1]).abs()
    order = torch.argsort(dy)
    kept = kept[order]
    if kept.shape[0] <= K:
        return kept
    idx = torch.linspace(0, kept.shape[0] - 1, steps=K).long()
    return kept[idx]


@torch.no_grad()
def gaussian_vocab_sample(
    vocab: torch.Tensor,            # (V, T, 3) shared across batch
    policy_traj: torch.Tensor,      # (B, T, 3) current mean policy trajectory
    g1: int = 8,                    # top-softmax samples for discrimination
    g2: int = 8,                    # neighborhood samples for exploration
    sigma_xy: float = 1.5,
    sigma_h: float = 0.2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Returns (B, g1 + g2, T, 3) sampled trajectories from the vocabulary.

    Computes Mahalanobis distance between vocabulary entries and the policy
    trajectory under a diagonal Gaussian, then:
        - g1 entries via categorical sampling from softmax(-d / temperature)
        - g2 entries via top-g2 by smallest distance (deterministic neighbourhood)

    The first axis of the output is batch; entries are independently sampled
    per batch element.
    """
    B, T, _ = policy_traj.shape
    V = vocab.shape[0]
    sigma = torch.tensor([sigma_xy, sigma_xy, sigma_h], device=policy_traj.device)

    diff = vocab[None] - policy_traj[:, None]  # (B, V, T, 3)
    diff[..., 2] = _wrap(diff[..., 2])
    d = ((diff / sigma) ** 2).sum(dim=(-1, -2))  # (B, V)

    logits = -d / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    discrim_idx = torch.multinomial(probs, num_samples=g1, replacement=True)  # (B, g1)
    # neighborhood: smallest distance
    nbh_idx = torch.topk(-d, k=g2, dim=-1).indices  # (B, g2)
    idx = torch.cat([discrim_idx, nbh_idx], dim=-1)  # (B, g1+g2)

    gathered = vocab[idx]  # (B, G, T, 3)
    return gathered, idx, d


def total_reward_from_dense(
    dense_rewards: torch.Tensor,    # (B, T_h, K) sigmoid probs in [0,1]
    safety_idx=(0, 1, 2),           # nc, dac, ttc
    task_idx=(3, 4),                # ep, comfort
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    DreamerAD eqs. (16-19): log-sigmoid aggregation of safety + log of task sum.
    Returns (B,) trajectory-level rewards (sum over horizons).
    """
    safety = dense_rewards[..., list(safety_idx)].clamp(min=eps)
    task = dense_rewards[..., list(task_idx)].clamp(min=eps)
    L = safety.log().sum(dim=-1)                       # (B, T_h)
    S = task.sum(dim=-1).clamp(min=eps).log()          # (B, T_h)
    per_horizon = L + S
    return per_horizon.sum(dim=-1)                     # (B,)
