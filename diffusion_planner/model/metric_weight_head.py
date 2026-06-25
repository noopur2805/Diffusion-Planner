"""
Context-conditional reward priorities.

``MetricWeightHead`` consumes the planner's scene context tokens and emits a
per-scene per-metric weight vector ``w(s) in R^K`` used by
``total_reward_from_dense`` to bias the safety-log and task-sum aggregations.
Weights are positive (softplus) and normalized so their mean is 1.0, which
makes the head a perturbation around the uniform aggregator and keeps the
overall reward scale comparable to the unweighted baseline.

The head is small (~25K params for hidden_dim=192, K=5) and can be:
    * trained alongside the AD-RM in ``train_reward.py`` to learn weights
      that improve a downstream objective (e.g. ranking GT > perturbed), or
    * frozen and reused at GRPO time as a scene-conditional aggregator.

Backward compatibility is preserved: if no head is constructed,
``total_reward_from_dense`` falls back to the original uniform aggregator.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricWeightHead(nn.Module):
    """Predict per-scene per-metric weights from encoder context tokens."""

    def __init__(
        self,
        hidden_dim: int = 192,
        n_metrics: int = 8,
        n_heads: int = 4,
        dropout: float = 0.1,
        weight_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_metrics = n_metrics
        self.weight_floor = float(weight_floor)

        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.pool = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_metrics),
        )

    def forward(
        self,
        context_tokens: torch.Tensor,                # (B, N, D) or (B, H, N, D)
        context_mask: Optional[torch.Tensor] = None,  # (B, N) True = padded
    ) -> torch.Tensor:
        """Returns weights of shape ``(B, n_metrics)``, positive, mean 1."""
        if context_tokens.dim() == 4:
            ctx = context_tokens.mean(dim=1)
        elif context_tokens.dim() == 3:
            ctx = context_tokens
        else:
            raise ValueError(
                f"context_tokens must be 3D or 4D, got {context_tokens.dim()}D"
            )

        B = ctx.shape[0]
        q = self.query[None].expand(B, -1, -1)
        kpm = context_mask if (context_mask is not None and context_mask.dim() == 2) else None
        pooled, _ = self.pool(q, ctx, ctx, key_padding_mask=kpm)
        logits = self.mlp(pooled.squeeze(1))                   # (B, K)
        w = F.softplus(logits) + self.weight_floor             # positive
        w = w * (self.n_metrics / w.sum(dim=-1, keepdim=True)) # normalize: mean 1
        return w
