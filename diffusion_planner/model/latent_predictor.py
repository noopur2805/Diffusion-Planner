"""
Vectorized Latent World Model for the AD-RM.

Predicts per-horizon "imagined" scene latents from the current scene context
tokens (produced by ``Encoder``) conditioned on a candidate ego trajectory.
This is the vectorized analogue of DreamerAD's video-DiT latent rollout: it
produces ``H`` future token sets ``[z_{t+1}, ..., z_{t+H}]`` that the AD-RM can
cross-attend to, instead of attending to the static ``z_t`` only.

Design choices:
    * Lightweight (a few cross-attention layers) so it adds little inference
      cost on top of the AD-RM.
    * Additive horizon + action conditioning, mirroring the AD-RM's query
      construction (eq. 12-13 of DreamerAD) for consistency.
    * Trained end-to-end with the AD-RM reward loss; no separate "next-frame"
      supervision target is required because the AD-RM's BCE loss already
      gives the predictor a useful gradient signal.

Inputs:
    context_tokens: (B, N, D)
    trajectory:     (B, T, traj_dim)
Outputs:
    future_latents: (B, H, N, D)
"""
from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import Mlp


class LatentWorldModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 192,
        traj_dim: int = 4,
        n_horizons: int = 8,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        adaptive_horizons: bool = False,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_horizons = n_horizons
        self.adaptive_horizons = adaptive_horizons
        self.dt = float(dt)

        self.traj_mlp = Mlp(
            in_features=traj_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            drop=dropout,
        )
        self.step_emb = nn.Embedding(n_horizons, hidden_dim)
        if adaptive_horizons:
            self.step_proj = Mlp(
                in_features=2, hidden_features=hidden_dim,
                out_features=hidden_dim, drop=dropout,
            )

        # Refinement stack: each block lets per-horizon tokens attend to the
        # *original* context tokens (so the predictor cannot drift away from
        # the observable scene) plus a small FFN.
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffns = nn.ModuleList([
            Mlp(in_features=hidden_dim, hidden_features=hidden_dim * 4,
                out_features=hidden_dim, act_layer=nn.GELU, drop=dropout)
            for _ in range(n_layers)
        ])
        self.norms_q = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.norms_kv = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.norms_ff = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def _compute_horizons(self, traj: torch.Tensor):
        """Pick H samples uniformly in time (legacy) or in cumulative distance
        along ``traj[..., :2]`` (when ``adaptive_horizons=True``). Returns the
        resampled trajectory ``(B, H, D)`` and per-horizon ``(time_sec, dist_m)``
        features ``(B, H, 2)`` used for continuous conditioning.
        """
        B, T, D = traj.shape
        H = self.n_horizons
        device = traj.device

        pos = traj[..., :2]
        seg = torch.norm(pos[:, 1:] - pos[:, :-1], dim=-1)
        cum_dist = torch.cat(
            [torch.zeros(B, 1, device=device, dtype=traj.dtype), seg.cumsum(dim=-1)], dim=1
        )

        if self.adaptive_horizons:
            d_max = cum_dist[:, -1].clamp(min=1e-3)
            fracs = torch.linspace(1.0 / H, 1.0, H, device=device, dtype=traj.dtype)
            targets = fracs[None] * d_max[:, None]
            idx = (cum_dist[:, None, :] - targets[..., None]).abs().argmin(dim=-1)
        else:
            idx_1d = torch.linspace(0, T - 1, steps=H, device=device).long()
            idx = idx_1d[None].expand(B, H).contiguous()

        idx_exp = idx[..., None].expand(B, H, D)
        traj_h = traj.gather(1, idx_exp)
        tau_time = idx.to(traj.dtype) * self.dt
        tau_dist = cum_dist.gather(1, idx)
        tau = torch.stack([tau_time, tau_dist], dim=-1)
        return traj_h, tau

    def forward(
        self,
        context_tokens: torch.Tensor,
        trajectory: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: (B, N, D) scene tokens from the planner's Encoder.
            trajectory:     (B, T, traj_dim) candidate ego trajectory.
            context_mask:   (B, N) optional bool padding mask (True = pad).
        Returns:
            future_latents: (B, H, N, D)
        """
        B, N, D = context_tokens.shape
        H = self.n_horizons

        traj_h, tau = self._compute_horizons(trajectory)         # (B, H, traj_dim), (B, H, 2)
        a_emb = self.traj_mlp(traj_h)                            # (B, H, D)
        if self.adaptive_horizons:
            s_emb = self.step_proj(tau)                          # (B, H, D)
        else:
            steps = torch.arange(H, device=trajectory.device)
            s_emb = self.step_emb(steps)[None].expand(B, -1, -1)
        cond = (a_emb + s_emb)[:, :, None, :]                    # (B, H, 1, D)

        # Broadcast context across horizons; add action+horizon conditioning.
        x = context_tokens[:, None].expand(B, H, N, D) + cond    # (B, H, N, D)
        x = x.reshape(B * H, N, D)
        ctx_flat = context_tokens[:, None].expand(B, H, N, D).reshape(B * H, N, D)

        if context_mask is not None:
            kv_mask = context_mask[:, None].expand(B, H, N).reshape(B * H, N)
        else:
            kv_mask = None

        for attn, ffn, n_q, n_kv, n_ff in zip(
            self.attn_blocks, self.ffns, self.norms_q, self.norms_kv, self.norms_ff
        ):
            q = n_q(x)
            kv = n_kv(ctx_flat)
            delta, _ = attn(q, kv, kv, key_padding_mask=kv_mask)
            x = x + delta
            x = x + ffn(n_ff(x))

        return x.reshape(B, H, N, D)
