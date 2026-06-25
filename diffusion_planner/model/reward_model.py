"""
Autoregressive Dense Reward Model (AD-RM) - DreamerAD eq. (10-15).

Predicts per-horizon, per-metric trajectory quality from the planner's scene
context tokens and a candidate ego trajectory. In the vectorized
Diffusion-Planner setting the "imagined latent" of DreamerAD is replaced by
the *scene context tokens* produced by ``Encoder``. When a per-horizon latent
predictor is wired in upstream the AD-RM can also consume a 4-D
``(B, H, N, D)`` context so each (horizon, metric) query attends to the
horizon-specific imagined latent (closer to the paper's video-DiT rollout).

Optional aleatoric-uncertainty head emits ``(mu, log_var)`` per (horizon,
metric); the mean is BCE-supervised while ``log_var`` is fitted to the
residual squared error so downstream GRPO can down-weight uncertain rewards.

Inputs:
    context_tokens: (B, N, D)   tokens from ``Encoder.forward(...)['encoding']``
                    (B, H, N, D) per-horizon imagined latents (optional path)
    context_mask:   (B, N) or (B, H, N) True = padding
    trajectory:     (B, T, traj_dim) ego candidate (x, y, cos, sin) or (x, y, heading)
Outputs:
    rewards:        (B, T_h, K) logits, OR (mu, log_var) if predict_uncertainty=True
"""
from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import Mlp


METRIC_NAMES = ("nc", "dac", "ttc", "ep", "comfort", "ddc", "mp", "sl")
# nc=collision-free, dac=drivable-area, ttc=time-to-collision, ep=ego-progress,
# comfort, ddc=driving-direction-compliance, mp=ego-is-making-progress, sl=speed-limit


class AutoregressiveDenseRewardModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 192,
        traj_dim: int = 4,
        n_horizons: int = 8,
        metric_names = METRIC_NAMES,
        n_heads: int = 4,
        n_layers: int = 3,
        query_compression: int = 32,
        dropout: float = 0.1,
        predict_uncertainty: bool = False,
        adaptive_horizons: bool = False,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_horizons = n_horizons
        self.metric_names = list(metric_names)
        self.n_metrics = len(self.metric_names)
        self.query_compression = query_compression
        self.predict_uncertainty = predict_uncertainty
        self.adaptive_horizons = adaptive_horizons
        self.dt = float(dt)

        self.traj_mlp = Mlp(in_features=traj_dim, hidden_features=hidden_dim, out_features=hidden_dim, drop=dropout)
        self.step_emb = nn.Embedding(n_horizons, hidden_dim)
        if adaptive_horizons:
            self.step_proj = Mlp(in_features=2, hidden_features=hidden_dim,
                                 out_features=hidden_dim, drop=dropout)

        # K learnable reward bases, one per metric (eq. 13).
        self.q_base = nn.Parameter(torch.randn(self.n_metrics, hidden_dim) * 0.02)

        # learnable query that compresses the context from N tokens to L=query_compression.
        self.context_query = nn.Parameter(torch.randn(query_compression, hidden_dim) * 0.02)
        self.ctx_compress = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        # Pre-LN on the query so cross-attn doesn't have to consume the raw
        # (q_base + traj + step) magnitude, and post-LN on the residual stream
        # before the head. Without the residual the trajectory only influences
        # the output through the attention weighting of V, which empirically
        # collapsed to scene-only predictions (std<0.003 across cand-types).
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.post_norm = nn.LayerNorm(hidden_dim)

        out_dim = 2 if predict_uncertainty else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _compute_horizons(self, traj: torch.Tensor):
        """Pick H horizon samples and return ``(traj_h, tau, idx)``.

        When ``adaptive_horizons=False`` indices are uniform in time
        (bit-identical to the legacy ``linspace`` rule). When True the
        indices are uniform in cumulative path distance along ``traj[..., :2]``
        (Option A1 / constant lookahead distance). ``tau`` is the per-horizon
        ``(time_sec, dist_m)`` pair used for continuous conditioning (Option B).
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
        return traj_h, tau, idx

    def forward(
        self,
        context_tokens: torch.Tensor,
        trajectory: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ):
        """
        Returns logits of shape ``(B, n_horizons, n_metrics)``. When
        ``predict_uncertainty=True`` returns ``(mu, log_var)`` of the same
        shape. Apply ``torch.sigmoid`` to ``mu`` to get probabilities for BCE.

        ``context_tokens`` may be ``(B, N, D)`` (legacy: same scene latent for
        every horizon) or ``(B, H, N, D)`` (per-horizon imagined latents from
        the latent world model).
        """
        B = trajectory.shape[0]
        H, K = self.n_horizons, self.n_metrics
        traj_h, tau, _ = self._compute_horizons(trajectory)
        traj_emb = self.traj_mlp(traj_h)

        if self.adaptive_horizons:
            step_emb = self.step_proj(tau)
        else:
            steps = torch.arange(H, device=trajectory.device)
            step_emb = self.step_emb(steps)[None].expand(B, -1, -1)

        cdyn = traj_emb + step_emb                                    # (B, H, D)
        q_r = self.q_base[None, None, :, :] + cdyn[:, :, None, :]      # (B, H, K, D)

        if context_tokens.dim() == 4:
            assert context_tokens.shape[1] == H, \
                f"per-horizon context expects H={H}, got {context_tokens.shape[1]}"
            N = context_tokens.shape[2]
            ctx_flat = context_tokens.reshape(B * H, N, self.hidden_dim)
            if context_mask is not None:
                if context_mask.dim() == 2:
                    mask_flat = context_mask[:, None].expand(B, H, N).reshape(B * H, N)
                else:
                    mask_flat = context_mask.reshape(B * H, N)
            else:
                mask_flat = None
            ctx_q = self.context_query[None].expand(B * H, -1, -1)
            ctx_compressed, _ = self.ctx_compress(ctx_q, ctx_flat, ctx_flat,
                                                  key_padding_mask=mask_flat)
            ctx_compressed = self.history_encoder(ctx_compressed)      # (B*H, L, D)
            q_per_h = q_r.reshape(B * H, K, self.hidden_dim)
            attn_out, _ = self.cross_attn(self.q_norm(q_per_h), ctx_compressed, ctx_compressed)
            decoded = self.post_norm(q_per_h + attn_out).reshape(B, H, K, self.hidden_dim)
        else:
            ctx_q = self.context_query[None].expand(B, -1, -1)
            ctx_compressed, _ = self.ctx_compress(ctx_q, context_tokens, context_tokens,
                                                  key_padding_mask=context_mask)
            ctx_compressed = self.history_encoder(ctx_compressed)
            q_flat = q_r.reshape(B, H * K, self.hidden_dim)
            attn_out, _ = self.cross_attn(self.q_norm(q_flat), ctx_compressed, ctx_compressed)
            decoded = self.post_norm(q_flat + attn_out).reshape(B, H, K, self.hidden_dim)

        out = self.head(decoded)                                       # (B, H, K, 1 or 2)
        if self.predict_uncertainty:
            mu = out[..., 0]
            log_var = out[..., 1]
            return mu, log_var
        return out.squeeze(-1)
