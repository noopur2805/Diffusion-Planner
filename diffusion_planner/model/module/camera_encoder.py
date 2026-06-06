"""
Multi-view (NavSim 8-camera) image encoder.

Produces a per-view token (or per-patch tokens) embedded into the planner's
``hidden_dim`` so it can be concatenated with the existing vector tokens
inside ``diffusion_planner.model.module.encoder.Encoder``.

The image backbone is created via ``timm.create_model`` so any ImageNet-pretrained
CNN (default: ``resnet18``) can be used without adding new dependencies.
"""
from typing import Optional

import torch
import torch.nn as nn
from timm import create_model
from timm.models.layers import Mlp


_NAVSIM_VIEWS = (
    "cam_f0", "cam_l0", "cam_l1", "cam_l2",
    "cam_r0", "cam_r1", "cam_r2", "cam_b0",
)


class CameraEncoder(nn.Module):
    """
    Encode NavSim's 8 surround-view RGB frames into a sequence of tokens.

    Args:
        hidden_dim: planner hidden dimension (must match Encoder.hidden_dim).
        num_views: number of camera views (8 for NavSim).
        backbone: timm model name (default 'resnet18'). Pooled global feature only.
        pretrained: whether to load ImageNet weights.
        use_view_embedding: add a learnable view-id embedding.
        cross_view_fusion: if True, run a small TransformerEncoder across the
            8 view tokens so each token sees the others before being passed
            to the planner's fusion stage.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_views: int = 8,
        backbone: str = "resnet18",
        pretrained: bool = True,
        use_view_embedding: bool = True,
        cross_view_fusion: bool = True,
        fusion_depth: int = 2,
        fusion_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.hidden_dim = hidden_dim

        self.backbone = create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.proj = Mlp(in_features=feat_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=dropout)

        self.view_embedding = nn.Embedding(num_views, hidden_dim) if use_view_embedding else None

        if cross_view_fusion:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=fusion_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                batch_first=True, activation="gelu",
            )
            self.cross_view = nn.TransformerEncoder(enc_layer, num_layers=fusion_depth)
        else:
            self.cross_view = None

    @staticmethod
    def view_names():
        return list(_NAVSIM_VIEWS)

    def forward(self, images: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            images: (B, V, 3, H, W) float tensor, already mean/std normalized.
            mask:   (B, V) optional bool padding mask (True = padded/missing).

        Returns:
            tokens: (B, V, hidden_dim)
            token_mask: (B, V) bool, True where token is invalid (matches encoder convention).
        """
        B, V, C, H, W = images.shape
        assert V == self.num_views, f"expected {self.num_views} views, got {V}"

        feat = self.backbone(images.reshape(B * V, C, H, W))  # (B*V, feat_dim)
        feat = self.proj(feat).reshape(B, V, self.hidden_dim)

        if self.view_embedding is not None:
            view_ids = torch.arange(V, device=feat.device)
            feat = feat + self.view_embedding(view_ids)[None]

        if self.cross_view is not None:
            key_padding_mask = mask if mask is not None else None
            feat = self.cross_view(feat, src_key_padding_mask=key_padding_mask)

        if mask is None:
            mask = torch.zeros((B, V), dtype=torch.bool, device=feat.device)

        return feat, mask
