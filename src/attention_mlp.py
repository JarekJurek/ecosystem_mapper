from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x
    

class FusionNetCBAM(nn.Module):
    """
    Fusion model with CBAM-enhanced EfficientNet image branch.

    - Image branch: EfficientNet → CBAM → GAP → feature vector
    - Variables branch: MLP → feature vector
    - Fusion: element-wise multiplication (same as original FusionNet)
    """

    def __init__(
        self,
        num_classes: int = 17,
        var_input_dim: Optional[int] = None,
        var_hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ================= Image branch =================
        self.image_enabled = True
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        img_feat_dim = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()

        self.cbam = CBAM(img_feat_dim)

        self.img_dropout = nn.Dropout(dropout)
        self.img_ln = nn.LayerNorm(img_feat_dim)

        # ================= Variables branch =================
        self.vars_enabled = var_input_dim is not None and var_input_dim > 0
        if self.vars_enabled:
            self.vars_mlp = nn.Sequential(
                nn.Linear(var_input_dim, var_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(var_hidden_dim, img_feat_dim),
                nn.Sigmoid(),  # gating, same as original
            )

        # ================= Classifier =================
        self.fusion_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(img_feat_dim, num_classes)

    def forward(
        self,
        images: Optional[torch.Tensor],
        variables: Optional[torch.Tensor],
    ) -> torch.Tensor:

        feats = []
        img_feats = None
        var_feats = None

        if self.image_enabled and images is not None:
            # EfficientNet conv features
            fmap = self.backbone.features(images)       # (B, C, H, W)
            fmap = self.cbam(fmap)                      # CBAM attention
            img_feats = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            img_feats = self.img_dropout(img_feats)
            img_feats = self.img_ln(img_feats)
            feats.append(img_feats)

        if self.vars_enabled and variables is not None:
            var_feats = self.vars_mlp(variables)
            feats.append(var_feats)

        if not feats:
            raise ValueError("Both image and variable inputs are None.")

        # Fusion strategy (same as your original model)
        if img_feats is not None and var_feats is not None:
            fused = img_feats * var_feats
        else:
            fused = feats[0]

        fused = self.fusion_dropout(fused)
        return self.classifier(fused)
