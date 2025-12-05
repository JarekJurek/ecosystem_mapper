from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class FusionNet(nn.Module):
    """
    Fusion model combining:
    - Image branch: EfficientNet backbone -> feature vector
    - Variables branch: MLP -> feature vector
    Both branches are optional; when only one is present, classification uses that branch.
    """

    def __init__(
        self,
        num_classes: int = 17,
        var_input_dim: Optional[int] = None,
        var_hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        ### Image branch ###
        self.image_enabled = True
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Cut off classifier head
        img_feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Dropout + LayerNorm on image features
        self.img_dropout = nn.Dropout(p=dropout)
        self.img_ln = nn.LayerNorm(img_feat_dim)

        ### Variables branch ###
        self.vars_enabled = var_input_dim is not None and var_input_dim > 0
        if self.vars_enabled:
            self.vars_mlp = nn.Sequential(
                nn.Linear(var_input_dim, var_hidden_dim),
                nn.LayerNorm(var_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(var_hidden_dim, var_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(var_hidden_dim, img_feat_dim),
            )
            self.var_ln = nn.LayerNorm(img_feat_dim)
        else:
            self.vars_mlp = None
            self.var_ln = None

        ### Fusion + classifier ###
        # If both branches present: concat(img, vars) -> fusion_fc(img_feat_dim)
        fused_in_dim = img_feat_dim * 2
        self.fusion_fc = nn.Sequential(
            nn.Linear(fused_in_dim, img_feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.fusion_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(img_feat_dim, num_classes)

    def forward(
        self, images: Optional[torch.Tensor], variables: Optional[torch.Tensor]
    ) -> torch.Tensor:
        feats = []
        img_feats = None
        var_feats = None
        if self.image_enabled and images is not None:
            img_feats = self.backbone(images)  # (B, D)
            img_feats = self.img_dropout(img_feats)
            img_feats = self.img_ln(img_feats)
            feats.append(img_feats)
        if self.vars_enabled and variables is not None:
            var_feats = self.vars_mlp(variables)  # (B, D)
            var_feats = self.var_ln(var_feats)
            feats.append(var_feats)
        if len(feats) == 0:
            raise ValueError("Both image and variable inputs are None; cannot forward.")

        # If both branches -> concat + fusion MLP
        if img_feats is not None and var_feats is not None:
            fused = torch.cat([img_feats, var_feats], dim=1)  # (B, 2D)
            fused = self.fusion_fc(fused)  # (B, D)
        else:
            fused = feats[0]
        fused = self.fusion_dropout(fused)
        logits = self.classifier(fused)
        return logits
