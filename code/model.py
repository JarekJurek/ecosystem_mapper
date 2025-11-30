from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class FusionNet(nn.Module):
    """
    Fusion model combining:
    - Image branch: ResNet backbone -> feature vector
    - Variables branch: simple MLP -> feature vector
    Both branches are optional; when only one is present, classification uses that branch.
    """

    def __init__(
        self,
        num_classes: int = 17,
        var_input_dim: Optional[int] = None,
        var_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.image_enabled = True
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Cutting the backbone head – nn.Identity() returns input as output
        img_feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Variables MLP
        self.vars_enabled = var_input_dim is not None and var_input_dim > 0
        if self.vars_enabled:
            self.vars_mlp = nn.Sequential(
                nn.Linear(var_input_dim, var_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(var_hidden_dim, img_feat_dim),
                nn.Sigmoid(),
            )
        else:
            pass

        fused_dim = img_feat_dim
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self, images: Optional[torch.Tensor], variables: Optional[torch.Tensor]
    ) -> torch.Tensor:
        feats = []
        img_feats = None
        var_feats = None
        if self.image_enabled and images is not None:
            img_feats = self.backbone(images)
            feats.append(img_feats)
        if self.vars_enabled and variables is not None:
            var_feats = self.vars_mlp(variables)
            feats.append(var_feats)
        if len(feats) == 0:
            raise ValueError("Both image and variable inputs are None; cannot forward.")
        if img_feats is not None and var_feats is not None:
            fused = img_feats * var_feats
        else:
            fused = feats[0]
        logits = self.classifier(fused)
        return logits
