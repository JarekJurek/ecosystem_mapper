from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


def build_backbone(name: str) -> Tuple[nn.Module, int]:
    name = name.lower()

    if name == "efficientnet_b0":
        bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = bb.classifier[1].in_features
        bb.classifier = nn.Identity()
        return bb, feat_dim

    if name == "resnet18":
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = bb.fc.in_features
        bb.fc = nn.Identity()
        return bb, feat_dim

    if name == "resnet50":
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = bb.fc.in_features
        bb.fc = nn.Identity()
        return bb, feat_dim

    if name == "densenet121":
        bb = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        feat_dim = bb.classifier.in_features
        bb.classifier = nn.Identity()
        return bb, feat_dim

    if name == "convnext_tiny":
        bb = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        feat_dim = bb.classifier[-1].in_features
        bb.classifier[-1] = nn.Identity()
        return bb, feat_dim

    if name == "vit_b_16":
        bb = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        feat_dim = bb.heads.head.in_features
        bb.heads = nn.Identity()
        return bb, feat_dim

    raise ValueError(f"Unknown backbone: {name}")


class FusionNetBackbone(nn.Module):
    """
    FusionNet style, but currently image-only (variables ignored/omitted).
    Keeps: backbone -> (B, D) -> dropout -> classifier.
    """

    def __init__(self, backbone_name: str, num_classes: int = 17, dropout: float = 0.3):
        super().__init__()
        self.image_enabled = True

        self.backbone, img_feat_dim = build_backbone(backbone_name)

        self.fusion_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(img_feat_dim, num_classes)

    def forward(self, images: torch.Tensor, variables=None) -> torch.Tensor:
        if images is None:
            raise ValueError("images is None; cannot forward.")
        feats = self.backbone(images)  # (B, D)
        feats = self.fusion_dropout(feats)
        return self.classifier(feats)
