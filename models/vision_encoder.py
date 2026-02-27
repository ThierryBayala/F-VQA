"""Vision encoder: pretrained ViT from timm (CLS token)."""

import timm
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Frozen ViT; returns CLS token embedding (B, 768)."""

    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,
        )
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.vit(x)  # (B, 768)
