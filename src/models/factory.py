"""Fabrika modela - napravi model po imenu (iz konfiguracije)."""
from __future__ import annotations

import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN
from src.models.transfer import build_efficientnet_b0, build_resnet50


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "baseline_cnn":
        return BaselineCNN(num_classes)
    if name == "resnet50":
        return build_resnet50(num_classes, pretrained=pretrained)
    if name in ("efficientnet", "efficientnet_b0"):
        return build_efficientnet_b0(num_classes, pretrained=pretrained)
    raise ValueError(f"Nepoznat model: {name}")