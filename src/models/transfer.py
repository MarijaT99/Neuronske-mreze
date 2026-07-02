"""Transfer learning modeli, pretrenirani na ImageNet-u.

Uzimamo mrežu koja je već naučila opšte vizuelne osobine na milionima slika i
zamenimo joj poslednji sloj svojim (broj naših klasa). Tako treniramo brže i
tačnije nego od nule - što poređenje sa baseline CNN i pokazuje.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # zameni poslednji sloj
    return model


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model