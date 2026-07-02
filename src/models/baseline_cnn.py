"""Jednostavna konvoluciona mreža napravljena od nule (baseline).

Referenca koja pokazuje šta se postiže BEZ transfer learning-a. Nekoliko blokova
(conv -> BatchNorm -> ReLU -> maxpool) smanjuju sliku i povećavaju broj kanala,
pa potpuno povezani sloj na kraju radi klasifikaciju.
"""
from __future__ import annotations

import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),     # 224 -> 112
            self._block(32, 64),    # 112 -> 56
            self._block(64, 128),   # 56 -> 28
            self._block(128, 256),  # 28 -> 14
        )
        reduced = img_size // 16    # posle 4 maxpool-a
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * reduced * reduced, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)