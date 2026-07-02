"""Provera da modeli daju izlaz oblika (batch, broj_klasa)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.models.factory import build_model  # noqa: E402


def main():
    x = torch.randn(4, 3, 224, 224)  # 4 lazne slike
    for name in ["baseline_cnn", "resnet50"]:
        model = build_model(name, num_classes=38, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:14s} izlaz={tuple(out.shape)}  parametara={params:,}")
    print("OK - modeli rade.")


if __name__ == "__main__":
    main()