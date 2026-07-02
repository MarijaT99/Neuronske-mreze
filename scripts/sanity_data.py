"""Brza provera da učitavanje radi: uzme par slika iz val skupa, napravi jedan
batch i ispiše oblik tenzora i oznake.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import build_dataset, build_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--target", default="flat")
    args = ap.parse_args()

    ds = build_dataset(args.data_root, "data/splits", "val", target=args.target)
    print(f"Broj primera u val: {len(ds)}")

    loader = build_loader(ds, batch_size=8, shuffle=True)
    images, labels = next(iter(loader))
    print(f"Oblik batch-a slika: {tuple(images.shape)}")   # ocekivano (8, 3, 224, 224)
    print(f"Oznake u batch-u: {labels.tolist()}")
    print(f"Min/max piksela: {images.min():.3f} / {images.max():.3f}")
    print("OK - ucitavanje radi.")


if __name__ == "__main__":
    main()