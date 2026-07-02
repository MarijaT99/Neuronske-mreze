"""Treniranje FLAT klasifikatora (svih 38 klasa odjednom).

Probni (lokalno, CPU):
    python scripts/train_flat.py --data-root "...\\color" --model baseline_cnn --epochs 1 --limit 200
Pun trening (Kaggle GPU):
    python scripts/train_flat.py --data-root "...\\color" --model resnet50 --epochs 20 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.loaders import build_dataset, build_loader  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.trainer import fit  # noqa: E402


def compute_class_weights(dataset, num_classes):
    """Teže klase (manje primera) dobijaju veći ponder — pomaže kod disbalansa."""
    counts = torch.zeros(num_classes)
    for label in dataset.df["class_id"]:
        counts[int(label)] += 1
    return counts.sum() / (num_classes * counts.clamp(min=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default="baseline_cnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="uzmi samo N primera (za brzu probu)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--use-class-weights", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or f"experiments/{args.model}_flat"

    train_ds = build_dataset(args.data_root, "data/splits", "train", target="flat")
    val_ds = build_dataset(args.data_root, "data/splits", "val", target="flat")

    if args.limit:  # probni rezim: nasumican mali uzorak (raznovrsne klase)
        train_ds.df = train_ds.df.sample(n=args.limit, random_state=42).reset_index(drop=True)
        val_ds.df = val_ds.df.sample(n=max(args.limit // 4, 40), random_state=42).reset_index(drop=True)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, num_classes=38, pretrained=not args.no_pretrained)
    weight = compute_class_weights(train_ds, 38) if args.use_class_weights else None

    print(f"Model: {args.model} | train: {len(train_ds)} | val: {len(val_ds)} | uredjaj: {args.device}")
    fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
        device=args.device, out_dir=out_dir, weight=weight)


if __name__ == "__main__":
    main()