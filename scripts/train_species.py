"""Treniranje modela za VRSTU biljke (nivo 1 hijerarhije, 14 klasa)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import build_dataset, build_loader  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.trainer import fit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    out_dir = f"experiments/{args.model}_hierarchical/species"

    train_ds = build_dataset(args.data_root, "data/splits", "train", target="species")
    val_ds = build_dataset(args.data_root, "data/splits", "val", target="species")
    if args.limit:
        train_ds.df = train_ds.df.sample(n=args.limit, random_state=42).reset_index(drop=True)
        val_ds.df = val_ds.df.sample(n=max(args.limit // 4, 40), random_state=42).reset_index(drop=True)

    train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, num_classes=14, pretrained=not args.no_pretrained)
    print(f"[VRSTA] Model: {args.model} | train: {len(train_ds)} | val: {len(val_ds)}")
    fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
        device=args.device, out_dir=out_dir)


if __name__ == "__main__":
    main()