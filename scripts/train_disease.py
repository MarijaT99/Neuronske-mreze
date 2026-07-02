"""Treniranje modela za BOLEST unutar svake vrste (nivo 2 hijerarhije).

Za svaku vrstu sa više od jedne bolesti trenira se poseban 'disease' model.
Vrste sa samo jednom klasom (Orange, Soybean, Raspberry, Squash, Blueberry) se preskaču.
"""
from __future__ import annotations

import argparse
import json
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
    args = ap.parse_args()

    maps = json.loads(Path("data/splits/label_maps.json").read_text(encoding="utf-8"))
    species_to_id = maps["species_to_id"]
    disease_maps = maps["disease_in_species_to_id"]

    for species, sid in species_to_id.items():
        num_diseases = len(disease_maps[species])
        if num_diseases < 2:
            print(f"Preskacem '{species}' (samo {num_diseases} klasa).")
            continue

        train_ds = build_dataset(args.data_root, "data/splits", "train", target="disease")
        val_ds = build_dataset(args.data_root, "data/splits", "val", target="disease")
        train_ds.df = train_ds.df[train_ds.df["species_id"] == sid].reset_index(drop=True)
        val_ds.df = val_ds.df[val_ds.df["species_id"] == sid].reset_index(drop=True)
        if args.limit:
            train_ds.df = train_ds.df.head(args.limit).reset_index(drop=True)
            val_ds.df = val_ds.df.head(max(args.limit // 4, 20)).reset_index(drop=True)

        train_loader = build_loader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = build_loader(val_ds, batch_size=args.batch_size, shuffle=False)

        model = build_model(args.model, num_classes=num_diseases, pretrained=not args.no_pretrained)
        out_dir = f"experiments/{args.model}_hierarchical/disease/{species}"
        print(f"\n[BOLEST] {species}: {num_diseases} klasa | train: {len(train_ds)} | val: {len(val_ds)}")
        fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
            device=args.device, out_dir=out_dir)


if __name__ == "__main__":
    main()