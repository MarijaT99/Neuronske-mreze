"""Napravi stratifikovanu podelu train/val/test i sačuvaj je u data/splits/.

Primer pokretanja:
    python scripts/prepare_splits.py --data-root "PUTANJA/do/color" --out-dir data/splits --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Dodaj koren projekta na put, da 'from src...' radi kad se skripta pokrene direktno.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import (  # noqa: E402
    add_ids,
    build_label_maps,
    make_splits,
    save_splits,
    scan_dataset,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder 'color' sa 38 klasa-foldera")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Skeniram dataset...")
    df = scan_dataset(args.data_root)
    print(f"Pronadjeno slika: {len(df)}")

    maps = build_label_maps(df)
    df = add_ids(df, maps)
    df = make_splits(df, seed=args.seed)

    out = Path(args.out_dir)
    save_splits(df, out)
    maps.save(out / "label_maps.json")

    print("\nPodela (broj slika po skupu):")
    print(df["split"].value_counts())
    print(f"\nBroj klasa: {df['class_id'].nunique()} | broj vrsta: {df['species_id'].nunique()}")
    print(f"Sacuvano u: {out.resolve()}")


if __name__ == "__main__":
    main()