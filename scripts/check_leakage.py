#!/usr/bin/env python
"""
check_leakage.py -- Detekcija near-duplikata izmedju train i test skupa (PlantVillage).
Ne trenira nista; samo racuna perceptualni hes (phash) i meri curenje podataka (data leakage).

Pokretanje:
    python scripts/check_leakage.py --data-root "PUTANJA_DO_color_foldera"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True,
                   help="Putanja do 'color' foldera koji sadrzi 38 klasa-foldera")
    p.add_argument("--splits-csv", default="data/splits/all.csv",
                   help="CSV sa kolonama filepath, class_name, split")
    p.add_argument("--hash-size", type=int, default=8,
                   help="Velicina phash-a (8 -> 64 bita)")
    p.add_argument("--thresholds", type=int, nargs="+", default=[0, 3, 5],
                   help="Hamming pragovi: 0=identicne, vise=sve slicnije")
    p.add_argument("--examples", type=int, default=8,
                   help="Koliko primera procurelih parova da ispise")
    return p.parse_args()


def compute_hashes(df, data_root, hash_size):
    bits = {}
    failed = 0
    for fp in tqdm(df["filepath"].tolist(), desc="Hesiranje slika"):
        try:
            with Image.open(data_root / fp) as im:
                h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
            bits[fp] = h.hash.flatten().astype(np.uint8)
        except Exception:
            failed += 1
            bits[fp] = None
    return bits, failed


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    df = pd.read_csv(args.splits_csv)

    for col in ("filepath", "class_name", "split"):
        if col not in df.columns:
            raise SystemExit(f"CSV nema kolonu '{col}'. Nadjene kolone: {list(df.columns)}")

    print(f"Ucitano {len(df)} redova iz {args.splits_csv}")
    print(f"Raspodela split-a:\n{df['split'].value_counts().to_string()}\n")

    bits, failed = compute_hashes(df, data_root, args.hash_size)
    if failed:
        print(f"UPOZORENJE: {failed} slika nije moglo da se procita (preskocene).")

    results = {t: 0 for t in args.thresholds}
    test_total = 0
    examples = []
    max_t = max(args.thresholds)

    for cls, g in df.groupby("class_name"):
        tr_paths = [fp for fp in g[g["split"] == "train"]["filepath"] if bits.get(fp) is not None]
        te_paths = [fp for fp in g[g["split"] == "test"]["filepath"] if bits.get(fp) is not None]
        if not tr_paths or not te_paths:
            continue
        tr_mat = np.stack([bits[fp] for fp in tr_paths])  # (n_train, nbits)

        for fp in te_paths:
            test_total += 1
            ham = np.count_nonzero(tr_mat != bits[fp], axis=1)
            j = int(ham.argmin())
            mn = int(ham[j])
            for t in args.thresholds:
                if mn <= t:
                    results[t] += 1
            if mn <= max_t and len(examples) < args.examples:
                examples.append((mn, fp, tr_paths[j]))

    print("\n" + "=" * 70)
    print("REZULTAT: curenje test skupa u trening skup (near-duplikati)")
    print("=" * 70)
    print(f"Ukupno test slika analizirano: {test_total}\n")
    for t in sorted(args.thresholds):
        n = results[t]
        pct = 100.0 * n / test_total if test_total else 0.0
        opis = "identicne (phash)" if t == 0 else f"Hamming <= {t}"
        print(f"  Prag {t:>2} ({opis:<18}): {n:>5} test slika  ({pct:5.2f}% test skupa)")

    if examples:
        print("\nPrimeri procurelih parova (hamming | TEST slika | najblizi TRAIN):")
        for mn, te, tr in sorted(examples):
            print(f"  d={mn:<2}  {te}")
            print(f"        <-> {tr}")

    print("\nInterpretacija: sto je veci procenat kod niskih pragova, to je "
          "vise test slika model vec 'video' na treningu -> naduvane metrike.")


if __name__ == "__main__":
    main()
