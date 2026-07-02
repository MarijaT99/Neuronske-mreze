"""Pravljenje i učitavanje stratifikovane podele PlantVillage skupa.

Skenira folder sa slikama (svaki podfolder je klasa oblika 'Vrsta___Bolest'),
gradi tabelu sa oznakama i deli je na train/val/test (70/15/15) stratifikovano
po klasi, sa fiksnim seed-om (reproducibilnost). Čita samo IMENA fajlova, ne
sadržaj slika, pa radi brzo i bez učitavanja gigabajta.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class LabelMaps:
    """Mapiranja imena -> celobrojni ID (koriste ih modeli i evaluacija)."""

    species_to_id: dict
    class_to_id: dict
    disease_in_species_to_id: dict  # {vrsta: {bolest: lokalni_id}}

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def load(path) -> "LabelMaps":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return LabelMaps(**d)


def scan_dataset(data_root) -> pd.DataFrame:
    """Prođe kroz foldere klasa i napravi tabelu: jedan red = jedna slika."""
    data_root = Path(data_root)
    rows = []
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        class_name = class_dir.name                 # npr. 'Apple___Apple_scab'
        species, disease = class_name.split("___", 1)
        for img in class_dir.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                rows.append(
                    {
                        "filepath": f"{class_name}/{img.name}",  # relativna, prenosiva putanja
                        "class_name": class_name,
                        "species": species,
                        "disease": disease,
                    }
                )
    if not rows:
        raise SystemExit(f"Nema slika u {data_root} — proveri putanju.")
    return pd.DataFrame(rows)


def build_label_maps(df: pd.DataFrame) -> LabelMaps:
    species = sorted(df["species"].unique())
    species_to_id = {s: i for i, s in enumerate(species)}

    classes = sorted(df["class_name"].unique())
    class_to_id = {c: i for i, c in enumerate(classes)}

    # Za svaku vrstu: lokalni ID bolesti (0..broj_bolesti_te_vrste - 1)
    disease_in_species_to_id = {}
    for s in species:
        diseases = sorted(df.loc[df["species"] == s, "disease"].unique())
        disease_in_species_to_id[s] = {d: i for i, d in enumerate(diseases)}
    return LabelMaps(species_to_id, class_to_id, disease_in_species_to_id)


def add_ids(df: pd.DataFrame, maps: LabelMaps) -> pd.DataFrame:
    df = df.copy()
    df["species_id"] = df["species"].map(maps.species_to_id)
    df["class_id"] = df["class_name"].map(maps.class_to_id)
    df["disease_id_in_species"] = [
        maps.disease_in_species_to_id[s][d] for s, d in zip(df["species"], df["disease"])
    ]
    return df


def make_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stratifikovana podela 70/15/15 po klasi (isti odnos klasa u sva tri skupa)."""
    df = df.reset_index(drop=True)
    idx = df.index.to_numpy()
    y = df["class_id"].to_numpy()

    # 1) izdvoji test = 15%
    train_val_idx, test_idx = train_test_split(
        idx, test_size=0.15, stratify=y, random_state=seed
    )
    # 2) iz preostalog izdvoji val tako da bude 15% CELOG skupa
    val_frac = 0.15 / 0.85
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_frac, stratify=y[train_val_idx], random_state=seed
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def save_splits(df: pd.DataFrame, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all.csv", index=False, encoding="utf-8")
    for name in ("train", "val", "test"):
        df[df["split"] == name].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")


def load_split(out_dir, which: str) -> pd.DataFrame:
    """Učitaj podelu 'train' / 'val' / 'test' / 'all' iz CSV-a."""
    return pd.read_csv(Path(out_dir) / f"{which}.csv", encoding="utf-8")