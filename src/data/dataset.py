"""PyTorch Dataset za PlantVillage — učitava slike po sačuvanoj podeli.

Svaki red podele (CSV) ima relativnu putanju do slike i oznake na oba nivoa
hijerarhije. Dataset učitava sliku sa diska, primeni transformaciju i vrati
(slika, oznaka). Koja se oznaka vraća zavisi od 'target':
    - 'flat'    -> class_id (jedna od 38 klasa)
    - 'species' -> species_id (jedna od 14 vrsta)          [nivo 1]
    - 'disease' -> disease_id_in_species (bolest u vrsti)  [nivo 2]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

TARGET_COLUMN = {
    "flat": "class_id",
    "species": "species_id",
    "disease": "disease_id_in_species",
}


class PlantVillageDataset(Dataset):
    def __init__(self, data_root, df: pd.DataFrame, transform=None, target: str = "flat"):
        if target not in TARGET_COLUMN:
            raise ValueError(f"target mora biti iz {list(TARGET_COLUMN)}, a dobijeno: {target}")
        self.data_root = Path(data_root)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = TARGET_COLUMN[target]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.data_root / row["filepath"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(row[self.label_col])
        return img, label