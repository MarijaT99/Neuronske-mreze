"""Pravljenje Dataset-a i DataLoader-a iz sačuvane podele (spaja sve prethodno)."""
from __future__ import annotations

from torch.utils.data import DataLoader

from src.data.dataset import PlantVillageDataset
from src.data.splits import load_split
from src.data.transforms import build_eval_transforms, build_train_transforms


def build_dataset(data_root, splits_dir, which, target="flat", img_size=224):
    df = load_split(splits_dir, which)
    # Augmentacija samo za train; val/test dobijaju determinističku transformaciju.
    transform = (
        build_train_transforms(img_size) if which == "train" else build_eval_transforms(img_size)
    )
    return PlantVillageDataset(data_root, df, transform=transform, target=target)


def build_loader(dataset, batch_size=32, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,                       # brži prenos na GPU
        persistent_workers=num_workers > 0,    # ne gasi radnike posle svake epohe
    )