"""Transformacije slika za treniranje i evaluaciju.

Train: nasumične augmentacije (flip, rotacija, boja) -> mreža vidi raznovrsnije
primere i bolje generalizuje. Eval/test: deterministički (samo resize + normalizacija)
da rezultat bude ponovljiv. Normalizacija koristi ImageNet statistike jer koristimo
modele pretrenirane na ImageNet-u.
"""
from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transforms(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )