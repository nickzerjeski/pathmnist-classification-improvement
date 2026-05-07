from __future__ import annotations

from dataclasses import dataclass

import medmnist
import torch
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms


@dataclass(frozen=True)
class DatasetMeta:
    n_classes: int
    class_names: list[str]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


PATHMNIST_MEAN = (0.7405, 0.5330, 0.7058)
PATHMNIST_STD = (0.1237, 0.1768, 0.1244)


def dataset_meta() -> DatasetMeta:
    info = INFO["pathmnist"]
    labels = info["label"]
    class_names = [labels[str(i)] for i in range(len(labels))]
    return DatasetMeta(len(class_names), class_names, PATHMNIST_MEAN, PATHMNIST_STD)


def build_transform(split: str, image_size: int, augment: str, norm: str = "pathmnist") -> transforms.Compose:
    ops: list[transforms.Compose] = []
    if image_size != 28:
        ops.append(transforms.Resize((image_size, image_size), antialias=True))
    if split == "train" and augment != "none":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation(90)], p=0.7),
            ]
        )
        if augment in {"strong", "histology"}:
            ops.extend(
                [
                    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18, hue=0.04),
                    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.9, 1.1)),
                ]
            )
    if norm == "official":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean, std = PATHMNIST_MEAN, PATHMNIST_STD
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if split == "train" and augment in {"strong", "histology"}:
        ops.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)))
    return transforms.Compose(ops)


def build_dataset(
    split: str, image_size: int = 28, augment: str = "none", root: str = "data", norm: str = "pathmnist"
):
    dataset_cls = getattr(medmnist, INFO["pathmnist"]["python_class"])
    return dataset_cls(
        split=split,
        transform=build_transform(split, image_size, augment, norm),
        download=True,
        as_rgb=True,
        root=root,
        size=28 if image_size == 28 else image_size,
    )


def build_loader(
    split: str,
    batch_size: int,
    image_size: int,
    augment: str,
    workers: int,
    root: str = "data",
    norm: str = "pathmnist",
) -> DataLoader:
    dataset = build_dataset(split, image_size=image_size, augment=augment, root=root, norm=norm)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
