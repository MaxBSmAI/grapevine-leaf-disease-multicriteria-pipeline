"""Manifest-backed PyTorch datasets, transforms, samplers, and loaders."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grape_disease.utils.dependencies import require_module

torch = require_module("torch")
transforms = require_module("torchvision.transforms", "torchvision")
pil = require_module("PIL.Image", "Pillow")


class ManifestImageDataset(torch.utils.data.Dataset):  # type: ignore[misc, name-defined]
    """Load portable image rows from a canonical split without copying images."""

    def __init__(
        self,
        dataset_root: Path,
        split_frame: pd.DataFrame,
        partition: str,
        transform: Callable[[Any], Any],
        allow_test: bool = False,
    ) -> None:
        if partition == "test" and not allow_test:
            raise PermissionError("Test dataset construction requires the test guard")
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown partition: {partition}")
        required = {
            "image_id",
            "original_id",
            "group_id",
            "class_id",
            "class_name",
            "relative_path",
            "split",
        }
        missing = sorted(required - set(split_frame.columns))
        if missing:
            raise ValueError(f"Canonical split missing columns: {missing}")
        rows = split_frame.loc[split_frame["split"] == partition].copy()
        if rows.empty:
            raise ValueError(f"Canonical split contains no {partition} rows")
        self.root = dataset_root.resolve()
        self.rows = rows.reset_index(drop=True)
        self.targets = self.rows["class_id"].astype(int).tolist()
        self.transform = transform
        self.partition = partition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        path = (self.root / Path(str(row["relative_path"]))).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Path escapes dataset root: {row['relative_path']}")
        with pil.open(path) as handle:
            image = handle.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": int(row["class_id"]),
            "image_id": str(row["image_id"]),
            "original_id": str(row["original_id"]),
            "group_id": str(row["group_id"]),
            "class_name": str(row["class_name"]),
            "relative_path": str(row["relative_path"]),
        }


def build_transforms(
    mean: list[float] | tuple[float, float, float],
    std: list[float] | tuple[float, float, float],
    input_size: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    """Build the moderate train and deterministic evaluation transforms."""

    interpolation = transforms.InterpolationMode.BILINEAR
    normalise = transforms.Normalize(mean=list(mean), std=list(std))
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=input_size,
                scale=(0.85, 1.0),
                ratio=(0.90, 1.10),
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, interpolation=interpolation),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            normalise,
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(size=input_size, interpolation=interpolation),
            transforms.ToTensor(),
            normalise,
        ]
    )
    return {"train": train, "validation": evaluation, "test": evaluation}


def seed_worker(worker_id: int) -> None:
    """Derive Python and NumPy worker seeds from PyTorch's worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_sampler(
    targets: list[int], regime: str, generator: Any
) -> tuple[Any | None, bool]:
    """Return the frozen regime sampler and whether DataLoader may shuffle."""

    if regime in {"standard_ce", "focal_loss"}:
        return None, True
    if regime != "balanced_ce":
        raise ValueError(f"Unknown training regime: {regime}")
    counts = Counter(targets)
    if not counts:
        raise ValueError("Balanced sampler requires at least one target")
    weights = torch.as_tensor(
        [1.0 / counts[target] for target in targets], dtype=torch.double
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(targets),
        replacement=True,
        generator=generator,
    )
    return sampler, False


def build_dataloader(
    dataset: ManifestImageDataset,
    batch_size: int,
    generator: Any,
    regime: str | None = None,
    num_workers: int = 0,
) -> Any:
    """Build a deterministic loader; balanced sampling is train-only."""

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    sampler = None
    shuffle = False
    if dataset.partition == "train":
        sampler, shuffle = build_sampler(
            dataset.targets, regime or "standard_ce", generator
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(torch.cuda.is_available()),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def load_canonical_split(path: Path) -> pd.DataFrame:
    """Load the split with strings preserved for portable identifiers."""

    return pd.read_csv(path, dtype=str, keep_default_na=False)
