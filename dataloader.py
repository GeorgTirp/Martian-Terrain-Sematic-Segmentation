# data.py
"""
AI4Mars dataloading utilities.

We use the Hugging Face dataset `hassanjbara/AI4MARS`, which provides:

- image: original rover image (PIL-like).
- label_mask: semantic segmentation mask with terrain classes encoded as:
    0 -> soil
    1 -> bedrock
    2 -> sand
    3 -> big rock
    255 -> null / no label
:contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import load_dataset  # pip install datasets


AI4MARS_CLASS_NAMES: List[str] = ["soil", "bedrock", "sand", "big_rock"]
AI4MARS_IGNORE_INDEX: int = 255  # null/no-label pixels


class AI4MarsHFDataset(Dataset):
    """
    PyTorch wrapper around a Hugging Face split of AI4Mars.

    Args:
        hf_split: a `datasets.Dataset` object (already split into train/val/test).
        image_size: resize (H,W) for both image and mask.
        to_rgb: if True, replicate grayscale channel to 3 channels.
    """

    def __init__(
        self,
        hf_split,
        image_size: int = 256,
        to_rgb: bool = False,
    ):
        super().__init__()
        self.ds = hf_split
        self.image_size = image_size
        self.to_rgb = to_rgb

        if to_rgb:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=InterpolationMode.BILINEAR
                    ),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=InterpolationMode.BILINEAR
                    ),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )

        self.mask_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST,
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        img = sample["image"]  # PIL.Image
        mask = sample["label_mask"]  # PIL.Image with encoded class indices as intensity

        # Some variants may store masks as RGB with repeated channels;
        # we convert to 'L' (single channel) to be safe.
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask))

        img_t = self.image_transform(img)  # [C,H,W], float 0–1

        mask_resized = self.mask_resize(mask)
        mask_np = np.array(mask_resized, dtype=np.uint8)

        # If masks are RGB, take one channel
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        mask_t = torch.from_numpy(mask_np.astype(np.int64))  # [H,W], long

        return img_t, mask_t


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def create_ai4mars_dataloaders(
    batch_size: int = 4,
    image_size: int = 256,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    to_rgb: bool = False,
    seed: int = 42,
) -> DataLoaders:
    """
    Create train/val/test dataloaders for AI4Mars via Hugging Face.

    This function:
      1. Downloads `hassanjbara/AI4MARS` (or loads from cache).
      2. Ensures we have train/val/test splits by splitting if necessary.
      3. Wraps each split with `AI4MarsHFDataset`.
    """

    raw = load_dataset("hassanjbara/AI4MARS")  # :contentReference[oaicite:5]{index=5}

    # Try to infer train and test splits; if not present, create them.
    if "train" in raw:
        full_train = raw["train"]
        if "test" in raw:
            test_hf = raw["test"]
        else:
            split = full_train.train_test_split(test_size=0.1, seed=seed)
            full_train, test_hf = split["train"], split["test"]
    else:
        # Single split dataset; create train/test from it.
        key = list(raw.keys())[0]
        full_train = raw[key]
        split = full_train.train_test_split(test_size=0.2, seed=seed)
        full_train, test_hf = split["train"], split["test"]

    # Now create train/val from full_train
    split2 = full_train.train_test_split(test_size=val_fraction, seed=seed + 1)
    train_hf, val_hf = split2["train"], split2["test"]

    train_ds = AI4MarsHFDataset(train_hf, image_size=image_size, to_rgb=to_rgb)
    val_ds = AI4MarsHFDataset(val_hf, image_size=image_size, to_rgb=to_rgb)
    test_ds = AI4MarsHFDataset(test_hf, image_size=image_size, to_rgb=to_rgb)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)
