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

# data.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import load_dataset

AI4MARS_CLASS_NAMES: List[str] = ["soil", "bedrock", "sand", "big_rock"]
AI4MARS_IGNORE_INDEX: int = 255


class AI4MarsHFDataset(Dataset):
    """
    PyTorch wrapper around a Hugging Face split of AI4Mars.

    Handles occasional corrupt / missing entries robustly by skipping them.
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
                        (image_size, image_size),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size),
                        interpolation=InterpolationMode.BILINEAR,
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

    def _get_raw_sample(self, idx: int):
        """Internal: get a sample, skipping over missing images/masks."""
        sample = self.ds[idx]
        img = sample.get("image", None)
        mask = sample.get("label_mask", None)

        # If something is None, skip to the next index (wrap-around).
        if img is None or mask is None:
            new_idx = (idx + 1) % len(self.ds)
            return self._get_raw_sample(new_idx)

        return img, mask

    def __getitem__(self, idx: int):
        img, mask = self._get_raw_sample(idx)

        # HF Image feature should give us PIL.Image already,
        # but if it's numpy, convert safely.
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))

        img_t = self.image_transform(img)  # [C,H,W]

        mask_resized = self.mask_resize(mask)
        mask_np = np.array(mask_resized, dtype=np.uint8)

        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        mask_t = torch.from_numpy(mask_np.astype(np.int64))  # [H,W], long

        return img_t, mask_t



from datasets import load_dataset, load_from_disk  # add load_from_disk if you want on-disk caching

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
    cache_dir: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    use_local_disk_copy: bool = False,
    local_disk_path: str = "./data/ai4mars_hf",
) -> DataLoaders:
    """
    Create train/val/test dataloaders for AI4Mars via Hugging Face.

    New knobs:
    - cache_dir: where Hugging Face stores its cache.
    - max_*_samples: limit number of samples per split (useful for quick dev runs).
    - use_local_disk_copy + local_disk_path:
        * On first prep run you can save the HF dataset to disk.
        * Later you can load from that folder instead of redownloading/processing.
    """

    if use_local_disk_copy and os.path.exists(local_disk_path):
        raw = load_from_disk(local_disk_path)
    else:
        raw = load_dataset(
            "hassanjbara/AI4MARS",
            cache_dir=cache_dir,
        )
        # Optional: save to disk once, then next runs can set use_local_disk_copy=True
        if use_local_disk_copy:
            os.makedirs(os.path.dirname(local_disk_path), exist_ok=True)
            raw.save_to_disk(local_disk_path)

    # --- same split logic as before ---
    if "train" in raw:
        full_train = raw["train"]
        if "test" in raw:
            test_hf = raw["test"]
        else:
            split = full_train.train_test_split(test_size=0.1, seed=seed)
            full_train, test_hf = split["train"], split["test"]
    else:
        key = list(raw.keys())[0]
        full_train = raw[key]
        split = full_train.train_test_split(test_size=0.2, seed=seed)
        full_train, test_hf = split["train"], split["test"]

    split2 = full_train.train_test_split(test_size=val_fraction, seed=seed + 1)
    train_hf, val_hf = split2["train"], split2["test"]

    # --- NEW: optionally limit split sizes for fast dev runs ---
    if max_train_samples is not None:
        train_hf = train_hf.select(range(min(max_train_samples, len(train_hf))))
    if max_val_samples is not None:
        val_hf = val_hf.select(range(min(max_val_samples, len(val_hf))))
    if max_test_samples is not None:
        test_hf = test_hf.select(range(min(max_test_samples, len(test_hf))))

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

