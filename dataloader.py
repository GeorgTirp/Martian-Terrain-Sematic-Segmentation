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
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

AI4MARS_CLASS_NAMES: List[str] = ["soil", "bedrock", "sand", "big_rock"]
AI4MARS_IGNORE_INDEX: int = 255


class AI4MarsHFDataset(Dataset):
    """
    PyTorch wrapper around a Hugging Face split of AI4Mars.

    Behavior controlled by `scan_spurious`:

    - scan_spurious = True:
        * Scan the HF split once to find valid (decodable) samples.
        * Build `valid_indices` (throwing out corrupted / None images or masks).
        * Save `valid_indices` to disk (in `cache_dir`) so next runs can reuse it.

    - scan_spurious = False:
        * Assume this cleanup has already been done.
        * Load `valid_indices` from disk (no expensive scan).
    """

    def __init__(
        self,
        hf_split,
        image_size: int = 256,
        to_rgb: bool = False,
        scan_spurious: bool = False,
        cache_dir: str = "./ai4mars_valid_indices",
        split_name: str = "train",
    ):
        super().__init__()
        self.ds = hf_split
        self.image_size = image_size
        self.to_rgb = to_rgb

        # --- transforms ---
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

        # --- caching for valid indices ---
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(
            cache_dir, f"valid_indices_{split_name}.npy"
        )

        self.valid_indices: List[int] = []

        if (not scan_spurious) and os.path.exists(self.cache_path):
            # Fast path: load precomputed valid indices
            self.valid_indices = np.load(self.cache_path).astype(int).tolist()
            print(
                f"[AI4MarsHFDataset] Loaded {len(self.valid_indices)} valid indices "
                f"from cache: {self.cache_path}"
            )
        else:
            # Slow path: scan HF split and build valid_indices
            print("[AI4MarsHFDataset] Scanning for corrupted samples...")
            for i in range(len(self.ds)):
                try:
                    sample = self.ds[i]
                    img = sample.get("image", None)
                    mask = sample.get("label_mask", None)

                    if img is None or mask is None:
                        continue

                    # Force PIL to decode image / mask (may raise UnidentifiedImageError)
                    _ = img.size
                    _ = mask.size

                    self.valid_indices.append(i)
                except UnidentifiedImageError:
                    print(
                        f"[AI4MarsHFDataset] Skipping corrupted sample at index {i}"
                    )
                    continue

            print(
                f"[AI4MarsHFDataset] Kept {len(self.valid_indices)} / {len(self.ds)} samples"
            )

            # Persist valid indices so future runs can skip scanning
            np.save(self.cache_path, np.array(self.valid_indices, dtype=np.int64))
            print(
                f"[AI4MarsHFDataset] Saved valid indices to cache: {self.cache_path}"
            )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        sample = self.ds[real_idx]

        img = sample["image"]
        mask = sample["label_mask"]

        # HF Image usually returns PIL.Image already, but keep numpy fallback
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))

        img_t = self.image_transform(img)  # [C,H,W]

        mask_resized = self.mask_resize(mask)
        mask_np = np.array(mask_resized, dtype=np.uint8)

        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        # Optional safety: clamp unknown labels to ignore_index
        valid_classes = [0, 1, 2, 3, AI4MARS_IGNORE_INDEX]
        mask_np = np.where(
            np.isin(mask_np, valid_classes), mask_np, AI4MARS_IGNORE_INDEX
        )

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
    cache_dir: str | None = None,            # HF cache (raw dataset)
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    use_local_disk_copy: bool = False,
    local_disk_path: str = "./data/ai4mars_hf",
    # NEW: control spurious image handling
    scan_spurious: bool = False,
    valid_indices_cache_dir: str = "./ai4mars_valid_indices",
) -> DataLoaders:
    """
    Create train/val/test dataloaders for AI4Mars via Hugging Face.

    Args (existing):
        cache_dir: where Hugging Face stores its cache.
        max_*_samples: limit split sizes (useful for quick dev runs).
        use_local_disk_copy / local_disk_path: save/load HF dataset to/from disk.

    Args (new):
        scan_spurious:
            - If True: scan each split for corrupted / undecodable samples,
              build valid_indices and save them to `valid_indices_cache_dir`.
            - If False: assume this has already been done and just load
              valid_indices from `valid_indices_cache_dir`.
        valid_indices_cache_dir:
            - Folder where per-split valid index files are stored, e.g.
              `valid_indices_train.npy`, `valid_indices_val.npy`, `valid_indices_test.npy`.
    """

    # --- Load HF dataset (raw) ---
    if use_local_disk_copy and os.path.exists(local_disk_path):
        raw = load_from_disk(local_disk_path)
    else:
        raw = load_dataset(
            "hassanjbara/AI4MARS",
            cache_dir=cache_dir,
        )
        if use_local_disk_copy:
            os.makedirs(os.path.dirname(local_disk_path), exist_ok=True)
            raw.save_to_disk(local_disk_path)

    # --- Split into train / test / val as before ---
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

    # --- Optional subsampling for quick experiments ---
    if max_train_samples is not None:
        train_hf = train_hf.select(range(min(max_train_samples, len(train_hf))))
    if max_val_samples is not None:
        val_hf = val_hf.select(range(min(max_val_samples, len(val_hf))))
    if max_test_samples is not None:
        test_hf = test_hf.select(range(min(max_test_samples, len(test_hf))))

    # --- Wrap in AI4MarsHFDataset with spurious-handling flags ---
    train_ds = AI4MarsHFDataset(
        train_hf,
        image_size=image_size,
        to_rgb=to_rgb,
        scan_spurious=scan_spurious,
        cache_dir=valid_indices_cache_dir,
        split_name="train",
    )
    val_ds = AI4MarsHFDataset(
        val_hf,
        image_size=image_size,
        to_rgb=to_rgb,
        scan_spurious=scan_spurious,
        cache_dir=valid_indices_cache_dir,
        split_name="val",
    )
    test_ds = AI4MarsHFDataset(
        test_hf,
        image_size=image_size,
        to_rgb=to_rgb,
        scan_spurious=scan_spurious,
        cache_dir=valid_indices_cache_dir,
        split_name="test",
    )

    # --- Dataloaders ---
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

