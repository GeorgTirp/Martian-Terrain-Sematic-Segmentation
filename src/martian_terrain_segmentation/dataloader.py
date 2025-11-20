# data.py
"""
AI4Mars dataloading utilities.

We use the Hugging Face dataset `hassanjbara/AI4MARS`, which provides:

- `image`: original rover image (Navcam, PIL-like).
- `label_mask`: semantic segmentation mask with terrain classes encoded as:

    * 0 -> soil  
    * 1 -> bedrock  
    * 2 -> sand  
    * 3 -> big rock  
    * 255 -> null / no label

This module wraps the dataset into PyTorch `Dataset` and `DataLoader` objects,
adds basic preprocessing (resize, grayscale/RGB conversion, tensor conversion),
and provides optional scanning to remove a small number of corrupted samples.
"""

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

from datasets import load_dataset, load_from_disk  # add load_from_disk if you want on-disk caching

AI4MARS_CLASS_NAMES: List[str] = ["soil", "bedrock", "sand", "big_rock"]
AI4MARS_IGNORE_INDEX: int = 255


class AI4MarsHFDataset(Dataset):
    """
    PyTorch wrapper around a Hugging Face split of the AI4Mars dataset.

    This class:

    - Applies resizing and grayscale/RGB conversion to images.
    - Resizes segmentation masks with nearest-neighbor interpolation.
    - Maps unknown label values to an ignore index (`AI4MARS_IGNORE_INDEX`).
    - Optionally scans the split once to drop corrupted or undecodable samples
      and caches the list of valid indices for future runs.

    Parameters
    ----------
    hf_split :
        A single split of the Hugging Face dataset (e.g. `raw["train"]`).
    image_size : int, default=256
        Target spatial size (height and width) for images and masks.
    to_rgb : bool, default=False
        If ``True``, convert images to 3-channel RGB tensors.
        If ``False``, keep them as 1-channel grayscale tensors.
    scan_spurious : bool, default=False
        If ``True``, scan the split to find valid (decodable) samples, build
        ``valid_indices``, and save them to disk in ``cache_dir``. This is
        useful for the **first** run on a new dataset cache.
        If ``False``, the dataset attempts to load precomputed valid indices
        from disk and skips the expensive scan.
    cache_dir : str, default="./ai4mars_valid_indices"
        Directory where per-split valid index files are stored
        (e.g. ``valid_indices_train.npy``).
    split_name : str, default="train"
        Name of the split (``"train"``, ``"val"``, or ``"test"``) used to build
        the cache file name.

    Attributes
    ----------
    ds :
        The underlying Hugging Face dataset split.
    valid_indices : list[int]
        Indices of valid (non-corrupted) samples in ``ds``.
    image_transform :
        Composed torchvision transform applied to images.
    mask_resize :
        torchvision transform used to resize label masks.
    cache_path : str
        Path to the ``.npy`` file storing ``valid_indices`` for this split.
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
        """Return the number of valid samples in this split."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        """
        Get a single (image, mask) pair.

        Parameters
        ----------
        idx : int
            Index in the filtered dataset (0 <= idx < len(self)).

        Returns
        -------
        img_t : torch.Tensor
            Image tensor of shape ``[C, H, W]``, where ``C`` is 1 (grayscale)
            or 3 (RGB) depending on ``to_rgb``.
        mask_t : torch.Tensor
            Integer segmentation mask of shape ``[H, W]`` with label values
            in ``{0, 1, 2, 3, AI4MARS_IGNORE_INDEX}``.
        """
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


@dataclass
class DataLoaders:
    """
    Container for the three data splits used in training and evaluation.

    Attributes
    ----------
    train : torch.utils.data.DataLoader
        Dataloader for the training split.
    val : torch.utils.data.DataLoader
        Dataloader for the validation split.
    test : torch.utils.data.DataLoader
        Dataloader for the test split.
    """
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
    scan_spurious: bool = False,
    valid_indices_cache_dir: str = "./ai4mars_valid_indices",
) -> DataLoaders:
    """
    Create train/validation/test dataloaders for the AI4Mars dataset.

    This function:

    - Downloads (or loads from disk) the Hugging Face dataset
      ``"hassanjbara/AI4MARS"``.
    - Splits it into train/val/test splits (using `val_fraction` and a fixed seed).
    - Optionally subsamples each split for faster experiments.
    - Wraps each split in an :class:`AI4MarsHFDataset`, which can scan for and
      cache valid (non-corrupted) samples.
    - Returns PyTorch dataloaders for each split.

    Parameters
    ----------
    batch_size : int, default=4
        Batch size used for all three dataloaders.
    image_size : int, default=256
        Spatial resolution to which images and masks are resized (square).
    num_workers : int, default=4
        Number of worker processes for data loading.
    val_fraction : float, default=0.1
        Fraction of the (non-test) data to reserve for validation.
    to_rgb : bool, default=False
        If ``True``, convert grayscale Navcam images to 3-channel RGB.
        If ``False``, keep them as 1-channel grayscale.
    seed : int, default=42
        Random seed used for splitting into train/val/test.
    cache_dir : str or None, default=None
        Directory used by Hugging Face to cache the raw dataset.
        If ``None``, the default HF cache location is used.
    max_train_samples : int or None, default=None
        If not ``None``, limit the training split to at most this many samples.
        Useful for quick debugging runs.
    max_val_samples : int or None, default=None
        If not ``None``, limit the validation split to at most this many samples.
    max_test_samples : int or None, default=None
        If not ``None``, limit the test split to at most this many samples.
    use_local_disk_copy : bool, default=False
        If ``True``, save the downloaded HF dataset to ``local_disk_path`` and
        load from there on subsequent runs (avoids re-downloading and reprocessing).
    local_disk_path : str, default="./data/ai4mars_hf"
        Path to store or load the local disk copy of the raw HF dataset.
    scan_spurious : bool, default=False
        Controls how spurious/corrupted samples are handled:

        - If ``True``, each split is scanned on this run, and a list of valid
          indices is built and saved into ``valid_indices_cache_dir``.
        - If ``False``, we assume the scanning was already done, and valid
          indices are loaded from disk (if present), avoiding the expensive scan.
    valid_indices_cache_dir : str, default="./ai4mars_valid_indices"
        Directory where per-split valid index files (``.npy``) are stored and
        loaded. Files are named like ``valid_indices_train.npy``,
        ``valid_indices_val.npy``, and ``valid_indices_test.npy``.

    Returns
    -------
    DataLoaders
        A dataclass bundle with ``train``, ``val``, and ``test`` PyTorch dataloaders.

    Examples
    --------
    Basic usage:

    .. code-block:: python

        loaders = create_ai4mars_dataloaders(
            batch_size=8,
            image_size=256,
            num_workers=4,
            val_fraction=0.1,
            to_rgb=False,
            scan_spurious=True,  # first run: build valid index cache
        )

        train_loader = loaders.train
        val_loader = loaders.val
        test_loader = loaders.test

    For later runs, you can skip scanning:

    .. code-block:: python

        loaders = create_ai4mars_dataloaders(
            batch_size=8,
            image_size=256,
            num_workers=4,
            val_fraction=0.1,
            to_rgb=False,
            scan_spurious=False,  # reuse cached valid indices
        )
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
