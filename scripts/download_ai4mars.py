#!/usr/bin/env python3
"""
Download the AI4Mars dataset through the project dataloader.

This script intentionally does one thing: call
`create_ai4mars_dataloaders(...)` so the Hugging Face AI4Mars dataset is
downloaded and optionally saved to a local on-disk copy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the AI4Mars dataset through the project dataloader."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size passed to the dataloader constructor.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size passed to the dataloader constructor.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for the dataloaders.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction used when splitting the training set.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-disk-path",
        default="data/ai4mars_hf",
        help="Path where the downloaded Hugging Face dataset will be saved.",
    )
    parser.add_argument(
        "--valid-indices-cache-dir",
        default="ai4mars_valid_indices",
        help="Path where valid sample index caches will be saved.",
    )
    parser.add_argument(
        "--scan-spurious",
        action="store_true",
        help="Force a scan for corrupted samples before caching valid indices.",
    )
    parser.add_argument(
        "--to-rgb",
        action="store_true",
        help="Convert grayscale rover images to 3-channel RGB tensors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sys.path.insert(0, str(SRC_DIR))

    try:
        from martian_terrain_segmentation.dataloader import create_ai4mars_dataloaders
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(
            "Unable to import the project dataloader because "
            f"`{missing}` is not installed.",
            file=sys.stderr,
        )
        print(
            "Install the project dependencies in your environment and rerun this script.",
            file=sys.stderr,
        )
        return 1

    local_disk_path = _resolve_path(args.local_disk_path)
    valid_indices_cache_dir = _resolve_path(args.valid_indices_cache_dir)
    cache_dir = _resolve_path(args.cache_dir) if args.cache_dir else None

    print("Downloading AI4Mars through the project dataloader...")
    print(f"Local dataset copy: {local_disk_path}")
    print(f"Valid index cache: {valid_indices_cache_dir}")
    if cache_dir is not None:
        print(f"Hugging Face cache: {cache_dir}")

    loaders = create_ai4mars_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        to_rgb=args.to_rgb,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        use_local_disk_copy=True,
        local_disk_path=str(local_disk_path),
        scan_spurious=args.scan_spurious,
        valid_indices_cache_dir=str(valid_indices_cache_dir),
    )

    print("Download complete.")
    print(f"Train samples: {len(loaders.train.dataset)}")
    print(f"Val samples:   {len(loaders.val.dataset)}")
    print(f"Test samples:  {len(loaders.test.dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
