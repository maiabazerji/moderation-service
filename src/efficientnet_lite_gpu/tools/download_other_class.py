#!/usr/bin/env python3
"""
Download non-food images for the "Other" class.

Uses Caltech-101 from tensorflow_datasets to populate an "Other" folder
in the raw dataset directory.  The Other class acts as a reject bin so
the model can learn to say "this is not food" instead of giving a
high-confidence food prediction for random images.

Usage:
    cd src/efficientnet_lite_gpu
    python -m tools.download_other_class                          # defaults
    python -m tools.download_other_class --count 800              # more images
    python -m tools.download_other_class --output train/dataset_raw_cleaned/Other
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


# Categories in Caltech-101 that might overlap with our food classes.
FOOD_KEYWORDS = frozenset({
    "pizza", "food", "sandwich", "burger", "fries",
    "hotdog", "hot_dog", "donut", "doughnut", "potato",
})


def _is_food_label(label_name: str) -> bool:
    lower = label_name.lower().replace("-", "_")
    return any(kw in lower for kw in FOOD_KEYWORDS)


def download_other_images(
    output_dir: str | Path,
    count: int = 700,
    image_size: tuple[int, int] = (224, 224),
    seed: int = 42,
) -> int:
    """
    Download *count* non-food images from Caltech-101 into *output_dir*.

    Returns the number of images actually saved.
    """
    # Import here so the rest of the codebase doesn't need tfds installed
    # unless this script is actually run.
    import tensorflow_datasets as tfds

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Caltech-101 dataset (first run may take a while)...")
    ds, info = tfds.load(
        "caltech101",
        split="train",
        with_info=True,
        as_supervised=True,
    )

    # Build label name list and identify food-related indices.
    label_names = info.features["label"].names
    food_indices = {
        idx
        for idx, name in enumerate(label_names)
        if _is_food_label(name)
    }
    print(f"Total Caltech-101 classes: {len(label_names)}")
    print(f"Excluded food-related classes ({len(food_indices)}): "
          f"{[label_names[i] for i in sorted(food_indices)]}")

    # Shuffle deterministically.
    ds = ds.shuffle(buffer_size=10_000, seed=seed)

    saved = 0
    skipped_food = 0
    for img_tensor, label_tensor in ds:
        if saved >= count:
            break

        label_idx = int(label_tensor.numpy())
        if label_idx in food_indices:
            skipped_food += 1
            continue

        img_np = img_tensor.numpy()

        # Caltech-101 has a few grayscale images — convert to RGB.
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[-1] == 1:
            img_np = np.concatenate([img_np] * 3, axis=-1)

        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.convert("RGB")
        img_pil = img_pil.resize(image_size, Image.LANCZOS)
        img_pil.save(output_dir / f"other_{saved:04d}.jpg", quality=92)
        saved += 1

    print(f"Saved {saved} 'Other' images to {output_dir}")
    print(f"Skipped {skipped_food} food-related images")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Download non-food images for the 'Other' class"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train/dataset_raw_cleaned/Other",
        help="Output directory for Other class images",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=700,
        help="Number of images to download (default: 700)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size (default: 224)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling (default: 42)",
    )
    args = parser.parse_args()

    size = (args.image_size, args.image_size)
    saved = download_other_images(
        output_dir=args.output,
        count=args.count,
        image_size=size,
        seed=args.seed,
    )

    if saved < args.count:
        print(f"WARNING: Only saved {saved}/{args.count} images. "
              f"Consider a larger source dataset.")

    print("\nDone! Next steps:")
    print("  1. Verify images:  ls -la", args.output)
    print("  2. Run training:   python main.py --action train")
    print("     (The split script will auto-distribute Other into Train/Val/Test)")


if __name__ == "__main__":
    main()
