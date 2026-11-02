"""Load images from the fixed split manifest."""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transform(image_size: int, training: bool = False):
    if image_size < 32:
        raise ValueError("image_size must be at least 32")

    operations = [
        # Preserve the complete image rather than cropping out small items.
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    ]

    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=15,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=(124, 116, 104),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(operations)


class WasteDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        split: str,
        image_size: int = 160,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.data_root = Path(data_root)
        frame = pd.read_csv(
            manifest_path,
            usecols=["path", "label", "class_id", "split"],
        )

        # Check the mapping across the complete manifest.
        mapping = frame[["label", "class_id"]].drop_duplicates()
        if (
            mapping["label"].duplicated().any()
            or mapping["class_id"].duplicated().any()
        ):
            raise ValueError("Class names and IDs must have a one-to-one mapping.")

        mapping = mapping.sort_values("class_id")
        if mapping["class_id"].tolist() != list(range(len(mapping))):
            raise ValueError("Class IDs must be contiguous and start at zero.")

        self.classes = mapping["label"].tolist()
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)

        if self.frame.empty:
            raise ValueError(f"No images found for split: {split}")

        self.transform = build_transform(
            image_size=image_size,
            training=(split == "train"),
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        record = self.frame.iloc[index]
        path = self.data_root / record["path"]

        try:
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                tensor = self.transform(rgb)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not process image: {path}") from exc

        return tensor, int(record["class_id"])


def create_loader(
    dataset: WasteDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Simple, reliable starting point on Windows.
        pin_memory=False,
        generator=generator,
    )


def main():
    parser = argparse.ArgumentParser(description="Check image loading.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/realwaste-main/RealWaste"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/splits/manifest.csv"),
    )
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()

    seed_everything(42)

    # Inspect training and validation only; leave test images untouched.
    for split in ("train", "validation"):
        dataset = WasteDataset(
            args.data_root, args.manifest, split, args.image_size
        )
        loader = create_loader(dataset, shuffle=(split == "train"))
        images, targets = next(iter(loader))

        expected_shape = (3, args.image_size, args.image_size)
        if tuple(images.shape[1:]) != expected_shape:
            raise RuntimeError(f"Unexpected image shape: {images.shape}")
        if not torch.isfinite(images).all():
            raise RuntimeError("Image batch contains non-finite values.")
        if targets.min() < 0 or targets.max() >= len(dataset.classes):
            raise RuntimeError("Target IDs are outside the expected range.")

        print(f"\nSplit: {split}")
        print(f"Dataset size: {len(dataset):,}")
        print(f"Image batch shape: {tuple(images.shape)}")
        print(f"Image dtype: {images.dtype}")
        print(f"Target batch shape: {tuple(targets.shape)}")
        print(f"Target dtype: {targets.dtype}")
        print(f"Classes: {dataset.classes}")

    print("\nData loading checks passed.")


if __name__ == "__main__":
    main()