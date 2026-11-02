"""Inspect dataset images and save a reproducible inventory."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def difference_hash(image: Image.Image) -> str:
    """Produce a similarity fingerprint for later near-duplicate review."""
    grayscale = image.convert("L").resize(
        (9, 8), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grayscale)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def duplicate_groups(frame: pd.DataFrame, column: str) -> list:
    """Return groups sharing an exact fingerprint."""
    groups = []
    for fingerprint, group in frame.groupby(column, sort=True):
        if len(group) > 1:
            groups.append({
                "fingerprint": fingerprint,
                "paths": group["path"].tolist(),
                "labels": sorted(group["label"].unique().tolist()),
            })
    return groups


def audit(data_root: Path, output_dir: Path) -> None:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_root}")

    paths = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found in {data_root}")

    records = []
    errors = []

    for path in tqdm(paths, desc="Auditing images"):
        relative_path = path.relative_to(data_root)

        # Expected layout: class_name/image_name.jpg
        if len(relative_path.parts) != 2:
            errors.append({
                "path": relative_path.as_posix(),
                "error": "Unexpected folder layout",
            })
            continue

        try:
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            with Image.open(path) as image:
                image.load()  # Decode the full image to detect unreadable files.
                width, height = image.size
                mode = image.mode

                # Match the orientation/color handling intended for inference.
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                pixel_hasher = hashlib.sha256()
                pixel_hasher.update(
                    f"{rgb.width}x{rgb.height}:RGB:".encode("ascii")
                )
                pixel_hasher.update(rgb.tobytes())

                records.append({
                    "path": relative_path.as_posix(),
                    "label": relative_path.parts[0],
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "file_sha256": file_hash,
                    "pixel_sha256": pixel_hasher.hexdigest(),
                    "dhash": difference_hash(rgb),
                })

        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            errors.append({
                "path": relative_path.as_posix(),
                "error": f"{type(exc).__name__}: {exc}",
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "image_errors.json"
    errors_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")

    if not records:
        raise RuntimeError(f"No readable images. See {errors_path}")

    frame = pd.DataFrame(records).sort_values("path")
    frame.to_csv(output_dir / "inventory.csv", index=False)

    class_counts = frame["label"].value_counts().sort_index()
    dimensions = (
        frame.groupby(["width", "height"])
        .size()
        .reset_index(name="count")
    )

    file_duplicates = duplicate_groups(frame, "file_sha256")
    pixel_duplicates = duplicate_groups(frame, "pixel_sha256")

    summary = {
        "data_root": data_root.as_posix(),
        "image_files_found": len(paths),
        "readable_images": len(frame),
        "image_errors": len(errors),
        "number_of_classes": len(class_counts),
        "class_counts": {
            label: int(count) for label, count in class_counts.items()
        },
        "largest_to_smallest_class_ratio": float(
            class_counts.max() / class_counts.min()
        ),
        "image_modes": {
            mode: int(count)
            for mode, count in frame["mode"].value_counts().items()
        },
        "dimensions": dimensions.to_dict(orient="records"),
        "file_duplicate_groups": file_duplicates,
        "pixel_duplicate_groups": pixel_duplicates,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Image files found: %s", len(paths))
    LOGGER.info("Readable images: %s", len(frame))
    LOGGER.info("Image errors: %s", len(errors))
    LOGGER.info("Classes: %s", len(class_counts))
    print("\nClass counts:")
    print(class_counts.to_string())
    print(f"\nImage modes: {summary['image_modes']}")
    print(f"Unique image dimensions: {len(dimensions)}")
    print(f"Exact file duplicate groups: {len(file_duplicates)}")
    print(f"Identical decoded-image groups: {len(pixel_duplicates)}")
    print(f"\nReports saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/realwaste-main/RealWaste"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/data_audit"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    audit(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
    