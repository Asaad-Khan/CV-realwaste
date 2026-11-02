"""Create sample grids and identify candidate near-duplicate image pairs."""

import argparse
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def show_image(axis, path: Path, title: str) -> None:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        axis.imshow(rgb)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def create_sample_grid(frame: pd.DataFrame, data_root: Path,
                       output_dir: Path) -> None:
    labels = sorted(frame["label"].unique())
    fig, axes = plt.subplots(
        len(labels), 3,
        figsize=(12, 2.5 * len(labels)),
        squeeze=False,
    )

    for row, label in enumerate(labels):
        examples = frame.loc[frame["label"] == label].sample(
            n=3, random_state=42
        )
        for column, record in enumerate(examples.to_dict("records")):
            show_image(
                axes[row, column],
                data_root / record["path"],
                f"{label}\n{Path(record['path']).name}",
            )

    fig.suptitle("Dataset samples — three images per class", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "class_samples.png", dpi=120)
    plt.close(fig)


def find_candidates(frame: pd.DataFrame, max_distance: int) -> pd.DataFrame:
    records = frame.to_dict("records")
    hashes = [int(record["dhash"], 16) for record in records]
    pairs = []

    # Compare every pair, including images with different labels.
    for i in tqdm(range(len(records)), desc="Checking image similarity"):
        for j in range(i + 1, len(records)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= max_distance:
                pairs.append({
                    "path_a": records[i]["path"],
                    "label_a": records[i]["label"],
                    "path_b": records[j]["path"],
                    "label_b": records[j]["label"],
                    "hash_distance": distance,
                })

    columns = ["path_a", "label_a", "path_b", "label_b", "hash_distance"]
    result = pd.DataFrame(pairs, columns=columns)
    return result.sort_values(
        ["hash_distance", "path_a", "path_b"]
    ).reset_index(drop=True)


def create_pair_grid(pairs: pd.DataFrame, data_root: Path,
                     output_dir: Path) -> None:
    # Show the 12 closest pairs initially; the CSV contains all candidates.
    examples = pairs.head(12).to_dict("records")
    if not examples:
        return

    rows = math.ceil(len(examples) / 2)
    fig, axes = plt.subplots(
        rows, 4, figsize=(16, 3.2 * rows), squeeze=False
    )
    for axis in axes.flat:
        axis.axis("off")

    for index, pair in enumerate(examples):
        row = index // 2
        column = (index % 2) * 2

        for offset, side in enumerate(("a", "b")):
            path = pair[f"path_{side}"]
            label = pair[f"label_{side}"]
            show_image(
                axes[row, column + offset],
                data_root / path,
                f"Pair {index + 1} | distance {pair['hash_distance']}\n"
                f"{label}: {Path(path).name}",
            )

    fig.suptitle("Closest similarity candidates — visual review required")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "near_duplicate_candidates.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/raw/realwaste-main/RealWaste"),
    )
    parser.add_argument(
        "--audit-dir", type=Path,
        default=Path("reports/data_audit"),
    )
    parser.add_argument("--max-distance", type=int, default=6)
    args = parser.parse_args()

    if not 0 <= args.max_distance <= 64:
        parser.error("--max-distance must be between 0 and 64")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    frame = pd.read_csv(
        args.audit_dir / "inventory.csv",
        dtype={"dhash": str},
    )

    create_sample_grid(frame, args.data_root, args.audit_dir)
    pairs = find_candidates(frame, args.max_distance)
    pairs.to_csv(
        args.audit_dir / "near_duplicate_candidates.csv", index=False
    )
    create_pair_grid(pairs, args.data_root, args.audit_dir)

    LOGGER.info("Similarity candidate pairs: %s", len(pairs))
    LOGGER.info(
        "Cross-label candidate pairs: %s",
        int((pairs["label_a"] != pairs["label_b"]).sum()),
    )
    LOGGER.info("Reports saved to: %s", args.audit_dir)


if __name__ == "__main__":
    main()