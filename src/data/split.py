"""Create reproducible splits while keeping similarity groups together."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


LOGGER = logging.getLogger(__name__)


def make_groups(frame: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    """Join candidate pairs into connected components."""
    paths = frame["path"].tolist()
    parent = {path: path for path in paths}

    def find(path: str) -> str:
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            # Deterministic representative for each component.
            first, second = sorted((left_root, right_root))
            parent[second] = first

    for pair in pairs.itertuples(index=False):
        if pair.path_a not in parent or pair.path_b not in parent:
            raise ValueError("Candidate CSV contains a path absent from inventory.")
        union(pair.path_a, pair.path_b)

    roots = [find(path) for path in paths]
    identifiers = {
        root: f"group_{index:05d}"
        for index, root in enumerate(sorted(set(roots)))
    }
    return [identifiers[root] for root in roots]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir", type=Path,
        default=Path("reports/data_audit"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/splits"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    frame = pd.read_csv(
        args.audit_dir / "inventory.csv",
        dtype={"dhash": str},
    ).sort_values("path").reset_index(drop=True)

    pairs = pd.read_csv(args.audit_dir / "near_duplicate_candidates.csv")

    if frame["path"].duplicated().any():
        raise ValueError("Inventory contains repeated paths.")

    frame["group_id"] = make_groups(frame, pairs)

    labels = sorted(frame["label"].unique())
    class_to_idx = {label: index for index, label in enumerate(labels)}
    frame["class_id"] = frame["label"].map(class_to_idx)

    # Seven approximately balanced folds:
    # five for training, one for validation, one for testing.
    splitter = StratifiedGroupKFold(
        n_splits=7,
        shuffle=True,
        random_state=args.seed,
    )

    frame["fold"] = -1
    for fold, (_, held_out_indices) in enumerate(
        splitter.split(
            X=frame["path"],
            y=frame["label"],
            groups=frame["group_id"],
        )
    ):
        frame.loc[held_out_indices, "fold"] = fold

    frame["split"] = "train"
    frame.loc[frame["fold"] == 0, "split"] = "test"
    frame.loc[frame["fold"] == 1, "split"] = "validation"

    # Explicit checks survive even when Python optimization is enabled.
    if (frame["fold"] < 0).any():
        raise RuntimeError("Some images were not assigned to a fold.")

    if frame.groupby("group_id")["split"].nunique().max() != 1:
        raise RuntimeError("A similarity group crosses split boundaries.")

    for column in ("file_sha256", "pixel_sha256"):
        if frame.groupby(column)["split"].nunique().max() != 1:
            raise RuntimeError(f"An exact duplicate crosses splits: {column}")

    path_to_split = frame.set_index("path")["split"].to_dict()
    for pair in pairs.itertuples(index=False):
        if path_to_split[pair.path_a] != path_to_split[pair.path_b]:
            raise RuntimeError("A candidate pair crosses split boundaries.")

    counts = pd.crosstab(frame["label"], frame["split"]).reindex(
        index=labels,
        columns=["train", "validation", "test"],
        fill_value=0,
    )

    if (counts == 0).any().any():
        raise RuntimeError("At least one split is missing a class.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "manifest.csv", index=False)
    counts.to_csv(args.output_dir / "class_counts.csv")

    (args.output_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2),
        encoding="utf-8",
    )

    group_sizes = frame.groupby("group_id").size()
    summary = {
        "seed": args.seed,
        "method": "StratifiedGroupKFold",
        "n_folds": 7,
        "test_fold": 0,
        "validation_fold": 1,
        "grouping": "Connected components of all dHash candidate pairs",
        "candidate_pairs": len(pairs),
        "total_images": len(frame),
        "total_groups": int(len(group_sizes)),
        "groups_with_multiple_images": int((group_sizes > 1).sum()),
        "largest_group_size": int(group_sizes.max()),
        "split_counts": {
            split: int((frame["split"] == split).sum())
            for split in ("train", "validation", "test")
        },
        "checks": {
            "group_overlap": False,
            "exact_duplicate_overlap": False,
            "candidate_pair_overlap": False,
            "all_classes_present_in_each_split": True,
        },
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nImages per class and split:")
    print(counts.to_string())

    print("\nSplit totals:")
    for split, count in summary["split_counts"].items():
        print(f"{split}: {count:,} ({count / len(frame):.1%})")

    print(f"\nGroups with multiple images: "
          f"{summary['groups_with_multiple_images']}")
    print(f"Largest group: {summary['largest_group_size']} images")
    print("All split integrity checks passed.")
    LOGGER.info("Saved split files to %s", args.output_dir)


if __name__ == "__main__":
    main()