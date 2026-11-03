import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WasteDataset,
    create_loader,
)
from src.models.baseline import SmallCNN
from src.models.transfer import MobileNetTransfer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("checkpoint", type=Path)
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
    args = parser.parse_args()
    torch.set_num_threads(4)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    metadata = checkpoint["metadata"]
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_hash != metadata["manifest_sha256"]:
        raise ValueError("Split manifest differs from the training manifest.")

    if (
        tuple(checkpoint["normalization_mean"]) != IMAGENET_MEAN
        or tuple(checkpoint["normalization_std"]) != IMAGENET_STD
    ):
        raise ValueError("Checkpoint normalization differs from the dataset.")

    dataset = WasteDataset(
        args.data_root, args.manifest, "test", checkpoint["image_size"]
    )
    classes = dataset.classes
    if classes != checkpoint["classes"]:
        raise ValueError("Checkpoint class order differs from the manifest.")

    model_name = checkpoint["model_name"]
    model_args = {
        "num_classes": len(classes),
        "dropout": checkpoint["config"]["dropout"],
    }
    if model_name == "small_cnn":
        model = SmallCNN(**model_args)
    elif model_name == "mobilenet_v3_small":
        model = MobileNetTransfer(pretrained=False, **model_args)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual = []
    predicted = []
    scores = []
    loader = create_loader(dataset, batch_size=32, shuffle=False)

    with torch.inference_mode():
        for images, targets in loader:
            probabilities = model(images).softmax(dim=1)
            top_scores, top_classes = probabilities.max(dim=1)
            actual.extend(targets.tolist())
            predicted.extend(top_classes.tolist())
            scores.extend(top_scores.tolist())

    labels = list(range(len(classes)))
    summary = {
        "model": model_name,
        "run_id": metadata["run_id"],
        "checkpoint_epoch": checkpoint["epoch"],
        "test_images": len(actual),
        "test_accuracy": accuracy_score(actual, predicted),
        "test_macro_f1": f1_score(
            actual, predicted, labels=labels, average="macro", zero_division=0
        ),
    }
    report_dir = Path("reports/evaluation") / metadata["run_id"]
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2)
    (report_dir / "summary.json").write_text(summary_text + "\n", encoding="utf-8")

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame.from_dict(
        {name: report[name] for name in classes}, orient="index"
    ).rename_axis("class").to_csv(report_dir / "per_class.csv")

    matrix = confusion_matrix(actual, predicted, labels=labels)
    pd.DataFrame(matrix, index=classes, columns=classes).rename_axis(
        "actual_class"
    ).to_csv(report_dir / "confusion_matrix.csv")

    pd.DataFrame(
        {
            "path": dataset.frame["path"].tolist(),
            "actual": [classes[i] for i in actual],
            "predicted": [classes[i] for i in predicted],
            "top_score": scores,
            "correct": [a == p for a, p in zip(actual, predicted)],
        }
    ).to_csv(report_dir / "predictions.csv", index=False)

    print(summary_text)
    print(f"Reports saved to {report_dir}")


if __name__ == "__main__":
    main()