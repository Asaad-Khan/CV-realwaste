"""Load a trained checkpoint and predict waste-material classes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transform,
)
from src.models.baseline import SmallCNN


class WastePredictor:
    def __init__(self, checkpoint_path: Path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["model_name"] != "small_cnn":
            raise ValueError("This inference version supports small_cnn only.")

        self.classes = checkpoint["classes"]
        self.image_size = checkpoint["image_size"]
        self.metadata = checkpoint["metadata"]
        self.validation_metrics = checkpoint["validation_metrics"]
        self.epoch = checkpoint["epoch"]

        if (
            len(self.classes) != checkpoint["num_classes"]
            or len(set(self.classes)) != len(self.classes)
        ):
            raise ValueError("Invalid class mapping in checkpoint.")

        # Ensure our shared transform matches the training checkpoint.
        if (
            tuple(checkpoint["normalization_mean"]) != IMAGENET_MEAN
            or tuple(checkpoint["normalization_std"]) != IMAGENET_STD
        ):
            raise ValueError("Checkpoint normalization differs from preprocessing.")

        self.transform = build_transform(
            self.image_size,
            training=False,
        )

        self.model = SmallCNN(
            num_classes=len(self.classes),
            dropout=checkpoint["config"]["dropout"],
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def predict_image(self, image: Image.Image, top_k: int = 3) -> dict:
        if not 1 <= top_k <= len(self.classes):
            raise ValueError("top_k must be between 1 and the class count.")

        rgb = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0)
        logits = self.model(tensor)

        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite outputs.")

        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = probabilities.topk(top_k)

        predictions = [
            {"label": self.classes[index], "score": float(score)}
            for score, index in zip(scores.tolist(), indices.tolist())
        ]

        return {
            "predicted_class": predictions[0]["label"],
            "top_predictions": predictions,
            "model_name": "small_cnn",
            "model_version": self.metadata["run_id"],
        }


@torch.inference_mode()
def verify_validation(predictor, data_root: Path, manifest: Path):
    # These imports are needed for verification, rather than prediction itself.
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm

    from src.data.dataset import WasteDataset, create_loader

    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if manifest_hash != predictor.metadata["manifest_sha256"]:
        raise ValueError("Manifest differs from the one used for training.")

    dataset = WasteDataset(
        data_root,
        manifest,
        split="validation",
        image_size=predictor.image_size,
    )
    if dataset.classes != predictor.classes:
        raise ValueError("Dataset and checkpoint class orders differ.")

    loader = create_loader(dataset, batch_size=32, shuffle=False)
    targets_all = []
    predictions_all = []

    for images, targets in tqdm(loader, desc="Verifying validation"):
        logits = predictor.model(images)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite validation outputs.")

        targets_all.extend(targets.tolist())
        predictions_all.extend(logits.argmax(dim=1).tolist())

    measured = {
        "accuracy": float(accuracy_score(targets_all, predictions_all)),
        "macro_f1": float(
            f1_score(
                targets_all,
                predictions_all,
                labels=list(range(len(predictor.classes))),
                average="macro",
                zero_division=0,
            )
        ),
    }

    for metric, value in measured.items():
        expected = predictor.validation_metrics[metric]
        if abs(value - expected) > 1e-6:
            raise RuntimeError(
                f"{metric} differs: checkpoint={expected}, measured={value}"
            )

    # Also exercise the single-image prediction route used by the future API.
    sample = dataset.frame.iloc[0]
    image_path = data_root / sample["path"]
    with Image.open(image_path) as image:
        result = predictor.predict_image(image)

    first_batch_route_label = predictor.classes[predictions_all[0]]
    if result["predicted_class"] != first_batch_route_label:
        raise RuntimeError("Single-image and batch prediction labels differ.")

    print("\nReproduced validation metrics:")
    print(json.dumps(measured, indent=2))
    print(f"Checkpoint epoch: {predictor.epoch}")
    print(f"\nExample image: {image_path}")
    print(f"Dataset label: {sample['label']}")
    print(json.dumps(result, indent=2))
    print("\nCheckpoint loading and inference checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--verify-validation", action="store_true")
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

    if args.image is None and not args.verify_validation:
        parser.error("Provide --image or --verify-validation.")

    torch.set_num_threads(4)
    predictor = WastePredictor(args.checkpoint)

    if args.verify_validation:
        verify_validation(predictor, args.data_root, args.manifest)

    if args.image is not None:
        with Image.open(args.image) as image:
            result = predictor.predict_image(image)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()