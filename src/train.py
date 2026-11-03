"""Train the baseline and select its checkpoint using validation macro F1."""

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torchvision
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm import tqdm

from src.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WasteDataset,
    create_loader,
    seed_everything,
)
from src.models.baseline import SmallCNN


LOGGER = logging.getLogger(__name__)


def run_epoch(model, loader, criterion, num_classes, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    targets_all = []
    predictions_all = []

    description = "Train" if training else "Validation"

    for images, targets in tqdm(loader, desc=description, leave=False):
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * targets.size(0)
        targets_all.extend(targets.tolist())
        predictions_all.extend(logits.detach().argmax(dim=1).tolist())

    return {
        "loss": total_loss / len(targets_all),
        "accuracy": float(accuracy_score(targets_all, predictions_all)),
        "macro_f1": float(
            f1_score(
                targets_all,
                predictions_all,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baseline.yaml")
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    supported_models = {"small_cnn", "mobilenet_v3_small"}
    if config["model_name"] not in supported_models:
        raise ValueError(f"Unsupported model: {config['model_name']}")
    if config["epochs"] < 1 or config["patience"] < 1:
        raise ValueError("epochs and patience must be positive.")

    seed_everything(config["seed"])
    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)

    run_id = (
        config["model_name"]
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    artifact_dir = Path("artifacts") / run_id
    report_dir = Path("reports/training") / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                report_dir / "training.log", encoding="utf-8"
            ),
        ],
        force=True,
    )

    (report_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    manifest_path = Path(config["manifest"])
    data_root = Path(config["data_root"])

    train_dataset = WasteDataset(
        data_root, manifest_path, "train", config["image_size"]
    )
    val_dataset = WasteDataset(
        data_root, manifest_path, "validation", config["image_size"]
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation class mappings differ.")

    train_loader = create_loader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        seed=config["seed"],
    )
    val_loader = create_loader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        seed=config["seed"],
    )

    classes = train_dataset.classes
    initial_checkpoint = config.get("initial_checkpoint")

    if config["model_name"] == "small_cnn":
        if initial_checkpoint:
            raise ValueError("Fine-tuning is configured for MobileNet only.")

        model = SmallCNN(
            num_classes=len(classes),
            dropout=config["dropout"],
        )
    else:
        from src.models.transfer import MobileNetTransfer

        model = MobileNetTransfer(
            num_classes=len(classes),
            pretrained=not bool(initial_checkpoint),
            freeze_backbone=True,
            dropout=config["dropout"],
        )

    if initial_checkpoint:
        checkpoint = torch.load(
            initial_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

        if (
            checkpoint["model_name"] != config["model_name"]
            or checkpoint["classes"] != classes
            or checkpoint["image_size"] != config["image_size"]
            or checkpoint["metadata"]["manifest_sha256"] != manifest_sha
        ):
            raise ValueError("Checkpoint does not match this training run.")

        model.load_state_dict(checkpoint["model_state_dict"])

        # The existing train() override keeps features in eval mode.
        # Eval mode fixes BatchNorm statistics while allowing gradients.
        final_features = model.network.features[-3:]
        final_features.requires_grad_(True)

        optimizer_parameters = [
            {
                "params": model.network.classifier.parameters(),
                "lr": config["learning_rate"],
            },
            {
                "params": final_features.parameters(),
                "lr": config["backbone_learning_rate"],
            },
        ]
    else:
        optimizer_parameters = [
            p for p in model.parameters() if p.requires_grad
        ]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    metadata = {
        "run_id": run_id,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "classes": classes,
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    (report_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    history = []
    best_f1 = -1.0
    best_epoch = None
    best_metrics = None
    epochs_without_improvement = 0
    checkpoint_path = artifact_dir / "best.pt"
    training_start = time.perf_counter()

    LOGGER.info("Run: %s", run_id)
    LOGGER.info(
        "Training images: %s | Validation images: %s",
        len(train_dataset),
        len(val_dataset),
    )

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model, train_loader, criterion, len(classes), optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, len(classes)
        )

        row = {
            "epoch": epoch,
            **{f"train_{key}": value
               for key, value in train_metrics.items()},
            **{f"val_{key}": value
               for key, value in val_metrics.items()},
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            report_dir / "history.csv", index=False
        )

        LOGGER.info(
            "Epoch %02d | train loss %.4f | val loss %.4f | "
            "val accuracy %.4f | val macro F1 %.4f | %.1fs",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            row["epoch_seconds"],
        )

        if val_metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_name": config["model_name"],
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(classes),
                    "classes": classes,
                    "image_size": config["image_size"],
                    "normalization_mean": list(IMAGENET_MEAN),
                    "normalization_std": list(IMAGENET_STD),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "config": config,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            LOGGER.info("Saved improved checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["patience"]:
            LOGGER.info("Early stopping: validation macro F1 stopped improving.")
            break

    summary = {
        **metadata,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "total_training_seconds": time.perf_counter() - training_start,
        "checkpoint": checkpoint_path.as_posix(),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    LOGGER.info("Best epoch: %s", best_epoch)
    LOGGER.info("Best validation macro F1: %.4f", best_f1)
    LOGGER.info("Checkpoint: %s", checkpoint_path)
    LOGGER.info("Reports: %s", report_dir)


if __name__ == "__main__":
    main()