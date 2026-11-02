"""Small CNN baseline trained from random initialization."""

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn

from src.data.dataset import WasteDataset, create_loader, seed_everything


def convolution_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 9, dropout: float = 0.25):
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.features = nn.Sequential(
            convolution_block(3, 32, stride=2),
            convolution_block(32, 64),
            convolution_block(64, 128),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 2 * 2, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def main():
    parser = argparse.ArgumentParser(
        description="Check a baseline training update on real images."
    )
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
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    dataset = WasteDataset(
        args.data_root,
        args.manifest,
        split="train",
        image_size=args.image_size,
    )
    loader = create_loader(dataset, batch_size=8, shuffle=True)
    images, targets = next(iter(loader))

    model = SmallCNN(num_classes=len(dataset.classes))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)

    expected_shape = (len(targets), len(dataset.classes))
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(f"Unexpected output shape: {logits.shape}")

    loss = criterion(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite.")

    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")

    optimizer.step()
    elapsed = time.perf_counter() - start

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Initial batch loss: {loss.item():.4f}")
    print(f"Training update time: {elapsed:.3f} seconds")
    print("Forward pass, loss, gradients, and optimizer update passed.")


if __name__ == "__main__":
    main()