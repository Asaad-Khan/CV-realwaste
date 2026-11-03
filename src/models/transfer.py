import time

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


class MobileNetTransfer(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        weights = (
            MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        self.network = mobilenet_v3_small(
            weights=weights,
            dropout=dropout,
        )

        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(
            in_features, num_classes
        )

        self.freeze_backbone = freeze_backbone

        for parameter in self.network.features.parameters():
            parameter.requires_grad = not freeze_backbone

        if freeze_backbone:
            self.network.features.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        # Frozen BatchNorm running statistics must stay unchanged too.
        if self.freeze_backbone:
            self.network.features.eval()

        return self

    def forward(self, images):
        return self.network(images)


def smoke_test():
    torch.manual_seed(42)
    torch.set_num_threads(4)

    model = MobileNetTransfer()
    model.train()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # Synthetic inputs test mechanics, not prediction quality.
    images = torch.randn(8, 3, 224, 224)
    targets = torch.arange(8, dtype=torch.long) % 9

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # Warm up CPU kernels before timing.
    model.eval()
    with torch.no_grad():
        model(images)
    model.train()

    head_before = model.network.classifier[-1].weight.detach().clone()
    backbone_before = {
        name: value.detach().clone()
        for name, value in model.network.features.state_dict().items()
    }

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()

    assert logits.shape == (8, 9)
    assert torch.isfinite(loss).item()
    assert all(
        p.grad is None
        for p in model.network.features.parameters()
    )

    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in trainable_parameters
    )

    optimizer.step()
    elapsed = time.perf_counter() - started

    assert not torch.equal(
        head_before, model.network.classifier[-1].weight
    )

    for name, value in model.network.features.state_dict().items():
        assert torch.equal(backbone_before[name], value), name

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Initial batch loss: {loss.item():.4f}")
    print(f"Training update time: {elapsed:.3f} seconds")
    print("Classifier updated; frozen backbone and buffers unchanged.")
    print("Transfer model checks passed.")


if __name__ == "__main__":
    smoke_test()