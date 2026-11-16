"""Example models, kept in an importable module so runs can rebuild them by ``model_factory``."""
from __future__ import annotations

from torch import nn


def mnist_mlp(hidden: int = 128) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 10))


def cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-18 with the usual CIFAR stem (3x3 conv, no max-pool)."""
    from torchvision.models import resnet18

    m = resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m
