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


def spiral_mlp(hidden: int = 64, classes: int = 3) -> nn.Module:
    return nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, classes))


def small_cnn(in_channels: int = 1, classes: int = 10) -> nn.Module:
    """Two conv blocks and a classifier; for 28x28 grayscale images."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, classes))


def cifar_cnn(width: int = 32, classes: int = 10) -> nn.Module:
    """A VGG-style CNN with BatchNorm for 32x32 colour images (about 0.3M parameters at width 32)."""
    def block(cin: int, cout: int) -> list[nn.Module]:
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]

    w = width
    return nn.Sequential(*block(3, w), *block(w, w), nn.MaxPool2d(2),
                         *block(w, 2 * w), *block(2 * w, 2 * w), nn.MaxPool2d(2),
                         *block(2 * w, 4 * w), nn.MaxPool2d(2),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, classes))


class ModularMLP(nn.Module):
    """Embed both operands, concatenate, and classify the sum with a one-hidden-layer MLP."""

    def __init__(self, p: int = 97, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(p, dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, p))

    def forward(self, x):
        return self.mlp(self.embed(x).flatten(1))


def modular_mlp(p: int = 97, dim: int = 128, hidden: int = 256) -> nn.Module:
    return ModularMLP(p, dim, hidden)
