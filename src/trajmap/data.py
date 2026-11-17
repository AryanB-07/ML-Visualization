"""Example datasets and a wrapper that makes a dataset yield its indices."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset, TensorDataset


class IndexedDataset(Dataset):
    """Wrap a dataset so each item is ``(*item, index)``; pass the index batch to ``tracer.step``."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, i: int) -> tuple[Any, ...]:
        return (*self.base[i], i)


def mnist(root: str = "data", train: bool = True) -> TensorDataset:
    """MNIST as normalized in-memory tensors (fast on CPU).  Downloads on first use."""
    from torchvision.datasets import MNIST

    ds = MNIST(root, train=train, download=True)
    x = (ds.data.float() / 255.0 - 0.1307) / 0.3081
    return TensorDataset(x.unsqueeze(1), ds.targets.clone())


def cifar10(root: str = "data", train: bool = True) -> Dataset:
    """CIFAR-10 with per-channel normalization and no augmentation.  Downloads on first use."""
    from torchvision import transforms as T
    from torchvision.datasets import CIFAR10

    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    return CIFAR10(root, train=train, download=True, transform=tf)


def fashion_mnist(root: str = "data", train: bool = True) -> TensorDataset:
    """Fashion-MNIST as normalized in-memory tensors.  Downloads (about 30 MB) on first use."""
    from torchvision.datasets import FashionMNIST

    ds = FashionMNIST(root, train=train, download=True)
    x = (ds.data.float() / 255.0 - 0.2860) / 0.3530
    return TensorDataset(x.unsqueeze(1), ds.targets.clone())


class _NormalizedImages(Dataset):
    """uint8 images kept in memory (a quarter of float32), normalized per item."""

    def __init__(self, images: torch.Tensor, targets: torch.Tensor, mean: tuple, std: tuple):
        self.images, self.targets = images, targets
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.images[i].float() / 255.0 - self.mean) / self.std, self.targets[i]


def cifar10_fast(root: str = "data", train: bool = True, limit: int | None = None) -> Dataset:
    """CIFAR-10 held in memory as uint8, normalized, no augmentation.  Downloads (about 170 MB) on first use.

    ``limit`` keeps the first N images (a fixed subset), which keeps CPU demos short.
    """
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets)
    if limit:
        x, y = x[:limit], y[:limit]
    return _NormalizedImages(x, y, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def spirals(train: bool = True, n_per_class: int = 1000, arms: int = 3, noise: float = 0.3,
            seed: int = 0) -> TensorDataset:
    """Interleaved spiral arms in 2D, one class per arm.  Synthetic, no download.

    A classic hard toy problem: the classes are not separable by anything
    simple, so the network has to bend its decision boundary a lot and the
    loss surface around the path is rugged.
    """
    g = torch.Generator().manual_seed(seed + (0 if train else 10_000))
    xs, ys = [], []
    for k in range(arms):
        t = torch.rand(n_per_class, generator=g) * 3.0 + 0.25         # radius / turn
        theta = t * 4.0 + 2 * torch.pi * k / arms
        pts = torch.stack([t * torch.cos(theta), t * torch.sin(theta)], 1)
        xs.append(pts + noise * 0.4 * torch.randn(n_per_class, 2, generator=g))
        ys.append(torch.full((n_per_class,), k, dtype=torch.long))
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def modular_addition(train: bool = True, p: int = 97, train_fraction: float = 0.4, seed: int = 0) -> TensorDataset:
    """All pairs (a, b) labelled (a + b) mod p, split at random into train and test.  No download.

    With strong weight decay a small network first memorizes the train split
    (test loss stays high) and much later generalizes ("grokking"); the map
    shows the long walk between the two.
    """
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    x = torch.stack([a.flatten(), b.flatten()], 1)
    y = (x[:, 0] + x[:, 1]) % p
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    cut = int(train_fraction * len(x))
    idx = perm[:cut] if train else perm[cut:]
    return TensorDataset(x[idx], y[idx])
