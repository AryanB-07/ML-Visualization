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
