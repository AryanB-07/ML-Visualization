"""Flatten a model's parameters and buffers into vectors and write them back.

Geometry (PCA, metrics, grid points) lives in *parameter* space: every tensor
in ``model.named_parameters()``, concatenated in iteration order.  Buffers
(BatchNorm running statistics and counters) are stored alongside each snapshot
so that a snapshot reconstructs the model exactly, but they are not part of the
coordinate system.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def param_spec(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": n, "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", "")}
        for n, p in model.named_parameters()
    ]


def buffer_spec(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": n, "shape": list(b.shape), "dtype": str(b.dtype).replace("torch.", ""),
         "float": bool(b.is_floating_point())}
        for n, b in model.named_buffers()
    ]


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def flatten_params(model: nn.Module) -> torch.Tensor:
    """All parameters as one float32 CPU vector (a copy)."""
    parts = [p.detach().reshape(-1).to("cpu", torch.float32) for p in model.parameters()]
    return torch.cat(parts) if parts else torch.zeros(0)


@torch.no_grad()
def flatten_buffers(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Float buffers as float32 and integer/bool buffers as int64, both flattened."""
    fl, it = [], []
    for b in model.buffers():
        (fl if b.is_floating_point() else it).append(
            b.detach().reshape(-1).to("cpu", torch.float32 if b.is_floating_point() else torch.int64)
        )
    return (torch.cat(fl) if fl else torch.zeros(0), torch.cat(it) if it else torch.zeros(0, dtype=torch.int64))


@torch.no_grad()
def assign_params(model: nn.Module, vec: torch.Tensor) -> None:
    vec = torch.as_tensor(vec)
    n = num_params(model)
    if vec.numel() != n:
        raise ValueError(f"parameter vector has {vec.numel()} entries, model expects {n}")
    i = 0
    for p in model.parameters():
        k = p.numel()
        p.copy_(vec[i:i + k].reshape(p.shape).to(p.device, p.dtype))
        i += k


@torch.no_grad()
def assign_buffers(model: nn.Module, fvec: torch.Tensor | None, ivec: torch.Tensor | None) -> None:
    fi = ii = 0
    for b in model.buffers():
        k = b.numel()
        if b.is_floating_point():
            if fvec is None or fvec.numel() == 0:
                continue
            b.copy_(torch.as_tensor(fvec)[fi:fi + k].reshape(b.shape).to(b.device, b.dtype))
            fi += k
        else:
            if ivec is None or ivec.numel() == 0:
                continue
            b.copy_(torch.as_tensor(ivec)[ii:ii + k].reshape(b.shape).to(b.device, b.dtype))
            ii += k
