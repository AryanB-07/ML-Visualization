"""Unlearning plugin interface.

Implement :class:`UnlearningMethod`, register it, and :func:`run_unlearning`
records it as a branch of an existing run::

    @register_method("my_method")
    class MyMethod(UnlearningMethod):
        def setup(self, model, forget_loader, retain_loader, config): ...
        def step(self, model): ...; return {"loss": ...}
        def is_done(self): ...

The single built-in method, ``gradient_ascent``, exists to validate the
pipeline end to end.  It is not a serious unlearning method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .params import assign_buffers, assign_params
from .store import Run, Trajectory, resolve
from .tracer import Recorder

_METHODS: dict[str, type["UnlearningMethod"]] = {}


class UnlearningMethod(ABC):
    """One unlearning procedure.  ``step`` performs one update of ``model`` in place."""

    name: str = ""

    @abstractmethod
    def setup(self, model: nn.Module, forget_loader: Iterable[Any], retain_loader: Iterable[Any] | None,
              config: dict[str, Any]) -> None: ...

    @abstractmethod
    def step(self, model: nn.Module) -> dict[str, float]:
        """Update the model once; return scalars to log (``"loss"`` is shown in the dashboard)."""

    @abstractmethod
    def is_done(self) -> bool: ...


def register_method(name: str) -> Callable[[type[UnlearningMethod]], type[UnlearningMethod]]:
    def deco(cls: type[UnlearningMethod]) -> type[UnlearningMethod]:
        if not issubclass(cls, UnlearningMethod):
            raise TypeError(f"{cls.__name__} must subclass UnlearningMethod")
        if name in _METHODS and _METHODS[name] is not cls:
            raise ValueError(f"unlearning method {name!r} is already registered")
        cls.name = name
        _METHODS[name] = cls
        return cls
    return deco


def get_method(name: str) -> type[UnlearningMethod]:
    if name not in _METHODS:
        raise KeyError(f"unknown unlearning method {name!r}; registered: {sorted(_METHODS)}")
    return _METHODS[name]


def list_methods() -> list[str]:
    return sorted(_METHODS)


@register_method("gradient_ascent")
class GradientAscent(UnlearningMethod):
    """Plain SGD ascent on the forget-set loss.  Config: ``lr`` (0.01), ``steps`` (40)."""

    def setup(self, model, forget_loader, retain_loader, config):
        self.loader, self.it = forget_loader, iter(forget_loader)
        self.opt = torch.optim.SGD(model.parameters(), lr=config.get("lr", 0.01))
        self.steps, self.done = config.get("steps", 40), 0

    def step(self, model):
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            batch = next(self.it)
        dev = next(model.parameters()).device
        loss = torch.nn.functional.cross_entropy(model(batch[0].to(dev)), batch[1].to(dev))
        self.opt.zero_grad()
        (-loss).backward()
        self.opt.step()
        self.done += 1
        return {"loss": loss.item()}

    def is_done(self):
        return self.done >= self.steps


def loaders_from_run(run: Run, batch_size: int = 64, seed: int = 0) -> tuple[DataLoader, DataLoader | None]:
    """Forget and retain loaders built from the run's ``dataset_factory`` and declared forget set."""
    meta = run.meta
    if not meta.get("dataset_factory"):
        raise RuntimeError("run has no dataset_factory; pass forget_loader/retain_loader to run_unlearning yourself")
    forget = run.forget_indices()
    if len(forget) == 0:
        raise RuntimeError("run declares no forget set (Tracer(forget_indices=...))")
    ds = resolve(meta["dataset_factory"])(**(meta.get("dataset_kwargs") or {}))
    retain = np.setdiff1d(np.arange(len(ds)), forget)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(Subset(ds, forget.tolist()), batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(Subset(ds, retain.tolist()), batch_size=batch_size, shuffle=True, generator=g))


def run_unlearning(run: Run | str, method: str, from_step: int,
                   forget_loader: Iterable[Any] | None = None, retain_loader: Iterable[Any] | None = None,
                   config: dict[str, Any] | None = None, branch: str | None = None,
                   every_n_steps: int = 1, max_steps: int = 100_000, device: str = "cpu",
                   metrics: bool = True, landscape: bool = True) -> Trajectory:
    """Run a registered method from a trunk snapshot and record it as a branch.

    The branch starts with the trunk snapshot at ``from_step`` (branch step 0),
    then stores a snapshot every ``every_n_steps`` method steps plus the final
    one.  With ``metrics=True`` all registered trajectory metrics are computed
    and saved when the method finishes.
    """
    run = Run(run) if not isinstance(run, Run) else run
    config = dict(config or {})
    if forget_loader is None:
        forget_loader, auto_retain = loaders_from_run(run, batch_size=config.get("batch_size", 64))
        retain_loader = retain_loader or auto_retain
    if branch is None:
        base, k, branch = f"{method}-from{from_step}", 1, f"{method}-from{from_step}"
        while branch in run.branch_names():
            k += 1
            branch = f"{base}-{k}"

    m = get_method(method)()
    snap = run.trunk.load(from_step)  # raises early if the snapshot is missing
    model = run.build_model()
    assign_params(model, snap.params)
    assign_buffers(model, snap.buffers, snap.int_buffers)
    model.to(device).train()

    writer = run.create_branch(branch, from_step, method=method, config=config, every_n_steps=every_n_steps)
    rec = Recorder(model, writer, every_n_steps, initial_snapshot=False)
    m.setup(model, forget_loader, retain_loader, config)
    while not m.is_done() and rec.step_count < max_steps:
        out = dict(m.step(model))
        rec.step(loss=out.pop("loss", None), **out)
    rec.finish()

    if metrics:
        from .metrics import compute_metrics
        compute_metrics(run, branch)
    if landscape:
        from .worker import ensure_worker
        ensure_worker(run.dir)
    return run.trajectory(branch)
