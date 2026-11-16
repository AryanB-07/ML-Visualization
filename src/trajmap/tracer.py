"""Recording hooks.

``Recorder`` writes any trajectory (trunk or branch); ``Tracer`` is the
user-facing recorder for the training trunk::

    tracer = Tracer(model, run_dir="runs/exp1", eval_loader=val_loader, every_n_steps=50)
    for x, y, idx in loader:
        ...
        optimizer.step()
        tracer.step(loss=loss.item(), batch_indices=idx)
    tracer.finish()

Call ``step`` *after* ``optimizer.step()``.  ``finish`` (also run at interpreter
exit and on ``with`` exit) stores the final weights.
"""
from __future__ import annotations

import atexit
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .params import buffer_spec, flatten_buffers, flatten_params, num_params, param_spec
from .store import Run, TrajectoryWriter, atomic_save_tensors


def _to_index_list(idx: Any) -> list[int]:
    if idx is None:
        return []
    if isinstance(idx, torch.Tensor):
        return idx.detach().reshape(-1).tolist()
    return [int(i) for i in np.asarray(idx).reshape(-1)]


class Recorder:
    """Snapshot a model along one trajectory and log its steps."""

    def __init__(self, model: nn.Module, writer: TrajectoryWriter, every_n_steps: int = 50, *,
                 forget_indices: Iterable[int] | None = None, max_snapshots: int | None = None,
                 save_forget_deltas: bool = False, log_batch_indices: bool = True,
                 initial_snapshot: bool = True):
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        if max_snapshots is not None and max_snapshots < 3:
            raise ValueError("max_snapshots must be >= 3 (init, one intermediate, final)")
        self.model = model
        self.writer = writer
        self.every_n_steps = every_n_steps
        self.max_snapshots = max_snapshots
        self.save_forget_deltas = save_forget_deltas
        self.log_batch_indices = log_batch_indices
        self.forget: frozenset[int] = frozenset(int(i) for i in (forget_indices if forget_indices is not None else []))
        self.step_count = 0
        self._last_loss: float | None = None
        self._last_saved = -1
        self._finished = False
        self._prev: torch.Tensor | None = flatten_params(model) if save_forget_deltas else None
        if initial_snapshot:
            self._snapshot(None)
        else:
            self._last_saved = 0

    def _snapshot(self, loss: float | None) -> None:
        fb, ib = flatten_buffers(self.model)
        self.writer.save_snapshot(self.step_count, flatten_params(self.model), fb, ib, loss=loss)
        self._last_saved = self.step_count
        if self.max_snapshots and len(self.writer._records) > self.max_snapshots:
            self.writer.thin()
            self.every_n_steps *= 2
            self.writer.set_info(every_n_steps=self.every_n_steps)

    def step(self, loss: float | None = None, batch_indices: Any = None, **metrics: float) -> None:
        """Record one optimizer step.  Call after the weights were updated."""
        self.step_count += 1
        idx = _to_index_list(batch_indices)
        n_forget = sum(1 for i in idx if i in self.forget) if self.forget else 0
        rec: dict[str, Any] = {"step": self.step_count, "loss": loss, "n": len(idx), "n_forget": n_forget}
        if self.log_batch_indices and idx:
            rec["idx"] = idx
        rec.update(metrics)
        self.writer.log_step(rec)
        self._last_loss = loss

        if self._prev is not None:
            cur = flatten_params(self.model)
            if n_forget > 0:
                self.writer.save_delta(self.step_count, cur - self._prev)
            self._prev = cur

        if self.step_count % self.every_n_steps == 0:
            self._snapshot(loss)

    def finish(self) -> None:
        """Store the final weights and mark the trajectory finished.  Idempotent."""
        if self._finished:
            return
        self._finished = True
        if self._last_saved != self.step_count:
            self._snapshot(self._last_loss)
        self.writer.flush()
        self.writer.set_info(finished=True, last_step=self.step_count, finished_time=time.time())

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()


class Tracer(Recorder):
    """Record the training trunk of a new run.

    Args:
        model: the model being trained (any ``nn.Module``).
        run_dir: where the run is stored.
        eval_loader: yields ``(x, y)`` batches.  The first ``eval_max_samples``
            samples are copied into the run directory and used for every
            landscape evaluation, so the run stays explorable later.
        every_n_steps: snapshot interval.  Init and final are always saved.
        forget_indices: dataset indices of the forget set, declared up front.
        dtype: ``"float32"`` (exact) or ``"float16"`` (half the disk, lossy).
        max_snapshots: cap on stored snapshots; when exceeded, every other
            intermediate snapshot is dropped and the interval doubles.
        save_forget_deltas: also store the parameter update of every step whose
            batch contains a forget sample (off by default; see README for cost).
        log_batch_indices: store the dataset indices of every batch in steps.jsonl.
        config, seed: free-form experiment config and the RNG seed, saved to meta.json.
        model_factory, model_kwargs: ``"module:callable"`` that rebuilds the
            architecture.  Without it the model object is pickled, which needs
            its class to live in an importable module.
        dataset_factory, dataset_kwargs: ``"module:callable"`` returning the
            training dataset; lets ``tracer unlearn`` build forget/retain loaders.
        loss: ``"cross_entropy"``, ``"mse"`` or ``"module:callable"`` taking (output, target).
        landscape: start the background landscape worker.
        landscape_options: initial worker settings (see ``landscape.DEFAULT_CONTROL``).
        worker_device: ``"cpu"`` (default), ``"cuda"``, ``"mps"`` or ``"auto"``.
    """

    def __init__(self, model: nn.Module, run_dir: str | Path, eval_loader: Iterable[Any] | None = None,
                 every_n_steps: int = 50, *, forget_indices: Sequence[int] | None = None,
                 dtype: str = "float32", max_snapshots: int | None = None,
                 save_forget_deltas: bool = False, log_batch_indices: bool = True,
                 config: dict[str, Any] | None = None, seed: int | None = None,
                 model_factory: str | None = None, model_kwargs: dict[str, Any] | None = None,
                 dataset_factory: str | None = None, dataset_kwargs: dict[str, Any] | None = None,
                 loss: str = "cross_entropy", eval_max_samples: int = 2048, eval_batch_size: int = 512,
                 landscape: bool = True, landscape_options: dict[str, Any] | None = None,
                 worker_device: str = "cpu", overwrite: bool = False):
        cls = type(model)
        meta = {
            "format_version": 1,
            "created": time.time(),
            "config": config or {},
            "seed": seed,
            "model_class": f"{cls.__module__}:{cls.__qualname__}",
            "model_factory": model_factory,
            "model_kwargs": model_kwargs or {},
            "dataset_factory": dataset_factory,
            "dataset_kwargs": dataset_kwargs or {},
            "num_params": num_params(model),
            "param_spec": param_spec(model),
            "buffer_spec": buffer_spec(model),
            "snapshot_dtype": dtype,
            "loss": loss,
            "eval_batch_size": eval_batch_size,
            "save_forget_deltas": save_forget_deltas,
            "n_forget": len(forget_indices) if forget_indices is not None else 0,
            "landscape": landscape_options or {},
            "worker_device": worker_device,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        self.run = Run.create(run_dir, meta, overwrite=overwrite)

        if model_factory is None:
            try:
                torch.save(model, self.run.dir / "model.pkl")
            except Exception as e:  # e.g. a class defined inside a function
                print(f"[trajmap] could not pickle the model ({e}); pass model_factory= "
                      "to make this run explorable after training")
        if forget_indices is not None:
            np.save(self.run.dir / "forget_indices.npy", np.asarray(list(forget_indices), dtype=np.int64))
        if eval_loader is not None:
            self._save_eval_data(eval_loader, eval_max_samples)

        writer = TrajectoryWriter(self.run.dir, dtype=dtype)
        writer.set_info(name="trunk", parent=None, from_step=0, every_n_steps=every_n_steps, finished=False)
        super().__init__(model, writer, every_n_steps, forget_indices=forget_indices,
                         max_snapshots=max_snapshots, save_forget_deltas=save_forget_deltas,
                         log_batch_indices=log_batch_indices)
        atexit.register(self.finish)
        if landscape:
            from .worker import ensure_worker
            ensure_worker(self.run.dir, device=worker_device)

    def _save_eval_data(self, loader: Iterable[Any], max_samples: int) -> None:
        xs, ys, n = [], [], 0
        for batch in loader:
            x, y = batch[0], batch[1]
            xs.append(x.detach().cpu())
            ys.append(y.detach().cpu())
            n += len(x)
            if n >= max_samples:
                break
        if not xs:
            return
        atomic_save_tensors(self.run.dir / "eval_data.safetensors",
                            {"x": torch.cat(xs)[:max_samples], "y": torch.cat(ys)[:max_samples]})
