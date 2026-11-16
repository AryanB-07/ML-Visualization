"""On-disk layout of a run.

    run_dir/
      meta.json               config, model class/factory, param + buffer spec, seed, eval spec
      model.pkl               pickled model (fallback when no model_factory is given)
      eval_data.safetensors   the fixed eval subset (x, y)
      forget_indices.npy      optional forget set (dataset indices)
      control.json            dashboard -> worker settings
      trunk/                  the training trajectory
      branches/<name>/        one directory per branch, same format as trunk/
      bases/<mode>/vNNNN.*    versioned projection bases (see basis.py)
      landscape/<basis-id>/   cached grid evaluations (see landscape.py)

A trajectory directory (trunk or branch) contains:

      trajectory.json         name, parent, from_step, method, config, finished, ...
      snapshots.json          index: [{step, file, loss, time}, ...]
      snapshots/step_*.safetensors   keys: params, buffers, int_buffers
      steps.jsonl             per-step log: step, loss, n, n_forget, idx
      deltas/step_*.safetensors      optional per-step updates for forget steps

Trunk and branches are read through the same :class:`Trajectory` class, so all
projection, landscape and metric code treats them uniformly.  Every write is
atomic (temp file + rename) or append-only, because the worker and dashboard
read while training writes.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .params import assign_buffers, assign_params

TRUNK = "trunk"


# ----------------------------------------------------------------- file helpers
def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic_save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an append-only JSONL file, ignoring a partially written last line."""
    out: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except FileNotFoundError:
        pass
    return out


def resolve(spec: str) -> Any:
    """Resolve an import path of the form ``"package.module:attribute"``."""
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attribute', got {spec!r}")
    return getattr(importlib.import_module(mod), attr)


# ------------------------------------------------------------------ trajectory
@dataclass
class Snapshot:
    step: int
    params: torch.Tensor  # float32, flat
    buffers: torch.Tensor  # float32, flat (may be empty)
    int_buffers: torch.Tensor  # int64, flat (may be empty)


class Trajectory:
    """Read access to one trajectory: the trunk or a branch."""

    def __init__(self, run_dir: Path | str, name: str = TRUNK):
        self.run_dir = Path(run_dir)
        self.name = name
        self.dir = self.run_dir / TRUNK if name == TRUNK else self.run_dir / "branches" / name

    # -- metadata
    @property
    def is_trunk(self) -> bool:
        return self.name == TRUNK

    def info(self) -> dict[str, Any]:
        return read_json(self.dir / "trajectory.json", {}) or {}

    @property
    def from_step(self) -> int:
        """Trunk step this trajectory started from (0 for the trunk itself)."""
        return int(self.info().get("from_step", 0))

    @property
    def finished(self) -> bool:
        return bool(self.info().get("finished", False))

    # -- snapshots
    def records(self) -> list[dict[str, Any]]:
        return read_json(self.dir / "snapshots.json", []) or []

    def steps(self) -> list[int]:
        return [int(r["step"]) for r in self.records()]

    def _file(self, step: int) -> Path:
        return self.dir / "snapshots" / f"step_{step:09d}.safetensors"

    def load(self, step: int) -> Snapshot:
        t = load_file(str(self._file(step)))
        return Snapshot(
            step=step,
            params=t["params"].to(torch.float32),
            buffers=t.get("buffers", torch.zeros(0)).to(torch.float32),
            int_buffers=t.get("int_buffers", torch.zeros(0, dtype=torch.int64)),
        )

    def load_params(self, step: int) -> np.ndarray:
        return load_file(str(self._file(step)))["params"].to(torch.float32).numpy()

    def iter_params(self, steps: list[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        for s in self.steps() if steps is None else steps:
            yield s, self.load_params(s)

    def matrix(self, steps: list[int] | None = None) -> np.ndarray:
        """Snapshots stacked as an [N, D] float32 array (loads them all into memory)."""
        rows = [w for _, w in self.iter_params(steps)]
        return np.stack(rows) if rows else np.zeros((0, 0), np.float32)

    # -- logs
    def step_log(self) -> list[dict[str, Any]]:
        return read_jsonl(self.dir / "steps.jsonl")

    def delta_steps(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in (self.dir / "deltas").glob("step_*.safetensors"))

    def load_delta(self, step: int) -> np.ndarray:
        t = load_file(str(self.dir / "deltas" / f"step_{step:09d}.safetensors"))
        return t["delta"].to(torch.float32).numpy()

    def metrics(self) -> dict[str, dict[str, Any]]:
        """Saved trajectory metrics, keyed by metric name."""
        out = {}
        for p in sorted((self.dir / "metrics").glob("*.json")):
            d = read_json(p)
            if d:
                out[p.stem] = d
        return out

    def __repr__(self) -> str:
        return f"Trajectory({self.name!r}, {len(self.records())} snapshots)"


class TrajectoryWriter(Trajectory):
    """Write access.  One writer per trajectory at a time."""

    def __init__(self, run_dir: Path | str, name: str = TRUNK, dtype: str = "float32"):
        super().__init__(run_dir, name)
        if dtype not in ("float32", "float16"):
            raise ValueError("dtype must be 'float32' or 'float16'")
        self.dtype = getattr(torch, dtype)
        self._records: list[dict[str, Any]] = self.records()
        self._log_buf: list[str] = []
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)

    def set_info(self, **kw: Any) -> None:
        info = self.info()
        info.update(kw)
        atomic_write_json(self.dir / "trajectory.json", info)

    def save_snapshot(self, step: int, params: torch.Tensor, buffers: torch.Tensor | None = None,
                      int_buffers: torch.Tensor | None = None, loss: float | None = None) -> None:
        tensors = {"params": params.to(self.dtype)}
        if buffers is not None and buffers.numel():
            tensors["buffers"] = buffers.to(torch.float32)
        if int_buffers is not None and int_buffers.numel():
            tensors["int_buffers"] = int_buffers.to(torch.int64)
        atomic_save_tensors(self._file(step), tensors)
        self._records = [r for r in self._records if r["step"] != step]
        self._records.append({"step": int(step), "loss": loss, "time": time.time()})
        self._records.sort(key=lambda r: r["step"])
        atomic_write_json(self.dir / "snapshots.json", self._records)
        self.flush()

    def thin(self) -> None:
        """Drop every other intermediate snapshot (first and last are always kept)."""
        keep = [self._records[0]] + self._records[1:-1][1::2] + [self._records[-1]]
        kept = {r["step"] for r in keep}
        drop = [r["step"] for r in self._records if r["step"] not in kept]
        self._records = keep
        atomic_write_json(self.dir / "snapshots.json", self._records)  # index first, then files
        for s in drop:
            self._file(s).unlink(missing_ok=True)

    def log_step(self, rec: dict[str, Any]) -> None:
        self._log_buf.append(json.dumps(rec))
        if len(self._log_buf) >= 200:
            self.flush()

    def flush(self) -> None:
        if self._log_buf:
            with open(self.dir / "steps.jsonl", "a") as f:
                f.write("\n".join(self._log_buf) + "\n")
            self._log_buf = []

    def save_delta(self, step: int, delta: torch.Tensor) -> None:
        atomic_save_tensors(self.dir / "deltas" / f"step_{step:09d}.safetensors", {"delta": delta.to(self.dtype)})


# ------------------------------------------------------------------------- run
class Run:
    """A training trunk plus zero or more named branches, in one directory."""

    def __init__(self, run_dir: Path | str):
        self.dir = Path(run_dir)
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"{self.dir} is not a run directory (no meta.json)")

    @staticmethod
    def create(run_dir: Path | str, meta: dict[str, Any], overwrite: bool = False) -> "Run":
        d = Path(run_dir)
        if (d / "meta.json").exists():
            if not overwrite:
                raise FileExistsError(f"{d} already holds a run; pass overwrite=True to replace it")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_json(d / "meta.json", meta)
        return Run(d)

    @property
    def meta(self) -> dict[str, Any]:
        return read_json(self.dir / "meta.json", {})

    # -- trajectories
    @property
    def trunk(self) -> Trajectory:
        return Trajectory(self.dir, TRUNK)

    def branch_names(self) -> list[str]:
        root = self.dir / "branches"
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if (p / "trajectory.json").exists())

    def trajectory(self, name: str) -> Trajectory:
        if name != TRUNK and name not in self.branch_names():
            raise KeyError(f"no branch {name!r}; have {self.branch_names()}")
        return Trajectory(self.dir, name)

    def branches(self) -> list[Trajectory]:
        return [Trajectory(self.dir, n) for n in self.branch_names()]

    def trajectories(self, branches: bool = True) -> list[Trajectory]:
        return [self.trunk] + (self.branches() if branches else [])

    def create_branch(self, name: str, from_step: int, method: str | None = None,
                      config: dict[str, Any] | None = None, every_n_steps: int = 1) -> TrajectoryWriter:
        """Start a branch at a trunk snapshot.  The snapshot becomes branch step 0."""
        if name == TRUNK or not name or "/" in name:
            raise ValueError(f"invalid branch name {name!r}")
        if name in self.branch_names():
            raise FileExistsError(f"branch {name!r} already exists")
        steps = self.trunk.steps()
        if from_step not in steps:
            raise ValueError(f"trunk has no snapshot at step {from_step}; available: {steps}")
        snap = self.trunk.load(from_step)
        w = TrajectoryWriter(self.dir, name, dtype=self.meta.get("snapshot_dtype", "float32"))
        w.set_info(name=name, parent=TRUNK, from_step=int(from_step), method=method, config=config or {},
                   every_n_steps=every_n_steps, finished=False, created=time.time())
        loss = next((r.get("loss") for r in self.trunk.records() if r["step"] == from_step), None)
        w.save_snapshot(0, snap.params, snap.buffers, snap.int_buffers, loss=loss)
        return w

    # -- reconstruction
    def build_model(self) -> nn.Module:
        """A fresh instance of the recorded model (weights are whatever the factory gives)."""
        m = self.meta
        if m.get("model_factory"):
            return resolve(m["model_factory"])(**(m.get("model_kwargs") or {}))
        pkl = self.dir / "model.pkl"
        if pkl.exists():
            return torch.load(pkl, map_location="cpu", weights_only=False)
        raise RuntimeError(
            "this run cannot rebuild its model: pass model_factory='module:fn' to Tracer, "
            "or define the model class in an importable module so it can be pickled")

    def load_model(self, trajectory: str = TRUNK, step: int | None = None) -> nn.Module:
        """The exact model recorded at a snapshot (last snapshot if step is None)."""
        t = self.trajectory(trajectory)
        snap = t.load(t.steps()[-1] if step is None else step)
        model = self.build_model()
        assign_params(model, snap.params)
        assign_buffers(model, snap.buffers, snap.int_buffers)
        return model

    def eval_data(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        p = self.dir / "eval_data.safetensors"
        if not p.exists():
            return None
        t = load_file(str(p))
        return t["x"], t["y"]

    def loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        name = self.meta.get("loss", "cross_entropy")
        if name == "cross_entropy":
            return torch.nn.functional.cross_entropy
        if name == "mse":
            return torch.nn.functional.mse_loss
        return resolve(name)

    def forget_indices(self) -> np.ndarray:
        p = self.dir / "forget_indices.npy"
        return np.load(p) if p.exists() else np.zeros(0, dtype=np.int64)

    # -- dashboard <-> worker
    def control(self) -> dict[str, Any]:
        from .landscape import DEFAULT_CONTROL
        return {**DEFAULT_CONTROL, **(self.meta.get("landscape") or {}), **(read_json(self.dir / "control.json", {}) or {})}

    def set_control(self, **kw: Any) -> dict[str, Any]:
        cur = read_json(self.dir / "control.json", {}) or {}
        cur.update(kw)
        atomic_write_json(self.dir / "control.json", cur)
        return self.control()
