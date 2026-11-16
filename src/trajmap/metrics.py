"""Trajectory metric plugin interface.

Metrics compare a branch with the trunk in FULL parameter space, never in the
2D projection.  A metric returns one value per branch snapshot for each named
series; results are saved under ``branches/<name>/metrics/`` and plotted by
the dashboard.

The two built-in metrics are reference implementations for validating the
pipeline, not research-grade diagnostics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .store import Run, Trajectory, atomic_write_json

_METRICS: dict[str, type["TrajectoryMetric"]] = {}


class TrajectoryMetric(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def compute(self, branch: Trajectory, trunk: Trajectory) -> dict[str, np.ndarray]:
        """Return ``{series_name: array}`` with one entry per branch snapshot (``branch.steps()`` order)."""


def register_metric(name: str) -> Callable[[type[TrajectoryMetric]], type[TrajectoryMetric]]:
    def deco(cls: type[TrajectoryMetric]) -> type[TrajectoryMetric]:
        if not issubclass(cls, TrajectoryMetric):
            raise TypeError(f"{cls.__name__} must subclass TrajectoryMetric")
        if name in _METRICS and _METRICS[name] is not cls:
            raise ValueError(f"metric {name!r} is already registered")
        cls.name = name
        _METRICS[name] = cls
        return cls
    return deco


def get_metric(name: str) -> type[TrajectoryMetric]:
    if name not in _METRICS:
        raise KeyError(f"unknown metric {name!r}; registered: {sorted(_METRICS)}")
    return _METRICS[name]


def list_metrics() -> list[str]:
    return sorted(_METRICS)


@register_metric("nearest_trunk")
class NearestTrunkSnapshot(TrajectoryMetric):
    """Rewind check: which stored trunk snapshot is each branch point closest to (L2)?"""

    description = ("Nearest trunk snapshot per branch step. If unlearning retraces training, "
                   "nearest_trunk_step moves back from the branch point and distance stays small.")

    def compute(self, branch, trunk):
        B = branch.matrix().astype(np.float64)
        best = np.full(len(B), np.inf)
        best_step = np.zeros(len(B))
        best_idx = np.zeros(len(B))
        for k, (step, w) in enumerate(trunk.iter_params()):
            d = np.linalg.norm(B - w, axis=1)
            closer = d < best
            best[closer], best_step[closer], best_idx[closer] = d[closer], step, k
        return {"nearest_trunk_step": best_step, "nearest_trunk_index": best_idx,
                "distance": best, "rewind_steps": branch.from_step - best_step}


@register_metric("off_path")
class OffPathDistance(TrajectoryMetric):
    """Minimum L2 distance from each branch point to the trunk polyline (snapshots joined by segments)."""

    description = ("Distance to the closest point on the piecewise-linear trunk path. "
                   "Zero means the branch point lies on the recorded training path.")

    def compute(self, branch, trunk):
        B = branch.matrix().astype(np.float64)
        best = np.full(len(B), np.inf)
        best_pos = np.zeros(len(B))
        prev: tuple[int, np.ndarray] | None = None
        for step, w in trunk.iter_params():
            w = w.astype(np.float64)
            if prev is None:
                d, pos = np.linalg.norm(B - w, axis=1), np.full(len(B), float(step))
            else:
                p_step, p = prev
                v = w - p
                t = np.clip((B - p) @ v / max(float(v @ v), 1e-300), 0.0, 1.0)
                d = np.linalg.norm(B - (p + t[:, None] * v), axis=1)
                pos = p_step + t * (step - p_step)
            closer = d < best
            best[closer], best_pos[closer] = d[closer], pos[closer]
            prev = (step, w)
        return {"off_path_distance": best, "trunk_position": best_pos}


def compute_metrics(run: Run | str, branch: str, names: list[str] | None = None) -> dict[str, dict]:
    """Compute metrics for a branch against the trunk, save them, and return them."""
    run = Run(run) if not isinstance(run, Run) else run
    b, trunk = run.trajectory(branch), run.trunk
    if b.is_trunk:
        raise ValueError("metrics are computed for a branch relative to the trunk")
    out = {}
    for name in names or list_metrics():
        metric = get_metric(name)()
        series = metric.compute(b, trunk)
        steps = b.steps()
        for k, v in series.items():
            if len(v) != len(steps):
                raise ValueError(f"metric {name!r} series {k!r} has {len(v)} values for {len(steps)} branch snapshots")
        out[name] = {"metric": name, "description": metric.description, "branch": branch,
                     "from_step": b.from_step, "branch_steps": steps,
                     "series": {k: np.asarray(v, dtype=float).tolist() for k, v in series.items()}}
        atomic_write_json(b.dir / "metrics" / f"{name}.json", out[name])
    return out
