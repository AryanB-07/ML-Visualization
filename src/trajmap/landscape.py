"""Loss evaluation on the 2D plane: evaluator, on-disk cache, refinement, interpolation.

Grid points live on an integer lattice.  A basis fixes ``coarse_h`` (the coarse
spacing); lattice key ``(i, j)`` is the point ``a = i*u, b = j*u`` with
``u = coarse_h / 2**MAX_LEVEL``.  Level 0 is the coarse grid, each further
level halves the spacing.  Because keys are absolute, the grid can grow with
the trajectory and change resolution without re-evaluating anything.

BatchNorm.  A grid point defines parameters only; BatchNorm running statistics
are not part of the plane.  Two modes:

    recalibrate (default)  reset the running statistics and re-estimate them
                           with one extra forward pass over the eval subset
                           (cumulative average), then evaluate in eval mode.
                           This is what the loss at those weights would be
                           with matched statistics; it doubles the cost.
    center                 reuse the statistics stored with the basis center.
                           Cheap, but increasingly wrong away from the center.

Models without BatchNorm are unaffected.  Dropout is always off (eval mode).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from torch import nn

from .basis import MARGIN, Basis
from .params import assign_buffers, assign_params
from .store import Run, atomic_write_json, read_json, read_jsonl

MAX_LEVEL = 3
RESOLUTIONS = {9: 0, 17: 1, 33: 2, 65: 3}  # nominal points per axis -> refinement level

DEFAULT_CONTROL: dict[str, Any] = {
    "mode": "pca_trunk",      # basis mode shown / evaluated
    "resolution": 33,          # finest grid, nominal points per axis (9, 17, 33, 65)
    "eval_batches": 2,         # eval batches per grid point (0 = the whole eval subset)
    "budget": 600,             # max evaluated grid points per basis version
    "refit_every": 5,          # refit the basis every K new snapshots
    "bn_mode": "recalibrate",  # see module docstring
}


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# -------------------------------------------------------------------- evaluator
class Evaluator:
    """Loads a run's model and eval subset once, then evaluates weight vectors."""

    def __init__(self, run: Run, device: str = "cpu"):
        data = run.eval_data()
        if data is None:
            raise RuntimeError("run has no eval data (Tracer was created without eval_loader)")
        self.device = pick_device(device)
        self.model = run.build_model().to(self.device).eval()
        self.x, self.y = data[0].to(self.device), data[1].to(self.device)
        self.loss_fn = run.loss_fn()
        self.batch_size = int(run.meta.get("eval_batch_size", 512))
        self.bn_layers = [m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def _batches(self, eval_batches: int | None):
        n = len(self.x) if not eval_batches else min(len(self.x), eval_batches * self.batch_size)
        for i in range(0, n, self.batch_size):
            yield self.x[i:min(i + self.batch_size, n)], self.y[i:min(i + self.batch_size, n)]

    @torch.no_grad()
    def _recalibrate(self, eval_batches: int | None) -> None:
        momenta = [m.momentum for m in self.bn_layers]
        for m in self.bn_layers:
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average
            m.train()
        for x, _ in self._batches(eval_batches):
            self.model(x)
        for m, mom in zip(self.bn_layers, momenta):
            m.momentum = mom
            m.eval()

    @torch.no_grad()
    def evaluate(self, params: np.ndarray | torch.Tensor, buffers: torch.Tensor | None = None,
                 int_buffers: torch.Tensor | None = None, eval_batches: int | None = None,
                 bn_mode: str = "stored") -> dict[str, Any]:
        """Loss (and accuracy for classification) of the given flat parameters.

        ``bn_mode="stored"`` uses ``buffers`` as given (exact snapshot evaluation);
        ``"center"`` is the same thing with the basis center's buffers;
        ``"recalibrate"`` re-estimates BatchNorm statistics first.
        """
        self.model.eval()
        assign_params(self.model, torch.as_tensor(params))
        assign_buffers(self.model, buffers, int_buffers)
        if bn_mode == "recalibrate" and self.bn_layers:
            self._recalibrate(eval_batches)
        tot, correct, n, has_acc = 0.0, 0, 0, False
        for x, y in self._batches(eval_batches):
            out = self.model(x)
            tot += float(self.loss_fn(out, y)) * len(x)
            if out.ndim == 2 and y.ndim == 1 and not y.is_floating_point():
                correct += int((out.argmax(1) == y).sum())
                has_acc = True
            n += len(x)
        loss = tot / max(n, 1)
        return {"loss": loss if np.isfinite(loss) else None, "acc": correct / n if has_acc and n else None, "n": n}

    def state_dict_at(self, params: np.ndarray, buffers: torch.Tensor | None, int_buffers: torch.Tensor | None,
                      eval_batches: int | None = None, bn_mode: str = "stored") -> dict[str, torch.Tensor]:
        """state_dict for the given parameters, with buffers handled as in ``evaluate``."""
        self.evaluate(params, buffers, int_buffers, eval_batches=eval_batches, bn_mode=bn_mode)
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


# ------------------------------------------------------------------------ cache
class GridCache:
    """Append-only cache of evaluated grid points for one basis version.

    The cache is bound to a basis id and to the evaluation settings.  Opening it
    with a different id or different settings discards what is on disk, so a
    stale point can never be mixed into a new surface.
    """

    def __init__(self, run_dir: Path | str, basis_id: str, settings: dict[str, Any] | None = None):
        self.dir = Path(run_dir) / "landscape" / basis_id
        self.basis_id = basis_id
        self.settings = settings or {}
        meta = read_json(self.dir / "meta.json")
        if meta is None or meta.get("basis_id") != basis_id or meta.get("settings") != self.settings:
            shutil.rmtree(self.dir, ignore_errors=True)
            atomic_write_json(self.dir / "meta.json", {"basis_id": basis_id, "settings": self.settings})
        self.points: dict[tuple[int, int], tuple[float | None, float | None]] = {
            (r["i"], r["j"]): (r["loss"], r.get("acc")) for r in read_jsonl(self.dir / "points.jsonl")}

    def add(self, key: tuple[int, int], loss: float | None, acc: float | None) -> None:
        self.points[key] = (loss, acc)
        with open(self.dir / "points.jsonl", "a") as f:
            f.write(json.dumps({"i": key[0], "j": key[1], "loss": loss, "acc": acc}) + "\n")

    @staticmethod
    def read(run_dir: Path | str, basis_id: str) -> dict[tuple[int, int], tuple[float | None, float | None]]:
        """Read-only view for the dashboard (never clears anything)."""
        d = Path(run_dir) / "landscape" / basis_id
        meta = read_json(d / "meta.json")
        if meta is None or meta.get("basis_id") != basis_id:
            return {}
        return {(r["i"], r["j"]): (r["loss"], r.get("acc")) for r in read_jsonl(d / "points.jsonl")}


# ------------------------------------------------------------------- refinement
def unit(coarse_h: float) -> float:
    return coarse_h / 2 ** MAX_LEVEL


def grid_bounds(points_ab: np.ndarray, coarse_h: float, min_extent: float = 0.0) -> tuple[int, int, int, int]:
    """Bounding box of the trajectories plus a margin, snapped outward to coarse cells.

    Returns (i_min, i_max, j_min, j_max) in lattice units.
    """
    pts = np.vstack([points_ab, [[-min_extent, -min_extent], [min_extent, min_extent]]])
    lo, hi = pts.min(0), pts.max(0)
    span = hi - lo
    pad = np.maximum(MARGIN * span.max(), coarse_h)
    c = 2 ** MAX_LEVEL
    lo_k = np.floor((lo - pad) / coarse_h).astype(int) * c
    hi_k = np.ceil((hi + pad) / coarse_h).astype(int) * c
    return int(lo_k[0]), int(hi_k[0]), int(lo_k[1]), int(hi_k[1])


def polyline_distance(q: np.ndarray, polylines: list[np.ndarray]) -> np.ndarray:
    """Distance from each row of q [M, d] to the nearest point on any polyline [N_k, d]."""
    best = np.full(len(q), np.inf)
    for pl in polylines:
        if len(pl) == 0:
            continue
        if len(pl) == 1:
            best = np.minimum(best, np.linalg.norm(q - pl[0], axis=1))
            continue
        p0, v = pl[:-1], pl[1:] - pl[:-1]
        vv = np.maximum((v * v).sum(1), 1e-300)
        t = np.clip(((q[:, None, :] - p0[None]) * v[None]).sum(2) / vv[None], 0.0, 1.0)
        d = np.linalg.norm(q[:, None, :] - (p0[None] + t[..., None] * v[None]), axis=2)
        best = np.minimum(best, d.min(1))
    return best


def plan_points(polylines: list[np.ndarray], coarse_h: float, evaluated: set[tuple[int, int]],
                level: int, budget: int, min_extent: float = 0.0) -> list[tuple[int, int]]:
    """Grid points still to evaluate, in priority order, within the remaining budget.

    Order: the whole coarse grid first; then, level by level, the points close
    to a trajectory (a band that narrows as the level gets finer); then, if
    budget remains, the rest of each level outward from the trajectories.
    """
    remaining = budget - len(evaluated)
    pts = np.vstack([p for p in polylines if len(p)]) if any(len(p) for p in polylines) else np.zeros((1, 2))
    if remaining <= 0:
        return []
    i0, i1, j0, j1 = grid_bounds(pts, coarse_h, min_extent)
    u = unit(coarse_h)
    near: list[tuple[int, int]] = []
    far: list[tuple[int, int]] = []
    for lvl in range(0, min(level, MAX_LEVEL) + 1):
        s = 2 ** (MAX_LEVEL - lvl)
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1, s), np.arange(j0, j1 + 1, s), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        if lvl > 0:  # points on the coarser lattice belong to an earlier level
            fresh = (ii % (2 * s) != 0) | (jj % (2 * s) != 0)
            ii, jj = ii[fresh], jj[fresh]
        dist = polyline_distance(np.stack([ii * u, jj * u], 1), polylines)
        order = np.argsort(dist, kind="stable")
        band = np.inf if lvl == 0 else 2.0 * (s * 2 * u)  # two cells of the previous level
        for k in order:
            key = (int(ii[k]), int(jj[k]))
            if key not in evaluated:
                (near if dist[k] <= band else far).append(key)
    return (near + far)[:remaining]


# ---------------------------------------------------------------- interpolation
class SurfaceModel:
    """Interpolates the evaluated points (scipy thin-plate RBF, in log-loss when possible)."""

    def __init__(self, points: dict[tuple[int, int], tuple[float | None, float | None]], coarse_h: float,
                 cap: float | None = None):
        u = unit(coarse_h)
        keys = list(points)
        self.xy = np.array([[i * u, j * u] for i, j in keys], dtype=float).reshape(-1, 2)
        raw = np.array([np.nan if points[k][0] is None else points[k][0] for k in keys], dtype=float)
        finite = raw[np.isfinite(raw)]
        self.ok = len(finite) >= 4
        if not self.ok:
            return
        ceiling = float(finite.max()) if cap is None else float(min(cap, finite.max()))
        self.z = np.clip(np.where(np.isfinite(raw), raw, ceiling), None, ceiling)
        self.zmin, self.zmax = float(self.z.min()), float(self.z.max())
        self.log = self.zmin > 0
        vals = np.log(self.z) if self.log else self.z
        self.tree = cKDTree(self.xy)
        try:
            self.rbf = RBFInterpolator(self.xy, vals, neighbors=min(len(vals), 24), kernel="thin_plate_spline")
        except Exception:  # degenerate layouts: fall back to nearest neighbour
            self.rbf = None
            self._vals = vals

    def __call__(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1, 2)
        try:
            v = self.rbf(q) if self.rbf is not None else self._vals[self.tree.query(q)[1]]
        except Exception:
            v = (np.log(self.z) if self.log else self.z)[self.tree.query(q)[1]]
        z = np.exp(v) if self.log else v
        return np.clip(z, self.zmin, self.zmax)

    def nearest_distance(self, q: np.ndarray) -> np.ndarray:
        return self.tree.query(np.asarray(q, dtype=float).reshape(-1, 2))[0]

    def mesh(self, bounds: tuple[int, int, int, int], coarse_h: float, level: int,
             max_n: int = 97) -> dict[str, np.ndarray]:
        """Display mesh: axes, interpolated z, and a mask of where z is backed by a nearby evaluation."""
        u = unit(coarse_h)
        i0, i1, j0, j1 = bounds
        s = 2 ** (MAX_LEVEL - min(level, MAX_LEVEL))
        na = min(max_n, (i1 - i0) // s * 2 + 1)
        nb = min(max_n, (j1 - j0) // s * 2 + 1)
        a = np.linspace(i0 * u, i1 * u, na)
        b = np.linspace(j0 * u, j1 * u, nb)
        A, B = np.meshgrid(a, b)  # rows = b, cols = a (plotly's surface convention)
        q = np.stack([A.ravel(), B.ravel()], 1)
        z = self(q).reshape(A.shape)
        supported = (self.nearest_distance(q) <= 0.75 * s * u).reshape(A.shape)
        return {"a": a, "b": b, "z": z, "supported": supported}
