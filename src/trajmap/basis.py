"""Projection bases: a shared 2D coordinate system for trunk and branches.

A basis is ``(center, d1, d2)`` in full parameter space.  Any weight vector
``w`` gets coordinates ``(a, b)`` by least-squares projection of ``w - center``
onto ``span(d1, d2)``, and a grid point is ``center + a*d1 + b*d2``.

Modes
    pca_trunk   top-2 principal directions of trunk snapshot displacements
                from the final trunk snapshot (latest one while training).
    pca_joint   same, fit on trunk and branch snapshots together.
    random      two filter-normalized random directions (Li et al., 2018).

Bases are versioned.  Fitting version N+1 deletes the cached landscape of
every older version of that mode, because cached (a, b) points only mean
something relative to the basis they were computed in.

The 2D picture hides everything orthogonal to the plane.  ``Basis.fidelity``
reports, per snapshot, the fraction of its squared displacement from the
center that lies in the plane (``captured``) and the norm of what does not
(``residual``).
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load_file as np_load_file

from .store import TRUNK, Run, atomic_save_tensors, atomic_write_json, read_json

MODES = ("pca_trunk", "pca_joint", "random")
COARSE_CELLS = 8  # the coarse grid spans the trajectory's bounding box with this many cells
MARGIN = 0.25


def _dot(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u.astype(np.float64, copy=False), v.astype(np.float64, copy=False)))


@dataclass
class Basis:
    mode: str
    version: int
    center: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        g = np.array([[_dot(self.d1, self.d1), _dot(self.d1, self.d2)],
                      [_dot(self.d1, self.d2), _dot(self.d2, self.d2)]])
        self._gram = g
        self._gram_inv = np.linalg.pinv(g)

    @property
    def id(self) -> str:
        return basis_id(self.mode, self.version)

    def fidelity(self, w: np.ndarray) -> dict[str, float]:
        """Coordinates of ``w`` and how much of its displacement the plane captures."""
        disp = w.astype(np.float64) - self.center
        rhs = np.array([float(disp @ self.d1), float(disp @ self.d2)])
        ab = self._gram_inv @ rhs
        in_plane_sq = float(ab @ self._gram @ ab)
        total_sq = float(disp @ disp)
        resid_sq = max(total_sq - in_plane_sq, 0.0)
        return {
            "a": float(ab[0]), "b": float(ab[1]),
            "captured": 1.0 if total_sq <= 1e-24 else min(in_plane_sq / total_sq, 1.0),
            "residual": resid_sq ** 0.5,
            "norm": total_sq ** 0.5,
        }

    def coords(self, w: np.ndarray) -> tuple[float, float]:
        f = self.fidelity(w)
        return f["a"], f["b"]

    def point(self, a: float, b: float) -> np.ndarray:
        """In-plane weights ``center + a*d1 + b*d2`` (float32)."""
        return (self.center + np.float32(a) * self.d1 + np.float32(b) * self.d2).astype(np.float32)


def basis_id(mode: str, version: int) -> str:
    return f"{mode}-v{version:04d}"


# ------------------------------------------------------------------- directions
def pca_directions(X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-2 orthonormal principal directions of the rows of X (not mean-centered).

    Uses the N x N Gram matrix, so cost is O(N^2 D) with N = number of snapshots.
    Returns (d1, d2, eigenvalues descending).
    """
    X64 = X.astype(np.float64)
    evals, evecs = np.linalg.eigh(X64 @ X64.T)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]
    dirs = []
    for k in range(2):
        if k < len(evals) and evals[k] > 1e-12 * max(evals[0], 1e-300):
            d = X64.T @ evecs[:, k]
        else:  # rank-deficient (e.g. only two snapshots so far): any orthogonal direction
            d = rng.standard_normal(X.shape[1])
        for prev in dirs:
            d -= (d @ prev) * prev
        dirs.append(d / np.linalg.norm(d))
    return dirs[0].astype(np.float32), dirs[1].astype(np.float32), evals


def filter_normalized_direction(center: np.ndarray, spec: list[dict[str, Any]],
                                rng: np.random.Generator) -> np.ndarray:
    """A random direction rescaled filter by filter to the norms of ``center``.

    Adapted from Li et al., "Visualizing the Loss Landscape of Neural Nets"
    (NeurIPS 2018) and their reference code: for each weight tensor, every
    filter (slice along dim 0) of the Gaussian direction is scaled to the norm
    of the corresponding filter of the weights; 1-D tensors (biases, BatchNorm
    scales and shifts) are zeroed.
    """
    out = np.zeros_like(center, dtype=np.float32)
    i = 0
    for s in spec:
        k = int(np.prod(s["shape"])) if s["shape"] else 1
        if len(s["shape"]) > 1:
            w = center[i:i + k].reshape(s["shape"][0], -1).astype(np.float64)
            d = rng.standard_normal(w.shape)
            d *= np.linalg.norm(w, axis=1, keepdims=True) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-10)
            out[i:i + k] = d.reshape(-1)
        i += k
    return out


# ---------------------------------------------------------------------- manager
class BasisManager:
    """Fits, versions, loads and projects with the bases of one run."""

    def __init__(self, run: Run, max_fit_snapshots: int = 128, seed: int = 0):
        self.run = run
        self.root = run.dir / "bases"
        self.max_fit_snapshots = max_fit_snapshots
        self.seed = seed
        self._loaded: dict[str, Basis] = {}

    # -- versions
    def versions(self, mode: str) -> list[int]:
        d = self.root / mode
        return sorted(int(p.stem[1:]) for p in d.glob("v[0-9][0-9][0-9][0-9].json")) if d.exists() else []

    def latest_info(self, mode: str) -> dict[str, Any] | None:
        vs = self.versions(mode)
        return read_json(self.root / mode / f"v{vs[-1]:04d}.json") if vs else None

    def latest(self, mode: str) -> Basis | None:
        info = self.latest_info(mode)
        return self.load(mode, info["version"]) if info else None

    def load(self, mode: str, version: int) -> Basis:
        bid = basis_id(mode, version)
        if bid not in self._loaded:
            stem = self.root / mode / f"v{version:04d}"
            t = np_load_file(str(stem) + ".safetensors")
            self._loaded = {bid: Basis(mode, version, t["center"], t["d1"], t["d2"], read_json(Path(str(stem) + ".json")))}
        return self._loaded[bid]

    # -- fitting
    def _scope(self, mode: str) -> list:
        return self.run.trajectories(branches=(mode == "pca_joint"))

    def scope_keys(self, mode: str) -> list[list[Any]]:
        return [[t.name, s] for t in self._scope(mode) for s in t.steps()]

    def needs_refit(self, mode: str, refit_every: int = 5) -> bool:
        """True when ``refit_every`` new snapshots arrived, or the scope finished with any new ones."""
        trunk_steps = self.run.trunk.steps()
        if len(trunk_steps) < 2:
            return False
        info = self.latest_info(mode)
        if info is None:
            return True
        seen = {tuple(k) for k in info["seen_keys"]}
        new = [k for k in self.scope_keys(mode) if tuple(k) not in seen]
        if len(new) >= max(1, refit_every):
            return True
        settled = all(t.finished for t in self._scope(mode))
        return settled and (bool(new) or info["center"][1] != trunk_steps[-1])

    def fit(self, mode: str) -> Basis | None:
        """Fit a new version of ``mode`` and invalidate the landscapes of older versions."""
        if mode not in MODES:
            raise ValueError(f"unknown basis mode {mode!r}; choose from {MODES}")
        trunk = self.run.trunk
        trunk_steps = trunk.steps()
        if len(trunk_steps) < 2:
            return None
        center_step = trunk_steps[-1]
        center = trunk.load_params(center_step)
        keys = self.scope_keys(mode)
        version = (self.versions(mode) or [0])[-1] + 1
        rng = np.random.default_rng(self.seed)
        info: dict[str, Any] = {"mode": mode, "version": version, "id": basis_id(mode, version),
                                "center": [TRUNK, center_step], "seen_keys": keys, "created": time.time()}

        if mode == "random":
            spec = self.run.meta["param_spec"]
            d1 = filter_normalized_direction(center, spec, rng)
            d2 = filter_normalized_direction(center, spec, rng)
            info.update(orthonormal=False, min_extent=1.0, fit_keys=[])
        else:
            fit_keys = [k for k in keys if not (k[0] == TRUNK and k[1] == center_step)]
            if len(fit_keys) > self.max_fit_snapshots:
                pick = np.linspace(0, len(fit_keys) - 1, self.max_fit_snapshots).round().astype(int)
                fit_keys = [fit_keys[i] for i in sorted(set(pick.tolist()))]
            X = np.stack([self.run.trajectory(n).load_params(s) - center for n, s in fit_keys])
            d1, d2, evals = pca_directions(X, rng)
            total = float(evals.sum()) or 1.0
            info.update(orthonormal=True, min_extent=0.0, fit_keys=fit_keys,
                        explained_variance_ratio=[float(e / total) for e in evals[:2]])

        basis = Basis(mode, version, center, d1, d2, info)
        ab = np.array([basis.coords(w) for t in self._scope(mode) for _, w in t.iter_params()])
        span = float(max(np.ptp(ab[:, 0]), np.ptp(ab[:, 1]), 2 * info["min_extent"]))
        info["coarse_h"] = (span if span > 0 else 1.0) * (1 + 2 * MARGIN) / COARSE_CELLS

        stem = self.root / mode / f"v{version:04d}"
        atomic_save_tensors(Path(str(stem) + ".safetensors"),
                            {"center": torch.from_numpy(center), "d1": torch.from_numpy(d1), "d2": torch.from_numpy(d2)})
        atomic_write_json(Path(str(stem) + ".json"), info)  # written last: its presence publishes the version
        self.invalidate_older(mode, version)
        return basis

    def invalidate_older(self, mode: str, keep_version: int) -> None:
        """Delete tensors, projections and cached grid points of older versions (their JSON stays as history)."""
        for v in self.versions(mode):
            if v == keep_version:
                continue
            stem = self.root / mode / f"v{v:04d}"
            Path(str(stem) + ".safetensors").unlink(missing_ok=True)
            Path(str(stem) + ".proj.json").unlink(missing_ok=True)
            shutil.rmtree(self.run.dir / "landscape" / basis_id(mode, v), ignore_errors=True)

    # -- projections
    def projections_path(self, basis_or_id: Basis | str) -> Path:
        bid = basis_or_id if isinstance(basis_or_id, str) else basis_or_id.id
        mode, _, v = bid.rpartition("-")
        return self.root / mode / f"{v}.proj.json"

    def read_projections(self, basis_or_id: Basis | str) -> dict[str, dict[str, list[float]]]:
        return read_json(self.projections_path(basis_or_id), {}) or {}

    def update_projections(self, basis: Basis) -> tuple[dict[str, dict[str, list[float]]], bool]:
        """Project every snapshot of every trajectory (trunk and all branches) that is not projected yet."""
        old = self.read_projections(basis)
        new: dict[str, dict[str, list[float]]] = {}
        changed = False
        for t in self.run.trajectories():
            steps = t.steps()
            prev = old.get(t.name, {})
            have = {s: i for i, s in enumerate(prev.get("steps", []))}
            cols: dict[str, list[float]] = {k: [] for k in ("a", "b", "captured", "residual", "norm")}
            for s in steps:
                if s in have:
                    for k in cols:
                        cols[k].append(prev[k][have[s]])
                else:
                    try:
                        f = basis.fidelity(t.load_params(s))
                    except FileNotFoundError:  # thinned away between listing and loading
                        f = None
                    if f is None:
                        steps = [x for x in steps if x != s]
                        continue
                    for k in cols:
                        cols[k].append(f[k])
                    changed = True
            changed |= len(steps) != len(prev.get("steps", []))
            new[t.name] = {"steps": steps, "from_step": t.from_step, **cols}
        changed |= set(new) != set(old)
        if changed:
            atomic_write_json(self.projections_path(basis), new)
        return new, changed
