"""Improve the map: descend from new starting points far from every recorded path.

The trunk shows one route through weight space.  An *exploration* starts
somewhere else, trains down to (or close to) a minimum, and is recorded like a
branch, so the map gains real trajectories and the worker refines the surface
around them.

Start points (``start=``)
    plane   a point of the current 2D plane, chosen at random among the
            candidates farthest from every projected path (default).  It is on
            the map by construction, so the descent begins where the picture
            is least informed.
    init    a fresh random initialization, the candidate farthest (full-space
            L2) from every stored snapshot.  An independent training run; most
            of its displacement is usually off the plane, which the fidelity
            numbers will show.

Descent
    SGD (or AdamW when the run's config says ``optimizer: adamw``) on the
    run's ``dataset_factory``, with lr / momentum / weight_decay / batch_size
    taken from the run's config (overridable).  It is not your training loop:
    schedules, augmentation and custom optimizers are not reproduced.  An
    exploration stops at the first of: recent loss
    reaches the trunk's final loss level, the loss plateaus, ``max_steps``,
    the per-exploration time limit, the overall time budget, or a stop request.

Modes (``improve_map``)
    one     a single exploration.
    auto    keep going until no start point is far from existing paths any
            more (or ``max_runs``).
    time    keep going for ``minutes``; the exploration in progress is cut
            off at the deadline and kept as it is.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .basis import BasisManager
from .landscape import pick_device
from .params import assign_buffers, assign_params, flatten_buffers, flatten_params
from .store import TRUNK, Run, Trajectory, atomic_write_json, read_json, resolve
from .tracer import Recorder

REGION_PAD = 0.35     # the search region is the trunk/branch bounding box grown by this fraction
AUTO_MIN_GAP = 0.12   # auto mode stops when the best start is closer than this fraction of the region diagonal
DEFAULTS: dict[str, Any] = {"optimizer": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 0.0, "batch_size": 128,
                            "max_seconds_each": 120.0, "grad_clip": 10.0, "plateau_tol": 0.02, "extra_budget": 150,
                            "max_steps": None,       # None = as many steps as the trunk took
                            "target_ratio": 1.05}    # stop once recent loss <= ratio * trunk's final loss level


# ------------------------------------------------------------------ start points
def pick_plane_start(run: Run, bm: BasisManager, rng: np.random.Generator, mode: str | None = None,
                     n_candidates: int = 512, top_k: int = 5) -> dict[str, Any] | None:
    """A random in-plane point among the ``top_k`` candidates farthest from all projected paths."""
    basis = bm.latest(mode or run.control()["mode"]) or bm.latest("pca_trunk") or bm.fit("pca_trunk")
    if basis is None:
        return None
    proj, _ = bm.update_projections(basis)
    kinds = {t.name: t.kind for t in run.trajectories()}
    pts = {n: np.column_stack([p["a"], p["b"]]) for n, p in proj.items() if p["steps"]}
    anchor = np.vstack([v for n, v in pts.items() if kinds.get(n) != "explore"])  # region ignores explorations
    lo, hi = anchor.min(0), anchor.max(0)
    span = np.maximum(hi - lo, 1e-12)
    pad = REGION_PAD * span.max()
    lo, hi = lo - pad, hi + pad
    cand = rng.uniform(lo, hi, size=(n_candidates, 2))
    existing = np.vstack(list(pts.values()))
    dist = np.sqrt(((cand[:, None, :] - existing[None]) ** 2).sum(2)).min(1)
    best = np.argsort(dist)[::-1][:top_k]
    k = int(rng.choice(best))
    try:
        snap = run.trunk.load(basis.info["center"][1])
        fb, ib = snap.buffers, snap.int_buffers
    except FileNotFoundError:
        fb = ib = None
    return {"start": "plane", "a": float(cand[k, 0]), "b": float(cand[k, 1]), "distance": float(dist[k]),
            "best_distance": float(dist[best[0]]), "region_diagonal": float(np.linalg.norm(hi - lo)),
            "basis_id": basis.id, "params": torch.from_numpy(basis.point(cand[k, 0], cand[k, 1])),
            "buffers": fb, "int_buffers": ib}


def pick_init_start(run: Run, rng: np.random.Generator, n_candidates: int = 4) -> dict[str, Any]:
    """A fresh random initialization: of a few candidates, the one farthest from every stored snapshot."""
    cands = []
    for _ in range(n_candidates):
        torch.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
        model = run.build_model()
        for m in model.modules():  # a pickled model comes back with its recorded weights; re-draw them
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        cands.append((flatten_params(model), *flatten_buffers(model)))
    P = torch.stack([c[0] for c in cands]).double().numpy()
    dist = np.full(len(cands), np.inf)
    for t in run.trajectories():
        for _, w in t.iter_params():
            dist = np.minimum(dist, np.linalg.norm(P - w, axis=1))
    k = int(np.argmax(dist))
    return {"start": "init", "distance": float(dist[k]), "params": cands[k][0], "buffers": cands[k][1],
            "int_buffers": cands[k][2]}


# ----------------------------------------------------------------------- descent
def trunk_loss_level(run: Run, window: int) -> float | None:
    losses = [r["loss"] for r in run.trunk.step_log()[-window:] if r.get("loss") is not None]
    return float(np.mean(losses)) if losses else None


def explore_once(run: Run, start: dict[str, Any], name: str | None = None, config: dict[str, Any] | None = None,
                 max_steps: int | None = None, deadline: float | None = None,
                 should_stop: Callable[[], bool] | None = None, device: str = "cpu",
                 on_progress: Callable[[dict[str, Any]], None] | None = None, seed: int = 0) -> Trajectory:
    """Train from ``start`` down to a minimum (or as close as the limits allow) and record the path."""
    meta = run.meta
    if not meta.get("dataset_factory"):
        raise RuntimeError("exploration needs the training data: pass dataset_factory='module:fn' to Tracer")
    cfg = {**DEFAULTS, **{k: v for k, v in (meta.get("config") or {}).items() if k in DEFAULTS}, **(config or {})}
    every = int(run.trunk.info().get("every_n_steps", 50))
    max_steps = int(max_steps or cfg["max_steps"] or run.trunk.info().get("last_step") or (run.trunk.steps() or [1000])[-1])
    window = max(100, 4 * every)
    target = trunk_loss_level(run, window)
    if name is None:
        k = 1 + sum(t.kind == "explore" for t in run.branches())
        name = f"explore-{k:02d}"
        while name in run.branch_names():
            k += 1
            name = f"explore-{k:02d}"

    dev = pick_device(device)
    model = run.build_model()
    assign_params(model, start["params"])
    assign_buffers(model, start.get("buffers"), start.get("int_buffers"))
    model.to(dev).train()
    info = {k: v for k, v in start.items() if k not in ("params", "buffers", "int_buffers")}
    writer = run.create_exploration(name, start["params"], start.get("buffers"), start.get("int_buffers"),
                                    method="sgd_descent", config=cfg, every_n_steps=every, **info)
    rec = Recorder(model, writer, every, initial_snapshot=False)
    from .demos import make_optimizer
    opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["momentum"], cfg["weight_decay"])
    loss_fn = run.loss_fn()
    ds = resolve(meta["dataset_factory"])(**(meta.get("dataset_kwargs") or {}))
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    t_end = time.time() + float(cfg["max_seconds_each"])
    losses: list[float] = []
    reason, last_report, flat = "max_steps", 0.0, 0
    while reason == "max_steps" and rec.step_count < max_steps:
        for batch in loader:
            loss = loss_fn(model(batch[0].to(dev)), batch[1].to(dev))
            if not bool(torch.isfinite(loss)):
                reason = "diverged"
                break
            opt.zero_grad()
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            losses.append(loss.item())
            rec.step(loss=losses[-1])
            n, now = rec.step_count, time.time()
            recent = float(np.mean(losses[-window:]))
            if on_progress and now - last_report > 1.0:
                on_progress({"name": name, "step": n, "loss": recent})
                last_report = now
            if n >= max_steps:
                break
            if should_stop and should_stop():
                reason = "stopped"
            elif deadline and now >= deadline:
                reason = "time budget"
            elif now >= t_end:
                reason = "time limit per exploration"
            elif n >= window and target is not None and recent <= cfg["target_ratio"] * target:
                reason = "reached trunk loss level"
            elif n >= 2 * window and n % window == 0:
                # plateau = two windows in a row without a real improvement over the window before
                improved = recent <= (1 - cfg["plateau_tol"]) * float(np.mean(losses[-2 * window:-window]))
                flat = 0 if improved else flat + 1
                if flat >= 2:
                    reason = "plateau"
            if reason != "max_steps":
                break
    rec.finish()
    writer.set_info(stopped=reason, steps=rec.step_count, final_loss=float(np.mean(losses[-window:])) if losses else None)
    return run.trajectory(name)


# ------------------------------------------------------------------------- modes
def improve_status(run_dir: Path | str) -> dict[str, Any]:
    st = read_json(Path(run_dir) / "explore.json", {}) or {}
    st["alive"] = bool(st) and st.get("state") == "running" and time.time() - st.get("heartbeat", 0) < 30
    return st


def request_stop(run_dir: Path | str) -> None:
    (Path(run_dir) / "explore.stop").touch()


def improve_map(run: Run | str, mode: str = "one", minutes: float | None = None, max_runs: int = 8,
                start: str = "plane", config: dict[str, Any] | None = None, device: str = "cpu",
                seed: int | None = None, metrics: bool = True, landscape: bool = True) -> list[Trajectory]:
    """Run explorations in ``mode`` ("one", "auto" or "time") and return the new trajectories."""
    run = Run(run) if not isinstance(run, Run) else run
    if mode not in ("one", "auto", "time"):
        raise ValueError("mode must be 'one', 'auto' or 'time'")
    if mode == "time" and not minutes:
        raise ValueError("mode='time' needs minutes")
    stop_file = run.dir / "explore.stop"
    stop_file.unlink(missing_ok=True)
    bm = BasisManager(run)
    n_prior = sum(t.kind == "explore" for t in run.branches())
    rng = np.random.default_rng(seed if seed is not None else 1000 + n_prior)
    deadline = time.time() + 60 * minutes if mode == "time" else None
    limit = 1 if mode == "one" else (max_runs if mode == "auto" else 10 ** 6)
    done: list[Trajectory] = []
    status: dict[str, Any] = {"pid": os.getpid(), "state": "running", "mode": mode, "deadline": deadline,
                              "done": 0, "started": time.time()}

    def report(**kw: Any) -> None:
        status.update(kw, heartbeat=time.time())
        atomic_write_json(run.dir / "explore.json", status)

    report(current=None)
    why = "finished"
    try:
        while len(done) < limit:
            if stop_file.exists():
                why = "stopped"
                break
            if deadline and time.time() >= deadline:
                why = "time budget used"
                break
            st = pick_plane_start(run, bm, rng) if start == "plane" else pick_init_start(run, rng)
            if st is None:
                why = "no basis yet (the trunk needs at least two snapshots)"
                break
            if mode == "auto" and start == "plane" and st["best_distance"] < AUTO_MIN_GAP * st["region_diagonal"]:
                why = "map covered: no start point is far from existing paths"
                break
            t = explore_once(run, st, config=config, deadline=deadline, should_stop=stop_file.exists, device=device,
                             on_progress=lambda p: report(current=p), seed=int(rng.integers(0, 2 ** 31 - 1)))
            done.append(t)
            if metrics:
                from .metrics import compute_metrics
                compute_metrics(run, t.name)
            extra = int({**DEFAULTS, **(config or {})}["extra_budget"])
            if extra:  # let the worker spend points on the surface around the new path
                run.set_control(budget=int(run.control()["budget"]) + extra)
            if landscape:
                from .worker import ensure_worker
                ensure_worker(run.dir)
            report(done=len(done), current=None, last=t.name, last_stopped=t.info().get("stopped"))
            if t.info().get("stopped") == "stopped":
                why = "stopped"
                break
    finally:
        stop_file.unlink(missing_ok=True)
        status.update(state="idle", result=why, heartbeat=time.time(), current=None)
        atomic_write_json(run.dir / "explore.json", status)
    return done


def start_improve(run_dir: Path | str, mode: str = "one", minutes: float | None = None, max_runs: int = 8,
                  start: str = "plane", device: str = "cpu") -> bool:
    """Launch ``improve_map`` in a background process (used by the dashboard).  False if one is already running."""
    run_dir = Path(run_dir)
    if improve_status(run_dir)["alive"]:
        return False
    cmd = [sys.executable, "-m", "trajmap.explore", str(run_dir), "--mode", mode, "--max-runs", str(max_runs),
           "--start", start, "--device", device]
    if minutes:
        cmd += ["--minutes", str(minutes)]
    atomic_write_json(run_dir / "explore.json", {"state": "running", "mode": mode, "done": 0, "heartbeat": time.time()})
    with open(run_dir / "explore.log", "a") as log:
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="trajmap: improve the map with explorations")
    ap.add_argument("run_dir")
    ap.add_argument("--mode", choices=["one", "auto", "time"], default="one")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--max-runs", type=int, default=8)
    ap.add_argument("--start", choices=["plane", "init"], default="plane")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    for t in improve_map(a.run_dir, a.mode, a.minutes, a.max_runs, a.start, device=a.device):
        i = t.info()
        print(f"{t.name}: {i.get('steps')} steps, final loss {i.get('final_loss'):.4g}, stopped: {i.get('stopped')}")
    print("result:", improve_status(a.run_dir).get("result"))


if __name__ == "__main__":
    main()
