"""Background landscape worker.

A separate process driven entirely by the run directory, so the same worker
serves a live training run (started by ``Tracer``) and a finished run reopened
in the dashboard.  It never touches the training process.

Loop: read ``control.json`` -> refit the basis if due (new version, old cache
invalidated) -> project new snapshots of trunk and branches -> evaluate the
next few grid points in priority order -> repeat.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .basis import BasisManager
from .landscape import RESOLUTIONS, Evaluator, GridCache, plan_points
from .store import Run, atomic_write_json, read_json

HEARTBEAT_STALE = 15.0


def worker_status(run_dir: Path | str) -> dict[str, Any]:
    st = read_json(Path(run_dir) / "worker.json", {}) or {}
    st["alive"] = bool(st) and time.time() - st.get("heartbeat", 0) < HEARTBEAT_STALE
    return st


def ensure_worker(run_dir: Path | str, device: str | None = None, threads: int | None = None) -> bool:
    """Start a worker for this run unless one is already alive.  Returns True if one was started."""
    run_dir = Path(run_dir)
    if worker_status(run_dir)["alive"]:
        return False
    device = device or (read_json(run_dir / "meta.json", {}) or {}).get("worker_device", "cpu")
    cmd = [sys.executable, "-m", "trajmap.worker", str(run_dir), "--device", device]
    if threads:
        cmd += ["--threads", str(threads)]
    atomic_write_json(run_dir / "worker.json", {"pid": None, "heartbeat": time.time(), "state": "starting"})
    with open(run_dir / "worker.log", "a") as log:
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return True


def run_worker(run_dir: Path | str, device: str = "cpu", threads: int | None = None,
               poll: float = 0.5, slice_seconds: float = 0.5, idle_exit: float = 60.0,
               once: bool = False) -> None:
    """Run the worker loop.  ``once=True`` works until nothing is left to do, then returns (used in tests)."""
    run = Run(run_dir)
    torch.set_num_threads(threads or max(1, (os.cpu_count() or 4) // 4))
    bm = BasisManager(run)
    evaluator: Evaluator | None = None
    evaluator_error: str | None = None
    cache: GridCache | None = None
    center_buffers: tuple[str, Any, Any] | None = None
    idle_since: float | None = None
    n_eval = 0

    while True:
        ctl = run.control()
        mode = ctl["mode"]
        level = RESOLUTIONS.get(int(ctl["resolution"]), 2)
        eval_batches = int(ctl["eval_batches"]) or None
        did, pending, basis_name = False, 0, None

        if bm.needs_refit(mode, int(ctl["refit_every"])):
            did = bm.fit(mode) is not None
        basis = bm.latest(mode)
        if basis is not None:
            basis_name = basis.id
            proj, changed = bm.update_projections(basis)
            did |= changed
            if evaluator is None and evaluator_error is None:
                try:
                    evaluator = Evaluator(run, device)
                except Exception as e:  # no eval data / model not rebuildable: projections only
                    evaluator_error = f"{type(e).__name__}: {e}"
                    print(f"[worker] landscape evaluation disabled: {evaluator_error}", flush=True)
            if evaluator is not None:
                settings = {"eval_batches": eval_batches or 0, "bn_mode": ctl["bn_mode"]}
                if cache is None or cache.basis_id != basis.id or cache.settings != settings:
                    cache = GridCache(run.dir, basis.id, settings)
                if center_buffers is None or center_buffers[0] != basis.id:
                    try:
                        snap = run.trunk.load(basis.info["center"][1])
                        center_buffers = (basis.id, snap.buffers, snap.int_buffers)
                    except FileNotFoundError:
                        center_buffers = (basis.id, None, None)
                polylines = [np.column_stack([p["a"], p["b"]]) for p in proj.values() if p["steps"]]
                todo = plan_points(polylines, basis.info["coarse_h"], set(cache.points), level,
                                   int(ctl["budget"]), basis.info.get("min_extent", 0.0))
                pending = len(todo)
                u = basis.info["coarse_h"] / 8
                t0 = time.time()
                for key in todo:
                    r = evaluator.evaluate(basis.point(key[0] * u, key[1] * u), center_buffers[1], center_buffers[2],
                                           eval_batches=eval_batches, bn_mode=ctl["bn_mode"])
                    cache.add(key, r["loss"], r["acc"])
                    n_eval += 1
                    pending -= 1
                    did = True
                    if time.time() - t0 > slice_seconds:
                        break

        atomic_write_json(run.dir / "worker.json", {
            "pid": os.getpid(), "heartbeat": time.time(), "state": "working" if did else "idle",
            "basis": basis_name, "pending": pending, "evaluated_total": n_eval,
            "device": device, "error": evaluator_error})

        if did:
            idle_since = None
            continue
        if once:
            return
        idle_since = idle_since or time.time()
        idle = time.time() - idle_since
        dash = read_json(run.dir / "dashboard.json", {}) or {}
        dash_alive = time.time() - dash.get("heartbeat", 0) < HEARTBEAT_STALE
        if (idle > idle_exit and run.trunk.finished and not dash_alive) or idle > 1800:
            atomic_write_json(run.dir / "worker.json", {"pid": os.getpid(), "heartbeat": 0, "state": "exited"})
            return
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(description="trajmap background landscape worker")
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    run_worker(a.run_dir, a.device, a.threads)


if __name__ == "__main__":
    main()
