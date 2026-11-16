"""Command line: ``tracer dashboard | unlearn | metrics | improve | info``."""
from __future__ import annotations

import argparse
import importlib
import json
import sys


def _import_plugins(mods: list[str]) -> None:
    sys.path.insert(0, "")
    for m in mods or []:
        importlib.import_module(m)


def _parse_config(items: list[str]) -> dict:
    cfg = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tracer", description="Training and unlearning trajectories on one map.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dashboard", help="open the live dashboard for a run directory")
    d.add_argument("run_dir")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8050)
    d.add_argument("--interval", type=float, default=2.0, help="refresh interval in seconds")
    d.add_argument("--device", default="cpu", help="device for evaluating clicked points")

    u = sub.add_parser("unlearn", help="run an unlearning method from a trunk snapshot and record it as a branch")
    u.add_argument("run_dir")
    u.add_argument("--method", required=True)
    u.add_argument("--from-step", type=int, required=True, help="trunk step to branch from (snapped to the nearest snapshot)")
    u.add_argument("--branch", default=None, help="branch name (default: <method>-from<step>)")
    u.add_argument("--every", type=int, default=1, help="snapshot every N method steps")
    u.add_argument("--config", "-c", action="append", metavar="KEY=VALUE", help="method config, e.g. -c lr=0.01 -c steps=40")
    u.add_argument("--device", default="cpu")
    u.add_argument("--plugin", "-p", action="append", metavar="MODULE", help="import a module that registers methods or metrics")

    m = sub.add_parser("metrics", help="compute trajectory metrics for a branch")
    m.add_argument("run_dir")
    m.add_argument("--branch", required=True)
    m.add_argument("--metric", action="append", help="metric name (default: all registered)")
    m.add_argument("--plugin", "-p", action="append", metavar="MODULE")

    e = sub.add_parser("improve", help="improve the map: descend from new start points far from recorded paths")
    e.add_argument("run_dir")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--auto", action="store_true", help="keep going until the map is covered (or --max-runs)")
    g.add_argument("--minutes", type=float, help="keep going for this many minutes")
    e.add_argument("--max-runs", type=int, default=8, help="cap for --auto")
    e.add_argument("--start", choices=["plane", "init"], default="plane",
                   help="start on the 2D plane far from existing paths (default) or from a fresh random init")
    e.add_argument("--config", "-c", action="append", metavar="KEY=VALUE", help="e.g. -c lr=0.05 -c max_seconds_each=60")
    e.add_argument("--device", default="cpu")

    i = sub.add_parser("info", help="summarize a run directory")
    i.add_argument("run_dir")

    a = ap.parse_args(argv)
    from .store import Run

    if a.cmd == "dashboard":
        from .dashboard import serve
        serve(a.run_dir, host=a.host, port=a.port, interval_ms=int(a.interval * 1000), device=a.device)

    elif a.cmd == "unlearn":
        _import_plugins(a.plugin)
        from .unlearning import list_methods, run_unlearning
        run = Run(a.run_dir)
        steps = run.trunk.steps()
        step = min(steps, key=lambda s: abs(s - a.from_step))
        if step != a.from_step:
            print(f"no snapshot at step {a.from_step}; using the nearest one, step {step}")
        if a.method not in list_methods():
            sys.exit(f"unknown method {a.method!r}; registered: {list_methods()}")
        t = run_unlearning(run, a.method, step, config=_parse_config(a.config), branch=a.branch,
                           every_n_steps=a.every, device=a.device)
        print(f"recorded branch {t.name!r}: {len(t.steps())} snapshots from trunk step {step}")
        for name, res in t.metrics().items():
            for s, v in res["series"].items():
                print(f"  {name}/{s}: first {v[0]:.4g}, last {v[-1]:.4g}")

    elif a.cmd == "metrics":
        _import_plugins(a.plugin)
        from .metrics import compute_metrics
        res = compute_metrics(Run(a.run_dir), a.branch, a.metric)
        for name, r in res.items():
            print(f"{name}: {r['description']}")
            print("  step  " + "  ".join(f"{s:>20}" for s in r["series"]))
            for k, st in enumerate(r["branch_steps"]):
                print(f"  {st:>4}  " + "  ".join(f"{v[k]:>20.6g}" for v in r["series"].values()))

    elif a.cmd == "improve":
        from .explore import improve_map, improve_status
        mode = "auto" if a.auto else ("time" if a.minutes else "one")
        for t in improve_map(Run(a.run_dir), mode, a.minutes, a.max_runs, a.start, _parse_config(a.config), a.device):
            inf = t.info()
            print(f"{t.name}: {inf.get('steps')} steps, final loss {inf.get('final_loss'):.4g}, stopped: {inf.get('stopped')}")
        print("result:", improve_status(a.run_dir).get("result"))

    elif a.cmd == "info":
        run = Run(a.run_dir)
        meta = run.meta
        print(f"run {run.dir}: {meta['num_params']:,} parameters, model {meta['model_class']}, seed {meta['seed']}")
        for t in run.trajectories():
            st = t.steps()
            extra = "" if t.is_trunk else (f", exploration, stopped: {t.info().get('stopped')}" if t.kind == "explore"
                                           else f", from trunk step {t.from_step}, method {t.info().get('method')}")
            print(f"  {t.name}: {len(st)} snapshots, steps {st[0]}..{st[-1]}, {'finished' if t.finished else 'running'}{extra}")


if __name__ == "__main__":
    main()
