"""Ready-made demo setups: a dataset, a model and a training recipe, recorded as a run.

    tracer demo list
    tracer demo spirals                    # trains into runs/spirals
    tracer demo grokking --watch --extras  # slowed down to watch, then a branch and explorations

The dashboard's "New run" drawer starts the same presets in the background.
Every preset records with ``model_factory`` / ``dataset_factory`` so its run
can be unlearned, explored and reopened later like any other.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import IndexedDataset
from .store import Run, read_json, resolve


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    description: str
    model: str                                  # "module:callable"
    dataset: str                                # "module:callable", called with train=True / False
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer: str = "sgd"                      # "sgd" or "adamw"
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 128
    steps: int = 1000                           # optimizer steps (loops over the data as needed)
    every: int = 25                             # snapshot interval
    n_forget: int = 500
    watch_delay: float = 0.02                   # seconds per step with --watch
    dtype: str = "float32"
    max_snapshots: int | None = None
    landscape: dict[str, Any] = field(default_factory=dict)
    download: str = ""                          # human note, e.g. "about 30 MB"
    data_check: str = ""                        # path under the data root that exists once downloaded
    cost: str = ""                              # rough CPU time

    def data_ready(self, root: str = "data") -> bool:
        return not self.data_check or (Path(root) / self.data_check).exists()


PRESETS: dict[str, Preset] = {p.name: p for p in [
    Preset(
        name="spirals", title="Three spirals (MLP)",
        description="Three interleaved spiral arms in 2D, one class per arm. Synthetic and tiny, so the whole "
                    "landscape is cheap to evaluate; the tanh MLP has to bend its boundary hard, which gives a "
                    "rugged surface with several basins.",
        model="trajmap.models:spiral_mlp", dataset="trajmap.data:spirals",
        optimizer="adamw", lr=0.004, weight_decay=0.0, batch_size=64, steps=4000, every=40, n_forget=150,
        watch_delay=0.01, landscape={"resolution": 65, "eval_batches": 4, "budget": 1500},
        cost="seconds; about 40 s when watched"),
    Preset(
        name="grokking", title="Modular addition (grokking)",
        description="Learn (a + b) mod 97 from 40% of all pairs with strong weight decay. The network memorizes the "
                    "train split quickly while test loss stays high, then much later generalizes. The map is drawn "
                    "on held-out pairs, so you see the long walk from the memorizing basin to the generalizing one.",
        model="trajmap.models:modular_mlp", dataset="trajmap.data:modular_addition",
        optimizer="adamw", lr=0.003, weight_decay=1.0, batch_size=512, steps=4000, every=40, n_forget=200,
        watch_delay=0.005, max_snapshots=120, landscape={"resolution": 33, "eval_batches": 2, "budget": 900},
        cost="about 10 s; 30 s when watched"),
    Preset(
        name="mnist", title="MNIST (MLP)",
        description="The original demo: a 3-layer MLP on handwritten digits. Smooth, nearly convex bowl.",
        model="trajmap.models:mnist_mlp", dataset="trajmap.data:mnist",
        lr=0.05, momentum=0.9, steps=1400, every=25, n_forget=500,
        download="about 12 MB", data_check="MNIST/raw/train-images-idx3-ubyte", cost="about 20 s; a minute when watched"),
    Preset(
        name="fashion", title="Fashion-MNIST (CNN)",
        description="Clothing images with a small two-block CNN. Harder than MNIST (shirts vs. T-shirts vs. "
                    "pullovers), with conv filters whose scale the filter-normalized directions account for.",
        model="trajmap.models:small_cnn", dataset="trajmap.data:fashion_mnist",
        lr=0.03, momentum=0.9, steps=1500, every=25, n_forget=500,
        download="about 30 MB", data_check="FashionMNIST/raw/train-images-idx3-ubyte", cost="about 30 s; a minute when watched"),
    Preset(
        name="cifar_cnn", title="CIFAR-10 (VGG-style CNN with BatchNorm)",
        description="Colour photos of 10 classes on a 5-conv network with BatchNorm, trained on a 20k-image subset. "
                    "Real-image difficulty; every map point re-estimates the BatchNorm statistics.",
        model="trajmap.models:cifar_cnn", dataset="trajmap.data:cifar10_fast", dataset_kwargs={"limit": 20000},
        lr=0.05, momentum=0.9, weight_decay=5e-4, steps=1200, every=30, n_forget=500,
        landscape={"resolution": 17, "eval_batches": 1, "budget": 300},
        download="about 170 MB", data_check="cifar-10-batches-py/data_batch_1", cost="a few minutes"),
    Preset(
        name="cifar_resnet18", title="CIFAR-10 (ResNet-18)",
        description="The full ResNet-18 (11M parameters). Slow on CPU: use --device cuda or mps if you have one. "
                    "Snapshots are float16 and capped at 60.",
        model="trajmap.models:cifar_resnet18", dataset="trajmap.data:cifar10_fast", dataset_kwargs={"limit": 20000},
        lr=0.05, momentum=0.9, weight_decay=5e-4, steps=800, every=20, n_forget=500, dtype="float16",
        max_snapshots=60, landscape={"resolution": 17, "eval_batches": 1, "budget": 200},
        download="about 170 MB", data_check="cifar-10-batches-py/data_batch_1", cost="30+ minutes on CPU"),
]}


def list_presets() -> list[Preset]:
    return list(PRESETS.values())


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown demo {name!r}; available: {', '.join(PRESETS)}")
    return PRESETS[name]


def make_optimizer(name: str, params, lr: float, momentum: float = 0.9, weight_decay: float = 0.0):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def train_demo(name: str, run_dir: str | Path | None = None, *, watch: bool = False, extras: bool = False,
               steps: int | None = None, device: str = "cpu", seed: int = 0, landscape: bool = True) -> Run:
    """Train preset ``name`` into ``run_dir`` (default ``runs/<name>``) and return the run.

    ``watch`` sleeps a little each step so the live map has time to form.
    ``extras`` afterwards adds a gradient-ascent unlearning branch from the
    middle of training and two explorations, so the map has more than one path.
    """
    from .landscape import pick_device
    from .tracer import Tracer

    p = get_preset(name)
    run_dir = Path(run_dir or f"runs/{name}")
    steps = int(steps or p.steps)
    torch.manual_seed(seed)
    dev = pick_device(device)
    factory = resolve(p.dataset)
    train = factory(train=True, **p.dataset_kwargs)
    test = factory(train=False, **{k: v for k, v in p.dataset_kwargs.items() if k != "limit"})
    forget = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))[:p.n_forget].tolist()
    loader = DataLoader(IndexedDataset(train), batch_size=p.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    eval_loader = DataLoader(test, batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(seed + 1))

    model = resolve(p.model)(**p.model_kwargs).to(dev)
    opt = make_optimizer(p.optimizer, model.parameters(), p.lr, p.momentum, p.weight_decay)
    tracer = Tracer(
        model, run_dir=run_dir, eval_loader=eval_loader, every_n_steps=p.every, forget_indices=forget,
        dtype=p.dtype, max_snapshots=p.max_snapshots, seed=seed, overwrite=True,
        config={"demo": p.name, "optimizer": p.optimizer, "lr": p.lr, "momentum": p.momentum,
                "weight_decay": p.weight_decay, "batch_size": p.batch_size, "steps": steps},
        model_factory=p.model, model_kwargs=p.model_kwargs,
        dataset_factory=p.dataset, dataset_kwargs={"train": True, **p.dataset_kwargs},
        landscape=landscape, landscape_options=p.landscape,
    )
    print(f"[demo] {p.title}: {tracer.run.meta['num_params']:,} parameters, {steps} steps -> {run_dir}", flush=True)
    delay = p.watch_delay if watch else 0.0
    t0 = time.time()
    while tracer.step_count < steps:
        for x, y, idx in loader:
            loss = nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tracer.step(loss=loss.item(), batch_indices=idx)
            if tracer.step_count % 200 == 0:
                print(f"[demo] step {tracer.step_count}/{steps}  loss {loss.item():.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if delay:
                time.sleep(delay)
            if tracer.step_count >= steps:
                break
    tracer.finish()
    run = Run(run_dir)
    print(f"[demo] training done in {time.time() - t0:.0f}s", flush=True)

    if extras:
        add_extras(run, device=device)
    return run


def add_extras(run: Run, device: str = "cpu", explorations: int = 2) -> None:
    """A gradient-ascent branch from the middle of the trunk, then a couple of explorations."""
    from .explore import improve_map
    from .unlearning import run_unlearning

    steps = run.trunk.steps()
    mid = steps[len(steps) // 2]
    print(f"[demo] unlearning branch: gradient_ascent from step {mid}", flush=True)
    run_unlearning(run, "gradient_ascent", mid, config={"lr": 0.01, "steps": 40}, device=device)
    if explorations:
        # Let the worker fit a basis that includes the branch before choosing start points.
        from .basis import BasisManager
        mode = run.control()["mode"]
        for _ in range(60):
            if BasisManager(run).latest_info(mode) is not None:
                break
            time.sleep(1)
        print(f"[demo] {explorations} explorations", flush=True)
        improve_map(run, mode="auto", max_runs=explorations, device=device,
                    config={"max_seconds_each": 60})
    print("[demo] extras done", flush=True)


# ------------------------------------------------------------- background start
def start_demo(name: str, run_dir: str | Path, *, watch: bool = True, extras: bool = False,
               device: str = "cpu") -> Path:
    """Launch ``train_demo`` in a background process; returns the log file path."""
    run_dir = Path(run_dir)
    log_dir = run_dir.parent / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{run_dir.name}.log"
    cmd = [sys.executable, "-m", "trajmap.demos", name, "--run-dir", str(run_dir), "--device", device]
    cmd += ["--watch"] if watch else []
    cmd += ["--extras"] if extras else []
    with open(log, "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return log


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    """Run directories directly under ``root``, newest first."""
    root = Path(root)
    out = []
    for d in root.iterdir() if root.is_dir() else []:
        meta = read_json(d / "meta.json")
        if not meta:
            continue
        trunk = read_json(d / "trunk" / "info.json", {}) or {}
        out.append({"name": d.name, "dir": d, "created": meta.get("created", 0),
                    "demo": (meta.get("config") or {}).get("demo"), "finished": bool(trunk.get("finished"))})
    return sorted(out, key=lambda r: -r["created"])


def unique_run_name(root: str | Path, base: str) -> str:
    root, name, k = Path(root), base, 1
    while (root / name).exists():
        k += 1
        name = f"{base}-{k}"
    return name


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tracer demo", description="Train a ready-made demo and record it as a run.")
    ap.add_argument("name", help="preset name, or 'list'")
    ap.add_argument("--run-dir", default=None, help="default: runs/<name>")
    ap.add_argument("--watch", action="store_true", help="slow training down so the live map can form")
    ap.add_argument("--extras", action="store_true", help="then add an unlearning branch and two explorations")
    ap.add_argument("--steps", type=int, default=None, help="override the preset's step count")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-landscape", action="store_true", help="do not start the landscape worker")
    a = ap.parse_args(argv)
    if a.name == "list":
        for p in list_presets():
            note = f"; downloads {p.download}" if p.download and not p.data_ready() else ""
            print(f"{p.name:16s} {p.title}  ({p.cost}{note})")
        return
    train_demo(a.name, a.run_dir, watch=a.watch, extras=a.extras, steps=a.steps, device=a.device,
               seed=a.seed, landscape=not a.no_landscape)


if __name__ == "__main__":
    main()
