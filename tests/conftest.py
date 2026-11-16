import numpy as np
import pytest
import torch
from torch import nn

from trajmap import Tracer


def tiny_model() -> nn.Module:
    """Importable factory so runs created in tests can rebuild their model."""
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))


def bn_model() -> nn.Module:
    return nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 3))


def make_data(n: int = 256, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 6, generator=g)
    return x, x[:, :3].argmax(1)


def train_run(run_dir, steps: int = 40, every: int = 4, factory: str = "conftest:tiny_model", **kw):
    """Train a tiny model for a few steps and return (run_dir, model, tracer)."""
    torch.manual_seed(0)
    model = tiny_model() if factory.endswith("tiny_model") else bn_model()
    x, y = make_data()
    kw.setdefault("forget_indices", list(range(10)))
    tracer = Tracer(model, run_dir, eval_loader=[(x[:128], y[:128])], every_n_steps=every,
                    model_factory=factory, landscape=False, seed=0, **kw)
    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (32,), generator=g)
        loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tracer.step(loss=loss.item(), batch_indices=idx)
    tracer.finish()
    return run_dir, model, tracer


@pytest.fixture
def run_dir(tmp_path):
    return train_run(tmp_path / "run")[0]


def write_trajectory(writer, points: np.ndarray) -> None:
    """Store rows of ``points`` as snapshots at steps 0, 10, 20, ..."""
    for k, p in enumerate(points):
        writer.save_snapshot(k * 10, torch.tensor(p, dtype=torch.float32))
    writer.set_info(finished=True)
