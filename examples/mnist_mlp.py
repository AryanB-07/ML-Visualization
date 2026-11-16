"""MNIST MLP on CPU: record a training trunk with a declared forget set.

    python examples/mnist_mlp.py                 # trains into runs/mnist (about a minute)
    tracer dashboard runs/mnist                  # in another terminal, while it trains or later
    tracer unlearn runs/mnist --method gradient_ascent --from-step 1000
"""
import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from trajmap import IndexedDataset, Tracer
from trajmap.data import mnist
from trajmap.models import mnist_mlp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/mnist")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--every", type=int, default=25, help="snapshot interval in steps")
    ap.add_argument("--delay", type=float, default=0.02,
                    help="sleep per step so the live map is watchable; 0 trains at full speed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    train = mnist(train=True)
    forget = torch.randperm(len(train), generator=torch.Generator().manual_seed(a.seed))[:500].tolist()
    loader = DataLoader(IndexedDataset(train), batch_size=128, shuffle=True)
    eval_loader = DataLoader(mnist(train=False), batch_size=512)

    model = mnist_mlp()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    tracer = Tracer(
        model, run_dir=a.run_dir, eval_loader=eval_loader, every_n_steps=a.every,
        forget_indices=forget, seed=a.seed, overwrite=True,
        config={"lr": 0.05, "momentum": 0.9, "batch_size": 128, "epochs": a.epochs},
        model_factory="trajmap.models:mnist_mlp", dataset_factory="trajmap.data:mnist",
    )
    print(f"recording to {a.run_dir}; watch it with:  tracer dashboard {a.run_dir}")
    for epoch in range(a.epochs):
        for x, y, idx in loader:
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tracer.step(loss=loss.item(), batch_indices=idx)
            if a.delay:
                time.sleep(a.delay)
        print(f"epoch {epoch + 1}/{a.epochs}  step {tracer.step_count}  loss {loss.item():.4f}")
    tracer.finish()
    print("done")


if __name__ == "__main__":
    main()
