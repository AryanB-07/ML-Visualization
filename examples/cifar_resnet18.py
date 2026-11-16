"""ResNet-18 on CIFAR-10 (11.2M parameters): the same hooks at a larger scale.

    python examples/cifar_resnet18.py --epochs 2
    tracer dashboard runs/cifar

Each float16 snapshot is about 22 MB, so the snapshot count is capped.  Each
grid point costs two forward passes over the eval batches (BatchNorm
recalibration + evaluation); the landscape defaults below are deliberately
small.  Use a GPU for training if you have one; the worker stays on CPU unless
you pass --worker-device.
"""
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from trajmap import IndexedDataset, Tracer
from trajmap.data import cifar10
from trajmap.landscape import pick_device
from trajmap.models import cifar_resnet18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/cifar")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early after this many steps (0 = no limit)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--worker-device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    train = cifar10(train=True)
    forget = torch.randperm(len(train), generator=torch.Generator().manual_seed(a.seed))[:500].tolist()
    loader = DataLoader(IndexedDataset(train), batch_size=128, shuffle=True, num_workers=2)
    eval_loader = DataLoader(cifar10(train=False), batch_size=256)

    model = cifar_resnet18().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    tracer = Tracer(
        model, run_dir=a.run_dir, eval_loader=eval_loader, every_n_steps=a.every,
        forget_indices=forget, seed=a.seed, overwrite=True, dtype="float16", max_snapshots=60,
        eval_max_samples=512, eval_batch_size=256,
        config={"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4, "batch_size": 128, "epochs": a.epochs},
        model_factory="trajmap.models:cifar_resnet18", dataset_factory="trajmap.data:cifar10",
        landscape_options={"resolution": 17, "eval_batches": 1, "budget": 150, "refit_every": 8},
        worker_device=a.worker_device,
    )
    print(f"training on {device}; recording to {a.run_dir}; watch with:  tracer dashboard {a.run_dir}")
    done = False
    for epoch in range(a.epochs):
        for x, y, idx in loader:
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tracer.step(loss=loss.item(), batch_indices=idx)
            if tracer.step_count % 20 == 0:
                print(f"epoch {epoch + 1}  step {tracer.step_count}  loss {loss.item():.4f}", flush=True)
            if a.max_steps and tracer.step_count >= a.max_steps:
                done = True
                break
        if done:
            break
    tracer.finish()
    print("done")


if __name__ == "__main__":
    main()
