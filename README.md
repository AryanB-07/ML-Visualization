# trajmap

Record a model's training trajectory and the trajectories of machine-unlearning
methods as **branches of one run**, in **one weight-space coordinate system**,
and explore them on a **live 3D loss-landscape map**.

Loss-surface plots and PCA trajectory projections are not new (see
[Credits](#credits)). What this tool adds:

- training and later unlearning runs stored as trunk + branches of the same run,
  projected with the same basis;
- enough logging during training for path diagnostics afterwards (which batches,
  which steps touched the forget set, optionally the per-step updates);
- a map that forms while training runs and stays explorable after everything is closed.

Status: MVP. The built-in unlearning method and metrics are reference
implementations for validating the pipeline, not research results.

## Install

```bash
pip install -e ".[examples]"     # examples need torchvision
pip install -e ".[dev]"          # plus pytest
```

Python 3.10+, PyTorch 2.1+.

## Quickstart

```bash
python examples/mnist_mlp.py          # trains an MLP into runs/mnist, about a minute on CPU
tracer dashboard runs/mnist           # second terminal; open http://127.0.0.1:8050
```

The surface fills in coarse-to-fine and the path extends while training runs.
Close everything, run `tracer dashboard runs/mnist` again later, and the run
reopens as it was.

Then branch off an unlearning run and look at it on the same map:

```bash
tracer unlearn runs/mnist --method gradient_ascent --from-step 1000 -c lr=0.05 -c steps=22
tracer metrics runs/mnist --branch gradient_ascent-from1000
```

In the dashboard, switch the basis to "PCA, trunk + branches" to fit the plane
on both.

### Hooking into your own loop

```python
from trajmap import Tracer, IndexedDataset

loader = DataLoader(IndexedDataset(train_set), batch_size=128, shuffle=True)  # yields (x, y, idx)

tracer = Tracer(
    model, run_dir="runs/exp1", eval_loader=val_loader, every_n_steps=50,
    forget_indices=forget_idx,                  # optional, declared up front
    model_factory="mypkg.models:build_model",   # lets the run be reopened later
    dataset_factory="mypkg.data:train_set",     # lets `tracer unlearn` build loaders
    config={"lr": 0.1}, seed=0,
)
for x, y, idx in loader:
    loss = criterion(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    tracer.step(loss=loss.item(), batch_indices=idx)   # after optimizer.step()
tracer.finish()
```

`Tracer` never evaluates the landscape itself. It starts a separate worker
process that reads the run directory, so training is not blocked. Pass
`landscape=False` to record only.

Useful options: `dtype="float16"`, `max_snapshots=N`, `save_forget_deltas=True`,
`loss="mse"` or `"module:fn"`, `eval_max_samples`, `worker_device="auto"`,
`landscape_options={"resolution": 17, "eval_batches": 1, "budget": 150}`.

Requirements for a run to be reopenable: the model must be rebuildable, either
through `model_factory` or because its class lives in an importable module (it
is then pickled). Batches of `eval_loader` must be `(x, y)`.

## How it works

```
training process ──writes──▶ run_dir ◀──reads/writes── worker process
   Tracer.step()                ▲                      basis fit, projection, grid evaluation
                                └──reads── dashboard (writes control.json, evaluates clicks)
```

**Run directory**

```
meta.json                 config, seed, model class/factory, parameter and buffer shapes
eval_data.safetensors     the fixed eval subset
forget_indices.npy
trunk/                    snapshots/, snapshots.json, steps.jsonl, deltas/
branches/<name>/          same format, plus the trunk step it started from
bases/<mode>/vNNNN.*      versioned bases and per-snapshot projections
landscape/<basis-id>/     cached grid evaluations
```

**Branches.** A branch starts from a trunk snapshot (which becomes branch step 0)
and stores its own snapshots, step log and metrics. Trunk and branches are read
through the same `Trajectory` class; projection, landscape and metric code does
not distinguish them.

**Bases.** `pca_trunk` and `pca_joint` take the top two principal directions of
snapshot displacements from the final trunk snapshot (the latest one during
training). `random` uses two filter-normalized random directions. While
training, the basis is refit every `refit_every` new snapshots. Each fit is a
new version; grid points cached under older versions are deleted, since their
coordinates no longer mean anything.

**Landscape.** A grid point is `center + a*d1 + b*d2`, evaluated in eval mode
on the stored eval subset. The whole coarse grid is evaluated first, then finer
levels in a band around the trajectories, then outward while budget remains.
Unevaluated regions are interpolated with a thin-plate RBF and drawn in grey.

**BatchNorm.** A grid point defines parameters, not running statistics. By
default the worker resets the statistics and re-estimates them on the eval
subset before evaluating (`bn_mode="recalibrate"`, one extra forward pass per
point). `bn_mode="center"` reuses the statistics of the basis center, which is
faster and increasingly wrong away from the center. Snapshots themselves store
their buffers and reconstruct exactly.

**Accuracy vs. latency.** Three controls, in the dashboard or
`landscape_options`: finest grid resolution (9, 17, 33, 65 points per axis),
eval batches per grid point, and the refinement budget (grid points per basis
version). Changing eval batches re-evaluates the surface; the other two keep the cache.

## The 2D projection can mislead

The map shows a 2D slice of a space with as many dimensions as the model has
parameters. Three consequences:

1. **A snapshot is usually not on the plane.** The path is drawn at each
   snapshot's in-plane position, on the surface. The loss there can differ a lot
   from the snapshot's real loss (in the MNIST example the initial weights sit
   at in-plane loss 0.4 while their recorded loss is 2.3). Hover a snapshot to
   see the recorded loss.
2. **The fidelity panel tells you how much to trust a position.** "Captured" is
   the share of a snapshot's squared distance from the map center that lies in
   the plane; "residual norm" is the rest. Snapshots very close to the center
   often have low captured values, because their small remaining displacement
   is not along the main directions. With the random basis, captured is near zero
   for everything: that view shows the local landscape, not the path.
3. **Two paths that overlap on the map can be far apart.** Whether unlearning
   retraces training is a question for the trajectory metrics, which are
   computed in full parameter space, not for the picture.

Weights reconstructed from a clicked point are labelled "in-plane
approximation" for the same reason.

## Storage costs

With P parameters:

| What | Size | Example: MNIST MLP (P = 109k) | ResNet-18 (P = 11.2M) |
|---|---|---|---|
| Snapshot, float32 | 4·P bytes + buffers | 0.44 MB | 45 MB |
| Snapshot, float16 | 2·P bytes + buffers | 0.22 MB | 22 MB |
| Step log with batch indices | about 7 bytes per sample seen | 1.3 MB for 3 epochs | about 0.4 MB per epoch of CIFAR-10 |
| Basis version | 12·P bytes (center, d1, d2); old versions are deleted | 1.3 MB | 134 MB |
| Grid cache | about 50 bytes per evaluated point | 30 kB | 30 kB |
| Eval subset | the raw tensors | 6.4 MB | 6 MB for 512 images |

The MNIST example writes 58 snapshots, about 25 MB. Use `max_snapshots` and
`dtype="float16"` for larger models; float16 snapshots are not bit-exact.

**Forget deltas (`save_forget_deltas=True`, off by default).** Stores the full
parameter update of every step whose batch contains at least one forget
sample, one snapshot-sized file per such step. With forget fraction f and batch
size B, a fraction 1 − (1 − f)^B of all steps qualifies: for 500 forget
samples out of 60,000 and B = 128 that is 66% of steps. For the MNIST example
this is about 400 MB; for ResNet-18 it is about 30 GB per 1,000 steps in
float32. It also costs one extra parameter copy per step. The stored update is
the whole batch's update (including momentum and weight decay), not the forget
samples' contribution alone.

## Plugins

```python
from trajmap import UnlearningMethod, register_method, TrajectoryMetric, register_metric

@register_method("my_method")
class MyMethod(UnlearningMethod):
    def setup(self, model, forget_loader, retain_loader, config): ...
    def step(self, model) -> dict: ...        # one update in place; return {"loss": ...}
    def is_done(self) -> bool: ...

@register_metric("my_metric")
class MyMetric(TrajectoryMetric):
    def compute(self, branch, trunk) -> dict:  # one value per branch snapshot, per series
        B = branch.matrix()                    # [n_snapshots, P], full parameter space
        ...
```

`run_unlearning(run, "my_method", from_step=N, ...)` or
`tracer unlearn RUN --method my_method --from-step N --plugin mypkg.methods`
loads the trunk snapshot, records every step as a branch, and computes all
registered metrics. Branches end at their last finite snapshot if a method diverges.

Built in:

- `gradient_ascent`: SGD ascent on the forget loss (`lr`, `steps`, `max_loss`).
- `nearest_trunk`: nearest stored trunk snapshot per branch step, its distance,
  and how many trunk steps the branch has "rewound".
- `off_path`: minimum L2 distance from each branch point to the trunk polyline.

## CLI

```
tracer dashboard RUN [--port 8050] [--interval 2]
tracer unlearn   RUN --method NAME --from-step N [--branch NAME] [--every K] [-c key=value ...] [--plugin MODULE]
tracer metrics   RUN --branch NAME [--metric NAME ...] [--plugin MODULE]
tracer info      RUN
```

## Examples and tests

- `examples/mnist_mlp.py`: MLP on MNIST, CPU, about a minute (`--delay 0` for full speed).
- `examples/cifar_resnet18.py`: ResNet-18 on CIFAR-10 with float16 snapshots,
  a snapshot cap and a small landscape budget.

```bash
pytest
```

## Limitations

- The model is loaded with `torch.load(..., weights_only=False)` when no
  `model_factory` is given. Only open run directories you trust.
- PCA fitting loads up to 128 snapshots into memory at once.
- The dashboard's step slider scrubs the trunk; branches appear once the
  selected step reaches their starting step.
- No distributed training, no resuming a trunk after `finish()`.

## Credits

- Li, Xu, Taylor, Studer, Goldstein, *Visualizing the Loss Landscape of Neural
  Nets* (NeurIPS 2018), and its reference code
  [tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape):
  filter normalization (`basis.filter_normalized_direction` is adapted from it)
  and PCA trajectory projection centered on the final weights.
- [loss-landscapes](https://github.com/marcellodebernardi/loss-landscapes) and
  [loss-landscape-anim](https://github.com/logancyang/loss-landscape-anim):
  prior art for the plane evaluation loop and animated PCA paths. Not
  dependencies; their APIs compute a full grid in one blocking call, which
  does not fit an incremental, cached, versioned surface.
- [SciPy](https://scipy.org) (`RBFInterpolator`, `cKDTree`) for interpolation,
  [safetensors](https://github.com/huggingface/safetensors) for snapshots,
  [Plotly Dash](https://dash.plotly.com) for the dashboard.
