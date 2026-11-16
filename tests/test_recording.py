import json

import numpy as np
import pytest
import torch

from conftest import bn_model, make_data, tiny_model, train_run
from trajmap import Run, Tracer
from trajmap.params import flatten_buffers, flatten_params


def test_snapshot_round_trip_is_exact(tmp_path):
    d, model, _ = train_run(tmp_path / "r", steps=40, every=4)
    run = Run(d)
    assert run.trunk.steps() == list(range(0, 41, 4))  # init and final included
    snap = run.trunk.load(40)
    assert torch.equal(snap.params, flatten_params(model))
    rebuilt = run.load_model()
    for (k, a), (_, b) in zip(model.state_dict().items(), rebuilt.state_dict().items()):
        assert torch.equal(a, b), k


def test_final_snapshot_saved_off_interval(tmp_path):
    d, model, _ = train_run(tmp_path / "r", steps=10, every=4)
    run = Run(d)
    assert run.trunk.steps() == [0, 4, 8, 10]
    assert run.trunk.finished
    assert torch.equal(run.trunk.load(10).params, flatten_params(model))


def test_batchnorm_buffers_round_trip(tmp_path):
    d, model, _ = train_run(tmp_path / "r", steps=12, every=4, factory="conftest:bn_model")
    rebuilt = Run(d).load_model()
    fb, ib = flatten_buffers(model)
    fb2, ib2 = flatten_buffers(rebuilt)
    assert fb.numel() > 0 and torch.equal(fb, fb2) and torch.equal(ib, ib2)
    x, _ = make_data()
    model.eval()
    assert torch.equal(model(x[:16]), rebuilt.eval()(x[:16]))


def test_float16_and_max_snapshots(tmp_path):
    d, model, tracer = train_run(tmp_path / "r", steps=64, every=2, dtype="float16", max_snapshots=9)
    run = Run(d)
    steps = run.trunk.steps()
    assert len(steps) <= 9 and steps[0] == 0 and steps[-1] == 64
    assert tracer.every_n_steps > 2  # interval doubled when thinning
    files = sorted(p.name for p in (run.dir / "trunk" / "snapshots").iterdir())
    assert len(files) == len(steps)  # thinned files are deleted
    snap = run.trunk.load(64)
    assert snap.params.dtype == torch.float32
    assert torch.allclose(snap.params, flatten_params(model), atol=1e-2)


def test_metadata_and_step_log(run_dir):
    run = Run(run_dir)
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["seed"] == 0 and meta["model_factory"] == "conftest:tiny_model"
    assert [s["shape"] for s in meta["param_spec"]] == [[8, 6], [8], [3, 8], [3]]
    assert meta["num_params"] == 8 * 6 + 8 + 3 * 8 + 3
    log = run.trunk.step_log()
    assert len(log) == 40 and log[0]["step"] == 1 and len(log[0]["idx"]) == 32
    forget = set(run.forget_indices().tolist())
    assert forget == set(range(10))
    for rec in log:
        assert rec["n_forget"] == sum(i in forget for i in rec["idx"])


def test_forget_deltas_off_by_default_and_exact_when_on(tmp_path):
    d, _, _ = train_run(tmp_path / "off", steps=8, every=1)
    assert Run(d).trunk.delta_steps() == []
    d, _, _ = train_run(tmp_path / "on", steps=8, every=1, save_forget_deltas=True)
    trunk = Run(d).trunk
    with_forget = [r["step"] for r in trunk.step_log() if r["n_forget"] > 0]
    assert with_forget and trunk.delta_steps() == with_forget
    s = with_forget[0]
    np.testing.assert_array_equal(trunk.load_delta(s), trunk.load_params(s) - trunk.load_params(s - 1))


def test_existing_run_is_not_overwritten(run_dir):
    with pytest.raises(FileExistsError):
        Tracer(tiny_model(), run_dir, landscape=False)


def test_nonfinite_weights_are_not_snapshotted(tmp_path):
    model = tiny_model()
    tracer = Tracer(model, tmp_path / "r", every_n_steps=1, landscape=False, model_factory="conftest:tiny_model")
    tracer.step(loss=1.0)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    tracer.step(loss=float("nan"))
    tracer.finish()
    assert Run(tmp_path / "r").trunk.steps() == [0, 1]
