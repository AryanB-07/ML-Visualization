import numpy as np
import pytest
import torch

from conftest import train_run
from trajmap import BasisManager, GridCache, Run
from trajmap.basis import Basis, filter_normalized_direction
from trajmap.landscape import MAX_LEVEL, Evaluator, SurfaceModel, grid_bounds, plan_points
from trajmap.params import assign_params, flatten_params
from trajmap.tracer import Recorder
from trajmap.store import TrajectoryWriter
from trajmap.worker import run_worker


def test_exact_weight_reconstruction_from_ab(run_dir):
    run = Run(run_dir)
    basis = BasisManager(run).fit("pca_trunk")
    rng = np.random.default_rng(0)
    for a, b in rng.uniform(-3, 3, size=(5, 2)):
        w = basis.point(a, b)
        expected = basis.center + np.float32(a) * basis.d1 + np.float32(b) * basis.d2
        np.testing.assert_array_equal(w, expected.astype(np.float32))
        a2, b2 = basis.coords(w)  # and (a, b) is recovered from the weights
        assert a2 == pytest.approx(a, abs=1e-4) and b2 == pytest.approx(b, abs=1e-4)
        assert basis.fidelity(w)["captured"] == pytest.approx(1.0, abs=1e-5)
    # the weights load into the model and give back the same vector
    model = run.build_model()
    w = basis.point(0.5, -1.25)
    assign_params(model, torch.from_numpy(w))
    np.testing.assert_array_equal(flatten_params(model).numpy(), w)
    # the center is the final trunk snapshot, at (0, 0)
    np.testing.assert_array_equal(basis.point(0, 0), run.trunk.load_params(run.trunk.steps()[-1]))


def test_pca_basis_is_orthonormal_and_reports_fidelity(run_dir):
    run = Run(run_dir)
    bm = BasisManager(run)
    basis = bm.fit("pca_trunk")
    assert np.dot(basis.d1, basis.d1) == pytest.approx(1, abs=1e-5)
    assert np.dot(basis.d1, basis.d2) == pytest.approx(0, abs=1e-5)
    evr = basis.info["explained_variance_ratio"]
    assert 0 < evr[1] <= evr[0] <= 1
    proj, changed = bm.update_projections(basis)
    assert changed and proj["trunk"]["steps"] == run.trunk.steps()
    cap = np.array(proj["trunk"]["captured"])
    assert ((cap >= 0) & (cap <= 1)).all()
    # captured + residual account for the whole displacement
    k = 0
    w = run.trunk.load_params(proj["trunk"]["steps"][k])
    total = np.linalg.norm(w.astype(np.float64) - basis.center)
    assert proj["trunk"]["norm"][k] == pytest.approx(total, rel=1e-6)
    assert proj["trunk"]["residual"][k] == pytest.approx(total * np.sqrt(1 - cap[k]), rel=1e-4)


def test_fidelity_known_answer():
    center = np.zeros(4, np.float32)
    basis = Basis("pca_trunk", 1, center, np.array([1, 0, 0, 0], np.float32), np.array([0, 1, 0, 0], np.float32))
    f = basis.fidelity(np.array([3, 0, 4, 0], np.float32))
    assert (f["a"], f["b"]) == (3.0, 0.0)
    assert f["captured"] == pytest.approx(9 / 25) and f["residual"] == pytest.approx(4) and f["norm"] == pytest.approx(5)


def test_filter_normalization_matches_filter_norms():
    spec = [{"name": "w", "shape": [4, 5]}, {"name": "b", "shape": [4]}]
    rng = np.random.default_rng(0)
    center = rng.standard_normal(24).astype(np.float32)
    d = filter_normalized_direction(center, spec, rng)
    np.testing.assert_allclose(np.linalg.norm(d[:20].reshape(4, 5), axis=1),
                               np.linalg.norm(center[:20].reshape(4, 5), axis=1), rtol=1e-5)
    assert not d[20:].any()  # biases are left out of the direction


def test_basis_versioning_and_cache_invalidation(tmp_path):
    torch.manual_seed(0)
    from conftest import make_data, tiny_model
    from trajmap import Tracer
    model = tiny_model()
    x, y = make_data()
    tracer = Tracer(model, tmp_path / "r", eval_loader=[(x[:64], y[:64])], every_n_steps=1,
                    model_factory="conftest:tiny_model", landscape=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.2)

    def train(n):
        for _ in range(n):
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tracer.step(loss=loss.item())

    run = Run(tmp_path / "r")
    bm = BasisManager(run)
    assert not bm.needs_refit("pca_trunk", 3)  # one snapshot is not enough
    train(3)
    assert bm.needs_refit("pca_trunk", 3)
    v1 = bm.fit("pca_trunk")
    assert v1.id == "pca_trunk-v0001" and not bm.needs_refit("pca_trunk", 3)

    cache = GridCache(run.dir, v1.id, {"eval_batches": 0, "bn_mode": "recalibrate"})
    cache.add((0, 0), 1.5, 0.5)
    cache.add((8, -8), 2.5, 0.25)
    assert GridCache.read(run.dir, v1.id) == {(0, 0): (1.5, 0.5), (8, -8): (2.5, 0.25)}

    train(2)
    assert not bm.needs_refit("pca_trunk", 3)  # only 2 new snapshots
    train(1)
    assert bm.needs_refit("pca_trunk", 3)
    v2 = bm.fit("pca_trunk")
    assert v2.id == "pca_trunk-v0002" and bm.versions("pca_trunk") == [1, 2]
    assert bm.latest("pca_trunk").id == v2.id
    assert v2.info["center"] == ["trunk", 6]
    # everything computed in v1 coordinates is gone
    assert GridCache.read(run.dir, v1.id) == {}
    assert not (run.dir / "landscape" / v1.id).exists()
    assert not (run.dir / "bases" / "pca_trunk" / "v0001.safetensors").exists()
    assert GridCache(run.dir, v2.id, {"eval_batches": 0, "bn_mode": "recalibrate"}).points == {}

    # a finished run with any new snapshot refits even below the threshold
    train(1)
    assert not bm.needs_refit("pca_trunk", 3)
    tracer.finish()
    assert bm.needs_refit("pca_trunk", 3)


def test_cache_cleared_when_eval_settings_change(run_dir):
    c = GridCache(run_dir, "pca_trunk-v0001", {"eval_batches": 1, "bn_mode": "center"})
    c.add((0, 0), 1.0, None)
    assert GridCache(run_dir, "pca_trunk-v0001", {"eval_batches": 1, "bn_mode": "center"}).points == {(0, 0): (1.0, None)}
    assert GridCache(run_dir, "pca_trunk-v0001", {"eval_batches": 2, "bn_mode": "center"}).points == {}


def test_refinement_is_coarse_first_then_near_the_path():
    path = [np.array([[0.0, 0.0], [4.0, 0.0]])]
    h = 1.0
    todo = plan_points(path, h, set(), level=2, budget=10_000)
    c = 2 ** MAX_LEVEL
    i0, i1, j0, j1 = grid_bounds(path[0], h)
    n_coarse = ((i1 - i0) // c + 1) * ((j1 - j0) // c + 1)
    assert all(i % c == 0 and j % c == 0 for i, j in todo[:n_coarse])  # the whole coarse grid comes first
    assert len(set(todo)) == len(todo)
    fine = np.array(todo[n_coarse:], dtype=float) / c
    dist = np.abs(fine[:, 1])  # distance to the path's line, good enough inside its span
    assert dist[:20].mean() < dist[-20:].mean()
    # budget and already-evaluated points are respected
    done = set(todo[:30])
    rest = plan_points(path, h, done, level=2, budget=50)
    assert len(rest) == 20 and not (set(rest) & done)
    assert plan_points(path, h, done, level=2, budget=30) == []


def test_worker_fills_surface_and_grid_matches_direct_evaluation(run_dir):
    run = Run(run_dir)
    run.set_control(budget=120, resolution=17, eval_batches=0)
    run_worker(run_dir, once=True)
    bm = BasisManager(run)
    basis = bm.latest("pca_trunk")
    points = GridCache.read(run_dir, basis.id)
    assert len(points) == 120
    assert bm.read_projections(basis)["trunk"]["steps"] == run.trunk.steps()
    # a cached grid value equals evaluating the reconstructed weights directly
    ev = Evaluator(run)
    u = basis.info["coarse_h"] / 2 ** MAX_LEVEL
    (i, j), (loss, acc) = next(iter(points.items()))
    direct = ev.evaluate(basis.point(i * u, j * u))
    assert direct["loss"] == pytest.approx(loss, rel=1e-6) and direct["acc"] == acc
    # the origin is the final trunk snapshot: its grid loss is the model's real eval loss
    x, y = run.eval_data()
    real = torch.nn.functional.cross_entropy(run.load_model()(x), y).item()
    assert points[(0, 0)][0] == pytest.approx(real, rel=1e-5)
    # the interpolated surface reproduces evaluated points
    surf = SurfaceModel(points, basis.info["coarse_h"])
    assert surf(np.array([[i * u, j * u]]))[0] == pytest.approx(loss, rel=1e-3)
    # nothing left to do: a second pass evaluates nothing
    run_worker(run_dir, once=True)
    assert len(GridCache.read(run_dir, basis.id)) == 120


def test_batchnorm_recalibration_vs_center_stats(tmp_path):
    d, _, _ = train_run(tmp_path / "r", steps=20, every=4, factory="conftest:bn_model")
    run = Run(d)
    ev = Evaluator(run)
    snap = run.trunk.load(20)
    assert len(ev.bn_layers) == 1
    stored = ev.evaluate(snap.params, snap.buffers, snap.int_buffers, bn_mode="stored")
    again = ev.evaluate(snap.params, snap.buffers, snap.int_buffers, bn_mode="stored")
    assert stored["loss"] == again["loss"]  # deterministic, eval mode
    far = snap.params + 0.5
    recal = ev.evaluate(far, snap.buffers, snap.int_buffers, bn_mode="recalibrate")
    center = ev.evaluate(far, snap.buffers, snap.int_buffers, bn_mode="center")
    assert recal["loss"] != center["loss"]
    assert not any(m.training for m in ev.bn_layers)
