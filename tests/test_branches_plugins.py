import numpy as np
import pytest
import torch

from conftest import make_data, write_trajectory
from trajmap import (BasisManager, Run, TrajectoryMetric, UnlearningMethod, compute_metrics, list_methods,
                     list_metrics, register_method, register_metric, run_unlearning)
from trajmap.metrics import get_metric
from trajmap.store import TrajectoryWriter
from trajmap.unlearning import get_method
from trajmap.worker import run_worker


# ------------------------------------------------------------------- branches
def test_branch_creation(run_dir):
    run = Run(run_dir)
    w = run.create_branch("b1", from_step=20, method="manual", config={"k": 1})
    b = run.trajectory("b1")
    assert run.branch_names() == ["b1"] and [t.name for t in run.trajectories()] == ["trunk", "b1"]
    assert b.from_step == 20 and b.info()["method"] == "manual" and b.info()["parent"] == "trunk"
    assert b.steps() == [0]
    np.testing.assert_array_equal(b.load_params(0), run.trunk.load_params(20))  # starts at the trunk snapshot
    w.save_snapshot(1, torch.from_numpy(b.load_params(0)) + 1.0)
    assert b.steps() == [0, 1]
    assert run.trunk.steps() == list(range(0, 41, 4))  # trunk untouched
    with pytest.raises(FileExistsError):
        run.create_branch("b1", from_step=20)
    with pytest.raises(ValueError):
        run.create_branch("b2", from_step=21)  # no snapshot at that step
    with pytest.raises(ValueError):
        run.create_branch("trunk", from_step=20)


def test_gradient_ascent_records_a_branch_on_the_shared_map(run_dir):
    run = Run(run_dir)
    x, y = make_data()
    branch = run_unlearning(run, "gradient_ascent", from_step=20, forget_loader=[(x[:10], y[:10])],
                            config={"steps": 6, "lr": 0.05}, landscape=False)
    assert branch.name == "gradient_ascent-from20" and branch.finished
    assert branch.steps() == list(range(7))
    losses = [r["loss"] for r in branch.step_log()]
    assert losses[-1] > losses[0]  # ascent on the forget loss
    assert set(branch.metrics()) == {"nearest_trunk", "off_path"}
    assert branch.metrics()["off_path"]["series"]["off_path_distance"][0] == pytest.approx(0, abs=1e-6)
    # a second run from the same step gets its own name
    again = run_unlearning(run, "gradient_ascent", from_step=20, forget_loader=[(x[:10], y[:10])],
                           config={"steps": 2}, landscape=False)
    assert again.name == "gradient_ascent-from20-2"

    # projections cover trunk and branches in the same basis, for trunk-only and joint fits
    bm = BasisManager(run)
    for mode in ("pca_trunk", "pca_joint"):
        basis = bm.fit(mode)
        proj, _ = bm.update_projections(basis)
        assert set(proj) == {"trunk", branch.name, again.name}
        k = proj["trunk"]["steps"].index(20)
        assert proj[branch.name]["a"][0] == pytest.approx(proj["trunk"]["a"][k], abs=1e-6)
        assert proj[branch.name]["b"][0] == pytest.approx(proj["trunk"]["b"][k], abs=1e-6)
    assert len(bm.latest("pca_trunk").info["fit_keys"]) == 10
    assert len(bm.latest("pca_joint").info["fit_keys"]) == 10 + 7 + 3


def test_joint_basis_refits_when_a_branch_appears(run_dir):
    run = Run(run_dir)
    run.set_control(mode="pca_joint", budget=20)
    run_worker(run_dir, once=True)
    bm = BasisManager(run)
    assert bm.latest("pca_joint").version == 1
    bm.fit("pca_trunk")
    x, y = make_data()
    run_unlearning(run, "gradient_ascent", 20, forget_loader=[(x[:10], y[:10])], config={"steps": 3}, landscape=False)
    assert bm.needs_refit("pca_joint", 5) and not bm.needs_refit("pca_trunk", 5)
    run_worker(run_dir, once=True)
    assert bm.latest("pca_joint").version == 2


# ------------------------------------------------------------------ registries
def test_method_registry():
    assert "gradient_ascent" in list_methods()

    @register_method("noop_test")
    class Noop(UnlearningMethod):
        def setup(self, model, forget_loader, retain_loader, config): self.n = 0
        def step(self, model): self.n += 1; return {"loss": 0.0, "extra": 1.0}
        def is_done(self): return self.n >= 3

    assert get_method("noop_test") is Noop and Noop.name == "noop_test"
    with pytest.raises(ValueError):
        @register_method("noop_test")
        class Other(UnlearningMethod):
            def setup(self, *a): ...
            def step(self, model): return {}
            def is_done(self): return True
    with pytest.raises(TypeError):
        register_method("bad")(object)
    with pytest.raises(KeyError):
        get_method("does_not_exist")


def test_registered_method_runs_as_branch(run_dir):
    @register_method("noop_run_test")
    class Noop(UnlearningMethod):
        def setup(self, model, forget_loader, retain_loader, config): self.n = 0
        def step(self, model): self.n += 1; return {"loss": 0.0, "extra": 1.0}
        def is_done(self): return self.n >= 3

    b = run_unlearning(Run(run_dir), "noop_run_test", 8, forget_loader=[None], branch="noop", landscape=False)
    assert b.steps() == [0, 1, 2, 3] and b.step_log()[0]["extra"] == 1.0
    np.testing.assert_array_equal(b.load_params(3), b.load_params(0))  # the method changed nothing
    assert max(b.metrics()["off_path"]["series"]["off_path_distance"]) == pytest.approx(0, abs=1e-6)


def test_metric_registry(run_dir):
    assert {"nearest_trunk", "off_path"} <= set(list_metrics())

    @register_metric("norm_test")
    class Norm(TrajectoryMetric):
        description = "L2 norm of each branch snapshot"
        def compute(self, branch, trunk):
            return {"norm": np.linalg.norm(branch.matrix(), axis=1)}

    assert get_metric("norm_test") is Norm
    run = Run(run_dir)
    run.create_branch("b", from_step=8)
    out = compute_metrics(run, "b", ["norm_test"])
    assert out["norm_test"]["series"]["norm"][0] == pytest.approx(np.linalg.norm(run.trunk.load_params(8)), rel=1e-6)
    assert "norm_test" in run.trajectory("b").metrics()
    with pytest.raises(ValueError):
        compute_metrics(run, "trunk")


# -------------------------------------------------- metrics with known answers
@pytest.fixture
def synthetic(tmp_path):
    """Trunk polyline (0,0,0) -> (10,0,0) -> (10,10,0) at steps 0, 10, 20, and a branch from step 20."""
    from trajmap.store import atomic_write_json
    d = tmp_path / "syn"
    atomic_write_json(d / "meta.json", {"num_params": 3, "snapshot_dtype": "float32"})
    trunk = TrajectoryWriter(d)
    write_trajectory(trunk, np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float))
    run = Run(d)
    w = run.create_branch("b", from_step=20)
    pts = [[10, 6, 0],     # on the second segment
           [10, 4, 3],     # 3 off the second segment, nearer the middle snapshot
           [4, 0, 0],      # on the first segment: the branch has rewound past the corner
           [-3, 4, 0],     # beyond the start: closest to the first snapshot
           [13, 14, 0]]    # beyond the end: closest to the last snapshot
    for k, p in enumerate(pts, start=1):
        w.save_snapshot(k, torch.tensor(p, dtype=torch.float32))
    return run


def test_nearest_trunk_snapshot_known_answers(synthetic):
    s = compute_metrics(synthetic, "b", ["nearest_trunk"])["nearest_trunk"]["series"]
    #                                   start       p1          p2             p3    p4    p5
    assert s["nearest_trunk_step"] == [20, 20, 10, 0, 0, 20]
    assert s["nearest_trunk_index"] == [2, 2, 1, 0, 0, 2]
    np.testing.assert_allclose(s["distance"], [0, 4, 5, 4, 5, 5], atol=1e-6)
    assert s["rewind_steps"] == [0, 0, 10, 20, 20, 0]


def test_off_path_distance_known_answers(synthetic):
    s = compute_metrics(synthetic, "b", ["off_path"])["off_path"]["series"]
    np.testing.assert_allclose(s["off_path_distance"], [0, 0, 3, 0, 5, 5], atol=1e-6)
    np.testing.assert_allclose(s["trunk_position"], [20, 16, 14, 4, 0, 20], atol=1e-6)
