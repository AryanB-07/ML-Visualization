import numpy as np
import pytest
import torch

from conftest import train_run
from trajmap import BasisManager, Run
from trajmap import explore
from trajmap.explore import explore_once, improve_map, improve_status, pick_init_start, pick_plane_start

CFG = {"lr": 0.2, "batch_size": 32, "momentum": 0.0}


@pytest.fixture
def run(tmp_path):
    d, _, _ = train_run(tmp_path / "r", steps=40, every=4, dataset_factory="conftest:tiny_dataset")
    return Run(d)


def test_plane_start_is_on_the_plane_and_far_from_paths(run):
    bm = BasisManager(run)
    st = pick_plane_start(run, bm, np.random.default_rng(0))
    basis = bm.latest("pca_trunk")
    np.testing.assert_array_equal(st["params"].numpy(), basis.point(st["a"], st["b"]))
    proj = bm.read_projections(basis)["trunk"]
    path = np.column_stack([proj["a"], proj["b"]])
    d = np.linalg.norm(path - [st["a"], st["b"]], axis=1).min()
    assert d == pytest.approx(st["distance"], rel=1e-6)
    # farther from the path than a typical point of the search region
    rng = np.random.default_rng(1)
    lo, hi = path.min(0), path.max(0)
    pad = explore.REGION_PAD * (hi - lo).max()
    rand = rng.uniform(lo - pad, hi + pad, size=(500, 2))
    typical = np.median(np.linalg.norm(rand[:, None] - path[None], axis=2).min(1))
    assert st["distance"] > 1.5 * typical
    # randomised: different seeds give different starts
    other = pick_plane_start(run, bm, np.random.default_rng(5))
    assert (other["a"], other["b"]) != (st["a"], st["b"])


def test_one_exploration_descends_and_joins_the_map(run):
    budget = run.control()["budget"]
    done = improve_map(run, "one", config=CFG, landscape=False, seed=0)
    assert [t.name for t in done] == ["explore-01"]
    t = done[0]
    info = t.info()
    assert t.kind == "explore" and t.finished and info["start"] == "plane" and info["stopped"]
    assert run.trunk.kind == "trunk"
    losses = [r["loss"] for r in t.step_log()]
    assert np.mean(losses[-5:]) < np.mean(losses[:5])  # it went downhill
    # the first snapshot is the chosen in-plane point
    basis = BasisManager(run).latest("pca_trunk")
    np.testing.assert_array_equal(t.load_params(0), basis.point(info["a"], info["b"]))
    # projected with everything else, metrics computed, worker given budget for the new region
    proj, _ = BasisManager(run).update_projections(basis)
    assert proj["explore-01"]["captured"][0] == pytest.approx(1.0, abs=1e-4)
    assert set(t.metrics()) >= {"nearest_trunk", "off_path"}
    assert run.control()["budget"] == budget + explore.DEFAULTS["extra_budget"]
    st = improve_status(run.dir)
    assert st["state"] == "idle" and st["done"] == 1 and not st["alive"]


def test_next_exploration_avoids_the_previous_one(run):
    improve_map(run, "one", config=CFG, landscape=False, seed=0, metrics=False)
    bm = BasisManager(run)
    st = pick_plane_start(run, bm, np.random.default_rng(3))
    proj = bm.read_projections(st["basis_id"])
    assert "explore-01" in proj
    first = np.column_stack([proj["explore-01"]["a"], proj["explore-01"]["b"]])
    assert np.linalg.norm(first - [st["a"], st["b"]], axis=1).min() >= st["distance"] - 1e-9 > 0


def test_time_mode_stops_at_the_deadline(run):
    done = improve_map(run, "time", minutes=0.01, config={**CFG, "plateau_tol": -1e9, "target_ratio": 0.0, "max_steps": 10 ** 9},
                       landscape=False, metrics=False)
    assert len(done) == 1  # only the deadline can end this descent
    assert done[-1].info()["stopped"] == "time budget"
    assert improve_status(run.dir)["result"] == "time budget used"
    with pytest.raises(ValueError):
        improve_map(run, "time")


def test_auto_mode_stops_on_max_runs_or_coverage(run, monkeypatch):
    done = improve_map(run, "auto", max_runs=2, config=CFG, landscape=False, metrics=False)
    assert len(done) == 2 and improve_status(run.dir)["result"] == "finished"
    monkeypatch.setattr(explore, "AUTO_MIN_GAP", 10.0)  # nothing counts as far any more
    assert improve_map(run, "auto", max_runs=5, config=CFG, landscape=False) == []
    assert improve_status(run.dir)["result"].startswith("map covered")


def test_stop_request_ends_the_exploration(run):
    bm = BasisManager(run)
    st = pick_plane_start(run, bm, np.random.default_rng(0))
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] >= 7

    t = explore_once(run, st, config=CFG, should_stop=stop)
    assert t.info()["stopped"] == "stopped" and t.info()["steps"] == 7
    assert t.steps()[-1] == 7  # the weights at the stop are kept


def test_init_start_is_a_fresh_initialization(run):
    st = pick_init_start(run, np.random.default_rng(0))
    assert st["start"] == "init" and st["distance"] > 0
    assert not torch.equal(st["params"], run.trunk.load(0).params)
    done = improve_map(run, "one", start="init", config=CFG, landscape=False, metrics=False)
    assert done[0].info()["start"] == "init"
