import torch

from trajmap.data import modular_addition, spirals
from trajmap.demos import PRESETS, list_runs, train_demo, unique_run_name
from trajmap.explore import improve_map
from trajmap.models import cifar_cnn, modular_mlp, small_cnn, spiral_mlp
from trajmap.store import Run


def test_synthetic_datasets_are_deterministic_and_split():
    a, b = spirals(train=True), spirals(train=True)
    assert torch.equal(a.tensors[0], b.tensors[0])
    assert not torch.equal(a.tensors[0], spirals(train=False).tensors[0])
    assert set(a.tensors[1].tolist()) == {0, 1, 2}

    tr, te = modular_addition(train=True, p=11), modular_addition(train=False, p=11)
    assert len(tr) + len(te) == 121
    pairs = {tuple(x) for x in tr.tensors[0].tolist()} & {tuple(x) for x in te.tensors[0].tolist()}
    assert not pairs                                           # no pair in both splits
    x, y = tr.tensors
    assert torch.equal(y, (x[:, 0] + x[:, 1]) % 11)


def test_preset_models_build():
    assert spiral_mlp()(torch.zeros(4, 2)).shape == (4, 3)
    assert modular_mlp(p=11)(torch.zeros(4, 2, dtype=torch.long)).shape == (4, 11)
    assert small_cnn()(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert cifar_cnn(width=8)(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    for p in PRESETS.values():
        assert p.model.count(":") == 1 and p.dataset.count(":") == 1


def test_train_demo_records_a_reopenable_run(tmp_path):
    run = train_demo("spirals", tmp_path / "s", steps=120, landscape=False)
    assert run.trunk.finished and run.trunk.steps()[-1] == 120
    assert run.meta["config"]["demo"] == "spirals"
    assert len(run.forget_indices()) == PRESETS["spirals"].n_forget
    x, _ = run.eval_data()
    assert len(x) == 2048
    m = Run(tmp_path / "s").load_model()                       # rebuilt from model_factory
    assert m(torch.zeros(1, 2)).shape == (1, 3)

    rows = list_runs(tmp_path)
    assert [r["name"] for r in rows] == ["s"] and rows[0]["finished"] and rows[0]["demo"] == "spirals"
    assert unique_run_name(tmp_path, "s") == "s-2"


def test_exploration_uses_the_runs_optimizer(tmp_path):
    run = train_demo("spirals", tmp_path / "s", steps=80, landscape=False)
    assert run.meta["config"]["optimizer"] == "adamw"
    [t] = improve_map(run, "one", start="init", config={"max_steps": 30}, landscape=False, metrics=False)
    assert t.info()["config"]["optimizer"] == "adamw"
    assert t.steps()[-1] == 30


def test_dashboard_switches_between_runs_in_a_folder(tmp_path):
    from trajmap.dashboard import _Runs, _run_options, build_app

    empty = _Runs(tmp_path / "none", "cpu")
    assert empty.name is None and _run_options(empty) == []
    build_app(tmp_path / "none", start_worker=False).layout()   # empty state renders

    train_demo("spirals", tmp_path / "a", steps=40, landscape=False)
    train_demo("spirals", tmp_path / "b", steps=40, landscape=False)
    runs = _Runs(tmp_path / "a", "cpu")                          # a run dir: its siblings are switchable
    assert runs.root == tmp_path and runs.name == "a"
    assert {o["value"] for o in _run_options(runs)} == {"a", "b"}
    runs.switch("b")
    assert runs.st.run.dir == tmp_path / "b"
    assert _Runs(tmp_path, "cpu").name == "b"                    # a folder opens the newest run
