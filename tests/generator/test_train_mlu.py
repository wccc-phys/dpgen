"""Tests for the MLU dual-track training (M4).

make_train_dp creates task.mlu when mlu_training_param is set and uparam
data exists; run_train_dp includes it in the dispatched tasks; post_train_dp
exposes the frozen model as 00.train/mlu_model* (not graph.*, which the
exploration stage would mistake for a committee energy model).
"""

import json
import os

import numpy as np
import pytest

from dpgen.generator.run import (
    _make_train_mlu_task,
    _resolve_mlu_model_path,
    _system_has_uparam,
    copy_model,
)


def _make_system(root, name, natoms=8, with_uparam=True, nframes=3):
    sys_dir = os.path.join(root, name)
    set_dir = os.path.join(sys_dir, "set.000")
    os.makedirs(set_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    np.save(os.path.join(set_dir, "box.npy"), np.tile(np.eye(3).reshape(9), (nframes, 1)))
    np.save(os.path.join(set_dir, "coord.npy"), rng.random((nframes, natoms * 3)))
    np.save(os.path.join(set_dir, "energy.npy"), rng.random(nframes))
    np.save(os.path.join(set_dir, "force.npy"), rng.random((nframes, natoms * 3)))
    np.save(os.path.join(set_dir, "fparam.npy"), np.full((nframes, 1), 300.0))
    if with_uparam:
        np.save(os.path.join(set_dir, "uparam.npy"), rng.uniform(3.0, 3.5, (nframes, 1)))
    with open(os.path.join(sys_dir, "type.raw"), "w") as fp:
        fp.write("".join("0\n" for _ in range(natoms)))
    with open(os.path.join(sys_dir, "type_map.raw"), "w") as fp:
        fp.write("Fe\n")
    return sys_dir


MLU_PARAM = {
    "model": {
        "type": "standard",
        "descriptor": {"type": "se_e2_a", "rcut": 6.0, "rcut_smth": 1.0,
                        "neuron": [25, 50, 100], "seed": 1},
        "fitting_net": {"type": "mlu", "numb_fparam": 1, "neuron": [120, 120, 120],
                         "intensive": None, "task_dim": 1, "seed": 1},
    },
    "loss": {"type": "mlu", "loss_func": "smooth_mae", "metric": ["mae"]},
    "training": {"numb_steps": 1000, "seed": 10,
                  "training_data": {"systems": "placeholder"}},
}


def test_system_has_uparam(tmp_path):
    with_u = _make_system(tmp_path, "mag", with_uparam=True)
    without_u = _make_system(tmp_path, "nonmag", with_uparam=False)
    assert _system_has_uparam(with_u)
    assert not _system_has_uparam(without_u)
    assert not _system_has_uparam(os.path.join(tmp_path, "does-not-exist"))


def test_make_train_mlu_task_filters_and_writes(tmp_path):
    work_path = tmp_path / "iter.000000" / "00.train"
    os.makedirs(work_path / "000")
    mag = _make_system(tmp_path, "init_mag", with_uparam=True)
    nonmag = _make_system(tmp_path, "init_nonmag", with_uparam=False)
    os.symlink(mag, work_path / "data.init.mag")
    os.symlink(nonmag, work_path / "data.init.nonmag")
    init_data_sys = ["../data.init.mag", "../data.init.nonmag"]
    jdata = {"mlu_training_param": json.loads(json.dumps(MLU_PARAM))}
    _make_train_mlu_task(0, jdata, str(work_path), init_data_sys, ["auto", "auto"], 1)
    task_dir = work_path / "mlu"
    assert (task_dir / "input.json").is_file()
    param = json.loads((task_dir / "input.json").read_text())
    systems = param["training"]["training_data"]["systems"]
    assert systems == ["../data.init.mag"]
    assert param["training"]["training_data"]["batch_size"] == ["auto"]
    # seeds must have been re-randomized
    assert param["model"]["descriptor"]["seed"] != 1
    assert param["training"]["seed"] != 10
    # idempotent
    _make_train_mlu_task(0, jdata, str(work_path), init_data_sys, ["auto", "auto"], 1)
    assert len(list(task_dir.iterdir())) == 1


def test_make_train_mlu_task_skips_without_uparam(tmp_path):
    work_path = tmp_path / "iter.000000" / "00.train"
    os.makedirs(work_path / "000")
    _make_system(tmp_path, "init_nonmag", with_uparam=False)
    os.symlink(
        os.path.join(tmp_path, "init_nonmag"), work_path / "data.init.nonmag"
    )
    jdata = {"mlu_training_param": json.loads(json.dumps(MLU_PARAM))}
    _make_train_mlu_task(
        0, jdata, str(work_path), ["../data.init.nonmag"], ["auto"], 1
    )
    assert not (work_path / "mlu").exists()


def test_make_train_mlu_task_disabled(tmp_path):
    work_path = tmp_path / "iter.000000" / "00.train"
    os.makedirs(work_path)
    _make_train_mlu_task(0, {}, str(work_path), [], [], 0)
    assert not (work_path / "mlu").exists()


def test_resolve_mlu_model_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_mlu_model_path(None, 3) is None
    assert _resolve_mlu_model_path("/abs/path/mlu.pth", 0) == "/abs/path/mlu.pth"
    # no model anywhere yet
    assert _resolve_mlu_model_path("auto", 2) is None
    # iter 1 has one (post_train product)
    os.makedirs("iter.000001/00.train/mlu")
    open("iter.000001/00.train/mlu_model.pth", "w").close()
    assert _resolve_mlu_model_path("auto", 3) == os.path.abspath(
        "iter.000001/00.train/mlu_model.pth"
    )
    # never resolves to the current iteration
    os.makedirs("iter.000002/00.train")
    open("iter.000002/00.train/mlu_model.pth", "w").close()
    assert _resolve_mlu_model_path("auto", 2) == os.path.abspath(
        "iter.000001/00.train/mlu_model.pth"
    )


def test_copy_model_carries_mlu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("iter.000000/00.train/task.000")
    open("iter.000000/00.train/task.000/frozen_model.pth", "w").close()
    os.makedirs("iter.000000/00.train/mlu")
    open("iter.000000/00.train/mlu_model.pth", "w").close()
    copy_model(1, 0, 1, ".pth")
    assert os.path.islink("iter.000001/00.train/graph.000.pth")
    assert os.path.islink("iter.000001/00.train/mlu_model.pth")
    assert os.path.isfile("iter.000001/00.train/mlu_model.pth")
    # a previous iteration without an MLU model links nothing
    os.makedirs("iter.000010/00.train/000")
    open("iter.000010/00.train/000/frozen_model.pth", "w").close()
    copy_model(1, 10, 11, ".pth")
    assert not os.path.exists("iter.000011/00.train/mlu_model.pth")
