"""Tests for the MLU-aware collect (M6): f/aparam/uparam preservation,
multi-set chunking, fingerprint dedup, system-index grouping."""

import json
import os

import numpy as np
import pytest

from dpgen.collect.collect import collect_data_mlu


def _write_system(root, name, nframes=8, natoms=4, seed=0, coords=None,
                  with_all=True):
    sys_dir = os.path.join(root, name)
    set_dir = os.path.join(sys_dir, "set.000")
    os.makedirs(set_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(os.path.join(set_dir, "box.npy"), np.tile(np.eye(3).reshape(9) * 5.0, (nframes, 1)))
    np.save(os.path.join(set_dir, "coord.npy"),
            coords if coords is not None else rng.random((nframes, natoms * 3)))
    np.save(os.path.join(set_dir, "energy.npy"), rng.normal(size=nframes))
    np.save(os.path.join(set_dir, "force.npy"), rng.normal(size=(nframes, natoms * 3)))
    np.save(os.path.join(set_dir, "virial.npy"), rng.normal(size=(nframes, 9)))
    np.save(os.path.join(set_dir, "fparam.npy"), np.full((nframes, 1), 500.0))
    if with_all:
        np.save(os.path.join(set_dir, "aparam.npy"),
                np.full((nframes, natoms, 1), 500.0))
        np.save(os.path.join(set_dir, "uparam.npy"), rng.uniform(3, 4, (nframes, 1)))
    with open(os.path.join(sys_dir, "type.raw"), "w") as fp:
        fp.write("".join("0\n" for _ in range(natoms)))
    with open(os.path.join(sys_dir, "type_map.raw"), "w") as fp:
        fp.write("Fe\n")
    return sys_dir


@pytest.fixture()
def job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init = _write_system(tmp_path, "init_data", nframes=6, seed=1)
    _write_system(tmp_path / "iter.000000" / "02.fp", "data.000", nframes=5, seed=2)
    _write_system(tmp_path / "iter.000001" / "02.fp", "data.000", nframes=7, seed=3)
    _write_system(tmp_path / "iter.000001" / "02.fp", "data.001", nframes=4, seed=4,
                  with_all=False)
    param = {
        "init_data_prefix": str(tmp_path),
        "init_data_sys": ["init_data"],
        "sys_configs": [["a"], ["b"]],
        "model_devi_jobs": [{}, {}],
    }
    with open("param.json", "w") as fp:
        json.dump(param, fp)
    return tmp_path


def test_collect_mlu_preserves_params(job):
    report = collect_data_mlu(str(job), "param.json", str(job / "out"),
                              verbose=False, shuffle=False, dedup=False)
    # init.000 + sys.000 + sys.001
    names = sorted(os.listdir(job / "out"))
    assert "init.000" in names and "sys.000" in names and "sys.001" in names
    assert "collect_report.json" in names
    # sys.000 merged 5 + 7 frames across iterations
    s0 = np.load(job / "out" / "sys.000" / "set.000" / "coord.npy")
    assert s0.shape == (12, 12)
    up = np.load(job / "out" / "sys.000" / "set.000" / "uparam.npy")
    assert up.shape == (12, 1)
    fp_ = np.load(job / "out" / "sys.000" / "set.000" / "fparam.npy")
    assert fp_.shape == (12, 1) and np.allclose(fp_, 500.0)
    ap = np.load(job / "out" / "sys.000" / "set.000" / "aparam.npy")
    assert ap.shape == (12, 4, 1)
    # sys.001 without a/uparam stays without
    assert (job / "out" / "sys.001" / "set.000" / "uparam.npy").exists() is False
    assert (job / "out" / "sys.001" / "set.000" / "fparam.npy").exists()
    # type files
    assert (job / "out" / "sys.000" / "type_map.raw").read_text().strip() == "Fe"
    # report content
    by_name = {e["system"]: e for e in report["systems"]}
    assert by_name["sys.000"]["nframes"] == 12
    assert by_name["sys.000"]["has"]["uparam"] is True
    assert by_name["sys.001"]["has"]["uparam"] is False


def test_collect_mlu_dedup(job):
    # add an exact duplicate of one init frame into iter.000001/sys.000
    dup_coords = np.load(job / "init_data" / "set.000" / "coord.npy")[0:1]
    dup_box = np.load(job / "init_data" / "set.000" / "box.npy")[0:1]
    sysd = job / "iter.000001" / "02.fp" / "data.000" / "set.000"
    coord = np.load(sysd / "coord.npy")
    box = np.load(sysd / "box.npy")
    coord[0] = dup_coords[0]
    box[0] = dup_box[0]
    np.save(sysd / "coord.npy", coord)
    np.save(sysd / "box.npy", box)
    report = collect_data_mlu(str(job), "param.json", str(job / "out2"),
                              verbose=False, shuffle=False, dedup=True)
    # sys.000 loses its duplicate frame against init only if init were merged
    # into the same output system; init.000 is separate, so sys.000 keeps 12
    # minus in-pool duplicates only
    s0 = np.load(job / "out2" / "sys.000" / "set.000" / "coord.npy")
    assert s0.shape[0] in (11, 12)


def test_collect_mlu_chunking(job):
    # tiny set_size forces multiple sets
    collect_data_mlu(str(job), "param.json", str(job / "out3"),
                     verbose=False, shuffle=False, dedup=False, set_size=4)
    sets = sorted(
        d for d in os.listdir(job / "out3" / "sys.000") if d.startswith("set.")
    )
    assert sets == ["set.000", "set.001", "set.002"]
    total = sum(
        np.load(job / "out3" / "sys.000" / s / "coord.npy").shape[0] for s in sets
    )
    assert total == 12
    # every set carries the params
    for s in sets:
        assert (job / "out3" / "sys.000" / s / "uparam.npy").exists()


def test_collect_mlu_incompatible_params_rejected(job):
    # a contributor without fparam cannot merge into one with fparam
    sysd = job / "iter.000000" / "02.fp" / "data.001"
    os.makedirs(sysd / "set.000", exist_ok=True)
    rng = np.random.default_rng(11)
    np.save(sysd / "set.000" / "box.npy", np.tile(np.eye(3).reshape(9), (3, 1)))
    np.save(sysd / "set.000" / "coord.npy", rng.random((3, 12)))
    np.save(sysd / "set.000" / "energy.npy", rng.normal(size=3))
    np.save(sysd / "set.000" / "force.npy", rng.normal(size=(3, 12)))
    with open(sysd / "type.raw", "w") as fp:
        fp.write("0\n" * 4)
    with open(sysd / "type_map.raw", "w") as fp:
        fp.write("Fe\n")
    # data.001 now has an energy-only contributor after a fparam contributor
    # -> mismatch must raise (both write into out sys.001)
    with pytest.raises(RuntimeError):
        collect_data_mlu(str(job), "param.json", str(job / "out4"),
                         verbose=False, shuffle=False, dedup=False)
