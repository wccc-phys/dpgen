"""Tests for the committee descriptor-union selection (M5).

Uses synthetic systems and monkeypatched descriptor computation (no real
models needed) to validate: pool assembly with magnetic split, per-model
minimal subsets, union semantics, already-labeled subtraction via
fingerprints, and fp task dir generation.
"""

import json
import os

import numpy as np
import pytest

from dpgen.generator.lib import mlu_select


def _write_system(root, name, nframes=20, natoms=4, seed=0, labeled=True,
                  with_uparam=True, coords=None):
    sys_dir = os.path.join(root, name)
    set_dir = os.path.join(sys_dir, "set.000")
    os.makedirs(set_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    box = np.tile(np.eye(3).reshape(9) * 5.0, (nframes, 1))
    coord = coords if coords is not None else rng.random((nframes, natoms * 3)) * 4
    np.save(os.path.join(set_dir, "box.npy"), box)
    np.save(os.path.join(set_dir, "coord.npy"), coord)
    if labeled:
        np.save(os.path.join(set_dir, "energy.npy"), rng.normal(size=nframes))
        np.save(os.path.join(set_dir, "force.npy"), rng.normal(size=(nframes, natoms * 3)))
        np.save(os.path.join(set_dir, "virial.npy"), rng.normal(size=(nframes, 9)) * 0.1)
    np.save(os.path.join(set_dir, "fparam.npy"), np.full((nframes, 1), 300.0))
    if with_uparam:
        np.save(os.path.join(set_dir, "uparam.npy"), rng.uniform(3, 4, (nframes, 1)))
    with open(os.path.join(sys_dir, "type.raw"), "w") as fp:
        fp.write("".join("0\n" for _ in range(natoms)))
    with open(os.path.join(sys_dir, "type_map.raw"), "w") as fp:
        fp.write("Fe\n")
    return sys_dir


def _write_modd_traj(task_dir, nframes=10, natoms=4, seed=1):
    os.makedirs(os.path.join(task_dir, "traj"), exist_ok=True)
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 6.0
    lines = []
    for f in range(nframes):
        lines.append("ITEM: TIMESTEP")
        lines.append(str((f + 1) * 10))
        lines.append("ITEM: NUMBER OF ATOMS")
        lines.append(str(natoms))
        lines.append("ITEM: BOX BOUNDS pp pp pp")
        for i in range(3):
            lines.append(f"0.0 {cell[i][i]:.6f}")
        lines.append("ITEM: ATOMS id type x y z")
        pos = rng.random((natoms, 3)) * 5.5
        for a in range(natoms):
            lines.append(f"{a + 1} 1 {pos[a][0]:.6f} {pos[a][1]:.6f} {pos[a][2]:.6f}")
    with open(os.path.join(task_dir, "traj", "10.lammpstrj"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    with open(os.path.join(task_dir, "job.json"), "w") as fp:
        json.dump({"ele_temp": 300.0, "press": 1.0}, fp)


def test_load_system_multiset_and_params(tmp_path):
    sys_dir = _write_system(tmp_path, "mag_sys")
    # add a second set with 5 more frames
    rng = np.random.default_rng(9)
    os.makedirs(os.path.join(sys_dir, "set.001"))
    np.save(os.path.join(sys_dir, "set.001", "box.npy"), np.tile(np.eye(9).ravel()[:9], (5, 1)))
    np.save(os.path.join(sys_dir, "set.001", "coord.npy"), rng.random((5, 12)))
    np.save(os.path.join(sys_dir, "set.001", "energy.npy"), rng.normal(size=5))
    np.save(os.path.join(sys_dir, "set.001", "force.npy"), rng.normal(size=(5, 12)))
    np.save(os.path.join(sys_dir, "set.001", "virial.npy"), rng.normal(size=(5, 9)))
    np.save(os.path.join(sys_dir, "set.001", "fparam.npy"), np.full((5, 1), 300.0))
    np.save(os.path.join(sys_dir, "set.001", "uparam.npy"), rng.uniform(3, 4, (5, 1)))
    sysd = mlu_select.load_system(sys_dir)
    assert sysd["coord"].shape == (25, 12)
    assert sysd["uparam"].shape == (25, 1)
    assert sysd["fparam"].shape == (25, 1)
    assert sysd["natoms"] == 4
    # unlabeled exploration-style system
    unlab = _write_system(tmp_path, "unlabeled", labeled=False, with_uparam=False)
    ud = mlu_select.load_system(unlab)
    assert ud["energy"] is None and ud["uparam"] is None


def test_fingerprint_dedup(tmp_path):
    sys_dir = _write_system(tmp_path, "s", nframes=3)
    sysd = mlu_select.load_system(sys_dir)
    fps = mlu_select.system_fingerprints(sysd)
    assert len(fps) == 3
    assert mlu_select.frame_fingerprint(sysd["box"][0], sysd["coord"][0]) in fps


def test_selection_convergence_and_metrics():
    rng = np.random.default_rng(0)
    n = 60
    feats = rng.normal(size=(n, 8))
    ref = mlu_select._fit_reference(feats)
    energy = rng.normal(size=n)
    force_norms = np.abs(rng.normal(size=n)) + 0.5
    thresholds = {"hull_coverage": 0.5, "kde_js": 0.3}
    idx, traj = mlu_select.select_minimal_subset(
        feats, ref, thresholds, energy=energy, force_norms=force_norms
    )
    assert len(idx) <= n
    assert 0.2 - 1e-9 <= len(idx) / n <= 1.0 + 1e-9
    # passing fraction found
    assert any(ok for _, _, ok in traj)
    # metrics sanity at full selection
    m_full = mlu_select.evaluate_metrics(np.arange(n), ref, feats, energy,
                                          force_norms, None)
    assert m_full["kde_js"] < 0.05
    assert m_full["hull_coverage"] == 1.0


def test_union_selects_exploration_not_labeled(tmp_path, monkeypatch):
    # workspace: init labeled system + iter0 fp data + iter1 exploration
    init = _write_system(tmp_path, "init_bcc", nframes=12, seed=2)  # magnetic by name
    os.makedirs(tmp_path / "iter.000000" / "02.fp" / "data.001")
    _write_system(
        tmp_path / "iter.000000" / "02.fp", "data.001", nframes=8, seed=3
    )  # non-magnetic (sys 1)
    modd_task = str(tmp_path / "iter.000001" / "01.model_devi" / "task.000.000000")
    _write_modd_traj(modd_task, nframes=10, seed=4)
    os.makedirs(tmp_path / "iter.000001" / "00.train")
    for m in ("graph.000.pth", "graph.001.pth"):
        open(tmp_path / "iter.000001" / "00.train" / m, "w").close()

    # deterministic synthetic descriptors: identity-like features from coords
    def fake_desc(system, model_path):
        rng = np.random.default_rng(hash(model_path) % (2**32))
        n = system["coord"].shape[0]
        base = system["coord"].copy()
        if "graph.001" in model_path:
            base = base * np.array([1.01, 0.99, 1.0] * (base.shape[1] // 3))
        feats = np.concatenate([base, np.zeros((n, 4))], axis=1) + rng.normal(
            scale=1e-3, size=(n, base.shape[1] + 4)
        )
        return feats

    monkeypatch.setattr(mlu_select, "compute_descriptors", fake_desc)
    monkeypatch.chdir(tmp_path)

    jdata = {
        "type_map": ["Fe"],
        "init_data_prefix": str(tmp_path),
        "init_data_sys": ["init_bcc"],
        "magnetic_sys_idx": [0],
        "mlu_select": {"seed": 1, "kde_js": 0.3, "hull_coverage": 0.6},
    }
    candidates, report = mlu_select.mlu_select_fp_candidates(1, jdata, {})
    # every candidate must be an exploration frame of the modd task
    assert report["n_candidates"] == len(candidates)
    for c in candidates:
        assert c["task"] == os.path.relpath(modd_task, str(tmp_path))
        assert 0 <= c["frame"] < 10
        assert c["sys_idx"] == 0
        assert c["magnetic"] is True
        assert len(c["models"]) >= 1
    # union produced something but bounded by pool size
    assert 1 <= len(candidates) <= 10
    # report has per-model per-system records for the magnetic group only
    assert report["groups"] == {"magnetic": 2, "non_magnetic": 1}  # init + modd | iter0 data.001
    assert len(report["models"]) == 2  # one record per committee model


def test_labeled_subtraction_by_fingerprint(tmp_path, monkeypatch):
    """Exploration frames identical to labeled init frames are dropped."""
    rng = np.random.default_rng(7)
    shared = rng.random((6, 12)) * 4
    init = _write_system(
        tmp_path, "init_bcc", nframes=6, seed=5,
        coords=shared,
    )
    modd_task = str(tmp_path / "iter.000000" / "01.model_devi" / "task.000.000000")
    # modd traj frames with the SAME coordinates (same box 5.0 vs 6.0 -> use 5)
    os.makedirs(os.path.join(modd_task, "traj"))
    cell = np.eye(3) * 5.0
    lines = []
    for f in range(6):
        lines += ["ITEM: TIMESTEP", str((f + 1) * 10), "ITEM: NUMBER OF ATOMS", "4",
                  "ITEM: BOX BOUNDS pp pp pp"]
        for i in range(3):
            lines.append(f"0.0 {cell[i][i]:.6f}")
        lines.append("ITEM: ATOMS id type x y z")
        for a in range(4):
            x, y, z = shared[f].reshape(4, 3)[a]
            lines.append(f"{a + 1} 1 {x:.6f} {y:.6f} {z:.6f}")
    open(os.path.join(modd_task, "traj", "10.lammpstrj"), "w").write("\n".join(lines) + "\n")
    json.dump({"ele_temp": 300.0}, open(os.path.join(modd_task, "job.json"), "w"))
    os.makedirs(tmp_path / "iter.000000" / "00.train")
    open(tmp_path / "iter.000000" / "00.train" / "graph.000.pth", "w").close()

    monkeypatch.setattr(
        mlu_select,
        "compute_descriptors",
        lambda system, model_path: np.asarray(system["coord"], dtype=float).copy(),
    )
    monkeypatch.chdir(tmp_path)
    jdata = {
        "type_map": ["Fe"],
        "init_data_prefix": str(tmp_path),
        "init_data_sys": ["init_bcc"],
        "magnetic_sys_idx": [0],
        "mlu_select": {"seed": 1},
    }
    candidates, report = mlu_select.mlu_select_fp_candidates(0, jdata, {})
    assert candidates == []
    assert report["n_candidates"] == 0
