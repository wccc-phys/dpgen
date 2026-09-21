"""Integration tests for the MLU pwscf labeling chain (M2).

Runs make_fp_pwscf_mlu on fake task dirs, fabricates converged QE outputs
from the validated mlu_model cluster samples, then checks
post_fp_pwscf_mlu writes DeepMD npy data with fparam/aparam/uparam.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from dpgen.generator.run import (
    _is_mlu_fp,
    _make_mlu_run_sh,
    make_fp_pwscf_mlu,
    post_fp_pwscf_mlu,
)

MLU_MODEL_SAMPLES = Path = (
    "/personal/wangc/fe/mlu/mlu_model/datasets/1bar-1000k/set.000/hp_calc/frame_0000"
)
HAS_SAMPLES = os.path.isfile(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"))

# bcc Fe 2x2x2, a=5.732 Angstrom, 16 atoms
CELL = np.eye(3) * 11.464
COORDS_FRAC = np.array(
    [
        (i / 2, j / 2, k / 2)
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    + [
        ((i + 0.25) / 2, (j + 0.25) / 2, (k + 0.25) / 2)
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
)


def _write_poscar(path):
    lines = ["bcc Fe", "1.0"]
    for row in CELL:
        lines.append("  %.8f  %.8f  %.8f" % tuple(row))
    lines.append("  Fe")
    lines.append("  16")
    lines.append("Direct")
    for c in COORDS_FRAC:
        lines.append("  %.8f  %.8f  %.8f" % tuple(c))
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")


def _make_jdata(tmp_path, uparam_mode="frame"):
    return {
        "type_map": ["Fe"],
        "mass_map": [55.845],
        "use_ele_temp": 1,
        "magnetic_sys_idx": [0],
        "fp_style": "pwscf",
        "fp_pp_path": str(tmp_path / "pp"),
        "fp_pp_files": ["Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"],
        "fp_params": {
            "ecutwfc": 80,
            "ecutrho": 400,
            "kspacing": 0.6,
            "conv_thr": 1e-5,
        },
        "fp_hubbard": {
            "element": "Fe",
            "manifold": "3d",
            "u_init": 1.0,
            "u_init_source": "const",
            "uparam_mode": uparam_mode,
            "starting_mag": 2.2,
        },
    }


@pytest.fixture()
def fp_workspace(tmp_path, monkeypatch):
    """Fake iter.000000/02.fp with one magnetic and one non-magnetic task."""
    monkeypatch.chdir(tmp_path)
    pp_dir = tmp_path / "pp"
    pp_dir.mkdir()
    upf = pp_dir / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"
    upf.write_text("<UPF fake/>")
    work = tmp_path / "iter.000000" / "02.fp"
    for sys_idx, magnetic in ((0, True), (1, False)):
        task = work / ("task.%03d.%06d" % (sys_idx, 0))
        task.mkdir(parents=True)
        _write_poscar(task / "POSCAR")
        with open(task / "job.json", "w") as fp:
            json.dump({"ele_temp": 1000.0, "press": 1.0}, fp)
    return tmp_path


def test_run_sh_content():
    sh = _make_mlu_run_sh(
        True,
        {"fp_pw_command": "mpirun -np 96 pw.x", "fp_hp_command": "mpirun -np 96 hp.x"},
        "Fe",
        "3d",
        "atomic",
        False,
        "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
        55.845,
    )
    assert "mpirun -np 96 pw.x -in pw.scf.in" in sh
    assert "mpirun -np 96 hp.x -in pw.hp.in" in sh
    assert "--u-mode atomic" in sh
    assert "--pseudo Fe.pbe-spn-kjpaw_psl.1.0.0.UPF" in sh
    assert "--mass 55.845" in sh
    assert "touch CHAIN_DONE" in sh
    # the continue-outside-loop bug must not be reintroduced
    assert "continue" not in sh
    sh_nm = _make_mlu_run_sh(
        False, {}, "Fe", "3d", "frame", False, "Fe.UPF", 55.845
    )
    assert "hp.x" not in sh_nm


def test_make_fp_pwscf_mlu(fp_workspace):
    jdata = _make_jdata(fp_workspace)
    mdata = {"fp_pw_command": "pw.x", "fp_hp_command": "hp.x"}
    assert _is_mlu_fp(jdata)
    make_fp_pwscf_mlu(0, jdata, mdata)
    work = fp_workspace / "iter.000000" / "02.fp"
    assert os.path.isfile(work / "mlu_frame.py")
    mag = work / "task.000.000000"
    nm = work / "task.001.000000"
    # magnetic: scf + hp + chain script
    scf_in = (mag / "pw.scf.in").read_text()
    assert "nspin = 2" in scf_in
    assert "U Fe-3d 1.000000" in scf_in
    assert "degauss = 0.006334" in scf_in  # 1000 K
    assert "starting_magnetization(1) = 2.2" in scf_in
    assert (mag / "pw.hp.in").exists()
    assert "hp.x" in (mag / "run.sh").read_text()
    # non-magnetic: plain scf, no hubbard, no hp
    scf_nm = (nm / "pw.scf.in").read_text()
    assert "HUBBARD" not in scf_nm
    assert "nspin = 1" in scf_nm
    assert not (nm / "pw.hp.in").exists()
    assert "hp.x" not in (nm / "run.sh").read_text()
    # pp linked into both task dirs
    for task in (mag, nm):
        assert (task / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF").exists()


@pytest.mark.skipif(not HAS_SAMPLES, reason="mlu_model cluster samples unavailable")
def test_post_fp_pwscf_mlu(fp_workspace):
    jdata = _make_jdata(fp_workspace)
    mdata = {"fp_pw_command": "pw.x", "fp_hp_command": "hp.x"}
    make_fp_pwscf_mlu(0, jdata, mdata)
    work = fp_workspace / "iter.000000" / "02.fp"

    # fabricate converged results: magnetic chain OK, non-magnetic chain OK
    mag, nm = work / "task.000.000000", work / "task.001.000000"
    shutil.copy(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"), mag / "pw.scf.out")
    shutil.copy(
        os.path.join(MLU_MODEL_SAMPLES, "Fe.Hubbard_parameters.dat"),
        mag / "pw.Hubbard_parameters.dat",
    )
    shutil.copy(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"), mag / "pw.scf_new.out")
    (mag / "CHAIN_DONE").touch()
    shutil.copy(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"), nm / "pw.scf.out")
    (nm / "CHAIN_DONE").touch()

    post_fp_pwscf_mlu(0, jdata)

    for sys_idx, has_u in ((0, True), (1, False)):
        setdir = work / ("data.%03d" % sys_idx) / "set.000"
        box = np.load(setdir / "box.npy")
        coord = np.load(setdir / "coord.npy")
        energy = np.load(setdir / "energy.npy")
        force = np.load(setdir / "force.npy")
        assert box.shape == (1, 9)
        assert coord.shape == (1, 48)
        assert energy.shape == (1,)
        assert force.shape == (1, 48)
        fparam = np.load(setdir / "fparam.npy")
        assert fparam.shape == (1, 1) and fparam[0, 0] == 1000.0
        assert np.isfinite(energy).all()
        if has_u:
            uparam = np.load(setdir / "uparam.npy")
            assert uparam.shape == (1, 1)  # frame mode averaged
            assert 2.5 < uparam[0, 0] < 4.0
        else:
            assert not (setdir / "uparam.npy").exists()
        # type files written
        assert (setdir.parent / "type.raw").exists()
        assert (setdir.parent / "type_map.raw").exists()


@pytest.mark.skipif(not HAS_SAMPLES, reason="mlu_model cluster samples unavailable")
def test_post_fp_pwscf_mlu_atomic(fp_workspace):
    jdata = _make_jdata(fp_workspace, uparam_mode="atomic")
    mdata = {"fp_pw_command": "pw.x", "fp_hp_command": "hp.x"}
    make_fp_pwscf_mlu(0, jdata, mdata)
    work = fp_workspace / "iter.000000" / "02.fp"
    mag = work / "task.000.000000"
    shutil.copy(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"), mag / "pw.scf.out")
    shutil.copy(
        os.path.join(MLU_MODEL_SAMPLES, "Fe.Hubbard_parameters.dat"),
        mag / "pw.Hubbard_parameters.dat",
    )
    shutil.copy(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"), mag / "pw.scf_new.out")
    (mag / "CHAIN_DONE").touch()
    nm = work / "task.001.000000"
    shutil.copy(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"), nm / "pw.scf.out")
    (nm / "CHAIN_DONE").touch()
    post_fp_pwscf_mlu(0, jdata)
    uparam = np.load(work / "data.000" / "set.000" / "uparam.npy")
    assert uparam.shape == (1, 16)  # atomic mode: per atom


def test_post_fp_skips_failed_chain(fp_workspace):
    jdata = _make_jdata(fp_workspace)
    jdata["ratio_failed"] = 0.6  # allow the 50% failure this scenario creates
    mdata = {"fp_pw_command": "pw.x", "fp_hp_command": "hp.x"}
    make_fp_pwscf_mlu(0, jdata, mdata)
    work = fp_workspace / "iter.000000" / "02.fp"
    # non-magnetic chain finished, magnetic chain did not (no CHAIN_DONE)
    nm = work / "task.001.000000"
    shutil.copy(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"), nm / "pw.scf.out")
    (nm / "CHAIN_DONE").touch()
    # scf converged but chain marked failed -> magnetic frame skipped
    mag = work / "task.000.000000"
    shutil.copy(os.path.join(MLU_MODEL_SAMPLES, "Fe.scf.out"), mag / "pw.scf.out")
    (mag / "CHAIN_FAILED").touch()
    post_fp_pwscf_mlu(0, jdata)
    assert not (work / "data.000").exists()
    assert (work / "data.001" / "set.000" / "energy.npy").exists()
