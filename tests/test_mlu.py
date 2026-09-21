"""Tests for the MLU init pipeline (M3): uinit task generation, AIMD inputs,
MSD amplitude-proportional picking (synthetic random walk).
"""

import json
import os

import numpy as np
import pytest

from dpgen.generator.run import _is_mlu_fp  # noqa: F401  (import sanity)
from dpgen.mlu.aimd import make_pw_md_input
from dpgen.mlu.msd import (
    compute_msd,
    distribute_by_amplitude,
    pick_frames,
)
from dpgen.mlu.uinit import gen_uinit_tasks

CELL = np.eye(3) * 11.464
COORDS = np.array(
    [(i / 2, j / 2, k / 2) for i in range(2) for j in range(2) for k in range(2)]
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
    for c in COORDS:
        lines.append("  %.8f  %.8f  %.8f" % tuple(c))
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")


@pytest.fixture()
def jdata(tmp_path):
    _write_poscar(tmp_path / "POSCAR")
    return {
        "type_map": ["Fe"],
        "mass_map": [55.845],
        "fp_pp_files": ["Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"],
        "uinit": {
            "points": [
                {"T": 300, "P": 1.0, "structure": str(tmp_path / "POSCAR")},
                {"T": 1000, "P": 10000.0, "structure": str(tmp_path / "POSCAR")},
            ],
            "element": "Fe",
            "manifold": "3d",
            "u0": 1.0,
            "tol": 0.02,
            "max_iter": 5,
        },
    }


def test_gen_uinit_tasks(jdata, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks = gen_uinit_tasks(jdata, {"fp_pw_command": "mpirun pw.x"})
    assert len(tasks) == 2
    for t in tasks:
        assert os.path.isfile(os.path.join(t, "pw.scf.in.000"))
        assert os.path.isfile(os.path.join(t, "pw.hp.in"))
        assert os.path.isfile(os.path.join(t, "run.sh"))
        assert os.path.isfile(os.path.join(t, "uinit.json"))
        assert os.path.isfile(os.path.join(t, "..", "uinit_loop.py"))
        assert os.path.isfile(os.path.join(t, "..", "mlu_frame.py"))
    scf = open(os.path.join(tasks[0], "pw.scf.in.000")).read()
    assert "U Fe-3d 1.000000" in scf
    # 300 K -> degauss
    assert "degauss = 0.001900" in scf
    cfg = json.load(open(os.path.join(tasks[1], "uinit.json")))
    assert cfg["npt"] is True and abs(cfg["press_kbar"] - 10.0) < 1e-9
    cfg0 = json.load(open(os.path.join(tasks[0], "uinit.json")))
    assert cfg0["npt"] is True  # 1 bar > 0 still runs the NPT cell loop


def test_make_pw_md_input():
    inp = make_pw_md_input(
        CELL,
        COORDS,
        [0] * 16,
        ["Fe"],
        {"Fe": "Fe.UPF"},
        {"Fe": 55.845},
        degauss=0.006334,
        ecutwfc=80,
        ecutrho=400,
        kpoints=(2, 2, 2),
        nstep=300,
        dt_fs=1.0,
        ensemble="npt",
        press_kbar=10.0,
        magnetic=True,
        u_value=3.1,
    )
    assert "calculation = 'md'" in inp
    assert "cell_dynamics = 'pr'" in inp
    assert "press = 10.000000" in inp
    assert "ion_temperature = 'nose-hoover'" in inp
    assert "nspin = 2" in inp
    assert "U Fe-3d 3.100000" in inp
    # 1 fs -> ~20.7 a.u.
    assert "dt = 20." in inp
    inp_nvt = make_pw_md_input(
        CELL,
        COORDS,
        [0] * 16,
        ["Fe"],
        {"Fe": "Fe.UPF"},
        {"Fe": 55.845},
        degauss=0.006334,
        ecutwfc=80,
        ecutrho=400,
        kpoints=(2, 2, 2),
        nstep=300,
        dt_fs=2.0,
        ensemble="nvt",
        magnetic=False,
    )
    assert "cell_dynamics" not in inp_nvt
    assert "nspin" not in inp_nvt
    assert "HUBBARD" not in inp_nvt


def _fake_structures(n_frames, amplitude, lattice=CELL):
    """Random-walk 'trajectory' as pymatgen Structures wrapping PBC."""
    from pymatgen.core import Lattice, Structure

    rng = np.random.default_rng(42)
    pos = COORDS.copy()
    lat = Lattice(lattice)
    species = ["Fe"] * 16
    structures = []
    for _ in range(n_frames):
        step = rng.normal(scale=amplitude, size=pos.shape)
        pos = pos + step
        wrapped = pos - np.floor(pos)
        structures.append(Structure(lat, species, wrapped, coords_are_cartesian=False))
    return structures


def test_compute_msd_recovers_amplitude():
    amp = 0.08
    structures = _fake_structures(200, amp)
    msd, amplitude = compute_msd(structures)
    # 3D random walk: MSD ~ 3 * amp^2 per step, grows linearly; late-time
    # sqrt(MSD) must be of the order of the walk extension and positive
    assert amplitude > 3 * amp
    assert msd[0] == 0.0
    # amplitude grows with the walk step size
    _, amp_big = compute_msd(_fake_structures(200, 0.2))
    assert amp_big > amplitude
    # a frozen trajectory has zero amplitude
    frozen = _fake_structures(50, 0.0)
    _, amp0 = compute_msd(frozen)
    assert amp0 == 0.0


def test_distribute_by_amplitude():
    counts = distribute_by_amplitude([1.0, 3.0], budget=20)
    assert counts[1] > counts[0]
    assert sum(counts) >= 20 or counts == [5, 15] or counts == [5, 15]
    counts_eq = distribute_by_amplitude([2.0, 2.0], budget=10)
    assert counts_eq == [5, 5]
    # zero amplitudes: equal split with minimum
    counts_zero = distribute_by_amplitude([0.0, 0.0], budget=4)
    assert counts_zero == [2, 2]


def test_pick_frames():
    idx = pick_frames(100, 5, skip=10)
    assert len(idx) == 5
    assert idx[0] >= 10 and idx[-1] <= 99
    assert np.all(np.diff(idx) > 0)
    idx1 = pick_frames(100, 1, skip=10)
    assert len(idx1) == 1
