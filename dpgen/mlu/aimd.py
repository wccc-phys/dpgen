"""AIMD coarse sampling with pw.x for the MLU initial dataset.

Generates one pw.x ``calculation='md'`` task per (region, T, P): NVT via
Nose-Hoover / Langevin ion control, NPT via Parrinello-Rahman cell dynamics.
Magnetic regions carry nspin=2 + starting_magnetization + a HUBBARD card
seeded from the u_tp bootstrap (or u0). Same pseudopotentials / cutoffs /
k-spacing as the labeling stage so the data is transfer-free.
"""

import json
import os

import numpy as np

from dpgen.generator.lib import qe_hp

FS_TO_AU = 1.0 / 0.0483776866  # femtoseconds -> Hartree a.u. of time


def _nearest_u_tp(u_tp, temp, press):
    if not u_tp:
        return None
    best, best_d = None, None
    for entry in u_tp:
        d = abs(float(entry["T"]) - temp)
        if press is not None and entry.get("P") is not None:
            d += abs(float(entry["P"]) - float(press)) * 1e-3
        if best_d is None or d < best_d:
            best_d, best = d, entry
    return float(best["U"])


def make_pw_md_input(
    cell,
    coords_crystal,
    atom_types,
    type_map,
    pseudo_files,
    mass_map,
    degauss,
    ecutwfc,
    ecutrho,
    kpoints,
    nstep,
    dt_fs,
    ensemble="nvt",
    press_kbar=0.0,
    thermostat="nose-hoover",
    prefix="pw",
    element="Fe",
    manifold="3d",
    magnetic=False,
    u_value=None,
    starting_mag=2.2,
    outdir="./tmp/",
):
    """Generate a pw.x MD input. `ensemble` is 'nvt' or 'npt'."""
    nat = len(atom_types)
    unique_types = sorted(set(atom_types))
    dt_au = dt_fs * FS_TO_AU
    lines = []
    lines.append("&CONTROL")
    lines.append("  calculation = 'md',")
    lines.append(f"  prefix = '{prefix}',")
    lines.append("  pseudo_dir = './',")
    lines.append(f"  outdir = '{outdir}',")
    lines.append(f"  nstep = {nstep},")
    lines.append(f"  dt = {dt_au:.1f},")
    lines.append("  tprnfor = .true.,")
    lines.append("  tstress = .true.,")
    lines.append("  disk_io = 'low',")
    if ensemble == "npt":
        lines.append("  cell_dynamics = 'pr',")
        lines.append(f"  press = {press_kbar:.6f},")
        lines.append("  cell_dofree = 'all',")
        lines.append("  cell_factor = 2.0,")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append("  ibrav = 0,")
    lines.append(f"  nat = {nat},")
    lines.append(f"  ntyp = {len(unique_types)},")
    lines.append(f"  ecutwfc = {ecutwfc},")
    lines.append(f"  ecutrho = {ecutrho},")
    lines.append("  occupations = 'smearing',")
    lines.append("  smearing = 'fd',")
    lines.append(f"  degauss = {degauss:.6f},")
    if magnetic:
        lines.append("  nspin = 2,")
        for i, elem in enumerate([type_map[t] for t in unique_types]):
            lines.append(f"  starting_magnetization({i + 1}) = {starting_mag},")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append("  conv_thr = 1.0e-5,")
    lines.append("  mixing_beta = 0.10,")
    lines.append("  electron_maxstep = 100,")
    lines.append("  mixing_mode = 'local-TF',")
    lines.append("/")
    lines.append("&IONS")
    if thermostat == "nose-hoover":
        lines.append("  ion_temperature = 'nose-hoover',")
        lines.append("  fnhscp = 1.0,")
    elif thermostat == "langevin":
        lines.append("  ion_temperature = 'langevin',")
        lines.append("  friction_therm = 0.02,")
    else:
        raise ValueError(f"unknown thermostat {thermostat}")
    lines.append("/")
    lines.append("CELL_PARAMETERS {angstrom}")
    for i in range(3):
        lines.append(f"  {cell[i, 0]:.6f}  {cell[i, 1]:.6f}  {cell[i, 2]:.6f}")
    lines.append("ATOMIC_SPECIES")
    for t in unique_types:
        elem = type_map[t]
        lines.append(f"  {elem}  {mass_map.get(elem, 1.0)}  {pseudo_files.get(elem, elem + '.UPF')}")
    lines.append("ATOMIC_POSITIONS crystal")
    for idx in range(nat):
        lines.append(
            f"  {type_map[atom_types[idx]]}  "
            f"{coords_crystal[idx, 0]:.8f}  {coords_crystal[idx, 1]:.8f}  "
            f"{coords_crystal[idx, 2]:.8f}"
        )
    k1, k2, k3 = kpoints
    lines.append("K_POINTS gamma")
    if magnetic and u_value is not None and u_value > 0:
        lines.append("HUBBARD atomic")
        lines.append(f"  U {element}-{manifold} {u_value:.6f}")
    return "\n".join(lines) + "\n"


def gen_aimd_tasks(jdata, mdata):
    """Generate AIMD tasks from jdata['aimd'].

    jdata["aimd"]: dict with
      regions: list of {name, structure, T:[K...], P:[bar...],
                        ensemble:'nvt'|'npt', magnetic:bool, nsteps, dt_fs}
      element/manifold/ecutwfc/ecutrho/kspacing/thermostat
      u_tp: optional bootstrap table [{T,P,U}]
      u0: fallback U for magnetic regions
    Task layout: aimd/<region>/<task>/ with pw.md.in + run.sh.
    """
    ai = jdata["aimd"]
    element = ai.get("element", jdata["type_map"][0])
    manifold = ai.get("manifold", "3d")
    kspacing = ai.get("kspacing", 0.6)
    thermostat = ai.get("thermostat", "nose-hoover")
    u_tp = ai.get("u_tp")
    u0 = ai.get("u0", 1.0)
    pseudo_files = dict(zip(jdata["type_map"], jdata.get("fp_pp_files", [])))
    mass_map = dict(zip(jdata["type_map"], jdata["mass_map"]))
    pw_cmd = mdata.get("fp_pw_command", "pw.x")

    import dpdata

    root = "aimd"
    os.makedirs(root, exist_ok=True)
    tasks = []
    for region in ai["regions"]:
        sys_data = dpdata.System(region["structure"], fmt="vasp/poscar").data
        cell = sys_data["cells"][0].reshape([3, 3])
        coords_crystal = qe_hp.cartesian_to_crystal(sys_data["coords"][0], cell)
        atom_types = sys_data["atom_types"]
        kmesh = qe_hp.compute_kmesh(cell, kspacing)
        nsteps = int(region.get("nsteps", 500))
        dt_fs = float(region.get("dt_fs", 1.0))
        ensemble = region.get("ensemble", "nvt")
        magnetic = bool(region.get("magnetic", False))
        for temp in region["T"]:
            for press in region.get("P", [0.0]):
                task_name = "T%04dK_P%08.2fbar" % (temp, press)
                task_dir = os.path.join(root, region["name"], task_name)
                os.makedirs(task_dir, exist_ok=True)
                u_value = None
                if magnetic:
                    u_value = _nearest_u_tp(u_tp, temp, press) or u0
                md_text = make_pw_md_input(
                    cell,
                    coords_crystal,
                    atom_types,
                    jdata["type_map"],
                    pseudo_files,
                    mass_map,
                    degauss=qe_hp.temperature_to_degauss(temp),
                    ecutwfc=ai.get("ecutwfc", 80),
                    ecutrho=ai.get("ecutrho", 400),
                    kpoints=kmesh,
                    nstep=nsteps,
                    dt_fs=dt_fs,
                    ensemble=ensemble,
                    press_kbar=press / 1e3,
                    thermostat=thermostat,
                    element=element,
                    manifold=manifold,
                    magnetic=magnetic,
                    u_value=u_value,
                    starting_mag=ai.get("starting_mag", 2.2),
                )
                with open(os.path.join(task_dir, "pw.md.in"), "w") as fp:
                    fp.write(md_text)
                with open(os.path.join(task_dir, "run.sh"), "w") as fp:
                    fp.write(
                        "#!/bin/bash\n"
                        "# AIMD task generated by dpgen mlu aimd\n"
                        "set -u\n"
                        "exec >> output 2>&1\n"
                        "mkdir -p tmp\n"
                        f"{pw_cmd} -in pw.md.in > pw.md.out 2>&1\n"
                        'if ! grep -q "JOB DONE" pw.md.out; then\n'
                        '  echo "AIMD_FAIL"; touch AIMD_FAILED; exit 1\n'
                        "fi\n"
                        'echo "AIMD_OK"\n'
                    )
                os.chmod(os.path.join(task_dir, "run.sh"), 0o755)
                with open(os.path.join(task_dir, "aimd.json"), "w") as fp:
                    json.dump(
                        {
                            "region": region["name"],
                            "T": temp,
                            "P": press,
                            "ensemble": ensemble,
                            "magnetic": magnetic,
                            "nsteps": nsteps,
                            "dt_fs": dt_fs,
                            "U": u_value,
                        },
                        fp,
                        indent=2,
                    )
                tasks.append(task_dir)
    return tasks
