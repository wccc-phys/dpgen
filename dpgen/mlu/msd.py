"""MSD-based frame picking for AIMD trajectories (pymatgen-based).

Reads the pw.x MD output trajectory (through ASE's espresso-out reader,
converted to pymatgen Structures), computes the PBC-unwrapped mean-square
displacement with pymatgen lattice primitives, and distributes the picking
budget across trajectories proportionally to the vibrational amplitude
(more vibration -> more representative configurations kept).

Output per trajectory: an unlabeled DeepMD npy system (box, coord, fparam=T)
plus a manifest recording amplitude / n_picked / magnetic flag, ready for the
fp labeling stage.
"""

import glob
import json
import os

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def read_pw_md_trajectory(md_out):
    """Read all MD frames from a pw.x output as pymatgen Structures."""
    from ase.io import read as ase_read

    images = ase_read(md_out, index=":", format="espresso-out")
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_structure(atoms) for atoms in images]


def compute_msd(structures):
    """PBC-unwrapped MSD(t) of a trajectory (list of pymatgen Structures).

    Returns (msd[nframes], amplitude) where amplitude = sqrt(MSD) evaluated
    over the last quarter of the run (plateau region); for a diffusing
    liquid this is the late-time displacement, for a vibrating solid the
    vibration amplitude. Frames with fewer than 4 entries return zeros.
    """
    if len(structures) < 4:
        return np.zeros(len(structures)), 0.0
    ref = structures[0]
    # fractional-space unwrapping: accumulate minimum-image deltas
    lat = ref.lattice
    frac_prev = np.array(ref.frac_coords)
    frac_unwrapped = np.zeros_like(frac_prev)
    unwrapped = []
    for s in structures:
        frac_curr = np.array(s.frac_coords)
        dfrac = frac_curr - frac_prev
        # minimum image in fractional space
        dfrac -= np.round(dfrac)
        frac_unwrapped = frac_unwrapped + dfrac
        frac_prev = frac_curr
        unwrapped.append(frac_unwrapped.copy())
    unwrapped = np.array(unwrapped)  # (nframes, natoms, 3) fractional
    d = unwrapped - unwrapped[0]
    disp_cart = np.einsum("fai,ij->faj", d, lat.matrix)  # frac -> cart
    msd = (disp_cart**2).sum(axis=2).mean(axis=1)  # average over atoms
    tail = msd[int(len(msd) * 3 / 4) :]
    amplitude = float(np.sqrt(max(tail.mean(), 0.0)))
    return msd, amplitude


def pick_frames(n_total, n_pick, skip=0):
    """Evenly spaced frame indices, skipping the equilibration prefix."""
    lo, hi = skip, n_total - 1
    if hi <= lo:
        return np.array([lo]) if n_pick > 0 else np.array([], dtype=int)
    if n_pick <= 1:
        return np.array([(lo + hi) // 2])
    return np.linspace(lo, hi, n_pick).round().astype(int)


def distribute_by_amplitude(amplitudes, budget, min_per_traj=1):
    """Split `budget` picks across trajectories proportional to amplitude."""
    amps = np.asarray(amplitudes, dtype=float)
    n = len(amps)
    if n == 0:
        return []
    if amps.sum() <= 0:
        raw = np.full(n, budget / n)
    else:
        raw = budget * amps / amps.sum()
    counts = np.maximum(np.round(raw).astype(int), min_per_traj)
    return counts.tolist()


def pick_aimd_frames(jdata):
    """Select initial-set frames from finished AIMD tasks.

    jdata["msd_pick"]: dict with
      traj_glob : glob of pw.x MD outputs (default 'aimd/*/*/pw.md.out')
      budget    : total frames to pick across all trajectories
      skip      : equilibration steps skipped (in frames)
      out_dir   : output root (default 'init_data')
    Writes <out_dir>/<region>.<T>K.<P>bar/set.000/{box,coord,fparam}.npy +
    type.raw/type_map.raw and a manifest.json.
    """
    mp = jdata["msd_pick"]
    traj_glob = mp.get("traj_glob", "aimd/*/*/pw.md.out")
    budget = int(mp["budget"])
    skip = int(mp.get("skip", 10))
    out_dir = mp.get("out_dir", "init_data")
    type_map = jdata["type_map"]
    os.makedirs(out_dir, exist_ok=True)

    metas, amplitudes = [], []
    for md_out in sorted(glob.glob(traj_glob)):
        task_dir = os.path.dirname(md_out)
        with open(os.path.join(task_dir, "aimd.json")) as fp:
            meta = json.load(fp)
        metas.append((md_out, meta))
    if not metas:
        raise RuntimeError(f"no AIMD outputs matched {traj_glob}")

    # amplitude first pass (also validates trajectories parse)
    for md_out, meta in metas:
        structures = read_pw_md_trajectory(md_out)
        _, amp = compute_msd(structures)
        amplitudes.append(amp)

    counts = distribute_by_amplitude(amplitudes, budget, min_per_traj=1)
    manifest = []
    for (md_out, meta), n_pick, amp in zip(metas, counts, amplitudes):
        structures = read_pw_md_trajectory(md_out)
        idx = pick_frames(len(structures), n_pick, skip=skip)
        sys_name = "%s.%04dK.%08.2fbar" % (meta["region"], meta["T"], meta["P"])
        sys_dir = os.path.join(out_dir, sys_name)
        set_dir = os.path.join(sys_dir, "set.000")
        os.makedirs(set_dir, exist_ok=True)
        boxes, coords = [], []
        for ii in idx:
            s = structures[ii]
            boxes.append(np.array(s.lattice.matrix).reshape(9))
            coords.append(np.array(s.cart_coords).reshape(-1))
        boxes, coords = np.array(boxes), np.array(coords)
        np.save(os.path.join(set_dir, "box.npy"), boxes)
        np.save(os.path.join(set_dir, "coord.npy"), coords)
        # fparam: frame temperature = MD target temperature
        np.save(
            os.path.join(set_dir, "fparam.npy"),
            np.full((len(idx), 1), float(meta["T"])),
        )
        with open(os.path.join(sys_dir, "type.raw"), "w") as fp:
            atom_types = [
                type_map.index(str(sp.symbol)) for sp in structures[0].species
            ]
            fp.write("".join(f"{t}\n" for t in atom_types))
        with open(os.path.join(sys_dir, "type_map.raw"), "w") as fp:
            fp.write("".join(f"{t}\n" for t in type_map))
        entry = dict(
            meta, trajectory=md_out, amplitude=amp, n_frames=len(structures),
            picked=len(idx), picked_idx=[int(i) for i in idx],
        )
        manifest.append(entry)
        print(
            f"[msd] {sys_name}: {len(structures)} frames, amplitude "
            f"{amp:.4f} A^2^0.5 -> picked {len(idx)}"
        )
    with open(os.path.join(out_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    return manifest
