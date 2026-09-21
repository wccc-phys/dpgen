"""QE DFT+U / Hubbard-U labeling support for the MLU workflow.

Ported from the validated ``mlu_model/qe_dfpt.py`` driver (QE 7.4.1 cluster)
and adapted for dpgen:

- pw.x SCF input generation with magnetism (nspin=2, starting_magnetization)
  and a HUBBARD card whose U comes from frame or per-atom ``uparam`` data
  (per-atom U is realized through the pseudo-species trick Fe1..FeN);
- hp.x input generation;
- extraction of per-atom U from hp.x ``{prefix}.Hubbard_parameters.dat``;
- pw.x output parsing (energy, forces, stress) and deepmd virial conversion;
- QE input parsing (geometry/species), including merging pseudo-species back
  to the element-level type map.

All unit conventions follow mlu_model: energies Ry->eV, forces Ry/au->eV/Ang,
stress Ry/bohr^3 -> virial W = -sigma * V in eV, degauss = T[K] * KB_RY.
"""

import math
import re

import numpy as np

BOHR_TO_ANG = 0.529177210903
RY_TO_EV = 13.605693009
RY_AU_TO_EV_ANG = RY_TO_EV / BOHR_TO_ANG  # Ry/au -> eV/Ang
KB_RY = 6.3336e-6  # Boltzmann constant in Ry/K

# Fallback atomic masses (amu); dpgen callers should pass jdata["mass_map"]
# keyed by element so pseudo-species blocks resolve correctly.
ATOMIC_MASSES = {"Fe": 55.845, "H": 1.008, "O": 15.999}


# ============================================================
# Temperature / k-mesh utilities
# ============================================================


def temperature_to_degauss(temperature_K):
    """Convert electronic temperature (K) to QE degauss (Ry)."""
    return temperature_K * KB_RY


def degauss_to_temperature(degauss_Ry):
    """Inverse of :func:`temperature_to_degauss` (Ry -> K)."""
    return degauss_Ry / KB_RY


def compute_kmesh(cell, kspacing):
    """Compute a K-point mesh from a cell matrix (3,3, Angstrom) and KSPACING (1/Ang)."""
    cell = np.asarray(cell, dtype=float)
    a1, a2, a3 = cell[0], cell[1], cell[2]
    vol = abs(np.dot(a1, np.cross(a2, a3)))
    if vol == 0:
        raise ValueError("Cell volume is zero.")
    b1 = np.cross(a2, a3) / vol * 2 * math.pi
    b2 = np.cross(a3, a1) / vol * 2 * math.pi
    b3 = np.cross(a1, a2) / vol * 2 * math.pi
    return (
        max(1, int(round(np.linalg.norm(b1) / kspacing))),
        max(1, int(round(np.linalg.norm(b2) / kspacing))),
        max(1, int(round(np.linalg.norm(b3) / kspacing))),
    )


def cartesian_to_crystal(coords_cart, cell):
    """Cartesian (Angstrom) -> fractional (crystal) coordinates."""
    return np.asarray(coords_cart, dtype=float) @ np.linalg.inv(np.asarray(cell, float))


# ============================================================
# Hubbard U extraction from hp.x output
# ============================================================


def extract_u_from_hubbard_dat(filename, element="Fe", manifold="3d"):
    """Extract the raw per-site U list (eV) from hp.x ``Hubbard_parameters.dat``.

    Rows are matched on ``parts[2]==element and parts[5]==element and
    parts[6]==manifold``; the value is ``parts[7]``.
    """
    with open(filename, errors="ignore") as fp:
        text = fp.read().splitlines()
    values = []
    for line in text:
        parts = line.split()
        if len(parts) >= 8:
            if parts[2] == element and parts[5] == element and parts[6] == manifold:
                try:
                    values.append(float(parts[7]))
                except ValueError:
                    pass
    return np.array(values, dtype=float)


# ============================================================
# pw.x output parsing
# ============================================================

_SCF_FAIL_MARKERS = (
    "Maximum number of iterations",
    "convergence NOT achieved",
    "too many iterations",
    "electronic convergence not achieved",
)
_SCF_OK_MARKERS = (
    "End of self-consistent calculation",
    "convergence has been achieved",
)


def check_scf_convergence(scf_out_path):
    """Inspect a pw.x output; return completed/converged/maxstep/reason."""
    import os

    if not os.path.isfile(scf_out_path):
        return {
            "completed": False,
            "converged": False,
            "maxstep": False,
            "reason": "scf_out_missing",
        }
    with open(scf_out_path, errors="ignore") as fp:
        text = fp.read()
    completed = "JOB DONE." in text
    maxstep = any(m in text for m in _SCF_FAIL_MARKERS)
    converged = (not maxstep) and any(m in text for m in _SCF_OK_MARKERS)
    if not converged and re.search(r"^!.*total energy\s*=", text, re.MULTILINE):
        converged = True
        maxstep = False
    if maxstep:
        reason = "scf_maxstep_not_converged"
    elif not converged:
        reason = "scf_not_converged"
    elif not completed:
        reason = "scf_no_job_done"
    else:
        reason = "ok"
    return {
        "completed": completed,
        "converged": converged,
        "maxstep": maxstep,
        "reason": reason,
    }


def parse_scf_out(scf_out_path, natoms):
    """Parse energy (Ry), forces (Ry/au) and stress (Ry/bohr^3) from a pw.x output."""
    path = str(scf_out_path)
    conv = check_scf_convergence(path)
    result = {
        "converged": conv["converged"],
        "reason": conv["reason"],
        "energy_Ry": None,
        "forces_Ry_au": None,
        "stress_Ry_bohr3": None,
    }
    import os

    if not os.path.isfile(path):
        return result
    with open(path, errors="ignore") as fp:
        text = fp.read()

    m = re.search(r"^!\s*total energy\s*=\s*(-?[0-9.Ee+-]+)\s*Ry", text, re.MULTILINE)
    if m:
        result["energy_Ry"] = float(m.group(1))

    forces = []
    for fm in re.finditer(
        r"atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*"
        r"(-?[0-9.Ee+-]+)\s+(-?[0-9.Ee+-]+)\s+(-?[0-9.Ee+-]+)",
        text,
    ):
        forces.append([float(fm.group(2)), float(fm.group(3)), float(fm.group(4))])
    if forces:
        forces = forces[-natoms:]
        if len(forces) == natoms:
            result["forces_Ry_au"] = np.array(forces, dtype=float)

    sm = re.search(
        r"total\s+stress\s*\(Ry/bohr\*\*3\)[^\n]*\n"
        r"\s*(-?[0-9.Ee+-]+)\s+(-?[0-9.Ee+-]+)\s+(-?[0-9.Ee+-]+)[^\n]*\n"
        r"\s*(-?[0-9.Ee+-]+)\s+(-?[0-9.Ee+-]+)\s+(-?[0-9.Ee+-]+)[^\n]*\n"
        r"\s*(-?[0-9.Ee+-]+)\s+(-?[0-9.Ee+-]+)\s+(-?[0-9.Ee+-]+)",
        text,
    )
    if sm:
        result["stress_Ry_bohr3"] = np.array(
            [float(sm.group(i)) for i in range(1, 10)], dtype=float
        ).reshape(3, 3)
    return result


def stress_to_virial(stress_Ry_bohr3, cell_angstrom):
    """QE stress (Ry/bohr^3) -> deepmd virial W = -sigma * V (eV)."""
    stress = np.asarray(stress_Ry_bohr3, dtype=float)
    cell = np.asarray(cell_angstrom, dtype=float)
    vol = abs(np.linalg.det(cell))
    stress_ev_A3 = stress * RY_TO_EV / (BOHR_TO_ANG**3)
    return -stress_ev_A3 * vol


# ============================================================
# QE input parsing (geometry / species / degauss)
# ============================================================

_QE_BLOCK_NAMES = [
    "ATOMIC_SPECIES",
    "ATOMIC_POSITIONS",
    "CELL_PARAMETERS",
    "K_POINTS",
    "CONSTRAINTS",
    "OCCUPATIONS",
    "ATOMIC_FORCES",
    "ATOMIC_VELOCITIES",
    "HUBBARD",
    "ADDITIONAL_K_POINTS",
    "SOLVENTS",
]


def _qe_load_block(lines, keyword, max_lines=None):
    for idx, ln in enumerate(lines):
        if keyword in ln:
            blk_idx = idx + 1
            while blk_idx < len(lines) and lines[blk_idx].strip() == "":
                blk_idx += 1
            body = []
            while blk_idx < len(lines):
                s = lines[blk_idx].strip()
                if s == "":
                    blk_idx += 1
                    continue
                first = s.split()[0]
                if first in _QE_BLOCK_NAMES or first in ("&", "/"):
                    break
                body.append(s)
                blk_idx += 1
                if max_lines and len(body) >= max_lines:
                    break
            return body
    return []


def _qe_load_key(lines, key):
    for line in lines:
        if key not in line:
            continue
        for token in line.replace(",", " ").split():
            if "=" in token:
                lhs, rhs = token.split("=", 1)
                if key in lhs:
                    return rhs
        stripped = line.replace(",", " ").replace("=", " = ").split()
        for i, token in enumerate(stripped):
            if token == "=" and i > 0 and i + 1 < len(stripped):
                if key in stripped[i - 1]:
                    return stripped[i + 1]
    return None


def parse_qe_input(text):
    """Parse a pw.x input; return (cell, species_labels, atom_types, natoms, degauss).

    cell is (3,3) Angstrom; atom_types indexes into species_labels; degauss in Ry
    (None when absent).
    """
    lines = text.splitlines()

    ibrav = 0
    for line in lines:
        if "ibrav" in line:
            for token in line.replace(",", " ").split():
                if "ibrav" in token and "=" in token:
                    try:
                        ibrav = int(token.split("=")[1])
                    except ValueError:
                        pass

    # ibrav=0 with CELL_PARAMETERS is what this workflow always generates;
    # a few common Bravais lattices are supported for user-provided templates.
    if ibrav != 0:
        raise ValueError(
            "qe_hp.parse_qe_input: only ibrav=0 (CELL_PARAMETERS) inputs are "
            f"supported, got ibrav={ibrav}"
        )
    blk = _qe_load_block(lines, "CELL_PARAMETERS", 3)
    if not blk:
        raise ValueError("CELL_PARAMETERS block not found (ibrav=0)")
    cell = np.array([[float(x) for x in ln.split()[:3]] for ln in blk])

    species_blk = _qe_load_block(lines, "ATOMIC_SPECIES")
    species_labels = [ln.split()[0] for ln in species_blk]

    pos_blk = _qe_load_block(lines, "ATOMIC_POSITIONS")
    atom_labels = [ln.split()[0] for ln in pos_blk]
    natoms = len(atom_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(species_labels)}
    atom_types = np.array([label_to_idx[lbl] for lbl in atom_labels], dtype=int)

    degauss = None
    val = _qe_load_key(lines, "degauss")
    if val is not None:
        degauss = float(val)

    return cell, list(species_labels), atom_types, natoms, degauss


def merge_pseudo_species(species_labels, atom_types):
    """Collapse per-atom pseudo-species (Fe1, Fe2, ...) back to one element.

    Returns (type_map, atom_types, is_merged).
    """
    if not species_labels:
        return [], np.array([], dtype=int), False
    bases = list(dict.fromkeys([re.sub(r"\d+$", "", lbl) for lbl in species_labels]))
    if len(bases) == 1 and len(species_labels) > 1:
        return bases, np.zeros(len(atom_types), dtype=int), True
    return list(species_labels), np.array(atom_types, dtype=int).copy(), False


# ============================================================
# pw.x SCF input generation
# ============================================================


def _normalize_u_mode(u_mode):
    um = (u_mode or "frame").lower()
    if um in ("fparam", "frame"):
        return "frame"
    if um in ("aparam", "atomic"):
        return "atomic"
    raise ValueError(f"unknown uparam mode: {u_mode}")


def _hubbard_card_lines(u_arr, u_mode, element, manifold, natoms, atom_labels=None,
                        uniform_fallback=False, u_init=0.0):
    """HUBBARD card body lines, or [] when no Hubbard correction applies."""
    if u_arr is None:
        if u_init and u_init > 0:
            return ["HUBBARD atomic", f"  U {element}-{manifold} {u_init:.6f}"]
        return []
    arr = np.atleast_1d(np.asarray(u_arr, dtype=float))
    per_atom = (
        u_mode == "atomic"
        and not uniform_fallback
        and atom_labels is not None
        and arr.size == natoms
    )
    if not per_atom:
        u_avg = float(arr.mean())
        if u_avg == 0.0:
            return []
        return ["HUBBARD atomic", f"  U {element}-{manifold} {u_avg:.6f}"]
    lines = ["HUBBARD atomic"]
    for u, label in zip(arr, atom_labels):
        lines.append(f"  U {label}-{manifold} {float(u):.6f}")
    return lines


def build_hubbard_card(u_arr, u_mode, element, manifold, natoms, atom_labels=None,
                       uniform_fallback=False, u_init=0.0):
    """Full HUBBARD card text (trailing newline) or ''."""
    body = _hubbard_card_lines(
        u_arr, u_mode, element, manifold, natoms, atom_labels, uniform_fallback, u_init
    )
    return "\n".join(body) + "\n" if body else ""


def _is_card_header_or_namelist(line):
    s = line.strip().upper()
    if s == "" or s.startswith("/"):
        return True
    if s.startswith("&"):
        return True
    return s.split()[0] in (
        "CELL_PARAMETERS",
        "ATOMIC_SPECIES",
        "ATOMIC_POSITIONS",
        "K_POINTS",
        "HUBBARD",
        "CONSTRAINTS",
        "OCCUPATIONS",
        "ATOMIC_FORCES",
        "ATOMIC_VELOCITIES",
        "SOLVENTS",
    )


def replace_hubbard_card(scf_in_text, new_card):
    """Replace (or append) the HUBBARD card in a pw.x input text."""
    lines = scf_in_text.splitlines()
    out = []
    i = 0
    replaced = False
    while i < len(lines):
        s = lines[i].strip()
        if not replaced and s.upper().startswith("HUBBARD"):
            i += 1
            while i < len(lines):
                hs = lines[i].strip()
                if hs == "" or _is_card_header_or_namelist(lines[i]):
                    break
                i += 1
            if new_card:
                out.append(new_card.rstrip("\n"))
            replaced = True
            continue
        out.append(lines[i])
        i += 1
    if not replaced and new_card:
        out.append(new_card.rstrip("\n"))
    return "\n".join(out) + "\n"


def generate_scf_input(
    cell_angstrom,
    atomic_positions_crystal,
    atom_type_indices,
    type_map,
    pseudo_dir,
    pseudo_files,
    degauss,
    ecutwfc,
    ecutrho,
    kpoints,
    mass_map=None,
    prefix="mlu",
    outdir="./tmp/",
    u_init=0.0,
    nspin=2,
    starting_magnetization=None,
    conv_thr=1e-5,
    mixing_beta=0.10,
    electron_maxstep=100,
    mixing_mode="local-TF",
    smearing="fd",
    verbosity="low",
    tprnfor=True,
    tstress=True,
    u_values=None,
    u_mode="frame",
    u_element=None,
    u_manifold="3d",
    aparam_uniform_fallback=False,
):
    """Generate a pw.x SCF input. See module docstring for U conventions.

    ``pseudo_files`` maps element -> filename; ``mass_map`` maps element -> mass.
    ``u_values``: frame scalar or per-atom array; ``u_init`` used when None.
    """
    if starting_magnetization is None:
        starting_magnetization = {}
    if mass_map is None:
        mass_map = {}

    cell = np.asarray(cell_angstrom, dtype=float)
    coords = np.asarray(atomic_positions_crystal, dtype=float)
    atom_type_indices = list(atom_type_indices)
    nat = len(atom_type_indices)
    base_element = u_element or type_map[atom_type_indices[0]]
    u_mode = _normalize_u_mode(u_mode)

    u_arr = None
    if u_values is not None:
        u_arr = np.atleast_1d(np.asarray(u_values, dtype=float))
    per_atom = (
        u_mode == "atomic"
        and not aparam_uniform_fallback
        and u_arr is not None
        and u_arr.size == nat
    )

    if per_atom:
        atom_labels = [f"{type_map[t]}{i + 1}" for i, t in enumerate(atom_type_indices)]
        species_labels = atom_labels
        ntyp = nat
    else:
        unique_types = sorted(set(atom_type_indices))
        unique_elements = [type_map[t] for t in unique_types]
        species_labels = unique_elements
        atom_labels = [type_map[t] for t in atom_type_indices]
        ntyp = len(unique_elements)

    def _mass(elem):
        return mass_map.get(elem, ATOMIC_MASSES.get(elem, 1.0))

    lines = []
    lines.append("&CONTROL")
    lines.append("  calculation = 'scf',")
    lines.append(f"  prefix = '{prefix}',")
    lines.append(f"  pseudo_dir = '{pseudo_dir}',")
    lines.append(f"  outdir = '{outdir}',")
    lines.append(f"  verbosity = '{verbosity}',")
    lines.append(f"  tprnfor = .{str(tprnfor).lower()}.,")
    lines.append(f"  tstress = .{str(tstress).lower()}.,")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append("  ibrav = 0,")
    lines.append(f"  nat = {nat},")
    lines.append(f"  ntyp = {ntyp},")
    lines.append(f"  ecutwfc = {ecutwfc},")
    lines.append(f"  ecutrho = {ecutrho},")
    lines.append("  occupations = 'smearing',")
    lines.append(f"  smearing = '{smearing}',")
    lines.append(f"  degauss = {degauss:.6f},")
    lines.append(f"  nspin = {nspin},")
    if per_atom:
        for i, label in enumerate(species_labels, start=1):
            mag = starting_magnetization.get(
                label, starting_magnetization.get(base_element, 0.0)
            )
            lines.append(f"  starting_magnetization({i}) = {mag},")
    else:
        for i, elem in enumerate(unique_elements):
            mag = starting_magnetization.get(elem, 0.0)
            lines.append(f"  starting_magnetization({i + 1}) = {mag},")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = {conv_thr:.1e},")
    lines.append(f"  mixing_beta = {mixing_beta:.2f},")
    lines.append(f"  electron_maxstep = {electron_maxstep},")
    lines.append(f"  mixing_mode = '{mixing_mode}',")
    lines.append("/")
    lines.append("CELL_PARAMETERS {angstrom}")
    for i in range(3):
        lines.append(f"  {cell[i, 0]:.6f}  {cell[i, 1]:.6f}  {cell[i, 2]:.6f}")
    lines.append("ATOMIC_SPECIES")
    for label in species_labels:
        elem_key = base_element if per_atom else label
        mass = _mass(elem_key)
        pseudo = pseudo_files.get(elem_key, f"{elem_key}.UPF")
        lines.append(f"  {label}  {mass}  {pseudo}")
    lines.append("ATOMIC_POSITIONS crystal")
    for idx in range(nat):
        lines.append(
            f"  {atom_labels[idx]}  "
            f"{coords[idx, 0]:.8f}  {coords[idx, 1]:.8f}  {coords[idx, 2]:.8f}"
        )
    k1, k2, k3 = kpoints
    lines.append("K_POINTS automatic")
    lines.append(f"  {k1} {k2} {k3} 0 0 0")
    hub_card = build_hubbard_card(
        u_arr,
        u_mode,
        base_element,
        u_manifold,
        nat,
        atom_labels if per_atom else None,
        uniform_fallback=aparam_uniform_fallback,
        u_init=u_init,
    )
    if hub_card:
        lines.append(hub_card.rstrip("\n"))
    return "\n".join(lines) + "\n"


def _parse_atomic_positions(scf_in_text):
    lines = scf_in_text.splitlines()
    coords = []
    in_block = False
    for line in lines:
        s = line.strip()
        if not in_block:
            if s.upper().startswith("ATOMIC_POSITIONS"):
                in_block = True
            continue
        if s == "" or s.startswith("/") or s.upper().startswith(
            (
                "CELL_PARAMETERS",
                "ATOMIC_SPECIES",
                "K_POINTS",
                "HUBBARD",
                "CONSTRAINTS",
                "OCCUPATIONS",
                "ATOMIC_FORCES",
                "ATOMIC_VELOCITIES",
                "&",
            )
        ):
            break
        parts = s.split()
        if len(parts) >= 4:
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                break
        else:
            break
    if not coords:
        return None, None
    return len(coords), np.array(coords, dtype=float)


def make_scf_new_input(
    scf_in_text,
    u_values,
    u_mode,
    element,
    manifold,
    natoms,
    uniform_fallback=False,
    starting_magnetization=None,
    mass_map=None,
    pseudo_file=None,
):
    """Rewrite a pw.x input with new Hubbard U values (the ``scf_new`` step).

    frame mode: only the HUBBARD card is replaced. atomic mode (no uniform
    fallback): full per-atom pseudo-species restructure (ntyp=nat, Fe1..FeN,
    per-species starting_magnetization, per-atom HUBBARD lines).
    """
    if starting_magnetization is None:
        starting_magnetization = {}
    if mass_map is None:
        mass_map = {}
    if pseudo_file is None:
        pseudo_file = f"{element}.UPF"
    u_mode = _normalize_u_mode(u_mode)
    arr = np.atleast_1d(np.asarray(u_values, dtype=float)) if u_values is not None else None
    per_atom = (
        u_mode == "atomic"
        and not uniform_fallback
        and arr is not None
        and arr.size == natoms
    )

    if not per_atom:
        new_card = build_hubbard_card(
            arr, u_mode, element, manifold, natoms, None, uniform_fallback
        )
        return replace_hubbard_card(scf_in_text, new_card)

    nat_parsed, coords = _parse_atomic_positions(scf_in_text)
    if nat_parsed is None or nat_parsed != natoms:
        raise ValueError(
            f"make_scf_new_input: could not parse {natoms} atomic positions from scf.in"
        )
    atom_labels = [f"{element}{i + 1}" for i in range(natoms)]

    lines = scf_in_text.splitlines()
    out = []
    i = 0
    in_system = False
    mag_injected = False
    while i < len(lines):
        s = lines[i].strip()
        up = s.upper()
        if up.startswith("&SYSTEM"):
            in_system = True
        elif up.startswith("&") and not up.startswith("&SYSTEM"):
            in_system = False
        if up.startswith("NTYP"):
            out.append(f"  ntyp = {natoms},")
            i += 1
            continue
        if in_system and re.match(r"STARTING_MAGNETIZATION\s*\(", up):
            i += 1
            continue
        if in_system and s == "/" and not mag_injected:
            for k, label in enumerate(atom_labels, start=1):
                mag = starting_magnetization.get(
                    label, starting_magnetization.get(element, 0.0)
                )
                out.append(f"  starting_magnetization({k}) = {mag},")
            mag_injected = True
            in_system = False
            out.append(lines[i])
            i += 1
            continue
        if up.startswith("ATOMIC_SPECIES"):
            i += 1
            while i < len(lines) and not _is_card_header_or_namelist(lines[i]):
                i += 1
            out.append("ATOMIC_SPECIES")
            for label in atom_labels:
                mass = mass_map.get(element, ATOMIC_MASSES.get(element, 1.0))
                out.append(f"  {label}  {mass}  {pseudo_file}")
            continue
        if up.startswith("ATOMIC_POSITIONS"):
            i += 1
            while i < len(lines) and not _is_card_header_or_namelist(lines[i]):
                i += 1
            out.append("ATOMIC_POSITIONS crystal")
            for idx in range(natoms):
                out.append(
                    f"  {atom_labels[idx]}  {coords[idx, 0]:.8f}  "
                    f"{coords[idx, 1]:.8f}  {coords[idx, 2]:.8f}"
                )
            continue
        if up.startswith("HUBBARD"):
            i += 1
            while i < len(lines) and not _is_card_header_or_namelist(lines[i]):
                hs = lines[i].strip()
                if hs == "" or _is_card_header_or_namelist(lines[i]):
                    break
                i += 1
            new_card = build_hubbard_card(
                arr, u_mode, element, manifold, natoms, atom_labels, uniform_fallback
            )
            if new_card:
                out.append(new_card.rstrip("\n"))
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out) + "\n"


def generate_hp_input(
    prefix="mlu",
    outdir="./tmp/",
    nq1=1,
    nq2=1,
    nq3=1,
    conv_thr_chi=1e-4,
    iverbosity=2,
    find_atpert=3,
    ethr_nscf=1e-6,
    niter_max=200,
    nmix=8,
):
    """Generate an hp.x input."""
    lines = [
        "&INPUTHP",
        f"  prefix = '{prefix}',",
        f"  outdir = '{outdir}',",
        f"  nq1 = {nq1},",
        f"  nq2 = {nq2},",
        f"  nq3 = {nq3},",
        f"  conv_thr_chi = {conv_thr_chi:.1e},",
        f"  iverbosity = {iverbosity},",
        f"  find_atpert = {find_atpert},",
        f"  ethr_nscf = {ethr_nscf:.1e},",
        f"  niter_max = {niter_max},",
        f"  nmix = {nmix},",
        "/",
    ]
    return "\n".join(lines) + "\n"
